// metal_pscan.mm - PyTorch C++ extension for Metal parallel scan
// Direct MPS tensor access - no CPU copies!

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <string>
#include <unordered_map>

// Get MTLBuffer from PyTorch MPS tensor using PyTorch's MPS utilities
static id<MTLBuffer> getMTLBuffer(const torch::Tensor& t) {
    return at::native::mps::getMTLBufferStorage(t);
}

// Get the default Metal device
static id<MTLDevice> getMetalDevice() {
    return MTLCreateSystemDefaultDevice();
}

// Kernel cache
static std::unordered_map<std::string, id<MTLComputePipelineState>> pipelineCache;
static id<MTLLibrary> metalLibrary = nil;

// Embedded Metal shader source
static const char* metalShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

// Function constants for shape
constant uint BATCH_SIZE [[function_constant(0)]];
constant uint SEQ_LEN [[function_constant(1)]];
constant uint D_INNER [[function_constant(2)]];
constant uint D_STATE [[function_constant(3)]];

inline uint idx_4d(uint b, uint l, uint d, uint n, uint L, uint D, uint N) {
    return ((b * L + l) * D + d) * N + n;
}

// Forward pass: one thread per (b, d, n) slice, sequential scan across L
kernel void pscan_forward(
    device const float* A [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* H [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint flat_idx = tgid.x * 256 + tid;
    uint n = flat_idx % D_STATE;
    flat_idx /= D_STATE;
    uint d = flat_idx % D_INNER;
    uint b = flat_idx / D_INNER;

    if (b >= BATCH_SIZE) return;

    float h = 0.0f;
    for (uint l = 0; l < SEQ_LEN; l++) {
        uint idx = idx_4d(b, l, d, n, SEQ_LEN, D_INNER, D_STATE);
        h = A[idx] * h + X[idx];
        H[idx] = h;
    }
}

// Fused SSM: ssm_prep + pscan in one kernel
// Computes: deltaA = exp(delta * A), BX = delta * B * x, H = pscan(deltaA, BX)
// Inputs:
//   delta: (B, L, D) - dt after softplus
//   A: (D, N) - state matrix (negative log form, we negate and exp)
//   B_ssm: (B, L, N) - input-dependent B
//   x: (B, L, D) - input after conv+silu
// Output:
//   H: (B, L, D, N) - hidden states
kernel void ssm_fused(
    device const float* delta [[buffer(0)]],   // (B, L, D)
    device const float* A [[buffer(1)]],       // (D, N) - already negated (-exp(A_log))
    device const float* B_ssm [[buffer(2)]],   // (B, L, N)
    device const float* x [[buffer(3)]],       // (B, L, D)
    device float* H [[buffer(4)]],             // (B, L, D, N)
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    // One thread per (b, d, n) - scans across L
    uint flat_idx = tgid.x * 256 + tid;
    uint n = flat_idx % D_STATE;
    flat_idx /= D_STATE;
    uint d = flat_idx % D_INNER;
    uint b = flat_idx / D_INNER;

    if (b >= BATCH_SIZE) return;

    // A is (D, N), index as A[d * N + n]
    float A_dn = A[d * D_STATE + n];

    float h = 0.0f;
    for (uint l = 0; l < SEQ_LEN; l++) {
        // delta is (B, L, D), index as delta[(b * L + l) * D + d]
        uint delta_idx = (b * SEQ_LEN + l) * D_INNER + d;
        float delta_val = delta[delta_idx];

        // deltaA = exp(delta * A), clamp to prevent overflow
        float da_arg = delta_val * A_dn;
        da_arg = clamp(da_arg, -20.0f, 20.0f);
        float deltaA = exp(da_arg);

        // B_ssm is (B, L, N), index as B_ssm[(b * L + l) * N + n]
        uint B_idx = (b * SEQ_LEN + l) * D_STATE + n;
        float B_val = B_ssm[B_idx];

        // x is (B, L, D), same indexing as delta
        float x_val = x[delta_idx];

        // BX = delta * B * x
        float BX = delta_val * B_val * x_val;

        // Scan: h = deltaA * h + BX
        h = deltaA * h + BX;

        // H is (B, L, D, N)
        uint H_idx = ((b * SEQ_LEN + l) * D_INNER + d) * D_STATE + n;
        H[H_idx] = h;
    }
}

// Super-fused SSM: ssm_prep + pscan + output matmul in one kernel
// Computes: y[b,l,d] = sum_n(h[b,l,d,n] * C[b,l,n]) + D[d] * x[b,l,d]
// One thread per (b, d) - loops over L, sums over N
kernel void ssm_output_fused(
    device const float* delta [[buffer(0)]],   // (B, L, D)
    device const float* A [[buffer(1)]],       // (D, N)
    device const float* B_ssm [[buffer(2)]],   // (B, L, N)
    device const float* x [[buffer(3)]],       // (B, L, D)
    device const float* C_ssm [[buffer(4)]],   // (B, L, N)
    device const float* D_param [[buffer(5)]], // (D)
    device float* y_out [[buffer(6)]],         // (B, L, D)
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    // One thread per (b, d) - scans across L, sums over N
    uint flat_idx = tgid.x * 256 + tid;
    uint d = flat_idx % D_INNER;
    uint b = flat_idx / D_INNER;

    if (b >= BATCH_SIZE) return;

    float D_d = D_param[d];

    // Hidden state for all N dimensions
    float h[32];  // max D_STATE = 32
    for (uint n = 0; n < D_STATE; n++) {
        h[n] = 0.0f;
    }

    // Load A values for this d
    float A_d[32];
    for (uint n = 0; n < D_STATE; n++) {
        A_d[n] = A[d * D_STATE + n];
    }

    for (uint l = 0; l < SEQ_LEN; l++) {
        uint delta_idx = (b * SEQ_LEN + l) * D_INNER + d;
        float delta_val = delta[delta_idx];
        float x_val = x[delta_idx];

        // Compute output: sum over N
        float y_val = 0.0f;
        for (uint n = 0; n < D_STATE; n++) {
            // deltaA = exp(delta * A), clamp to prevent overflow
            float da_arg = delta_val * A_d[n];
            da_arg = clamp(da_arg, -20.0f, 20.0f);  // exp(20) ~ 485M, safe for float32
            float deltaA = exp(da_arg);

            // B_ssm is (B, L, N)
            uint B_idx = (b * SEQ_LEN + l) * D_STATE + n;
            float B_val = B_ssm[B_idx];

            // BX = delta * B * x
            float BX = delta_val * B_val * x_val;

            // Scan: h = deltaA * h + BX
            h[n] = deltaA * h[n] + BX;

            // C_ssm is (B, L, N)
            uint C_idx = (b * SEQ_LEN + l) * D_STATE + n;
            float C_val = C_ssm[C_idx];

            // Accumulate: y += h * C
            y_val += h[n] * C_val;
        }

        // Add skip connection: y += D * x
        y_val += D_d * x_val;

        // Write output
        y_out[delta_idx] = y_val;
    }
}

// Fused depthwise conv1d + SiLU
// PyTorch's conv1d on MPS is slow for depthwise convolutions
// Input: (B, D, L) - channels first
// Output: (B, D, L) - after conv + silu
// Conv is depthwise (groups=D), kernel_size=D_CONV, padding=D_CONV-1, then slice [:,:,:L]

constant uint D_CONV [[function_constant(4)]];

inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

kernel void conv1d_silu_fused(
    device const float* x_in [[buffer(0)]],        // (B, D, L) - input
    device const float* weight [[buffer(1)]],      // (D, 1, D_CONV) - depthwise weights
    device const float* bias [[buffer(2)]],        // (D) - bias
    device float* y_out [[buffer(3)]],             // (B, D, L) - output
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    // One thread per (b, d) - processes all L positions
    uint flat_idx = tgid.x * 256 + tid;
    uint d = flat_idx % D_INNER;
    uint b = flat_idx / D_INNER;

    if (b >= BATCH_SIZE) return;

    // Load conv weights for this channel
    float w[8];  // max D_CONV = 8
    for (uint k = 0; k < D_CONV && k < 8; k++) {
        w[k] = weight[d * D_CONV + k];
    }
    float b_val = bias[d];

    // Circular buffer for causal conv (pad with zeros on left)
    float buf[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // Process each position
    for (uint l = 0; l < SEQ_LEN; l++) {
        // Read input: x_in is (B, D, L), index = b*D*L + d*L + l
        uint in_idx = b * D_INNER * SEQ_LEN + d * SEQ_LEN + l;
        float x_val = x_in[in_idx];

        // Shift buffer left and insert new value
        for (uint k = 0; k < D_CONV - 1; k++) {
            buf[k] = buf[k + 1];
        }
        buf[D_CONV - 1] = x_val;

        // Compute conv output
        float conv_out = b_val;
        for (uint k = 0; k < D_CONV; k++) {
            conv_out += w[k] * buf[k];
        }

        // Apply SiLU and write output
        y_out[in_idx] = silu(conv_out);
    }
}

// Backward pass: one thread per (b, d, n) slice
kernel void pscan_backward(
    device const float* A [[buffer(0)]],
    device const float* H [[buffer(1)]],
    device const float* grad_H [[buffer(2)]],
    device float* grad_A [[buffer(3)]],
    device float* grad_X [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint flat_idx = tgid.x * 256 + tid;
    uint n = flat_idx % D_STATE;
    flat_idx /= D_STATE;
    uint d = flat_idx % D_INNER;
    uint b = flat_idx / D_INNER;

    if (b >= BATCH_SIZE) return;

    // Reverse scan for grad_X: grad_X[t] = grad_H[t] + A[t+1] * grad_X[t+1]
    float grad_x_acc = 0.0f;
    for (int l = SEQ_LEN - 1; l >= 0; l--) {
        uint idx = idx_4d(b, l, d, n, SEQ_LEN, D_INNER, D_STATE);
        float gh = grad_H[idx];
        grad_x_acc = gh + (l + 1 < SEQ_LEN ? A[idx_4d(b, l+1, d, n, SEQ_LEN, D_INNER, D_STATE)] : 0.0f) * grad_x_acc;
        grad_X[idx] = grad_x_acc;

        // grad_A[t] = grad_H[t] * H[t-1]
        grad_A[idx] = (l > 0) ? gh * H[idx_4d(b, l-1, d, n, SEQ_LEN, D_INNER, D_STATE)] : 0.0f;
    }
}
)";

// Compile shader with function constants
static id<MTLComputePipelineState> getOrCreatePipeline(
    id<MTLDevice> device,
    const std::string& kernelName,
    uint32_t B, uint32_t L, uint32_t D, uint32_t N
) {
    std::string key = kernelName + "_" + std::to_string(B) + "_" +
                      std::to_string(L) + "_" + std::to_string(D) + "_" + std::to_string(N);

    auto it = pipelineCache.find(key);
    if (it != pipelineCache.end()) {
        return it->second;
    }

    NSError* error = nil;

    // Compile library if needed
    if (metalLibrary == nil) {
        NSString* source = [NSString stringWithUTF8String:metalShaderSource];
        metalLibrary = [device newLibraryWithSource:source options:nil error:&error];
        if (error) {
            throw std::runtime_error("Failed to compile Metal library: " +
                std::string([[error localizedDescription] UTF8String]));
        }
    }

    // Create function constants
    MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&B type:MTLDataTypeUInt atIndex:0];
    [constants setConstantValue:&L type:MTLDataTypeUInt atIndex:1];
    [constants setConstantValue:&D type:MTLDataTypeUInt atIndex:2];
    [constants setConstantValue:&N type:MTLDataTypeUInt atIndex:3];

    // Get function
    NSString* funcName = [NSString stringWithUTF8String:kernelName.c_str()];
    id<MTLFunction> function = [metalLibrary newFunctionWithName:funcName
                                                  constantValues:constants
                                                           error:&error];
    if (error) {
        throw std::runtime_error("Failed to create function: " +
            std::string([[error localizedDescription] UTF8String]));
    }

    // Create pipeline
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (error) {
        throw std::runtime_error("Failed to create pipeline: " +
            std::string([[error localizedDescription] UTF8String]));
    }

    pipelineCache[key] = pipeline;
    return pipeline;
}

// Forward pass
torch::Tensor metal_pscan_forward(
    torch::Tensor A,  // (B, L, D, N)
    torch::Tensor X   // (B, L, D, N)
) {
    TORCH_CHECK(A.device().is_mps(), "A must be on MPS device");
    TORCH_CHECK(X.device().is_mps(), "X must be on MPS device");
    TORCH_CHECK(A.sizes() == X.sizes(), "A and X must have same shape");
    TORCH_CHECK(A.dim() == 4, "Expected 4D tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "Expected float32");

    // Make contiguous
    A = A.contiguous();
    X = X.contiguous();

    auto B = A.size(0);
    auto L = A.size(1);
    auto D = A.size(2);
    auto N = A.size(3);

    // Create output tensor
    auto H = torch::empty_like(X);

    // Get Metal resources
    id<MTLDevice> device = getMetalDevice();

    // Use simple sequential kernel for all lengths (still faster than PyTorch)
    std::string kernelName = "pscan_forward";
    auto pipeline = getOrCreatePipeline(device, kernelName, B, L, D, N);

    // Get MPS stream and use PyTorch's shared command encoder (zero-sync)
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        [encoder setComputePipelineState:pipeline];

        id<MTLBuffer> A_buf = getMTLBuffer(A);
        id<MTLBuffer> X_buf = getMTLBuffer(X);
        id<MTLBuffer> H_buf = getMTLBuffer(H);

        [encoder setBuffer:A_buf offset:A.storage_offset() * sizeof(float) atIndex:0];
        [encoder setBuffer:X_buf offset:X.storage_offset() * sizeof(float) atIndex:1];
        [encoder setBuffer:H_buf offset:H.storage_offset() * sizeof(float) atIndex:2];

        uint32_t blockSize = 256;
        uint32_t totalThreads = B * D * N;
        uint32_t numThreadgroups = (totalThreads + blockSize - 1) / blockSize;

        [encoder dispatchThreadgroups:MTLSizeMake(numThreadgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(blockSize, 1, 1)];

        // Don't endEncoding/commit - PyTorch manages encoder lifecycle
    }

    return H;
}

// Backward pass
std::vector<torch::Tensor> metal_pscan_backward(
    torch::Tensor A,
    torch::Tensor X,
    torch::Tensor H,
    torch::Tensor grad_H
) {
    TORCH_CHECK(A.device().is_mps(), "A must be on MPS device");
    TORCH_CHECK(H.device().is_mps(), "H must be on MPS device");
    TORCH_CHECK(grad_H.device().is_mps(), "grad_H must be on MPS device");

    A = A.contiguous();
    H = H.contiguous();
    grad_H = grad_H.contiguous();

    auto B = A.size(0);
    auto L = A.size(1);
    auto D = A.size(2);
    auto N = A.size(3);

    auto grad_A = torch::empty_like(A);
    auto grad_X = torch::empty_like(A);

    id<MTLDevice> device = getMetalDevice();
    auto pipeline = getOrCreatePipeline(device, "pscan_backward", B, L, D, N);

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        [encoder setComputePipelineState:pipeline];

        [encoder setBuffer:getMTLBuffer(A) offset:A.storage_offset() * sizeof(float) atIndex:0];
        [encoder setBuffer:getMTLBuffer(H) offset:H.storage_offset() * sizeof(float) atIndex:1];
        [encoder setBuffer:getMTLBuffer(grad_H) offset:grad_H.storage_offset() * sizeof(float) atIndex:2];
        [encoder setBuffer:getMTLBuffer(grad_A) offset:grad_A.storage_offset() * sizeof(float) atIndex:3];
        [encoder setBuffer:getMTLBuffer(grad_X) offset:grad_X.storage_offset() * sizeof(float) atIndex:4];

        uint32_t blockSize = 256;
        uint32_t totalThreads = B * D * N;
        uint32_t numThreadgroups = (totalThreads + blockSize - 1) / blockSize;

        [encoder dispatchThreadgroups:MTLSizeMake(numThreadgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(blockSize, 1, 1)];

        // Don't endEncoding/commit - PyTorch manages encoder lifecycle
    }

    return {grad_A, grad_X};
}

// Fused SSM forward: ssm_prep + pscan in one kernel - async version
torch::Tensor metal_ssm_fused(
    torch::Tensor delta,  // (B, L, D) - after softplus
    torch::Tensor A,      // (D, N) - negative state matrix
    torch::Tensor B_ssm,  // (B, L, N)
    torch::Tensor x       // (B, L, D)
) {
    TORCH_CHECK(delta.device().is_mps(), "delta must be on MPS device");
    TORCH_CHECK(A.device().is_mps(), "A must be on MPS device");
    TORCH_CHECK(B_ssm.device().is_mps(), "B_ssm must be on MPS device");
    TORCH_CHECK(x.device().is_mps(), "x must be on MPS device");

    // Cast to float32 if needed (autocast may send float16)
    delta = delta.to(torch::kFloat32).contiguous();
    A = A.to(torch::kFloat32).contiguous();
    B_ssm = B_ssm.to(torch::kFloat32).contiguous();
    x = x.to(torch::kFloat32).contiguous();

    auto B = delta.size(0);
    auto L = delta.size(1);
    auto D = delta.size(2);
    auto N = A.size(1);

    TORCH_CHECK(A.size(0) == D, "A shape mismatch");
    TORCH_CHECK(B_ssm.size(0) == B && B_ssm.size(1) == L && B_ssm.size(2) == N, "B_ssm shape mismatch");
    TORCH_CHECK(x.size(0) == B && x.size(1) == L && x.size(2) == D, "x shape mismatch");

    // Output: (B, L, D, N)
    auto H = torch::empty({B, L, D, N}, delta.options());

    id<MTLDevice> device = getMetalDevice();
    auto pipeline = getOrCreatePipeline(device, "ssm_fused", B, L, D, N);

    // Use MPS stream with PyTorch's shared encoder (zero-sync)
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        [encoder setComputePipelineState:pipeline];

        [encoder setBuffer:getMTLBuffer(delta) offset:delta.storage_offset() * sizeof(float) atIndex:0];
        [encoder setBuffer:getMTLBuffer(A) offset:A.storage_offset() * sizeof(float) atIndex:1];
        [encoder setBuffer:getMTLBuffer(B_ssm) offset:B_ssm.storage_offset() * sizeof(float) atIndex:2];
        [encoder setBuffer:getMTLBuffer(x) offset:x.storage_offset() * sizeof(float) atIndex:3];
        [encoder setBuffer:getMTLBuffer(H) offset:H.storage_offset() * sizeof(float) atIndex:4];

        uint32_t blockSize = 256;
        uint32_t totalThreads = B * D * N;
        uint32_t numThreadgroups = (totalThreads + blockSize - 1) / blockSize;

        [encoder dispatchThreadgroups:MTLSizeMake(numThreadgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(blockSize, 1, 1)];

        // Don't endEncoding/commit - PyTorch manages encoder lifecycle
    }

    return H;
}

bool metal_pscan_available() {
    return MTLCreateSystemDefaultDevice() != nil;
}

// Fused conv1d + silu - async version using MPS stream
torch::Tensor metal_conv1d_silu(
    torch::Tensor x,       // (B, D, L) - input
    torch::Tensor weight,  // (D, 1, d_conv) - depthwise weights
    torch::Tensor bias     // (D) - bias
) {
    TORCH_CHECK(x.device().is_mps(), "x must be on MPS device");

    // Cast to float32 if needed (autocast may send float16)
    x = x.to(torch::kFloat32).contiguous();
    weight = weight.to(torch::kFloat32).contiguous();
    bias = bias.to(torch::kFloat32).contiguous();

    auto B = x.size(0);
    auto D = x.size(1);
    auto L = x.size(2);
    auto d_conv = weight.size(2);

    auto y = torch::empty_like(x);

    id<MTLDevice> device = getMetalDevice();

    // Need to compile with d_conv as function constant
    std::string key = "conv1d_silu_" + std::to_string(B) + "_" +
                      std::to_string(L) + "_" + std::to_string(D) + "_" +
                      std::to_string(16) + "_" + std::to_string(d_conv);

    id<MTLComputePipelineState> pipeline;
    auto it = pipelineCache.find(key);
    if (it != pipelineCache.end()) {
        pipeline = it->second;
    } else {
        NSError* error = nil;
        if (metalLibrary == nil) {
            NSString* source = [NSString stringWithUTF8String:metalShaderSource];
            metalLibrary = [device newLibraryWithSource:source options:nil error:&error];
            if (error) {
                throw std::runtime_error("Failed to compile Metal library: " +
                    std::string([[error localizedDescription] UTF8String]));
            }
        }

        MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
        uint32_t B32 = B, L32 = L, D32 = D, N32 = 16, dc32 = d_conv;
        [constants setConstantValue:&B32 type:MTLDataTypeUInt atIndex:0];
        [constants setConstantValue:&L32 type:MTLDataTypeUInt atIndex:1];
        [constants setConstantValue:&D32 type:MTLDataTypeUInt atIndex:2];
        [constants setConstantValue:&N32 type:MTLDataTypeUInt atIndex:3];
        [constants setConstantValue:&dc32 type:MTLDataTypeUInt atIndex:4];

        id<MTLFunction> function = [metalLibrary newFunctionWithName:@"conv1d_silu_fused"
                                                      constantValues:constants
                                                               error:&error];
        if (error) {
            throw std::runtime_error("Failed to create conv1d function: " +
                std::string([[error localizedDescription] UTF8String]));
        }

        pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (error) {
            throw std::runtime_error("Failed to create conv1d pipeline: " +
                std::string([[error localizedDescription] UTF8String]));
        }
        pipelineCache[key] = pipeline;
    }

    // Use MPS stream with PyTorch's shared encoder (zero-sync)
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        [encoder setComputePipelineState:pipeline];

        [encoder setBuffer:getMTLBuffer(x) offset:x.storage_offset() * sizeof(float) atIndex:0];
        [encoder setBuffer:getMTLBuffer(weight) offset:weight.storage_offset() * sizeof(float) atIndex:1];
        [encoder setBuffer:getMTLBuffer(bias) offset:bias.storage_offset() * sizeof(float) atIndex:2];
        [encoder setBuffer:getMTLBuffer(y) offset:y.storage_offset() * sizeof(float) atIndex:3];

        uint32_t blockSize = 256;
        uint32_t totalThreads = B * D;
        uint32_t numThreadgroups = (totalThreads + blockSize - 1) / blockSize;

        [encoder dispatchThreadgroups:MTLSizeMake(numThreadgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(blockSize, 1, 1)];

        // Don't endEncoding/commit - PyTorch manages encoder lifecycle
    }

    return y;
}

// Super-fused SSM + output: ssm_prep + pscan + output matmul in one kernel
// Returns y directly instead of H
torch::Tensor metal_ssm_output_fused(
    torch::Tensor delta,  // (B, L, D) - after softplus
    torch::Tensor A,      // (D, N) - negative state matrix
    torch::Tensor B_ssm,  // (B, L, N)
    torch::Tensor x,      // (B, L, D)
    torch::Tensor C_ssm,  // (B, L, N)
    torch::Tensor D_param // (D)
) {
    TORCH_CHECK(delta.device().is_mps(), "delta must be on MPS device");

    // Cast to float32 if needed (autocast may send float16)
    delta = delta.to(torch::kFloat32).contiguous();
    A = A.to(torch::kFloat32).contiguous();
    B_ssm = B_ssm.to(torch::kFloat32).contiguous();
    x = x.to(torch::kFloat32).contiguous();
    C_ssm = C_ssm.to(torch::kFloat32).contiguous();
    D_param = D_param.to(torch::kFloat32).contiguous();

    auto B = delta.size(0);
    auto L = delta.size(1);
    auto D = delta.size(2);
    auto N = A.size(1);

    // Output: (B, L, D) - directly the y output
    auto y = torch::empty({B, L, D}, delta.options());

    id<MTLDevice> device = getMetalDevice();
    auto pipeline = getOrCreatePipeline(device, "ssm_output_fused", B, L, D, N);

    // Use MPS stream with PyTorch's shared encoder (zero-sync)
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        [encoder setComputePipelineState:pipeline];

        [encoder setBuffer:getMTLBuffer(delta) offset:delta.storage_offset() * sizeof(float) atIndex:0];
        [encoder setBuffer:getMTLBuffer(A) offset:A.storage_offset() * sizeof(float) atIndex:1];
        [encoder setBuffer:getMTLBuffer(B_ssm) offset:B_ssm.storage_offset() * sizeof(float) atIndex:2];
        [encoder setBuffer:getMTLBuffer(x) offset:x.storage_offset() * sizeof(float) atIndex:3];
        [encoder setBuffer:getMTLBuffer(C_ssm) offset:C_ssm.storage_offset() * sizeof(float) atIndex:4];
        [encoder setBuffer:getMTLBuffer(D_param) offset:D_param.storage_offset() * sizeof(float) atIndex:5];
        [encoder setBuffer:getMTLBuffer(y) offset:y.storage_offset() * sizeof(float) atIndex:6];

        uint32_t blockSize = 256;
        uint32_t totalThreads = B * D;  // One thread per (b, d)
        uint32_t numThreadgroups = (totalThreads + blockSize - 1) / blockSize;

        [encoder dispatchThreadgroups:MTLSizeMake(numThreadgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(blockSize, 1, 1)];

        // Don't endEncoding/commit - PyTorch manages encoder lifecycle
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &metal_pscan_forward, "Metal PScan forward");
    m.def("backward", &metal_pscan_backward, "Metal PScan backward");
    m.def("ssm_fused", &metal_ssm_fused, "Fused SSM (ssm_prep + pscan)");
    m.def("ssm_output_fused", &metal_ssm_output_fused, "Super-fused SSM + output");
    m.def("conv1d_silu", &metal_conv1d_silu, "Fused conv1d + silu");
    m.def("is_available", &metal_pscan_available, "Check Metal availability");
}
