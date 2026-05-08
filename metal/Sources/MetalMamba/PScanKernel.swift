// PScanKernel.swift
// MetalMamba
//
// Swift wrapper for Metal parallel scan kernel

import Metal
import Foundation

/// Configuration for the parallel scan kernel
public struct PScanConfig {
    public var batchSize: UInt32
    public var seqLen: UInt32
    public var dInner: UInt32
    public var dState: UInt32

    public init(batchSize: UInt32, seqLen: UInt32, dInner: UInt32, dState: UInt32) {
        self.batchSize = batchSize
        self.seqLen = seqLen
        self.dInner = dInner
        self.dState = dState
    }
}

/// Metal-accelerated parallel scan for Mamba SSM
public class PScanKernel {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var forwardPipeline: MTLComputePipelineState?
    private var backwardPipeline: MTLComputePipelineState?
    private var forwardSIMDPipeline: MTLComputePipelineState?

    // Cached configuration
    private var cachedConfig: PScanConfig?

    // Block size must match the shader
    private let blockSize: Int = 256

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw PScanError.noMetalDevice
        }
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else {
            throw PScanError.failedToCreateCommandQueue
        }
        self.commandQueue = commandQueue
    }

    /// Compile the kernels for a specific configuration
    public func compile(config: PScanConfig) throws {
        // Check if we can reuse cached pipelines
        if let cached = cachedConfig,
           cached.batchSize == config.batchSize,
           cached.seqLen == config.seqLen,
           cached.dInner == config.dInner,
           cached.dState == config.dState {
            return  // Already compiled
        }

        // Create function constants
        let constants = MTLFunctionConstantValues()
        var batchSize = config.batchSize
        var seqLen = config.seqLen
        var dInner = config.dInner
        var dState = config.dState

        constants.setConstantValue(&batchSize, type: .uint, index: 0)
        constants.setConstantValue(&seqLen, type: .uint, index: 1)
        constants.setConstantValue(&dInner, type: .uint, index: 2)
        constants.setConstantValue(&dState, type: .uint, index: 3)

        // Load shader source
        let shaderSource = try loadShaderSource()

        // Compile library
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: shaderSource, options: nil)
        } catch {
            // If runtime compilation fails (macOS 15+), try xcrun fallback
            library = try compileWithXcrun(source: shaderSource)
        }

        // Create pipelines
        let forwardFunc = try library.makeFunction(name: "pscan_forward", constantValues: constants)
        forwardPipeline = try device.makeComputePipelineState(function: forwardFunc)

        let backwardFunc = try library.makeFunction(name: "pscan_backward", constantValues: constants)
        backwardPipeline = try device.makeComputePipelineState(function: backwardFunc)

        let forwardSIMDFunc = try library.makeFunction(name: "pscan_forward_simd", constantValues: constants)
        forwardSIMDPipeline = try device.makeComputePipelineState(function: forwardSIMDFunc)

        cachedConfig = config

        print("PScan kernels compiled for B=\(config.batchSize), L=\(config.seqLen), D=\(config.dInner), N=\(config.dState)")
    }

    /// Run forward pass
    /// - Parameters:
    ///   - A: Decay coefficients (B, L, D, N) as MTLBuffer
    ///   - X: Input values (B, L, D, N) as MTLBuffer
    ///   - H: Output buffer (B, L, D, N) as MTLBuffer
    ///   - useSIMD: Use SIMD-optimized kernel (faster for seq_len <= 256)
    public func forward(A: MTLBuffer, X: MTLBuffer, H: MTLBuffer, useSIMD: Bool = true) throws {
        guard let config = cachedConfig else {
            throw PScanError.notCompiled
        }

        let pipeline = useSIMD ? forwardSIMDPipeline : forwardPipeline
        guard let pipeline = pipeline else {
            throw PScanError.notCompiled
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PScanError.failedToCreateCommandBuffer
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(A, offset: 0, index: 0)
        encoder.setBuffer(X, offset: 0, index: 1)
        encoder.setBuffer(H, offset: 0, index: 2)

        // Threadgroup memory for A and X partial results
        let tgMemSize = blockSize * MemoryLayout<Float>.size
        encoder.setThreadgroupMemoryLength(tgMemSize, index: 0)
        encoder.setThreadgroupMemoryLength(tgMemSize, index: 1)

        // Dispatch: one threadgroup per (batch, d_inner, d_state) combination
        let numThreadgroups = Int(config.batchSize * config.dInner * config.dState)
        let threadsPerGroup = MTLSize(width: blockSize, height: 1, depth: 1)
        let threadgroups = MTLSize(width: numThreadgroups, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PScanError.executionFailed(error.localizedDescription)
        }
    }

    /// Run backward pass
    public func backward(
        A: MTLBuffer,
        H: MTLBuffer,
        gradH: MTLBuffer,
        gradA: MTLBuffer,
        gradX: MTLBuffer
    ) throws {
        guard let config = cachedConfig,
              let pipeline = backwardPipeline else {
            throw PScanError.notCompiled
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PScanError.failedToCreateCommandBuffer
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(A, offset: 0, index: 0)
        encoder.setBuffer(H, offset: 0, index: 1)
        encoder.setBuffer(gradH, offset: 0, index: 2)
        encoder.setBuffer(gradA, offset: 0, index: 3)
        encoder.setBuffer(gradX, offset: 0, index: 4)

        let tgMemSize = blockSize * MemoryLayout<Float>.size
        encoder.setThreadgroupMemoryLength(tgMemSize, index: 0)
        encoder.setThreadgroupMemoryLength(tgMemSize, index: 1)

        let numThreadgroups = Int(config.batchSize * config.dInner * config.dState)
        let threadsPerGroup = MTLSize(width: blockSize, height: 1, depth: 1)
        let threadgroups = MTLSize(width: numThreadgroups, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PScanError.executionFailed(error.localizedDescription)
        }
    }

    // MARK: - Helper Methods

    private func loadShaderSource() throws -> String {
        // First, try to load from bundle
        if let url = Bundle.module.url(forResource: "pscan", withExtension: "metal"),
           let source = try? String(contentsOf: url) {
            return source
        }

        // Fallback: embedded source (for development)
        return Self.embeddedShaderSource
    }

    private func compileWithXcrun(source: String) throws -> MTLLibrary {
        // Write source to temp file
        let tempDir = FileManager.default.temporaryDirectory
        let sourceFile = tempDir.appendingPathComponent("pscan_\(UUID().uuidString).metal")
        let libFile = tempDir.appendingPathComponent("pscan_\(UUID().uuidString).metallib")

        try source.write(to: sourceFile, atomically: true, encoding: .utf8)

        // Compile with xcrun
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        process.arguments = [
            "-sdk", "macosx",
            "metal",
            "-c", sourceFile.path,
            "-o", libFile.path.replacingOccurrences(of: ".metallib", with: ".air")
        ]

        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            throw PScanError.compilationFailed("xcrun metal failed")
        }

        // Link
        let airFile = libFile.path.replacingOccurrences(of: ".metallib", with: ".air")
        let linkProcess = Process()
        linkProcess.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        linkProcess.arguments = [
            "-sdk", "macosx",
            "metallib",
            airFile,
            "-o", libFile.path
        ]

        try linkProcess.run()
        linkProcess.waitUntilExit()

        guard linkProcess.terminationStatus == 0 else {
            throw PScanError.compilationFailed("xcrun metallib failed")
        }

        // Load library
        let library = try device.makeLibrary(URL: libFile)

        // Cleanup
        try? FileManager.default.removeItem(at: sourceFile)
        try? FileManager.default.removeItem(at: URL(fileURLWithPath: airFile))
        try? FileManager.default.removeItem(at: libFile)

        return library
    }

    /// Create a Metal buffer from data
    public func makeBuffer<T>(from data: [T]) -> MTLBuffer? {
        let size = data.count * MemoryLayout<T>.stride
        return device.makeBuffer(bytes: data, length: size, options: .storageModeShared)
    }

    /// Create an empty Metal buffer
    public func makeBuffer(size: Int) -> MTLBuffer? {
        return device.makeBuffer(length: size, options: .storageModeShared)
    }

    // Embedded shader source for when bundle resource isn't available
    private static let embeddedShaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    constant uint BATCH_SIZE [[function_constant(0)]];
    constant uint SEQ_LEN [[function_constant(1)]];
    constant uint D_INNER [[function_constant(2)]];
    constant uint D_STATE [[function_constant(3)]];
    constant uint BLOCK_SIZE = 256;

    inline uint idx_4d(uint b, uint l, uint d, uint n, uint L, uint D, uint N) {
        return ((b * L + l) * D + d) * N + n;
    }

    kernel void pscan_forward(
        device const float* A [[buffer(0)]],
        device const float* X [[buffer(1)]],
        device float* H [[buffer(2)]],
        threadgroup float* tg_A [[threadgroup(0)]],
        threadgroup float* tg_X [[threadgroup(1)]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]
    ) {
        uint flat_idx = tgid.x;
        uint n = flat_idx % D_STATE;
        flat_idx /= D_STATE;
        uint d = flat_idx % D_INNER;
        uint b = flat_idx / D_INNER;
        if (b >= BATCH_SIZE) return;

        uint elems_per_thread = (SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE;
        float local_a = 1.0f, local_x = 0.0f;

        for (uint i = 0; i < elems_per_thread; i++) {
            uint l = tid * elems_per_thread + i;
            if (l < SEQ_LEN) {
                uint idx = idx_4d(b, l, d, n, SEQ_LEN, D_INNER, D_STATE);
                float a_val = A[idx], x_val = X[idx];
                local_x = a_val * local_x + x_val;
                local_a = a_val * local_a;
                H[idx] = local_x;
            }
        }

        tg_A[tid] = local_a;
        tg_X[tid] = local_x;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 1; stride < BLOCK_SIZE; stride *= 2) {
            uint idx = (tid + 1) * stride * 2 - 1;
            if (idx < BLOCK_SIZE) {
                uint left = idx - stride;
                tg_X[idx] = tg_X[idx] + tg_A[idx] * tg_X[left];
                tg_A[idx] = tg_A[idx] * tg_A[left];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint stride = BLOCK_SIZE / 4; stride > 0; stride /= 2) {
            uint idx = (tid + 1) * stride * 2 - 1;
            if (idx + stride < BLOCK_SIZE) {
                uint right = idx + stride;
                tg_X[right] = tg_X[right] + tg_A[right] * tg_X[idx];
                tg_A[right] = tg_A[right] * tg_A[idx];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid > 0) {
            float prefix_x = tg_X[tid - 1];
            float running_x = prefix_x;
            for (uint i = 0; i < elems_per_thread; i++) {
                uint l = tid * elems_per_thread + i;
                if (l < SEQ_LEN) {
                    uint idx = idx_4d(b, l, d, n, SEQ_LEN, D_INNER, D_STATE);
                    float a_val = A[idx], x_val = X[idx];
                    running_x = a_val * running_x + x_val;
                    H[idx] = running_x;
                }
            }
        }
    }

    kernel void pscan_backward(
        device const float* A [[buffer(0)]],
        device const float* H [[buffer(1)]],
        device const float* grad_H [[buffer(2)]],
        device float* grad_A [[buffer(3)]],
        device float* grad_X [[buffer(4)]],
        threadgroup float* tg_A [[threadgroup(0)]],
        threadgroup float* tg_grad [[threadgroup(1)]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]
    ) {
        uint flat_idx = tgid.x;
        uint n = flat_idx % D_STATE;
        flat_idx /= D_STATE;
        uint d = flat_idx % D_INNER;
        uint b = flat_idx / D_INNER;
        if (b >= BATCH_SIZE) return;

        uint elems_per_thread = (SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE;
        float local_a = 1.0f, local_grad = 0.0f;

        for (int i = elems_per_thread - 1; i >= 0; i--) {
            uint l = tid * elems_per_thread + i;
            if (l < SEQ_LEN) {
                uint idx = idx_4d(b, l, d, n, SEQ_LEN, D_INNER, D_STATE);
                float a_val = (l + 1 < SEQ_LEN) ? A[idx_4d(b, l+1, d, n, SEQ_LEN, D_INNER, D_STATE)] : 0.0f;
                float gh_val = grad_H[idx];
                local_grad = gh_val + a_val * local_grad;
                local_a = a_val * local_a;
                grad_X[idx] = local_grad;
                grad_A[idx] = (l > 0) ? grad_H[idx] * H[idx_4d(b, l-1, d, n, SEQ_LEN, D_INNER, D_STATE)] : 0.0f;
            }
        }

        tg_A[tid] = local_a;
        tg_grad[tid] = local_grad;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    inline float2 simd_scan_step(float a, float x, uint lane, uint delta) {
        float other_a = simd_shuffle_up(a, delta);
        float other_x = simd_shuffle_up(x, delta);
        if (lane >= delta) { x = x + a * other_x; a = a * other_a; }
        return float2(a, x);
    }

    inline float2 simd_inclusive_scan(float a, float x, uint lane) {
        float2 r;
        r = simd_scan_step(a, x, lane, 1); a = r.x; x = r.y;
        r = simd_scan_step(a, x, lane, 2); a = r.x; x = r.y;
        r = simd_scan_step(a, x, lane, 4); a = r.x; x = r.y;
        r = simd_scan_step(a, x, lane, 8); a = r.x; x = r.y;
        r = simd_scan_step(a, x, lane, 16); a = r.x; x = r.y;
        return float2(a, x);
    }

    kernel void pscan_forward_simd(
        device const float* A [[buffer(0)]],
        device const float* X [[buffer(1)]],
        device float* H [[buffer(2)]],
        threadgroup float* tg_A [[threadgroup(0)]],
        threadgroup float* tg_X [[threadgroup(1)]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint simd_lane [[thread_index_in_simdgroup]],
        uint simd_idx [[simdgroup_index_in_threadgroup]]
    ) {
        uint flat_idx = tgid.x;
        uint n = flat_idx % D_STATE;
        flat_idx /= D_STATE;
        uint d = flat_idx % D_INNER;
        uint b = flat_idx / D_INNER;
        if (b >= BATCH_SIZE) return;

        constexpr uint SIMD_SIZE = 32;
        uint num_simds = BLOCK_SIZE / SIMD_SIZE;
        uint chunk_start = simd_idx * SIMD_SIZE;
        uint l = chunk_start + simd_lane;

        float a_val = 1.0f, x_val = 0.0f;
        if (l < SEQ_LEN) {
            uint idx = idx_4d(b, l, d, n, SEQ_LEN, D_INNER, D_STATE);
            a_val = A[idx]; x_val = X[idx];
        }

        float2 scanned = simd_inclusive_scan(a_val, x_val, simd_lane);
        float scan_a = scanned.x, scan_x = scanned.y;

        if (l < SEQ_LEN) {
            uint idx = idx_4d(b, l, d, n, SEQ_LEN, D_INNER, D_STATE);
            H[idx] = scan_x;
        }

        if (simd_lane == SIMD_SIZE - 1) {
            tg_A[simd_idx] = scan_a;
            tg_X[simd_idx] = scan_x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_idx == 0 && simd_lane < num_simds) {
            float2 cs = simd_inclusive_scan(tg_A[simd_lane], tg_X[simd_lane], simd_lane);
            tg_A[simd_lane] = cs.x;
            tg_X[simd_lane] = cs.y;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_idx > 0 && l < SEQ_LEN) {
            uint idx = idx_4d(b, l, d, n, SEQ_LEN, D_INNER, D_STATE);
            H[idx] = scan_x + scan_a * tg_X[simd_idx - 1];
        }
    }
    """
}

// MARK: - Errors

public enum PScanError: Error {
    case noMetalDevice
    case failedToCreateCommandQueue
    case failedToCreateCommandBuffer
    case notCompiled
    case compilationFailed(String)
    case executionFailed(String)
}
