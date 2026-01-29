// TestPScan - Test the Metal parallel scan kernel

import Foundation
import MetalMamba

print("=== Metal Mamba PScan Test ===\n")

// Test configuration
let B: UInt32 = 2      // Batch size
let L: UInt32 = 64     // Sequence length
let D: UInt32 = 128    // d_inner
let N: UInt32 = 16     // d_state

let totalElements = Int(B * L * D * N)

print("Configuration:")
print("  Batch size:  \(B)")
print("  Seq length:  \(L)")
print("  d_inner:     \(D)")
print("  d_state:     \(N)")
print("  Total elems: \(totalElements)")
print()

// Initialize kernel
do {
    let kernel = try PScanKernel()
    let config = PScanConfig(batchSize: B, seqLen: L, dInner: D, dState: N)
    try kernel.compile(config: config)
    print("Kernel compiled successfully!")

    // Create test data
    // A values should be in (0, 1) for numerical stability
    var A_data = [Float](repeating: 0, count: totalElements)
    var X_data = [Float](repeating: 0, count: totalElements)

    for i in 0..<totalElements {
        A_data[i] = Float.random(in: 0.5...0.99)  // Decay factors
        X_data[i] = Float.random(in: -1...1)      // Input values
    }

    // Create buffers
    guard let A_buf = kernel.makeBuffer(from: A_data),
          let X_buf = kernel.makeBuffer(from: X_data),
          let H_buf = kernel.makeBuffer(size: totalElements * MemoryLayout<Float>.size) else {
        print("Failed to create buffers")
        exit(1)
    }

    print("\nRunning forward pass...")

    // Run forward pass
    let start = CFAbsoluteTimeGetCurrent()
    try kernel.forward(A: A_buf, X: X_buf, H: H_buf, useSIMD: true)
    let elapsed = CFAbsoluteTimeGetCurrent() - start

    print("Forward pass completed in \(String(format: "%.3f", elapsed * 1000)) ms")

    // Read results
    let H_ptr = H_buf.contents().bindMemory(to: Float.self, capacity: totalElements)
    let H_result = Array(UnsafeBufferPointer(start: H_ptr, count: totalElements))

    // Verify against CPU reference implementation
    print("\nVerifying against CPU reference...")

    var H_ref = [Float](repeating: 0, count: totalElements)

    // CPU reference: sequential scan
    // For each (b, d, n), scan across L
    for b in 0..<Int(B) {
        for d in 0..<Int(D) {
            for n in 0..<Int(N) {
                var h: Float = 0
                for l in 0..<Int(L) {
                    let idx = ((b * Int(L) + l) * Int(D) + d) * Int(N) + n
                    h = A_data[idx] * h + X_data[idx]
                    H_ref[idx] = h
                }
            }
        }
    }

    // Compare results
    var maxError: Float = 0
    var totalError: Float = 0
    var errorCount = 0

    for i in 0..<totalElements {
        let error = abs(H_result[i] - H_ref[i])
        maxError = max(maxError, error)
        totalError += error

        if error > 1e-4 {
            errorCount += 1
            if errorCount <= 5 {
                print("  Mismatch at \(i): GPU=\(H_result[i]), CPU=\(H_ref[i]), error=\(error)")
            }
        }
    }

    let avgError = totalError / Float(totalElements)

    print("\nResults:")
    print("  Max error:   \(maxError)")
    print("  Avg error:   \(avgError)")
    print("  Error count: \(errorCount) / \(totalElements)")

    if maxError < 1e-3 {
        print("\n✓ TEST PASSED!")
    } else {
        print("\n✗ TEST FAILED - errors too large")
    }

    // Benchmark
    print("\n=== Benchmark ===")

    let iterations = 100
    let warmup = 10

    // Warmup
    for _ in 0..<warmup {
        try kernel.forward(A: A_buf, X: X_buf, H: H_buf, useSIMD: true)
    }

    // Benchmark SIMD version
    let benchStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        try kernel.forward(A: A_buf, X: X_buf, H: H_buf, useSIMD: true)
    }
    let simdTime = (CFAbsoluteTimeGetCurrent() - benchStart) / Double(iterations) * 1000

    // Benchmark non-SIMD version
    let benchStart2 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        try kernel.forward(A: A_buf, X: X_buf, H: H_buf, useSIMD: false)
    }
    let regularTime = (CFAbsoluteTimeGetCurrent() - benchStart2) / Double(iterations) * 1000

    print("SIMD kernel:    \(String(format: "%.3f", simdTime)) ms")
    print("Regular kernel: \(String(format: "%.3f", regularTime)) ms")
    print("SIMD speedup:   \(String(format: "%.2f", regularTime / simdTime))x")

    // Test with larger sequence
    print("\n=== Scaling Test ===")

    for testL in [64, 128, 256, 512, 1024] as [UInt32] {
        let testConfig = PScanConfig(batchSize: B, seqLen: testL, dInner: D, dState: N)
        try kernel.compile(config: testConfig)

        let testElems = Int(B * testL * D * N)
        var testA = [Float](repeating: 0.9, count: testElems)
        var testX = [Float](repeating: 1.0, count: testElems)

        guard let testABuf = kernel.makeBuffer(from: testA),
              let testXBuf = kernel.makeBuffer(from: testX),
              let testHBuf = kernel.makeBuffer(size: testElems * MemoryLayout<Float>.size) else {
            continue
        }

        // Warmup
        for _ in 0..<warmup {
            try kernel.forward(A: testABuf, X: testXBuf, H: testHBuf, useSIMD: true)
        }

        let scaleStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            try kernel.forward(A: testABuf, X: testXBuf, H: testHBuf, useSIMD: true)
        }
        let scaleTime = (CFAbsoluteTimeGetCurrent() - scaleStart) / Double(iterations) * 1000

        print("L=\(String(format: "%4d", testL)): \(String(format: "%.3f", scaleTime)) ms")
    }

} catch {
    print("Error: \(error)")
    exit(1)
}

print("\n=== Done ===")
