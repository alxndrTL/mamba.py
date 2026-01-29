// MambaBridge.swift
// C-callable interface for Python integration

import Foundation
import Metal
import MetalMamba

// Global kernel instance
private var globalKernel: PScanKernel?
private var globalDevice: MTLDevice?

// MARK: - Initialization

@_cdecl("mamba_init")
public func mamba_init() -> Bool {
    do {
        globalKernel = try PScanKernel()
        globalDevice = MTLCreateSystemDefaultDevice()
        print("[MetalMamba] Initialized successfully")
        return true
    } catch {
        print("[MetalMamba] Failed to initialize: \(error)")
        return false
    }
}

@_cdecl("mamba_is_available")
public func mamba_is_available() -> Bool {
    return MTLCreateSystemDefaultDevice() != nil
}

// MARK: - Kernel Compilation

@_cdecl("mamba_compile_pscan")
public func mamba_compile_pscan(
    batchSize: UInt32,
    seqLen: UInt32,
    dInner: UInt32,
    dState: UInt32
) -> Bool {
    guard let kernel = globalKernel else {
        print("[MetalMamba] Not initialized")
        return false
    }

    do {
        let config = PScanConfig(
            batchSize: batchSize,
            seqLen: seqLen,
            dInner: dInner,
            dState: dState
        )
        try kernel.compile(config: config)
        return true
    } catch {
        print("[MetalMamba] Compilation failed: \(error)")
        return false
    }
}

// MARK: - Buffer Management

/// Create a Metal buffer from a pointer
/// Returns an opaque handle (actually the MTLBuffer pointer)
@_cdecl("mamba_create_buffer")
public func mamba_create_buffer(
    data: UnsafeRawPointer?,
    size: Int
) -> UnsafeMutableRawPointer? {
    guard let device = globalDevice else { return nil }

    let buffer: MTLBuffer?
    if let data = data {
        buffer = device.makeBuffer(bytes: data, length: size, options: .storageModeShared)
    } else {
        buffer = device.makeBuffer(length: size, options: .storageModeShared)
    }

    guard let buf = buffer else { return nil }

    // Return the buffer as an unmanaged pointer
    return Unmanaged.passRetained(buf as AnyObject).toOpaque()
}

/// Create a buffer that wraps existing GPU memory (for PyTorch MPS tensors)
@_cdecl("mamba_wrap_buffer")
public func mamba_wrap_buffer(
    gpuAddress: UInt64,
    size: Int
) -> UnsafeMutableRawPointer? {
    guard let device = globalDevice else { return nil }

    // Note: This requires the buffer to already exist in Metal's address space
    // For PyTorch MPS, we need to use the MTLBuffer directly from the tensor
    // This is a placeholder - actual implementation needs PyTorch MPS internals

    // For now, create a new buffer (PyTorch integration will override this)
    guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else {
        return nil
    }

    return Unmanaged.passRetained(buffer as AnyObject).toOpaque()
}

/// Get the contents pointer of a buffer (for CPU access)
@_cdecl("mamba_buffer_contents")
public func mamba_buffer_contents(
    bufferHandle: UnsafeMutableRawPointer
) -> UnsafeMutableRawPointer? {
    let buffer = Unmanaged<AnyObject>.fromOpaque(bufferHandle).takeUnretainedValue() as! MTLBuffer
    return buffer.contents()
}

/// Release a buffer
@_cdecl("mamba_release_buffer")
public func mamba_release_buffer(
    bufferHandle: UnsafeMutableRawPointer
) {
    Unmanaged<AnyObject>.fromOpaque(bufferHandle).release()
}

// MARK: - PScan Execution

/// Run forward pass of parallel scan
@_cdecl("mamba_pscan_forward")
public func mamba_pscan_forward(
    A_handle: UnsafeMutableRawPointer,
    X_handle: UnsafeMutableRawPointer,
    H_handle: UnsafeMutableRawPointer,
    useSIMD: Bool
) -> Bool {
    guard let kernel = globalKernel else {
        print("[MetalMamba] Not initialized")
        return false
    }

    let A = Unmanaged<AnyObject>.fromOpaque(A_handle).takeUnretainedValue() as! MTLBuffer
    let X = Unmanaged<AnyObject>.fromOpaque(X_handle).takeUnretainedValue() as! MTLBuffer
    let H = Unmanaged<AnyObject>.fromOpaque(H_handle).takeUnretainedValue() as! MTLBuffer

    do {
        try kernel.forward(A: A, X: X, H: H, useSIMD: useSIMD)
        return true
    } catch {
        print("[MetalMamba] Forward pass failed: \(error)")
        return false
    }
}

/// Run backward pass of parallel scan
@_cdecl("mamba_pscan_backward")
public func mamba_pscan_backward(
    A_handle: UnsafeMutableRawPointer,
    H_handle: UnsafeMutableRawPointer,
    gradH_handle: UnsafeMutableRawPointer,
    gradA_handle: UnsafeMutableRawPointer,
    gradX_handle: UnsafeMutableRawPointer
) -> Bool {
    guard let kernel = globalKernel else {
        print("[MetalMamba] Not initialized")
        return false
    }

    let A = Unmanaged<AnyObject>.fromOpaque(A_handle).takeUnretainedValue() as! MTLBuffer
    let H = Unmanaged<AnyObject>.fromOpaque(H_handle).takeUnretainedValue() as! MTLBuffer
    let gradH = Unmanaged<AnyObject>.fromOpaque(gradH_handle).takeUnretainedValue() as! MTLBuffer
    let gradA = Unmanaged<AnyObject>.fromOpaque(gradA_handle).takeUnretainedValue() as! MTLBuffer
    let gradX = Unmanaged<AnyObject>.fromOpaque(gradX_handle).takeUnretainedValue() as! MTLBuffer

    do {
        try kernel.backward(A: A, H: H, gradH: gradH, gradA: gradA, gradX: gradX)
        return true
    } catch {
        print("[MetalMamba] Backward pass failed: \(error)")
        return false
    }
}

// MARK: - Direct Pointer Interface (for PyTorch MPS)

/// Run forward pass using raw GPU pointers from PyTorch MPS
/// This is the fast path that avoids buffer copies
@_cdecl("mamba_pscan_forward_raw")
public func mamba_pscan_forward_raw(
    A_ptr: UInt64,
    X_ptr: UInt64,
    H_ptr: UInt64,
    A_offset: Int,
    X_offset: Int,
    H_offset: Int,
    batchSize: UInt32,
    seqLen: UInt32,
    dInner: UInt32,
    dState: UInt32
) -> Bool {
    // This would require direct Metal buffer access from PyTorch
    // For now, this is a stub - actual implementation needs PyTorch MPS internals
    print("[MetalMamba] Raw pointer interface not yet implemented")
    return false
}
