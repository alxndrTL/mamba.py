# Metal Mamba

Native Metal implementation of Mamba's parallel scan for Apple Silicon (M1/M2/M3/M4).

## Performance

```
============================================================
Metal PScan vs PyTorch PScan on MPS
============================================================

Config: B=2, D=128, N=16

 Seq Len |   Metal (ms) |  PyTorch (ms) |  Speedup
----------------------------------------------------
      64 |        0.654 |         0.869 |    1.33x
     128 |        0.694 |         1.344 |    1.94x
     256 |        0.772 |         2.132 |    2.76x
     512 |        0.548 |         3.407 |    6.21x
    1024 |        1.024 |         6.359 |    6.21x
```

Metal parallelizes across all B*D*N slices while PyTorch has Python overhead.

## How it Works

The parallel scan computes:
```
H[t] = A[t] * H[t-1] + X[t]  with H[0] = X[0]
```

This is the core operation in Mamba's selective scan mechanism.

### Algorithm: Blelloch Scan

We implement Blelloch's work-efficient parallel prefix scan:

1. **Up-sweep (Reduce)**: Combine pairs of elements up the tree
2. **Down-sweep**: Propagate partial sums back down

For a sequence of length L, this requires only O(log L) parallel steps instead of O(L) sequential steps.

### Metal Optimizations

- **SIMD shuffle operations**: Use `simd_shuffle_up` for efficient intra-warp communication
- **Threadgroup memory**: Cache intermediate results for cross-SIMD communication
- **Coalesced memory access**: Process (B, D, N) slices in parallel across sequence length
- **Function constants**: Compile-time specialization for each tensor shape

## Project Structure

```
mamba-metal/
├── mambapy/                    # Original mamba.py (pure PyTorch)
│   ├── mamba.py               # Mamba model
│   └── pscan.py               # PyTorch parallel scan
│
├── metal/                      # Metal implementation
│   ├── Package.swift
│   └── Sources/
│       ├── MetalMamba/
│       │   ├── pscan.metal    # Metal shader for parallel scan
│       │   └── PScanKernel.swift
│       ├── MetalMambaBridge/
│       │   └── MambaBridge.swift  # C-callable interface
│       └── TestPScan/
│           └── main.swift     # Swift test
│
└── metal_pscan/               # Python wrapper
    └── __init__.py            # PyTorch integration
```

## Installation

### Prerequisites

- macOS 14+ (Sonoma) or macOS 15+ (Sequoia)
- Xcode Command Line Tools
- Python 3.10+ with PyTorch 2.0+

### Build

```bash
# Clone the repo
git clone https://github.com/alxndrTL/mamba.py.git mamba-metal
cd mamba-metal

# Build the Metal library
cd metal
swift build -c release
cd ..

# Set environment variable (add to ~/.zshrc)
export METAL_MAMBA_BRIDGE_PATH=/path/to/mamba-metal/metal/.build/release/libMetalMambaBridge.dylib
```

## Usage

### Python (Drop-in replacement)

```python
from metal_pscan import metal_pscan, is_available

if is_available():
    # Same interface as mambapy.pscan.pscan
    H = metal_pscan(A, X)
```

### With Mamba model

```python
import sys
sys.path.insert(0, '/path/to/mamba-metal')

from mambapy.mamba import Mamba, MambaConfig

# Patch pscan to use Metal
from metal_pscan import metal_pscan, is_available
if is_available():
    import mambapy.pscan
    mambapy.pscan.pscan = metal_pscan

# Create and use Mamba model
config = MambaConfig(d_model=256, n_layers=4)
model = Mamba(config).to('mps')

x = torch.randn(2, 1024, 256, device='mps')
y = model(x)
```

### Swift (Direct)

```swift
import MetalMamba

let kernel = try PScanKernel()
let config = PScanConfig(batchSize: 2, seqLen: 1024, dInner: 256, dState: 16)
try kernel.compile(config: config)

// Create buffers and run
try kernel.forward(A: A_buf, X: X_buf, H: H_buf, useSIMD: true)
```

## Shader Details

The Metal shader (`pscan.metal`) implements:

1. **`pscan_forward`**: Basic parallel scan using threadgroup memory
2. **`pscan_forward_simd`**: Optimized version using SIMD shuffle operations
3. **`pscan_backward`**: Gradient computation (reverse scan)

Key optimization: Each threadgroup processes one `(batch, d_inner, d_state)` element across the entire sequence, enabling maximum parallelism.

## Current Status

**Working:**
- Forward pass (verified against CPU reference)
- SIMD-optimized kernel
- Python wrapper with PyTorch integration

**TODO:**
- Metal backward pass (currently falls back to PyTorch)
- Direct MPS tensor integration (avoid CPU round-trip)
- Half precision (fp16) support
- Benchmark with full Mamba model

## Credits

- [mamba.py](https://github.com/alxndrTL/mamba.py) by alxndrTL - Pure PyTorch Mamba implementation
- [Mamba paper](https://arxiv.org/abs/2312.00752) by Albert Gu and Tri Dao
- [Blelloch scan](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf) algorithm

## License

MIT
