import math
import mlx.core as mx

"""

An implementation of the parallel scan algorithm in MLX (Blelloch version).
The PyTorch implementation is a bit easier to read.

If you want explanation about what's happening here, please see docs/pscan.ipynb.

There are a few points which are different from PyTorch :
- when taking a reshape, we have a new tensor rather than a simple view of it.
  this is quite problematic because this algorithm works with in-place updates
  thus, at the end of each iteration (down sweep and up sweep) we actually need to write back to A and X the computations done in the iteration.
- there is no need for hand-written backward computation !

Unfortunately, this parallel scan implementation is not worth it (compared to sequential implementation).
I suspect it is mainly caused by all the re-write we have to do at the end of each iteration.

"""

def pscan_f(A, X):
    # A : (B, D, L, N)
    # X : (B, D, L, N)

    # modifies X in place by doing a parallel scan.
    # more formally, X will be populated by these values :
    # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
    # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)
    
    Aa = A
    Xa = X

    B, D, L, _ = A.shape

    num_steps = int(math.log2(L))

    # up sweep
    for k in range(num_steps):
        T = 2 * (Xa.shape[2] // 2)

        Aa = Aa[:, :, :T].reshape(B, D, T//2, 2, -1)
        Xa = Xa[:, :, :T].reshape(B, D, T//2, 2, -1)

        Xa[:, :, :, 1] += Aa[:, :, :, 1] * Xa[:, :, :, 0]
        Aa[:, :, :, 1] *= Aa[:, :, :, 0]

        A[:, :, 2**(k+1)-1::2**(k+1)] = Aa[:, :, :, 1]
        X[:, :, 2**(k+1)-1::2**(k+1)] = Xa[:, :, :, 1]

        Aa = Aa[:, :, :, 1]
        Xa = Xa[:, :, :, 1]

    # down sweep
    for k in range(num_steps-1, -1, -1):
        Aa = A[:, :, 2**k-1::2**k]
        Xa = X[:, :, 2**k-1::2**k]

        step_len = Xa.shape[2]
        T = 2 * (step_len // 2)

        if T < step_len:
            last_val_aa = Aa[:, :, -1] * Aa[:, :, -2]
            last_val_xa = Xa[:, :, -1] + Aa[:, :, -1] * Xa[:, :, -2]

        Aa = Aa[:, :, :T].reshape(B, D, T//2, 2, -1)
        Xa = Xa[:, :, :T].reshape(B, D, T//2, 2, -1)

        Xa[:, :, 1:, 0] += Aa[:, :, 1:, 0] * Xa[:, :, :-1, 1]
        Aa[:, :, 1:, 0] *= Aa[:, :, :-1, 1]

        if T == step_len:
            A[:, :, 2**k-1::2**(k+1)] = Aa[:, :, :, 0]
            X[:, :, 2**k-1::2**(k+1)] = Xa[:, :, :, 0]
        else:
            A[:, :, 2**k-1::2**(k+1)] = mx.concatenate([Aa[:, :, :, 0], mx.array([last_val_aa]).reshape(B, D, 1, -1)], axis=2)
            X[:, :, 2**k-1::2**(k+1)] = mx.concatenate([Xa[:, :, :, 0], mx.array([last_val_xa]).reshape(B, D, 1, -1)], axis=2)

def pscan(A_in, X_in):
    """
    Applies the parallel scan operation, as defined above. Returns a new tensor.

    Args:
        A_in : (B, L, ED, N)
        X_in : (B, L, ED, N)

    Returns:
        H : (B, L, ED, N)
    """

    A = A_in[:].transpose(0, 2, 1, 3)
    X = X_in[:].transpose(0, 2, 1, 3)

    pscan_f(A, X)

    return X.transpose(0, 2, 1, 3)
