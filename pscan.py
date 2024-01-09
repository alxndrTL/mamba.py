import math

import torch

# TODO eviter les .flip() en codant un pscan reverse (avec flag)
# TODO commentaires en docstring

class PScan(torch.autograd.Function):
    # an implementation of the Blelloch parallel scan algorithm
    
    @staticmethod
    def pscan(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # modifies X in place by doing a parallel scan.
        # more formally, X will be populated by these values :
        # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)
        
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep
        Aa = A
        Xa = X
        for k in range(num_steps):
            T = 2 * (Xa.size(2) // 2)

            Aa = Aa[:, :, :T].view(B, D, T//2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T//2, 2, -1)
            
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        # down sweep
        for k in range(num_steps-1, -1, -1):
            Aa = A[:, :, 2**k-1:L:2**k]
            Xa = X[:, :, 2**k-1:L:2**k]

            T = 2 * (Xa.size(2) // 2)

            if T < Xa.size(2):
                Xa[:, :, -1].add_(Aa[:, :, -1].mul(Xa[:, :, -2]))
                Aa[:, :, -1].mul_(Aa[:, :, -2])

            Aa = Aa[:, :, :T].view(B, D, T//2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T//2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def forward(ctx, A_in, X_in):
        # A_in : (B, L, D, N)
        # X_in : (B, L, D, N)

        # H : (B, L, D, N)

        # applies the parallel scan operation, as defined above.
        # returns a new tensor.

        # clone tensor (in-place ops)
        A = A_in.clone() # (B, L, D, N)
        X = X_in.clone() # (B, L, D, N)
        
        # prepare tensors
        A = A.transpose(2, 1) # (B, D, L, N)
        X = X.transpose(2, 1) # (B, D, L, N)

        # parallel scan
        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)

        return X.transpose(2, 1)
    
    @staticmethod
    def backward(ctx, grad_output_in):
        # ctx : A_in : (B, L, D, N), X : (B, D, L, N)
        # grad_output_in : (B, L, D, N)

        # gradA : (B, L, D, N), gradX : (B, L, D, N)

        # flows the gradient from the output to the input.
        # return two new tensors.

        A_in, X = ctx.saved_tensors

        # clone tensors 
        A = A_in.clone()
        # grad_output_in will be cloned with flip()

        # prepare tensors
        A = A.transpose(2, 1) # (B, D, L, N)
        A = torch.cat((A[:, :, :1], A[:, :, 1:].flip(2)), dim=2)
        grad_output_b = grad_output_in.transpose(2, 1)

        # reverse parallel scan
        grad_output_b = grad_output_b.flip(2)
        PScan.pscan(A, grad_output_b)
        grad_output_b = grad_output_b.flip(2)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output_b[:, :, 1:])

        return Q.transpose(2, 1), grad_output_b.transpose(2, 1)
    
pscan = PScan.apply