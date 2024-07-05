import torch

from mambapy.pscan import pscan

B, L, D, N = 100, 1024, 64, 16

# chunked pscan

torch.cuda.reset_peak_memory_stats(device=None)

torch.manual_seed(123456)
A_chunks = torch.ones(B, L, D, N, requires_grad=True, device="cuda")
X_chunks = torch.randn(B, L, D, N, requires_grad=True, device="cuda")

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

chunk_size = 50 # best is power of 2 minus 1
num_chunks = L // chunk_size
remainder = L % chunk_size

print(f"number of chunks: {num_chunks} and remainder: {remainder}")

last_hidden = None

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size
    A_chunk = A_chunks[:, start_idx:end_idx]
    X_chunk = X_chunks[:, start_idx:end_idx]
    
    # pytorch bug : when you send 3 args, backward has to send back 3 tensors (even if one is None)
    if last_hidden is None:
        hs_chunk = pscan(A_chunk, X_chunk)
    else:
        hs_chunk = pscan(A_chunk, X_chunk, last_hidden)
    last_hidden = hs_chunk[:, -1]#.detach()

    loss = hs_chunk.sum()
    loss.backward(retain_graph=not(i==num_chunks-1 and remainder==0))
    print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

if remainder > 0:
    remainder_start_idx = num_chunks * chunk_size
    A_chunk = A_chunks[:, remainder_start_idx:]
    X_chunk = X_chunks[:, remainder_start_idx:]

    hs_chunk = pscan(A_chunk, X_chunk, last_hidden)

    loss = hs_chunk.sum()
    loss.backward()
    print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

print(f"max gpu used {torch.cuda.max_memory_allocated(device=None) / (1024**2)} MB")

#print(J_chunks)
print(A_chunks.grad.mean())
print(X_chunks.grad.mean())
