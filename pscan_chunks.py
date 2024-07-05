import torch

from mambapy.pscan import pscan

B, L, D, N = 16, 256, 58, 6

# chunked pscan

torch.cuda.reset_peak_memory_stats(device=None)
print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

torch.manual_seed(123456)
A_chunks = torch.randn(B, L, D, N, requires_grad=True, device="cuda")
X_chunks = torch.randn(B, L, D, N, requires_grad=True, device="cuda")

chunk_size = 63 # best is power of 2 minus 1
num_chunks = L // chunk_size
remainder = L % chunk_size

print(f"number of chunks: {num_chunks} and remainder: {remainder}")

last_hidden = None
hs_chunks = []

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size
    A_chunk = A_chunks[:, start_idx:end_idx]
    X_chunk = X_chunks[:, start_idx:end_idx]
    
    # kind of a bug of pytorch : we you send 3 argd, backward has to send back 3 tensors
    if last_hidden is None:
        hs_chunk = pscan(A_chunk, X_chunk)
    else:
        hs_chunk = pscan(A_chunk, X_chunk, last_hidden)
    last_hidden = hs_chunk[:, -1]

    hs_chunks.append(hs_chunk)

if remainder > 0:
    remainder_start_idx = num_chunks * chunk_size
    A_chunk = A_chunks[:, remainder_start_idx:]
    X_chunk = X_chunks[:, remainder_start_idx:]

    hs_chunk = pscan(A_chunk, X_chunk, last_hidden)

    hs_chunks.append(hs_chunk)

hs_chunks = torch.cat(hs_chunks, dim=1)

J_chunks = hs_chunks.sum()
J_chunks.backward()

print(f"gpu used {torch.cuda.max_memory_allocated(device=None) / (1024**2)} MB")
