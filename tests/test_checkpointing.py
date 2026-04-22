
import gc

import torch

from mambapy.mamba import Mamba, MambaConfig
from mambapy.checkpointing import checkpointed_sequence


def _make_mamba(d_model=32, n_layers=2, d_state=16):
    cfg = MambaConfig(
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        pscan=True,
        use_cuda=False,
    )
    return Mamba(cfg)


def _empty_caches(model, B, device='cpu'):
    # caches : [cache(layer) for all layers], cache : (h, inputs)
            # h : (B, ED, N)
            # inputs : (B, ED, d_conv-1)
    caches = []
    for layer in model.layers:
        cfg = layer.mixer.config
        ED = cfg.d_inner
        k = max(cfg.d_conv - 1, 0)
        inputs = torch.zeros(B, ED, k, device=device)
        caches.append((None, inputs))
    return caches


def test_full_forward_vs_checkpointed_gradients():
    # full forward+backward on a single sequence must produce
    # the same loss and gradients as 2-segment checkpointed
    torch.manual_seed(42)

    B, L, D = 4, 160, 64
    n_layers = 4
    segment_size = 80 # 2 segments
    num_segments = L // segment_size

    config = MambaConfig(d_model=D, n_layers=n_layers, d_state=16, pscan=True, use_cuda=False)

    torch.manual_seed(123)
    model_full = Mamba(config)
    torch.manual_seed(123)
    model_ckpt = Mamba(config)

    for (n1, p1), (_, p2) in zip(model_full.named_parameters(), model_ckpt.named_parameters()):
        assert torch.equal(p1, p2), f"Parameter {n1} not identical at init"

    torch.manual_seed(456)
    x = torch.randn(B, L, D, requires_grad=True)

    # full forward (standard training path)
    output_full = model_full(x)
    loss_full = output_full.sum()
    loss_full.backward()

    # 2-segment checkpointed
    segments = [x[:, i * segment_size : (i + 1) * segment_size] for i in range(num_segments)]
    caches = _empty_caches(model_ckpt, B)
    outputs_ckpt, _ = checkpointed_sequence(model_ckpt.chunk_step, segments, caches)
    output_ckpt = torch.cat(outputs_ckpt, dim=1)
    loss_ckpt = output_ckpt.sum()
    loss_ckpt.backward()

    assert torch.allclose(loss_full, loss_ckpt, rtol=1e-5, atol=1e-5)
    assert torch.allclose(output_full, output_ckpt, rtol=1e-5, atol=1e-5)

    for (name_f, p_f), (_, p_c) in zip(model_full.named_parameters(), model_ckpt.named_parameters()):
        assert p_f.grad is not None and p_c.grad is not None
        max_diff = (p_f.grad - p_c.grad).abs().max().item()
        assert torch.allclose(p_f.grad, p_c.grad, rtol=1e-5, atol=1e-5), \
            f"Gradient mismatch for {name_f}: max diff={max_diff:.2e}"


def test_checkpointing_reduces_memory():
    # checkpointed (10 x 100) must use at least 5x less peak memory than full (1 x 1000)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        return # skip, no GPU

    B, L, D = 4, 1000, 128
    n_layers = 8
    segment_size = 100
    num_segments = L // segment_size

    config = MambaConfig(d_model=D, n_layers=n_layers, d_state=16, pscan=True, use_cuda=False)

    def run_full():
        torch.manual_seed(42)
        model = Mamba(config).to(device)
        x = torch.randn(B, L, D, device=device, requires_grad=True)
        loss = model(x).sum()
        loss.backward()

    def run_checkpointed():
        torch.manual_seed(42)
        model = Mamba(config).to(device)
        x = torch.randn(B, L, D, device=device, requires_grad=True)
        segments = [x[:, i * segment_size : (i + 1) * segment_size] for i in range(num_segments)]
        caches = _empty_caches(model, B, device)
        outputs, _ = checkpointed_sequence(model.chunk_step, segments, caches)
        loss = torch.cat(outputs, dim=1).sum()
        loss.backward()

    def measure(fn):
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            fn()
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated(device)
        else: # mps
            torch.mps.empty_cache()
            torch.mps.synchronize()
            baseline = torch.mps.driver_allocated_memory()
            fn()
            torch.mps.synchronize()
            return torch.mps.driver_allocated_memory() - baseline

    # warmup
    measure(run_full)
    measure(run_checkpointed)

    mem_full = measure(run_full)
    mem_ckpt = measure(run_checkpointed)

    print(f"\nFull (1 x {L}): {mem_full / 1e6:.1f} MB")
    print(f"Checkpointed ({num_segments} x {segment_size}): {mem_ckpt / 1e6:.1f} MB")
    print(f"Reduction: {(1 - mem_ckpt / mem_full) * 100:.1f}%")

    assert mem_ckpt * 5 < mem_full
