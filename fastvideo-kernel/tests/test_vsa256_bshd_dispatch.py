"""Keep native256 training opt-in to its validated public-wrapper domain."""
import importlib
import pytest
import torch


@pytest.mark.parametrize('requires_grad,grad_enabled,expected_blocks', [
    (True, True, 2), (True, False, 4), (False, True, 4),
])
def test_vsa256_bshd_training_only_dispatch(monkeypatch, requires_grad, grad_enabled, expected_blocks):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip('SM10x required')
    wrapper = importlib.import_module('fastvideo_kernel.block_sparse_attn_256')
    adapter = importlib.import_module('fastvideo_kernel.block_sparse_attn_cute_fwd')
    monkeypatch.setattr(wrapper, '_resolve_backend', lambda: 'cute')
    seen = []

    def record(q, k, v, routes, sizes):
        seen.append(routes.shape[-1])
        return torch.empty_like(q), torch.empty(1, 2, 512, device='cuda', requires_grad=True)

    monkeypatch.setattr(adapter, 'block_sparse_attn_cute_fwd_bshd', record)
    qkv = [torch.randn(1, 512, 2, 128, device='cuda', dtype=torch.bfloat16,
                       requires_grad=requires_grad) for _ in range(3)]
    routes = torch.ones(1, 2, 2, 2, device='cuda', dtype=torch.bool)
    sizes = torch.tensor([131, 0], device='cuda', dtype=torch.int32)
    with torch.set_grad_enabled(grad_enabled):
        _, lse = wrapper.block_sparse_attn_256_bshd(*qkv, routes, sizes)
    assert seen == [expected_blocks]
    # The public auxiliary remains informational even when native training
    # internally retains a differentiable normalization state.
    if expected_blocks == 2:
        assert not lse.requires_grad
