"""BSHD-backed BHSD calls preserve routing and gradients across the layout path."""
import pytest
import torch
from fastvideo_kernel import ops


def _close(reference, actual):
    delta = (reference.float() - actual.float()).abs()
    assert torch.isfinite(actual).all()
    assert delta.mean() < 1e-3
    assert delta.max() / (reference.float().abs().mean() + 1e-6) < .25


def test_bhsd_weighted_layout_parent_grad_and_backend_stride(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip('Blackwell CuTe required')
    torch.manual_seed(751)
    packed = torch.randn(8, 4096, 32, 128, device='cuda', dtype=torch.bfloat16)
    packed[6:].mul_(.1)
    packed.requires_grad_()
    q, k, v, gate = [x.transpose(1, 2) for x in packed.chunk(4)]
    sizes = torch.tensor([128, 67, 17, 127] * 8, device='cuda', dtype=torch.int32)
    dy = torch.randn_like(q)
    original = ops._fork_bhsd_admitted
    # Independent original BHSD pooling/add path: no new layout dispatch.
    monkeypatch.setattr(ops, '_fork_bhsd_admitted', lambda *args: False)
    reference = ops.video_sparse_attn(q, k, v, sizes, sizes, 4, (4, 8, 4), gate)
    reference_grad = torch.autograd.grad(reference, packed, dy)[0]
    monkeypatch.setattr(ops, '_fork_bhsd_admitted', original)
    actual = ops.video_sparse_attn(q, k, v, sizes, sizes, 4, (4, 8, 4), gate)
    actual_grad = torch.autograd.grad(actual, packed, dy)[0]
    assert actual.transpose(1, 2).is_contiguous()
    _close(reference, actual)
    _close(reference_grad, actual_grad)


def test_bhsd_small_input_keeps_original_dispatch(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip('Blackwell CuTe required')
    def unexpected(*args, **kwargs):
        raise AssertionError('small input used BSHD weighted dispatch')
    monkeypatch.setattr(ops, 'block_sparse_attn_128_bshd', unexpected)
    x = torch.randn(4, 256, 2, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    q, k, v, gate = [t.transpose(1, 2) for t in x.chunk(4)]
    sizes = torch.tensor([128, 67], device='cuda', dtype=torch.int32)
    out = ops.video_sparse_attn(q, k, v, sizes, sizes, 1, (4, 8, 4), gate)
    assert torch.isfinite(torch.autograd.grad(out.sum(), x)[0]).all()
