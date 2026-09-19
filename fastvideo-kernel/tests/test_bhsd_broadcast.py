"""Compare BHSD coarse broadcasting with the original repeated-tensor expression."""
import pytest
import torch
from fastvideo_kernel import ops


@pytest.mark.parametrize('block,gate_kind', [(128, 'full'), (128, 'broadcast'), (128, 'none'),
                                           (64, 'full'), (256, 'full')])
def test_bhsd_coarse_combine_matches_repeat(block, gate_kind, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    torch.manual_seed(672)
    x = [torch.randn(2, 3, block * 4, 128, device='cuda', dtype=torch.bfloat16,
                     requires_grad=True) for _ in range(3)]
    sizes = torch.tensor([block, block // 2, 17, block - 1], device='cuda', dtype=torch.int32)
    gate = None if gate_kind == 'none' else torch.randn(
        x[0].shape if gate_kind == 'full' else (1, 3, 1, 128), device='cuda',
        dtype=torch.bfloat16, requires_grad=True)

    def fine(q, k, v):
        return q * .125 + k * .25 + v * .5

    def attention(q, k, v, *_):
        return fine(q, k, v), None

    for name in ('block_sparse_attn', 'block_sparse_attn_128', 'block_sparse_attn_256'):
        monkeypatch.setattr(ops, name, attention)
    pooled = [ops.fused_block_mean(t, sizes, block) for t in x]
    coarse = torch.softmax((pooled[0] @ pooled[1].transpose(-1, -2)) / (128**.5), -1) @ pooled[2]
    repeated = coarse.unsqueeze(-2).repeat(1, 1, 1, block, 1).reshape_as(x[0])
    expected = repeated + fine(*x) if gate is None else repeated * gate + fine(*x)
    actual = ops.video_sparse_attn(*x, sizes, sizes, 2, (1, 1, block), gate)
    leaves = x + ([] if gate is None else [gate])
    dy = torch.randn_like(actual)
    expected_grad = torch.autograd.grad(expected, leaves, dy)
    actual_grad = torch.autograd.grad(actual, leaves, dy)
    for a, b in zip((actual, *actual_grad), (expected, *expected_grad)):
        delta = (a.float() - b.float()).abs()
        assert torch.isfinite(a).all()
        assert delta.mean() < 1e-3
        assert delta.max() / (b.float().abs().mean() + 1e-6) < .25
