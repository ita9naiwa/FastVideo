"""BSHD compression keeps the original padded-block mean and its gradient."""
import pytest
import torch

from fastvideo_kernel import ops
from fastvideo_kernel.triton_kernels.fused_compress_topk import fused_block_mean_bshd


def _reference(x, sizes, block):
    b, n, h, d = x.shape
    pooled = x.view(b, n // block, block, h, d).float().sum(2)
    return (pooled / sizes.view(1, -1, 1, 1)).to(x.dtype).permute(0, 2, 1, 3).contiguous()


@pytest.mark.parametrize('block', [128, 256])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('dim', [64, 128])
def test_bshd_compression_strides_and_gradient(block, dtype, dim):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    torch.manual_seed(44)
    # Strided sequence and sizes exercise both address calculations. Nonzero
    # padding is intentional: compression sums it rather than silently masking.
    x = torch.randn(2, block * 6, 3, dim, device='cuda', dtype=dtype)[:, ::2].requires_grad_()
    sizes = torch.tensor([block, 0, 19, 0, 1, 0], device='cuda', dtype=torch.int64)[::2]
    ref, got = _reference(x, sizes, block), fused_block_mean_bshd(x, sizes, block)
    dy = torch.randn_like(ref)
    expected = torch.autograd.grad(ref, x, dy)[0]
    actual = torch.autograd.grad(got, x, dy)[0]
    torch.testing.assert_close(got, ref, atol=2e-6, rtol=.008 if dtype == torch.bfloat16 else .001)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    # Non-unit channel stride takes the original expression fallback.
    narrow = x[..., ::2]
    torch.testing.assert_close(fused_block_mean_bshd(narrow, sizes, block),
                               _reference(narrow, sizes, block), atol=0, rtol=0)


@pytest.mark.parametrize('block', [128, 256])
def test_bshd_wrapper_changed_graph(block, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip('SM10x required')
    monkeypatch.setenv('FASTVIDEO_VSA_PACK_TAILS', '0')
    torch.manual_seed(516)
    sizes = torch.tensor([block, 67, block, 17], device='cuda', dtype=torch.int32)
    n = 4 * block
    valid = (torch.arange(n, device='cuda') % block < sizes[torch.arange(n, device='cuda') // block])[None, :, None, None]
    qkv = [torch.randn(1, n, 2, 128, device='cuda', dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
    gate = torch.randn_like(qkv[0]).requires_grad_()
    dy = torch.randn_like(gate) * valid

    def call():
        out = ops.video_sparse_attn_bshd(*qkv, sizes, sizes, 2, (4, 8, block // 32), gate)
        return (out, *torch.autograd.grad(out, (*qkv, gate), dy))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    for _ in range(4):
        with torch.no_grad():
            for x in qkv:
                x.normal_().mul_(valid)
            gate.normal_().mul_(.1)
            for x in captured:
                x.fill_(float('nan'))
        with monkeypatch.context() as patch:
            patch.setattr(ops, 'fused_block_mean_bshd', _reference)
            ref = call()
        graph.replay()
        for a, b in zip(ref, captured):
            error = (a.float() - b.float()).abs()
            assert torch.isfinite(b).all()
            assert error.mean() < 1e-3
            assert error.max() / (a.float().abs().mean() + 1e-6) < .25


def test_bshd_compression_differentiable_divisor_fallback():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    x = torch.randn(1, 256, 2, 64, device='cuda', requires_grad=True)
    sizes = torch.tensor([128., 67.], device='cuda', requires_grad=True)
    ref, got = _reference(x, sizes, 128), fused_block_mean_bshd(x, sizes, 128)
    dy = torch.randn_like(ref)
    for a, b in zip(torch.autograd.grad(ref, (x, sizes), dy),
                    torch.autograd.grad(got, (x, sizes), dy)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_bshd_compression_higher_order_gradient():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    x = torch.randn(1, 256, 2, 64, device='cuda', requires_grad=True)
    sizes = torch.tensor([128, 67], device='cuda', dtype=torch.int32)
    ref, got = _reference(x, sizes, 128), fused_block_mean_bshd(x, sizes, 128)
    dy = torch.randn_like(ref, requires_grad=True)
    dx = torch.randn_like(x)
    expected = torch.autograd.grad(ref, x, dy, create_graph=True)[0]
    actual = torch.autograd.grad(got, x, dy, create_graph=True)[0]
    torch.testing.assert_close(torch.autograd.grad(actual, dy, dx)[0],
                               torch.autograd.grad(expected, dy, dx)[0])
