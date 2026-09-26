"""Keep native256 training opt-in to its validated public-wrapper domain."""
import importlib
import pytest
import torch


@pytest.mark.parametrize('requires_grad,grad_enabled,expected_blocks', [
    (True, True, 2), (True, False, 4), (False, True, 4),
])
@pytest.mark.parametrize('layout', ['contiguous', 'packed', 'misaligned_packed'])
def test_vsa256_bshd_training_only_dispatch(monkeypatch, requires_grad, grad_enabled, expected_blocks, layout):
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
    if layout != 'contiguous':
        offset = int(layout == 'misaligned_packed')
        packed = torch.randn(512 * 3 * 2 * 128 + offset, device='cuda', dtype=torch.bfloat16)
        packed = packed[offset:].view(1, 512, 3, 2, 128).requires_grad_(requires_grad)
        qkv = packed.unbind(2)
        if offset:
            expected_blocks = 4
    routes = torch.ones(1, 2, 2, 2, device='cuda', dtype=torch.bool)
    sizes = torch.tensor([131, 0], device='cuda', dtype=torch.int32)
    native = expected_blocks == 2
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof, \
            torch.set_grad_enabled(grad_enabled):
        _, lse = wrapper.block_sparse_attn_256_bshd(*qkv, routes, sizes)
    # Native Q256 training (256-token map) takes the fullgraph custom op fastvideo_kernel::vsa256_fwd; every other case
    # keeps the 128-token-map adapter path (4 blocks here).
    assert seen == ([] if native else [expected_blocks])
    assert ("fastvideo_kernel::vsa256_fwd" in {e.key for e in prof.key_averages()}) == native
    # The public auxiliary remains informational even when native training
    # internally retains a differentiable normalization state.
    if expected_blocks == 2:
        assert not lse.requires_grad


@pytest.mark.parametrize('dim', [64, 128])
@pytest.mark.parametrize('layout', ['packed', 'sequence_step2', 'head_transpose'])
def test_vsa256_aligned_views_gradients_and_versions(monkeypatch, dim, layout):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip('SM10x required')
    wrapper = importlib.import_module('fastvideo_kernel.block_sparse_attn_256')
    monkeypatch.setattr(wrapper, '_resolve_backend', lambda: 'cute')
    monkeypatch.setenv('FASTVIDEO_VSA_PACK_TAILS', '0')
    torch.manual_seed(832)
    if layout == 'head_transpose':
        leaf = torch.randn(2, 3, 3, 1024, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        packed = leaf.permute(0, 3, 1, 2, 4)
    else:
        leaf = torch.randn(2, 2048 if layout == 'sequence_step2' else 1024, 3, 3, dim,
                           device='cuda', dtype=torch.bfloat16, requires_grad=True)
        packed = leaf[:, ::2] if layout == 'sequence_step2' else leaf
    qkv = packed.unbind(2)
    routes = torch.rand(2, 3, 4, 4, device='cuda') > .5
    routes[:, :, 0] = False
    routes[:, :, 1] = True
    sizes = torch.tensor([256, 131, 67, 0], device='cuda', dtype=torch.int32)
    dy = torch.randn_like(qkv[0])
    got, lse = wrapper.block_sparse_attn_256_bshd(*qkv, routes, sizes)
    ref, _ = wrapper.block_sparse_attn_256_bshd(*(x.contiguous() for x in qkv), routes, sizes)
    got_grad = torch.autograd.grad(got, leaf, dy)[0]
    ref_grad = torch.autograd.grad(ref, leaf, dy)[0]
    # Native layout parity, including untouched backing rows. This does not
    # assert universal agreement of native BF16 attention with an FP32 oracle.
    for actual, expected in ((got, ref), (got_grad, ref_grad)):
        diff = (actual.float() - expected.float()).abs()
        assert torch.isfinite(actual).all()
        assert diff.mean() < 1e-3
        assert diff.max() / (expected.float().abs().mean() + 1e-6) < .25
    assert not lse.requires_grad
    out, _ = wrapper.block_sparse_attn_256_bshd(*qkv, routes, sizes)
    with torch.no_grad():
        leaf.add_(1)
    with pytest.raises(RuntimeError, match='modified|inplace'):
        torch.autograd.grad(out, leaf, dy)
