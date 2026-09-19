"""BHSD fork retains the original helper's derivative and pruning behavior."""
import pytest
import torch
from fastvideo_kernel.triton_kernels.fused_compress_topk import (
    _ForkBlockMeanBHSD, _fork_bhsd_admitted, fused_block_mean,
)


def _inputs(layout='bhsd'):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    torch.manual_seed(742)
    x = torch.randn(2, 3, 512, 128, device='cuda', dtype=torch.bfloat16)
    if layout == 'bshd_backed':
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    sizes = torch.tensor([128, 67, 17, 127], device='cuda', dtype=torch.int32)
    return x.requires_grad_(), sizes


@pytest.mark.parametrize('layout', ['bhsd', 'bshd_backed'])
@pytest.mark.parametrize('branch', ['fine', 'coarse', 'both'])
def test_bhsd_fork_gradient_branches(layout, branch):
    x, sizes = _inputs(layout)
    df = torch.randn_like(x)
    dc = torch.randn(2, 3, 4, 128, device='cuda', dtype=x.dtype)
    results = []
    for fork in (False, True):
        fine, coarse = _ForkBlockMeanBHSD.apply(x, sizes, 128) if fork else (x, fused_block_mean(x, sizes, 128))
        outputs, gradients = ((fine,), (df,)) if branch == 'fine' else (
            ((coarse,), (dc,)) if branch == 'coarse' else ((fine, coarse), (df, dc)))
        first = torch.autograd.grad(outputs, x, gradients, retain_graph=True)[0]
        again = torch.autograd.grad(outputs, x, gradients)[0]
        torch.testing.assert_close(first, again, atol=0, rtol=0)
        results.append(first)
    torch.testing.assert_close(*results, atol=0, rtol=0)


@pytest.mark.parametrize('branch', ['fine', 'coarse'])
def test_bhsd_fork_prunes_unused_sizes(branch):
    x, sizes = _inputs()
    fine, coarse = _ForkBlockMeanBHSD.apply(x, sizes, 128)
    sizes.add_(1)
    output = fine if branch == 'fine' else coarse
    if branch == 'coarse':
        with pytest.raises(RuntimeError, match='modified by an inplace operation'):
            torch.autograd.grad(output, x, torch.ones_like(output))
    else:
        result = torch.autograd.grad(output, x, torch.ones_like(output))[0]
        torch.testing.assert_close(result, torch.ones_like(result), atol=0, rtol=0)


def test_bhsd_fork_preserves_existing_higher_order_helper_behavior():
    x, sizes = _inputs()
    results = []
    for fork in (False, True):
        fine, coarse = _ForkBlockMeanBHSD.apply(x, sizes, 128) if fork else (x, fused_block_mean(x, sizes, 128))
        first = torch.autograd.grad(fine.square().sum() + coarse.square().sum(), x, create_graph=True)[0]
        second = torch.autograd.grad(first, x, torch.ones_like(first))[0]
        results.append((first, second))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_bhsd_fork_distinguishes_disjoint_views_from_overlap():
    x, sizes = _inputs('bshd_backed')
    packed = torch.stack([x, x, x]).detach().requires_grad_()
    q, k, v = packed.unbind(0)
    assert _fork_bhsd_admitted((q, k, v), sizes, sizes, 128)
    assert not _fork_bhsd_admitted((q, q, v), sizes, sizes, 128)
