"""Broadcast weighting preserves BF16 products and native fine-only pruning."""
import pytest
import torch

from fastvideo_kernel.ops import _combine_bshd
from fastvideo_kernel.triton_kernels.fused_compress_topk import _WeightedBSHD, _combine_weighted_bshd


@pytest.mark.parametrize('block', [128, 256])
def test_weighted_bshd_exact_branches(block):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    torch.manual_seed(925)
    for flags in ((True, True, True), (True, False, False), (False, True, False), (False, False, True)):
        fine = torch.randn(2, block * 3, 3, 128, device='cuda', dtype=torch.bfloat16, requires_grad=flags[0])
        coarse = torch.randn(2, 3, 3, 128, device='cuda', dtype=fine.dtype, requires_grad=flags[1])
        gate = torch.randn_like(fine, requires_grad=flags[2])
        for strided in (False, True):
            dy = torch.randn(2, block * 6, 3, 128, device='cuda', dtype=fine.dtype)[:, ::2]
            if not strided:
                dy = dy.contiguous()
            ref = _combine_bshd(fine, coarse, gate, block)
            got = fine + _WeightedBSHD.apply(coarse, gate, block)
            leaves = [x for x in (fine, coarse, gate) if x.requires_grad]
            assert torch.equal(ref, got)
            for a, b in zip(torch.autograd.grad(ref, leaves, dy), torch.autograd.grad(got, leaves, dy)):
                assert torch.equal(a, b)


def test_weighted_bshd_fine_only_prunes_saved_values():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    fine = torch.randn(1, 256, 1, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    coarse = torch.randn(1, 2, 1, 128, device='cuda', dtype=fine.dtype, requires_grad=True)
    gate = torch.randn_like(fine, requires_grad=True)
    out = fine + _WeightedBSHD.apply(coarse, gate, 128)
    with torch.no_grad():
        coarse.add_(1)
        gate.add_(1)
    dy = torch.randn_like(fine)
    assert torch.equal(torch.autograd.grad(out, fine, dy)[0], dy)


def test_weighted_bshd_dispatch_bounds():
    class Metadata:
        requires_grad = True
        ndim = 4

        def __init__(self, n):
            self.shape = (1, n, 1, 128)

        def numel(self):
            return self.shape[1] * 128

    # Reject before querying CUDA pointers or allocating a huge tensor.
    for n in (0, 256, 2**24):
        fine = Metadata(n)
        assert _combine_weighted_bshd(fine, fine, fine, 128, lambda *args: 'fallback') == 'fallback'
