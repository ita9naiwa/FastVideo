"""Native FA4 accepts aligned row-strided dO without an adapter copy."""
import math

import pytest
import torch

from fastvideo_kernel import block_sparse_attn_cute_fwd as adapter
from .test_vsa256_backward import _GRAD_TOL, _OUT_TOL
from .test_vsa256_triton import _metrics


@pytest.mark.parametrize("block,d", [(128, 64), (128, 128), (256, 64), (256, 128)])
def test_row_strided_dout_fp32_reference(block, d):
    pytest.importorskip("flash_attn.cute.block_sparsity")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM10x CUDA required")
    torch.manual_seed(271 + block + d)
    n, nk, h = 3 * block, 4 * block, 2
    inputs = [torch.randn(1, length, h, d, device="cuda", dtype=torch.bfloat16)
              for length in (n, nk, nk)]
    routes = torch.tensor([[[[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                            [[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 0]]]],
                          device="cuda", dtype=torch.bool)
    sizes = torch.tensor([block, block - 1, 37, 0], device="cuda", dtype=torch.int32)
    dout = torch.randn(1, n * 2, h, d, device="cuda", dtype=torch.bfloat16)[:, ::2]
    dlse = torch.randn(1, h, n, device="cuda") * .1 if block == 256 else None
    if dlse is not None:
        dlse[:, 0, 2 * block:] = 0
    refs = [x.float().detach().requires_grad_() for x in inputs]
    mask = routes.repeat_interleave(block, 2).repeat_interleave(block, 3)
    valid = (torch.arange(block, device="cuda")[None, :] < sizes[:, None]).reshape(1, 1, 1, -1)
    mask = mask & valid
    has = mask.any(-1, keepdim=True)
    logits = refs[0].transpose(1, 2) @ refs[1].transpose(1, 2).transpose(-2, -1) / math.sqrt(d)
    safe = torch.where(has, logits.masked_fill(~mask, -torch.inf), 0.)
    prob = safe.softmax(-1) * has
    ref_out = (prob @ refs[2].transpose(1, 2)).transpose(1, 2).to(torch.bfloat16)
    ref_lse = torch.where(has.squeeze(-1), safe.logsumexp(-1), -torch.inf)
    ref_grad = (torch.autograd.grad((ref_out, ref_lse), refs, (dout, dlse)) if dlse is not None
                else torch.autograd.grad(ref_out, refs, dout))
    xs = [x.detach().requires_grad_() for x in inputs]
    function = adapter._CuteAttentionQ128 if block == 128 else adapter._CuteAttentionQ256Training
    out, lse = function.apply(*xs, routes, sizes)
    grads = (torch.autograd.grad((out, lse), xs, (dout, dlse)) if dlse is not None
             else torch.autograd.grad(out, xs, dout))
    for ref, got, limits in [(ref_out, out, _OUT_TOL)] + [(a, b, _GRAD_TOL) for a, b in zip(ref_grad, grads)]:
        avg, scaled = _metrics(ref.float(), got.float())
        assert torch.isfinite(got).all()
        assert avg < limits[0] and scaled < limits[1]
    assert torch.equal(torch.isfinite(ref_lse), torch.isfinite(lse))
