# SPDX-License-Identifier: Apache-2.0
"""Fused-QKV gradient at tile 128: the stacked [3B, S, H, D] route of vsa_train_fwd/bwd (one gradient allocation written
through views, as vsa256 at tile 256) equals the three-gradient route (O, dK, dV bitwise; dQ within the suite _GRAD_TOL),
and H3 forward_qkv takes it when fused_qkv_grad is on (no chunk-backward concat)."""

import pytest
import torch

from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAImpl
from fastvideo.tests.attention.test_vsa_h3_tile128 import (_DIM, _GPU_CASES, _GPU_IDS, _GRAD_TOL, _HEADS, _gpu_meta,
                                                           _inputs, _metrics, cute)  # noqa: F401  (cute: fixture)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                       reason="SM100 GPU required"),
    pytest.mark.usefixtures("cute"),
]


def _impl():
    impl = MiniMaxH3VSAImpl(num_heads=_HEADS, head_size=_DIM, causal=False, softmax_scale=_DIM**-0.5)
    impl.layer_idx = 0
    return impl


def _check(fused, ref):
    for name, j in (("O", 0), ("dK", 2), ("dV", 3)):
        assert torch.equal(fused[j], ref[j]), name
    avg, rel = _metrics(ref[1], fused[1])
    assert avg < _GRAD_TOL[0] and rel < _GRAD_TOL[1], (avg, rel)


@pytest.mark.parametrize("case", _GPU_CASES, ids=_GPU_IDS)
def test_tile128_stacked_op_equals_three_gradient_op(case):
    """Op level: vsa_train_fwd(qkv, None, None, tile=128) vs the same kernels on the chunks; one dqkv gradient."""
    meta = _gpu_meta(case)
    impl = _impl()
    inputs = _inputs(meta, seed=3)
    x = impl.preprocess_qkv(torch.cat([t.detach() for t in inputs[:3]], dim=0), meta).detach()
    mask = torch.ops.fastvideo_kernel.vsa_h3_block_map(*x[:2].chunk(2, dim=0), meta.variable_block_sizes, 128,
                                                       meta.num_prefix_tiles, meta.num_video_tiles, meta.video_topk,
                                                       meta.exempt)
    dout = torch.randn_like(x[:1])
    ops = torch.ops.fastvideo_kernel
    qkv = x.clone().requires_grad_(True)
    out, _ = ops.vsa_train_fwd(qkv, None, None, mask, meta.variable_block_sizes, 128)
    (dqkv, ) = torch.autograd.grad(out, (qkv, ), dout)
    fused = (out.detach(), *dqkv.chunk(3, dim=0))
    q, k, v = (t.clone().requires_grad_(True) for t in x.chunk(3, dim=0))
    out3, _ = ops.vsa_train_fwd(q, k, v, mask, meta.variable_block_sizes, 128)
    ref = (out3.detach(), *torch.autograd.grad(out3, (q, k, v), dout))
    _check(fused, ref)


def _grad_fn_names(t):
    seen, stack, names = set(), [t.grad_fn], set()
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        names.add(type(node).__name__)
        stack.extend(fn for fn, _ in node.next_functions)
    return names


@pytest.mark.parametrize("case", _GPU_CASES[1:2], ids=_GPU_IDS[1:2])
def test_h3_forward_qkv_tile128_fused_vs_unfused(case):
    """H3 forward_qkv at tile 128: fused_qkv_grad on == off (O/dK/dV bitwise, dQ tolerance); ON has no chunk/split
    backward between the stacked input and the attention op, OFF has one."""
    impl = _impl()
    inputs = _inputs(_gpu_meta(case), seed=5)
    results, graphs = {}, {}
    for fused in (True, False):
        meta = _gpu_meta(case)
        meta.fused_qkv_grad = fused
        q, k, v = (t.detach().clone().requires_grad_(True) for t in inputs[:3])
        x = impl.preprocess_qkv(torch.cat([q, k, v], dim=0), meta)
        attn = impl.forward_qkv(x, meta)
        graphs[fused] = _grad_fn_names(attn)
        out = impl.postprocess_output(attn, meta)
        results[fused] = (out.detach(), *torch.autograd.grad(out, (q, k, v), inputs[3]))
    _check(results[True], results[False])
    assert not any("Split" in n or "Chunk" in n for n in graphs[True]), graphs[True]
    assert any("Split" in n or "Chunk" in n for n in graphs[False]), graphs[False]
