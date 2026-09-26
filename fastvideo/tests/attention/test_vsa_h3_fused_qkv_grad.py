# SPDX-License-Identifier: Apache-2.0
"""VSA-H3 forward_qkv: one fused Q/K/V gradient, same math as the chunked forward."""

import pytest
import torch

from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAImpl, MiniMaxH3VSAMetadataBuilder

_SPEC = dict(raw_latent_shape=(37, 48, 84), patch_size=(1, 2, 2), prefix_segments=(300, 0, 414))
_LAYOUTS = [dict(), dict(tile_layout="chunk256", merge_prefix=True)]
_H = 3  # [3, S, 3, 128] >= 2**25 elements: the invocation-owned _TilePermutation training tile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                                reason="SM10x CUDA required")


def _impl():
    return MiniMaxH3VSAImpl(num_heads=_H, head_size=128, causal=False, softmax_scale=128**-0.5, prefix="blocks.0.attn")


def _metadata(**layout):
    return MiniMaxH3VSAMetadataBuilder().build(current_timestep=0,
                                               VSA_sparsity=0.75,
                                               device=torch.device("cuda"),
                                               tile_size=256,
                                               **_SPEC,
                                               **layout)


def _inputs(md, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(3, md.total_seq_length, _H, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    up = torch.randn(1, md.total_seq_length, _H, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    return x.requires_grad_(True), up


def _reaches(fn, name):
    stack, seen = [fn], set()
    while stack:
        f = stack.pop()
        if f is None or f in seen:
            continue
        seen.add(f)
        if type(f).__name__ == name:
            return True
        stack.extend(n for n, _ in f.next_functions)
    return False


def _run(impl, md, x, up, fused):
    t = impl.preprocess_qkv(x, md)
    out = impl.forward_qkv(t, md) if fused else impl.forward(*t.chunk(3, dim=0), None, md)
    split = _reaches(out.grad_fn, "SplitBackward0")
    o = impl.postprocess_output(out, md)
    grad, = torch.autograd.grad(o, x, up)
    return o.detach(), grad, split


@pytest.mark.parametrize("pack_tails", ["1", "0"])
@pytest.mark.parametrize("layout", _LAYOUTS, ids=["cube", "chunk256"])
def test_fused_matches_chunked(monkeypatch, layout, pack_tails):
    monkeypatch.setenv("FASTVIDEO_VSA_PACK_TAILS", pack_tails)
    impl, md = _impl(), _metadata(**layout)
    x, up = _inputs(md, 0)
    # Fused first: a freed chunked-path gradient of the same size must not be able to masquerade as its result.
    o, g, split = _run(impl, md, x, up, fused=True)
    ref_o, ref_g, ref_split = _run(impl, md, x, up, fused=False)
    _, rep_g, _ = _run(impl, md, x, up, fused=False)
    assert ref_split and not split, "fused path must not leave a ChunkBackward concat"
    assert torch.equal(o, ref_o) and torch.equal(g[1:], ref_g[1:])  # O, dK, dV bitwise
    spread = (rep_g[0].float() - ref_g[0].float()).abs().max()  # dQ uses FP32 atomics (ruling 23)
    assert (g[0].float() - ref_g[0].float()).abs().max() <= max(2 * spread, 2**-7)


def test_repeated_live_calls():
    impl, md = _impl(), _metadata(tile_layout="chunk256", merge_prefix=True)
    xs = [_inputs(md, s) for s in (1, 2)]
    refs = [_run(impl, md, x, up, fused=False) for x, up in xs]
    outs = [impl.postprocess_output(impl.forward_qkv(impl.preprocess_qkv(x, md), md), md) for x, _ in xs]
    for (x, up), o, (ref_o, ref_g, _) in zip(xs, outs, refs, strict=True):  # two live graphs before any backward
        grad, = torch.autograd.grad(o, x, up)
        assert torch.equal(o, ref_o) and torch.equal(grad[1:], ref_g[1:])


def test_no_grad_matches_forward():
    impl, md = _impl(), _metadata()
    x, _ = _inputs(md, 3)
    with torch.no_grad():
        t = impl.preprocess_qkv(x, md).clone()
        assert torch.equal(impl.forward_qkv(t, md), impl.forward(*t.chunk(3, dim=0), None, md))
