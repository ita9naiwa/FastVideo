# SPDX-License-Identifier: Apache-2.0
"""VSA-H3 training tiles through the shared _TilePermutation: byte-identical to the index_put path."""

import pytest
import torch

from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAImpl, MiniMaxH3VSAMetadataBuilder

# token grid (37, 24, 42) + text/audio prefix: 38,010 rows; [4, S, 2, 128] is >= 2**25 elements (eligible).
_SPEC = dict(raw_latent_shape=(37, 48, 84), patch_size=(1, 2, 2), prefix_segments=(300, 0, 414))
_LAYOUTS = [dict(), dict(tile_layout="chunk256", merge_prefix=True)]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _impl():
    return MiniMaxH3VSAImpl(num_heads=2, head_size=128, causal=False, softmax_scale=128**-0.5, prefix="blocks.0.attn")


def _metadata(spec=_SPEC, **layout):
    return MiniMaxH3VSAMetadataBuilder().build(current_timestep=0,
                                               VSA_sparsity=0.75,
                                               device=torch.device("cuda"),
                                               **spec,
                                               **layout)


def _qkvg(md, heads=2, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(4, md.total_seq_length, heads, 128, device="cuda", dtype=torch.bfloat16, generator=g)


def _index_put_path(impl, md, x):
    """The unchanged holder/index_put route: drop the trusted index state for this call."""
    state = md.__dict__.pop("_tile_index_state")
    try:
        md.tile_buf_holder.buffer = None
        return impl.tile(x, md)
    finally:
        md._tile_index_state = state


@pytest.mark.parametrize("layout", _LAYOUTS, ids=["cube", "chunk256"])
def test_training_tile_matches_index_put_path(layout):
    impl, md = _impl(), _metadata(**layout)
    x = _qkvg(md).requires_grad_(True)
    up = torch.randn(4, md.variable_block_sizes.numel() * 256, 2, 128, device="cuda", dtype=torch.bfloat16)
    reference = _index_put_path(impl, md, x)
    expected_grad, = torch.autograd.grad(reference, x, up)
    actual = impl.tile(x, md)
    assert actual.data_ptr() != md.tile_buf_holder.buffer.data_ptr(), "training output must not alias the holder"
    grad, = torch.autograd.grad(actual, x, up)
    assert torch.equal(actual, reference) and torch.equal(grad, expected_grad)
    assert torch.equal(impl.postprocess_output(actual.detach(), md), x.detach())


@pytest.mark.parametrize("layout", _LAYOUTS, ids=["cube", "chunk256"])
def test_repeated_live_training_calls(layout):
    impl, md = _impl(), _metadata(**layout)
    xs = [_qkvg(md, seed=s).requires_grad_(True) for s in (1, 2)]
    outs = [impl.tile(x, md) for x in xs]  # two live graphs before any backward
    assert outs[0].data_ptr() != outs[1].data_ptr()
    for x, out in zip(xs, outs, strict=True):
        grad, = torch.autograd.grad(out, x, torch.ones_like(out))
        assert torch.equal(grad, torch.ones_like(x))  # each row lands in exactly one slot


def test_fallbacks_keep_holder_path():
    # Own geometry: the in-place write below bumps the version of cached index tensors.
    impl, md = _impl(), _metadata(dict(_SPEC, prefix_segments=(301, 0, 414)))
    x = _qkvg(md)
    assert impl.tile(x, md) is md.tile_buf_holder.buffer  # no grad
    small = torch.randn(4, md.total_seq_length, 1, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    assert impl.tile(small, md) is md.tile_buf_holder.buffer  # below the 2**25-element threshold
    md.non_pad_index.add_(0)  # in-place write: cached proof no longer trusted
    md.tile_buf_holder.buffer = None
    # The training op re-checks the recorded versions itself and falls back to the untile scatter into a fresh buffer:
    # same values as the holder path, but never the shared holder (see bugs/...-h3-tile-holder-autograd-reuse).
    out = impl.tile(x.requires_grad_(True), md)
    assert md.tile_buf_holder.buffer is None
    with torch.no_grad():
        assert torch.equal(out, impl.tile(x.detach(), md))
