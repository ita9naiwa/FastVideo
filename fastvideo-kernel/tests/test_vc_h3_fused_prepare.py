"""H3 VSA backend, fused VC route: packed rows -> prepare_vsa (padded FP8 + FP32 tile pools) must match the generic
route (tile() -> _pool_tiles -> vc_preprocess.prepare) up to the accepted K-centering tolerance, select the same
blocks, and fall back to the tiled path wherever the generic route would not reach VC attention."""
import math
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root: the fastvideo package

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10
                                or not os.environ.get("FASTVIDEO_VSA_VC_ROOT"),
                                reason="needs SM10x and FASTVIDEO_VSA_VC_ROOT (VC-enabled FA4 checkout)")


def _setup(monkeypatch, thw=(9, 20, 26), prefix=(300, 5, 130), sparsity=0.75, heads=4):
    monkeypatch.setenv("FASTVIDEO_VSA_VC", "1")
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    from fastvideo.attention.backends import video_sparse_attn_h3 as h3
    meta = h3.MiniMaxH3VSAMetadataBuilder().build(current_timestep=0, raw_latent_shape=(thw[0], 2 * thw[1], 2 * thw[2]),
                                                  patch_size=(1, 2, 2), VSA_sparsity=sparsity, prefix_segments=prefix,
                                                  device=torch.device("cuda"))
    impl = h3.MiniMaxH3VSAImpl(num_heads=heads, head_size=128, causal=False, softmax_scale=128**-0.5)
    impl.layer_idx = 0
    return h3, meta, impl


def _routes(h3, meta, impl, q, k, v):
    """Generic (tiled) and fused outputs through the backend entry points, plus both block masks."""
    # Generic: tiled inputs have the padded length, which never takes the fused branch in forward().
    tq, tk, tv = (impl.tile(t, meta).clone() for t in (q, k, v))
    generic = impl.postprocess_output(impl.forward(tq, tk, tv, None, meta), meta)
    qkv = torch.cat([q, k, v], dim=0)
    assert impl.preprocess_qkv(qkv, meta) is qkv, "fused route must hand the packed rows through untouched"
    fused = impl.postprocess_output(impl.forward(*qkv.chunk(3, dim=0), None, meta), meta)
    return generic, fused, tq, tk, tv


@pytest.mark.parametrize("thw,prefix", [((9, 20, 26), (300, 5, 130)), ((12, 10, 18), (250, 1, 800)),
                                        ((8, 15, 28), (420, 1, 420, 810))])
@torch.inference_mode()
def test_fused_route_matches_generic(monkeypatch, thw, prefix):
    h3, meta, impl = _setup(monkeypatch, thw, prefix)
    from fastvideo_kernel.block_sparse_attn_cute_fwd import prepare_vsa_vc_fwd_bshd
    from flash_attn.cute.vc_preprocess import prepare
    torch.manual_seed(11)
    n = meta.total_seq_length
    q, k, v = (torch.randn(1, n, 4, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    generic, fused, tq, tk, tv = _routes(h3, meta, impl, q, k, v)
    # Prepared tensors: Q/V codes exact, K codes within 1 E4M3 ulp on <= 1e-4 of elements (K-mean summation order).
    pg = prepare(tq, tk, tv, smooth=False, bshd=True)
    padded = meta.variable_block_sizes.numel() * 256
    p2o = torch.full((padded, ), -1, dtype=torch.int64, device="cuda")
    p2o[meta.untile_combined_index] = torch.arange(n, device="cuda")
    pf, pools = prepare_vsa_vc_fwd_bshd(q, k, v, p2o, meta.variable_block_sizes, 256,
                                        torch.arange(padded, device="cuda"), padded, 0)
    for name in ("q", "v"):
        assert torch.equal(pg[name].view(torch.uint8), pf[name].view(torch.uint8)), name

    def ordered(c):
        c = c.view(torch.uint8).to(torch.int32)
        return torch.where(c >= 128, -(c - 128), c)

    dk = (ordered(pg["k"]) - ordered(pf["k"])).abs()
    assert int(dk.max()) <= 1 and float((dk > 0).float().mean()) <= 1e-4
    for name in ("qs", "ks", "vs"):
        assert torch.allclose(pg[name], pf[name], rtol=2e-6, atol=1e-7), name
    pool_q, pool_k = (h3._pool_tiles(t, meta.variable_block_sizes, 256) for t in (tq, tk))
    assert torch.allclose(pools[0], pool_q, rtol=3e-5, atol=3e-6) and torch.allclose(pools[1], pool_k, rtol=3e-5, atol=3e-6)
    # Same block selection from either set of pools.
    def mask(pq, pk):
        scores = torch.matmul(pq, pk.transpose(-2, -1)) / math.sqrt(128)
        return h3._build_block_mask(scores, meta.num_prefix_tiles, meta.num_video_tiles, meta.VSA_sparsity, meta.exempt)
    assert torch.equal(mask(pool_q, pool_k), mask(pools[0], pools[1]))
    # Outputs: same kernel and blocks; only the K 1-ulp codes differ.
    assert generic.shape == fused.shape == q.shape and torch.isfinite(fused).all()
    rel = ((fused.float() - generic.float()).norm() / generic.float().norm()).item()
    assert rel < 1e-3, rel


@torch.inference_mode()
def test_fused_route_shuffled_untile_and_dense_layer(monkeypatch):
    h3, meta, impl = _setup(monkeypatch, sparsity=0.0)
    torch.manual_seed(3)
    n = meta.total_seq_length
    q, k, v = (torch.randn(1, n, 4, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    generic, fused, *_ = _routes(h3, meta, impl, q, k, v)
    rel = ((fused.float() - generic.float()).norm() / generic.float().norm()).item()
    assert rel < 1e-3, rel
    # A different geometry with the same padded length must rebuild the cached inverse map.
    meta2 = h3.MiniMaxH3VSAMetadataBuilder().build(current_timestep=0, raw_latent_shape=(9, 40, 52), patch_size=(1, 2, 2),
                                                   VSA_sparsity=0.75, prefix_segments=(301, 4, 130),
                                                   device=torch.device("cuda"))
    meta2.tile_buf_holder = meta.tile_buf_holder
    generic2, fused2, *_ = _routes(h3, meta2, impl, q, k, v)
    rel2 = ((fused2.float() - generic2.float()).norm() / generic2.float().norm()).item()
    assert rel2 < 1e-3, rel2


def test_fused_route_fallbacks(monkeypatch):
    h3, meta, impl = _setup(monkeypatch)
    x = torch.randn(3, meta.total_seq_length, 4, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        assert impl._vc_fused_route(x, meta)
    assert not impl._vc_fused_route(x.clone().requires_grad_(True), meta)          # training keeps the tiled path
    with torch.inference_mode():
        assert not impl._vc_fused_route(x.float(), meta)                            # FP32
        monkeypatch.setenv("FASTVIDEO_VSA_VC", "0")
        assert not impl._vc_fused_route(x, meta)                                    # VC off
        monkeypatch.setenv("FASTVIDEO_VSA_VC", "1")
        monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "0")
        assert not impl._vc_fused_route(x, meta)                                    # Triton backend
        monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
        monkeypatch.setattr(h3, "probe_enabled", lambda: "/tmp/probe")
        assert not impl._vc_fused_route(x, meta)                                    # probe recording
        monkeypatch.setattr(h3, "probe_enabled", lambda: None)
        meta64 = h3.MiniMaxH3VSAMetadataBuilder().build(current_timestep=0, raw_latent_shape=(9, 20, 26),
                                                        patch_size=(1, 2, 2), VSA_sparsity=0.75,
                                                        prefix_segments=(300, 5, 130), device=torch.device("cuda"),
                                                        tile_size=64)
        assert not impl._vc_fused_route(x, meta64)                                  # tile-64 route
        assert impl.preprocess_qkv(x, meta).data_ptr() == x.data_ptr()
