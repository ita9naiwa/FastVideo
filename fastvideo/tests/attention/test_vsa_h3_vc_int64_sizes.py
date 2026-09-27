"""Non-fused VC route (FASTVIDEO_VSA_VC=1 through the public 128/256 entries) with the H3 builder's int64 tile sizes: the
tile-128 call used to raise in _validate_vc_prepared. It must run and equal the int32-sizes call bitwise; tile 256 is
unchanged; malformed sizes still raise."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10
                                or not os.environ.get("FASTVIDEO_VSA_VC_ROOT"),
                                reason="needs SM10x and FASTVIDEO_VSA_VC_ROOT (VC-enabled FA4 checkout)")

HEADS = 4


def _inputs(tile, seed=0):
    from fastvideo.attention.backends import video_sparse_attn_h3 as h3
    # Real s085k32 pack 1 doc 0 geometry (latent 62x10x17, prefix segments 64/1/692), chunk layout with partial tiles.
    meta = h3.MiniMaxH3VSAMetadataBuilder().build(current_timestep=0,
                                                  raw_latent_shape=(62, 20, 34),
                                                  patch_size=(1, 2, 2),
                                                  VSA_sparsity=0.85,
                                                  prefix_segments=(64, 1, 692),
                                                  device=torch.device("cuda"),
                                                  tile_size=tile,
                                                  tile_layout=f"chunk{tile}",
                                                  merge_prefix=True)
    sizes = meta.variable_block_sizes
    assert sizes.dtype == torch.int64 and bool((sizes < tile).any())
    n = sizes.numel()
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (torch.randn(1, n * tile, HEADS, 128, device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(3))
    block_map = torch.rand(1, HEADS, n, n, device="cuda", generator=g) < 0.3
    block_map[..., 0] = True
    return q, k, v, block_map, sizes


@pytest.fixture
def vc(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_VSA_VC", "1")
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    from fastvideo_kernel import block_sparse_attn_256 as entry
    return entry


@pytest.mark.parametrize("tile", [128, 256])
@torch.no_grad()
def test_vc_int64_sizes_match_int32(vc, tile):
    fwd = vc.block_sparse_attn_128_bshd if tile == 128 else vc.block_sparse_attn_256_bshd
    q, k, v, block_map, sizes = _inputs(tile)
    out64 = fwd(q, k, v, block_map, sizes)[0]
    out32 = fwd(q, k, v, block_map, sizes.to(torch.int32))[0]
    assert torch.equal(out64, out32)


@torch.no_grad()
def test_vc_malformed_sizes_still_raise(vc):
    q, k, v, block_map, sizes = _inputs(128)
    for bad in (sizes[:-1], sizes.cpu(), sizes.float()):
        with pytest.raises(ValueError):
            vc.block_sparse_attn_128_bshd(q, k, v, block_map, bad)
