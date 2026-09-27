"""Tile-parametric VSA call shapes used by external trainers (TILE = 256 and 128) match the eager adapter paths.

Forward: _build_sparse_tensors(q_block=kv_block=TILE) + flash_attn_fwd(mask_mod=_build_vbs_vector_mask_mod(TILE)).
Backward: _build_sparse_tensors(map repeated x2, child sizes, q_block=TILE, kv_block=TILE // 2,
force_q_sparse_block_size=TILE) + flash_attn_bwd(mask_mod=_build_vbs_mask_mod(TILE // 2)).
At TILE = 256 that is the Q256 training route; at TILE = 128 it must equal the Q128 route (one list per 128-row tile).
"""
import pytest
import torch

from fastvideo_kernel import block_sparse_attn_cute_fwd as K


def _fwd(q, k, v, bmap, sizes, tile):
    _, _, fa_fwd, _ = K._load_fa4_cute()
    fwd, _ = K._build_sparse_tensors(bmap, sizes, q_len=q.shape[1], q_block_size=tile, kv_block_size=tile,
                                     need_backward=False)
    return fa_fwd(q, k, v, mask_mod=K._build_vbs_vector_mask_mod(tile), aux_tensors=[sizes],
                  block_sparse_tensors=fwd, return_lse=True)[:2]


def _bwd(q, k, v, out, dout, lse, bmap, sizes, tile):
    half = tile // 2
    child = torch.stack((sizes.clamp(0, half), (sizes - half).clamp(0, half)), -1).flatten()
    _, bwd = K._build_sparse_tensors(bmap.repeat_interleave(2, -1), child, q_len=q.shape[1], q_block_size=tile,
                                     kv_block_size=half, need_backward=True, need_forward=False,
                                     force_q_sparse_block_size=tile)
    _, _, _, fa_bwd = K._load_fa4_cute()
    return fa_bwd(q, k, v, out, dout, lse, mask_mod=K._build_vbs_mask_mod(half), aux_tensors=[child],
                  block_sparse_tensors=bwd)[:3]


@pytest.mark.parametrize("tile", [256, 128])
def test_tile_callshapes_match_eager(tile):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM10x required")
    pytest.importorskip("flash_attn.cute.interface")
    from .test_vsa256_backward import _check, _GRAD_TOL
    torch.manual_seed(1128 + tile)
    nb, heads, dim = (12, 4, 128) if tile == 256 else (24, 4, 128)
    n = tile * nb
    sizes = torch.full((nb, ), tile, device="cuda", dtype=torch.int32)
    sizes[3], sizes[-1] = tile * 131 // 256, tile // 3  # partial KV blocks, including a tail
    bmap = torch.rand(1, heads, nb, nb, device="cuda") > 0.6
    bmap[..., 0] = True
    q, k, v = [torch.randn(1, n, heads, dim, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    dout = torch.randn_like(q)
    leaves = [t.detach().requires_grad_(True) for t in (q, k, v)]
    ref = K._CuteAttentionQ256Training if tile == 256 else K._CuteAttentionQ128
    out_ref, lse_ref = ref.apply(*leaves, bmap, sizes)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, leaves, dout)
    try:
        out, lse = _fwd(q, k, v, bmap, sizes, tile)
    except ValueError as exc:
        if tile == 128 and "multiple of 256" in str(exc):
            pytest.skip("FA4 provider predates single-stage block-sparse 128 (attention-impl q_stage rule)")
        raise
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(lse, lse_ref, rtol=0, atol=0)
    dq, dk, dv = _bwd(q, k, v, out_ref.detach(), dout, lse_ref, bmap, sizes, tile)
    _check("tile call-shape dQ", dq_ref, dq, _GRAD_TOL)  # dQ accumulates atomically: order may differ
    torch.testing.assert_close(dk, dk_ref, rtol=0, atol=0)
    torch.testing.assert_close(dv, dv_ref, rtol=0, atol=0)


def test_fine_kv_blocks_merge_to_fa4_tile():
    """kv_block_size 64 at tile 128 builds the same lists as kv_block_size 128 on the parent map."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("flash_attn.cute.interface")
    torch.manual_seed(7)
    nb = 10
    sizes = torch.randint(0, 129, (nb, ), device="cuda", dtype=torch.int32)
    bmap = torch.rand(1, 2, nb, nb, device="cuda") > 0.5
    child = torch.stack((sizes.clamp(0, 64), (sizes - 64).clamp(0, 64)), -1).flatten()
    _, fine = K._build_sparse_tensors(bmap.repeat_interleave(2, -1), child, q_len=128 * nb, q_block_size=128,
                                      kv_block_size=64, need_backward=True, need_forward=False,
                                      force_q_sparse_block_size=128)
    _, coarse = K._build_sparse_tensors(bmap, sizes, q_len=128 * nb, q_block_size=128, kv_block_size=128,
                                        need_backward=True, need_forward=False)
    for a, b in zip(fine[:4], coarse[:4], strict=True):
        assert torch.equal(a, b)
    assert tuple(fine.block_size) == tuple(coarse.block_size) == (128, 128)
