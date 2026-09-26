"""Custom-op seam for BF16 VSA-256 CuTe training: torch.library opcheck, eager == previous autograd.Function numerics,
and fullgraph torch.compile of the H3 block with O/dK/dV bitwise equal to eager (dQ uses nondeterministic atomics)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root: the fastvideo package

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                                reason="needs SM10x (FA4 CuTe VSA-256 training)")


def _inputs(b=1, n_tiles=12, heads=4, seed=0, requires_grad=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    sizes = torch.tensor([256 if i % 3 else 131 for i in range(n_tiles)], device="cuda", dtype=torch.int64)
    sizes[-1] = 37
    scores = torch.rand(b, heads, n_tiles, n_tiles, device="cuda", generator=g)
    block_map = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, scores.topk(4, -1).indices, True)
    q, k, v = (torch.randn(b, n_tiles * 256, heads, 128, device="cuda", dtype=torch.bfloat16, generator=g)
               .requires_grad_(requires_grad) for _ in range(3))
    return q, k, v, block_map, sizes


@pytest.mark.parametrize("pack_tails", [True, False])
def test_vsa256_ops_opcheck(pack_tails):
    from fastvideo_kernel import vsa256_ops  # noqa: F401
    q, k, v, block_map, sizes = _inputs()
    torch.library.opcheck(torch.ops.fastvideo_kernel.vsa256_fwd.default, (q, k, v, block_map, sizes, pack_tails),
                          test_utils=("test_schema", "test_faketensor", "test_autograd_registration"))
    x = torch.randn(2, 1000, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    index = torch.randperm(1536, device="cuda")[:1000]
    torch.library.opcheck(torch.ops.fastvideo_kernel.vsa_tile_permute_fwd.default, (x, index, 1536),
                          test_utils=("test_schema", "test_faketensor", "test_autograd_registration"))


@pytest.mark.parametrize("pack_tails", ["1", "0"])
@pytest.mark.parametrize("b", [1, 2])
def test_vsa256_ops_match_autograd_function(monkeypatch, pack_tails, b):
    """The public training entry (now the op pair) equals the previous TailTraining / Q256Training autograd paths:
    O, LSE, dK, dV bitwise; dQ within the eager repeat spread of the old path."""
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    monkeypatch.setenv("FASTVIDEO_VSA_PACK_TAILS", pack_tails)
    from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_256_bshd
    from fastvideo_kernel.block_sparse_attn_cute_fwd import _CuteAttentionQ256Training
    from fastvideo_kernel.vsa_tail_backward import TailTraining
    old_fn = TailTraining if pack_tails == "1" else _CuteAttentionQ256Training
    q, k, v, block_map, sizes = _inputs(b=b)
    dout = torch.randn_like(q)

    def grads(fn):
        out, lse = fn()
        return (out.detach(), lse.detach(), *torch.autograd.grad(out, (q, k, v), dout))

    new = grads(lambda: block_sparse_attn_256_bshd(q, k, v, block_map, sizes))
    old1 = grads(lambda: old_fn.apply(q, k, v, block_map, sizes))
    old2 = grads(lambda: old_fn.apply(q, k, v, block_map, sizes))
    for i, name in ((0, "out"), (1, "lse"), (3, "dk"), (4, "dv")):
        assert torch.equal(new[i], old1[i]), name
    spread = (old2[2].float() - old1[2].float()).abs().max()
    assert (new[2].float() - old1[2].float()).abs().max() <= max(2 * spread, 1e-3)


def test_vsa256_ops_changed_maps_and_outstanding_graphs(monkeypatch):
    """Two forwards outstanding before their backwards, and a different block map / sizes of the same shape: each
    backward uses its own saved metadata (no stale capture)."""
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_256_bshd
    q, k, v, map1, sizes1 = _inputs(seed=1)
    _, _, _, map2, sizes2 = _inputs(seed=2)
    sizes2 = sizes2.flip(0).contiguous()
    dout = torch.randn_like(q)
    o1 = block_sparse_attn_256_bshd(q, k, v, map1, sizes1)[0]
    o2 = block_sparse_attn_256_bshd(q, k, v, map2, sizes2)[0]
    g2 = torch.autograd.grad(o2, (k, v), dout, retain_graph=True)
    g1 = torch.autograd.grad(o1, (k, v), dout)
    r1 = torch.autograd.grad(block_sparse_attn_256_bshd(q, k, v, map1, sizes1)[0], (k, v), dout)
    r2 = torch.autograd.grad(block_sparse_attn_256_bshd(q, k, v, map2, sizes2)[0], (k, v), dout)
    assert all(torch.equal(a, b) for a, b in zip(g1, r1, strict=True)) and all(torch.equal(a, b) for a, b in zip(g2, r2, strict=True))


def test_tile_permute_pair_roundtrip():
    from fastvideo_kernel import vsa256_ops  # noqa: F401
    x = torch.randn(3, 700, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    index = torch.randperm(1024, device="cuda")[:700]
    padded = torch.ops.fastvideo_kernel.vsa_tile_permute_fwd(x, index, 1024)
    assert torch.equal(padded[:, index], x) and not padded.index_fill(1, index, 0).any()
    g = torch.randn_like(padded)
    (gx, ) = torch.autograd.grad(padded, x, g)
    assert torch.equal(gx, g[:, index])
    assert padded.data_ptr() != torch.ops.fastvideo_kernel.vsa_tile_permute_fwd(x, index, 1024).data_ptr()


@pytest.mark.parametrize("layout,merge", [("cube", False), ("chunk256", False), ("chunk256", True)])
def test_h3_block_fullgraph(monkeypatch, layout, merge):
    """torch.compile(dynamic=True, fullgraph=True) of the H3 training block (tile -> pool/top-k -> attention -> untile)
    compiles without graph breaks; O, dK, dV bitwise equal to eager."""
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAImpl, MiniMaxH3VSAMetadataBuilder
    impl = MiniMaxH3VSAImpl(num_heads=4, head_size=128, causal=False, softmax_scale=128**-0.5)
    impl.layer_idx = 0
    meta = MiniMaxH3VSAMetadataBuilder().build(current_timestep=0, raw_latent_shape=(20, 20, 36), patch_size=(1, 2, 2),
                                               VSA_sparsity=0.75, prefix_segments=(250, 1, 0, 300),
                                               device=torch.device("cuda"), tile_layout=layout, merge_prefix=merge)

    def block(q, k, v):
        x = impl.preprocess_qkv(torch.cat([q, k, v], dim=0), meta)
        q2, k2, v2 = x.chunk(3, dim=0)
        return impl.postprocess_output(impl.forward(q2, k2, v2, None, meta), meta)

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, meta.total_seq_length, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
               for _ in range(3))
    dout = torch.randn_like(q)
    eager = block(q, k, v)
    ge = torch.autograd.grad(eager, (q, k, v), dout)
    torch._dynamo.reset()
    compiled = torch.compile(block, fullgraph=True, dynamic=True)(q, k, v)
    gc = torch.autograd.grad(compiled, (q, k, v), dout)
    assert torch.equal(compiled, eager)
    assert torch.equal(gc[1], ge[1]) and torch.equal(gc[2], ge[2])
