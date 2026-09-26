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
    import fastvideo.attention.backends.video_sparse_attn  # noqa: F401  (registers vsa_tile_permute_*)
    x = torch.randn(2, 1000, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    partition, nonpad = torch.randperm(1000, device="cuda"), torch.randperm(1536, device="cuda")[:1000]
    untile = nonpad[torch.argsort(partition)]
    torch.library.opcheck(torch.ops.fastvideo_kernel.vsa_tile_permute_fwd.default,
                          (x, partition, nonpad, untile, 1536, partition._version, nonpad._version),
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


@pytest.mark.parametrize("loss_on", ["out", "lse"])
def test_vsa256_ops_single_output_losses(monkeypatch, loss_on):
    """O-only and LSE-only losses: the unused output's gradient stays None (no materialized zeros), matching the old
    autograd.Function behaviour, and gradients equal the old path (dQ within the eager repeat spread)."""
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    from fastvideo_kernel import vsa256_ops  # noqa: F401
    from fastvideo_kernel.vsa_tail_backward import TailTraining
    q, k, v, block_map, sizes = _inputs(seed=5)
    seen = {}
    orig = torch.ops.fastvideo_kernel.vsa256_bwd

    def spy(dout, *args):
        seen["dlse_is_none"] = args[-1] is None
        return orig(dout, *args)

    monkeypatch.setattr(vsa256_ops, "vsa256_bwd", spy)

    def grads(fn):
        out, lse = fn()
        target = out if loss_on == "out" else lse
        return torch.autograd.grad(target.float().square().sum(), (q, k, v))

    new = grads(lambda: torch.ops.fastvideo_kernel.vsa256_fwd(q, k, v, block_map, sizes, True))
    if loss_on == "out":
        assert seen["dlse_is_none"]
    old1 = grads(lambda: TailTraining.apply(q, k, v, block_map, sizes))
    old2 = grads(lambda: TailTraining.apply(q, k, v, block_map, sizes))
    assert torch.equal(new[1], old1[1]) and torch.equal(new[2], old1[2])
    spread = (old2[0].float() - old1[0].float()).abs().max()
    assert (new[0].float() - old1[0].float()).abs().max() <= max(2 * spread, 1e-3)


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


def test_tile_permute_pair_matches_autograd_function():
    """Trusted indices: the op equals _TilePermutation (shared body) bitwise, forward and gradient, fresh output per
    call. After an index tensor is modified in place, the op falls back to the untile scatter (still correct)."""
    from fastvideo.attention.backends.video_sparse_attn import _TilePermutation
    x = torch.randn(3, 700, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    partition, nonpad = torch.randperm(700, device="cuda"), torch.randperm(1024, device="cuda")[:700]
    untile = nonpad[torch.argsort(partition)]
    op = torch.ops.fastvideo_kernel.vsa_tile_permute_fwd
    padded = op(x, partition, nonpad, untile, 1024, partition._version, nonpad._version)
    ref = _TilePermutation.apply(x, partition, nonpad, 1024)
    assert torch.equal(padded, ref)
    assert torch.equal(padded[:, untile], x) and not padded.index_fill(1, untile, 0).any()
    g = torch.randn_like(padded)
    (gx, ) = torch.autograd.grad(padded, x, g)
    (gr, ) = torch.autograd.grad(ref, x, g)
    assert torch.equal(gx, gr)
    assert padded.data_ptr() != op(x, partition, nonpad, untile, 1024, partition._version, nonpad._version).data_ptr()
    stale_version = partition._version
    partition.add_(0)  # bumps the version: the builder-trust contract no longer holds
    fallback = op(x, partition, nonpad, untile, 1024, stale_version, nonpad._version)
    assert torch.equal(fallback, ref)
    (gf, ) = torch.autograd.grad(fallback, x, g)
    assert torch.equal(gf, gr)



def test_tile_permute_backward_uses_forward_index():
    """The backward is the adjoint of the index the forward consumed: trusted partition/nonpad with a diverged untile
    (mutated in place before the forward) still routes gradient through partition/nonpad."""
    x = torch.zeros(1, 3, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    partition, nonpad = torch.tensor([0, 1, 2], device="cuda"), torch.tensor([0, 1, 2], device="cuda")
    untile = nonpad.clone()
    untile[1:] = torch.tensor([0, 0], device="cuda")  # diverged; partition/nonpad keep their recorded versions
    padded = torch.ops.fastvideo_kernel.vsa_tile_permute_fwd(x, partition, nonpad, untile, 4, partition._version,
                                                             nonpad._version)
    g = torch.tensor([3.0, 5.0, 1.0, 7.0], device="cuda", dtype=torch.bfloat16).view(1, 4, 1, 1).expand(1, 4, 1, 128)
    (gx, ) = torch.autograd.grad(padded, x, g)
    assert torch.equal(gx[0, :, 0, 0].float().cpu(), torch.tensor([3.0, 5.0, 1.0]))

@pytest.mark.parametrize("layout,merge", [("cube", False), ("chunk256", False), ("chunk256", True)])
def test_h3_block_fullgraph(monkeypatch, layout, merge):
    """torch.compile(dynamic=True, fullgraph=True) of the H3 training block (tile -> pool/top-k -> attention -> untile)
    compiles without graph breaks; O, dK, dV bitwise equal to eager."""
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAImpl, MiniMaxH3VSAMetadataBuilder
    # 8 heads x ~15k rows keeps q/k/v above the tile-permutation eligibility size (the product training path; smaller
    # inputs keep the pre-existing holder scatter, which is outside this seam)
    impl = MiniMaxH3VSAImpl(num_heads=8, head_size=128, causal=False, softmax_scale=128**-0.5)
    impl.layer_idx = 0
    meta = MiniMaxH3VSAMetadataBuilder().build(current_timestep=0, raw_latent_shape=(20, 40, 72), patch_size=(1, 2, 2),
                                               VSA_sparsity=0.75, prefix_segments=(250, 1, 0, 300),
                                               device=torch.device("cuda"), tile_layout=layout, merge_prefix=merge)

    def block(q, k, v):
        x = impl.preprocess_qkv(torch.cat([q, k, v], dim=0), meta)
        q2, k2, v2 = x.chunk(3, dim=0)
        return impl.postprocess_output(impl.forward(q2, k2, v2, None, meta), meta)

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, meta.total_seq_length, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
               for _ in range(3))
    assert 3 * q.numel() >= 2**25
    dout = torch.randn_like(q)
    eager = block(q, k, v)
    ge = torch.autograd.grad(eager, (q, k, v), dout)
    torch._dynamo.reset()
    compiled = torch.compile(block, fullgraph=True, dynamic=True)(q, k, v)
    gc = torch.autograd.grad(compiled, (q, k, v), dout)
    assert torch.equal(compiled, eager)
    assert torch.equal(gc[1], ge[1]) and torch.equal(gc[2], ge[2])
    # the real-size training call takes the permutation op (not the holder scatter)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        block(q, k, v)
    names = {e.key for e in prof.key_averages()}
    assert "fastvideo_kernel::vsa_tile_permute_fwd" in names and "fastvideo_kernel::vsa256_fwd" in names, names


def test_h3_block_sac_policy_sees_op_keys(monkeypatch):
    """Selective activation checkpointing: the policy sees the same op keys (vsa256_fwd, vsa_tile_permute_fwd) in eager
    and compiled mode; MUST_SAVE on vsa256_fwd with recompute elsewhere gives the eager gradients."""
    import functools

    from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAImpl, MiniMaxH3VSAMetadataBuilder
    impl = MiniMaxH3VSAImpl(num_heads=8, head_size=128, causal=False, softmax_scale=128**-0.5)
    impl.layer_idx = 0
    meta = MiniMaxH3VSAMetadataBuilder().build(current_timestep=0, raw_latent_shape=(20, 40, 72), patch_size=(1, 2, 2),
                                               VSA_sparsity=0.75, prefix_segments=(250, 1, 0, 300),
                                               device=torch.device("cuda"), tile_layout="chunk256", merge_prefix=True)

    def block(q, k, v):
        x = impl.preprocess_qkv(torch.cat([q, k, v], dim=0), meta)
        q2, k2, v2 = x.chunk(3, dim=0)
        return impl.postprocess_output(impl.forward(q2, k2, v2, None, meta), meta)

    keys = {"eager": set(), "compiled": set()}
    mode = {"now": "eager"}
    must_save = {torch.ops.fastvideo_kernel.vsa256_fwd.default}

    def policy(ctx, op, *args, **kwargs):
        keys[mode["now"]].add(str(op))
        return CheckpointPolicy.MUST_SAVE if op in must_save else CheckpointPolicy.PREFER_RECOMPUTE

    def sac(q, k, v):
        return checkpoint(block, q, k, v, use_reentrant=False,
                          context_fn=functools.partial(create_selective_checkpoint_contexts, policy))

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, meta.total_seq_length, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
               for _ in range(3))
    dout = torch.randn_like(q)
    ref = torch.autograd.grad(block(q, k, v), (q, k, v), dout)
    got = torch.autograd.grad(sac(q, k, v), (q, k, v), dout)
    assert torch.equal(got[1], ref[1]) and torch.equal(got[2], ref[2])
    mode["now"] = "compiled"
    torch._dynamo.reset()
    # aot_eager: the policy/op-key contract without Inductor codegen (Inductor + SAC + dynamic is recorded separately in
    # the candidate notes: a PyTorch codegen failure in the untile-gather backward, outside these ops)
    got_c = torch.autograd.grad(torch.compile(sac, backend="aot_eager", fullgraph=True, dynamic=True)(q, k, v),
                                (q, k, v), dout)
    assert torch.equal(got_c[1], ref[1]) and torch.equal(got_c[2], ref[2])
    for m in ("eager", "compiled"):
        assert any("vsa256_fwd" in k for k in keys[m]), (m, keys[m])
        assert any("vsa_tile_permute_fwd" in k for k in keys[m]), (m, keys[m])


@pytest.mark.parametrize("sparsity", [0.75, 0.5])  # both H3 operating points (ruling 67; spec f1c85686 / cd24cc2f)
def test_h3_block_h3mh_compile_config(monkeypatch, sparsity):
    """h3mh's training compile configuration (conductor ruling 31942): torch.compile(checkpointed(block), backend='inductor',
    mode='default', dynamic=True, fullgraph=True) with fail_on_recompile_limit_hit and use_duck_shape=False, SAC inside the
    compiled region (MUST_SAVE vsa256_fwd, PREFER_RECOMPUTE elsewhere), attention metadata passed as an argument; one run per
    operating point (s=0.75 and s=0.5, a fresh compile each, as a training run uses one sparsity). Ten H3
    geometries (small ones take the size-gated tile path in eager) compile once: 0 graph breaks, 0 recompiles, O/dK/dV bitwise
    vs eager, dQ within the nondeterministic-atomics tolerance."""
    import functools
    import torch.fx.experimental._config as fx_config
    from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    monkeypatch.setattr(torch._dynamo.config, "fail_on_recompile_limit_hit", True)
    monkeypatch.setattr(fx_config, "use_duck_shape", False)
    from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAImpl, MiniMaxH3VSAMetadataBuilder
    from .test_vsa256_backward import _metrics
    impl = MiniMaxH3VSAImpl(num_heads=8, head_size=128, causal=False, softmax_scale=128**-0.5)
    impl.layer_idx = 0

    def block(q, k, v, meta):
        x = impl.preprocess_qkv(torch.cat([q, k, v], dim=0), meta)
        q2, k2, v2 = x.chunk(3, dim=0)
        return impl.postprocess_output(impl.forward(q2, k2, v2, None, meta), meta)

    policy = lambda ctx, op, *a, **k: (CheckpointPolicy.MUST_SAVE if op == torch.ops.fastvideo_kernel.vsa256_fwd.default
                                       else CheckpointPolicy.PREFER_RECOMPUTE)
    compiled = torch.compile(lambda q, k, v, meta: checkpoint(block, q, k, v, meta, use_reentrant=False, context_fn=functools.partial(
        create_selective_checkpoint_contexts, policy)), backend="inductor", mode="default", dynamic=True, fullgraph=True)
    geometries = [((42, 14, 24), (175, 1, 170, 402)), ((42, 20, 20), (175, 1, 170, 402)), ((37, 16, 56), (250, 1, 0, 300)),
                  ((102, 14, 24), (120, 1, 0, 402)), ((62, 26, 24), (175, 1, 170, 0)), ((42, 30, 34), (300, 1, 0, 402)),
                  ((47, 32, 30), (175, 1, 170, 402)), ((37, 24, 62), (175, 0, 0, 402)), ((77, 14, 52), (250, 1, 170, 402)),
                  ((72, 26, 36), (175, 1, 170, 402))]
    torch._dynamo.reset()
    counters = torch._dynamo.utils.counters  # process-global: compare deltas from this test's start
    graphs0, breaks0 = counters["stats"]["unique_graphs"], sum(counters["graph_break"].values())
    for i, (raw, prefix) in enumerate(geometries):
        meta = MiniMaxH3VSAMetadataBuilder().build(current_timestep=0, raw_latent_shape=raw, patch_size=(1, 2, 2),
                                                   VSA_sparsity=sparsity, prefix_segments=prefix, device=torch.device("cuda"),
                                                   tile_layout="chunk256", merge_prefix=True)
        before = counters["stats"]["unique_graphs"]
        outs = []
        for fn in (compiled, block):
            torch.manual_seed(100 + i)
            q, k, v = (torch.randn(1, meta.total_seq_length, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                       for _ in range(3))
            out = fn(q, k, v, meta)
            dout = torch.randn_like(out)
            outs.append((out.detach(), *torch.autograd.grad(out, (q, k, v), dout)))
        (co, cq, ck, cv), (eo, eq, ek, ev) = outs
        assert i == 0 or counters["stats"]["unique_graphs"] == before, f"recompiled at geometry {i} {raw}"
        assert torch.equal(co, eo) and torch.equal(ck, ek) and torch.equal(cv, ev), f"geometry {i} {raw}"
        avg_abs, max_rel = _metrics(eq.float(), cq)
        assert avg_abs < 1e-3 and max_rel < 0.25, (i, avg_abs, max_rel)
    assert sum(counters["graph_break"].values()) == breaks0
    assert counters["stats"]["unique_graphs"] - graphs0 == 1
    torch._dynamo.reset()
