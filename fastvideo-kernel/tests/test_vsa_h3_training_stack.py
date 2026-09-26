"""H3 training stack, ACCEPTED SUBSET (conductor 33324): fullgraph custom op 5d13c02f (+ pad rows 2e805f18, block-map op,
exact capped top-k) + empty-query children with compiled pruning through the op. The five correctness families of
candidate h3-training-stack-integration, adapted to this subset (the conditional inputs pack_tails policy, prefix split
and fused-QKV gradient are not present; their tests return with them).

F1 dispatch / schema / option reach, F2 padding and lifetime, F3 gradient contracts, F4 compile / shape / layout
(fullgraph=True, dynamic=True, SAC, CUDA Graph replay), F5 cross-stack controls.

Reach is observed inside the opaque op bodies (forward sparse plan, tail backward plan), which run real Python in eager
and compiled mode alike; op identity reach via the profiler. Numerics: O, LSE, dK, dV bitwise where the kernels are unchanged; dQ uses FP32
atomics (order may differ), so it is held to the suite-wide _GRAD_TOL.
"""
import functools
import inspect
import json
import os
import sys
from pathlib import Path

import pytest
import torch

from fastvideo_kernel import block_sparse_attn_cute_fwd as adapter
from .test_vsa256_backward import _check, _GRAD_TOL
from .test_vsa256_tail_backward import _poison_case

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root: the fastvideo package

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                                reason="needs SM10x (FA4 CuTe VSA-256 training)")

# Real-size H3 geometry (tile-permutation op path: 3 * q.numel() >= 2**25); cube has partial tiles with <= 128 rows,
# so pruning has wholly padded Q128 children to skip; 4 prefix tiles give an interior split.
_BIG = dict(raw_latent_shape=(20, 40, 72), patch_size=(1, 2, 2), prefix_segments=(250, 1, 0, 300))
_SMALL = dict(raw_latent_shape=(16, 16, 24), patch_size=(1, 2, 2), prefix_segments=(64, 32, 16))
_ALL_ON = dict(query_pad_pruning=True)
_ALL_OFF = dict(query_pad_pruning=False)


@pytest.fixture(autouse=True)
def _cute(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    for name in ("FASTVIDEO_VSA_VC", "FASTVIDEO_VSA_TRITON", "FASTVIDEO_KERNEL_VSA_FORCE_TRITON",
                 "FASTVIDEO_VSA_PACK_TAILS"):
        monkeypatch.delenv(name, raising=False)
    yield
    os.environ.pop("FASTVIDEO_VSA_PACK_TAILS", None)  # _public may set it on trees without the policy input
    torch._dynamo.reset()


@pytest.fixture
def reach(monkeypatch):
    """Records what the op bodies actually did: the forward sparse plan (and, opt-in, the full block map it received)
    and the packed-tail plan (pruned or not)."""
    from fastvideo_kernel import vsa_tail_backward as tail
    from fastvideo_kernel import vsa256_ops
    rec = dict(fwd=[], prepare=[], prepare_sizes=[], tail=[], maps=None, stacked=[], bwd=[])
    chunks = vsa256_ops._chunks

    def spy_chunks(q, k, v):  # op bodies: k = v = None is the stacked [3B, S, H, D] input
        rec["stacked"].append(k is None)
        return chunks(q, k, v)

    load = adapter._load_fa4_cute

    def spy_load():
        a, b, c, bwd = load()

        @functools.wraps(bwd)  # keeps the signature: _check_workspace_support inspects it
        def spy_bwd(*args, **kwargs):
            grads = bwd(*args, **kwargs)
            bufs = [kwargs.get(n) for n in ("dq", "dk", "dv")]
            rec["bwd"].append(dict(fused=all(b_ is not None for b_ in bufs)
                                   and len({b_.untyped_storage().data_ptr() for b_ in bufs}) == 1
                                   and all(g is b_ for g, b_ in zip(grads[:3], bufs, strict=True)),
                                   storages=len({g.untyped_storage().data_ptr() for g in grads[:3] if g is not None})))
            return grads

        return a, b, c, spy_bwd
    bst = adapter._build_sparse_tensors

    def spy_bst(block_map, *args, **kwargs):
        if kwargs.get("need_backward") is False and kwargs.get("q_block_size") == 256:  # Q256 training forward plan
            rec["fwd"].append(tuple(block_map.shape))
            if rec["maps"] is not None:
                rec["maps"].append(block_map.clone())
        return bst(block_map, *args, **kwargs)

    prepare = tail._prepare

    def spy_prepare(routes, sizes, query_sizes=None):
        rec["prepare"].append(query_sizes is not None)
        rec["prepare_sizes"].append(query_sizes)  # no host sync here (CUDA Graph capture); counted by the test
        return prepare(routes, sizes, query_sizes)

    tail_backward = tail.tail_backward

    def spy_tail(*args, **kwargs):
        rec["tail"].append(True)
        return tail_backward(*args, **kwargs)

    monkeypatch.setattr(adapter, "_build_sparse_tensors", spy_bst)
    monkeypatch.setattr(vsa256_ops, "_chunks", spy_chunks)
    monkeypatch.setattr(adapter, "_load_fa4_cute", spy_load)
    monkeypatch.setattr(tail, "_prepare", spy_prepare)
    monkeypatch.setattr(tail, "tail_backward", spy_tail)
    return rec


def _op_calls(fn, *args):
    """Run fn under the profiler; return (result, number of fastvideo_kernel::vsa256_fwd calls)."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        res = fn(*args)
    return res, sum(e.count for e in prof.key_averages() if e.key == "fastvideo_kernel::vsa256_fwd")


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


def _clear(rec):
    for v in rec.values():
        if v is not None:
            v.clear()


def _impl(heads):
    from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAImpl
    impl = MiniMaxH3VSAImpl(num_heads=heads, head_size=128, causal=False, softmax_scale=128**-0.5)
    impl.layer_idx = 0
    return impl


def _features():
    """Which conditional inputs this tree carries (one module serves the stack and its single-input branches)."""
    from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAMetadataBuilder
    params = inspect.signature(MiniMaxH3VSAMetadataBuilder.build).parameters
    return dict(policy="pack_tails" in params, fused="fused_qkv_grad" in params)


def _need(feature):
    if not _features()[feature]:
        pytest.skip(f"this tree does not carry the {feature} input")


def _pack(meta):
    """Effective pack policy: the metadata bool (policy input) or the seam's env default (on)."""
    return getattr(meta, "pack_tails", os.environ.get("FASTVIDEO_VSA_PACK_TAILS", "1") == "1")


def _meta(spec=_BIG, layout="cube", merge=False, **opts):
    """Test metadata; pack_tails defaults to True (always pack) so pruning is reachable, "auto" = the policy (policy
    trees only; without the policy input the seam's env default applies)."""
    from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAMetadataBuilder
    feats = _features()
    opts.setdefault("pack_tails", True)
    if not feats["policy"]:
        assert opts.pop("pack_tails") is True, "pack policy requested on a tree without the policy input"
    if not feats["fused"]:
        opts.pop("fused_qkv_grad", None)
    return MiniMaxH3VSAMetadataBuilder().build(current_timestep=0, VSA_sparsity=opts.pop("VSA_sparsity", 0.75),
                                               device=torch.device("cuda"),
                                               tile_layout=layout, merge_prefix=merge, **spec, **opts)


def _layer_block(impl, meta):
    """DistributedAttention_VSA's single-rank data path: stack, tile, gate-free forward_qkv, untile."""

    def block(q, k, v):
        x = impl.preprocess_qkv(torch.cat([q, k, v], dim=0), meta)
        if hasattr(impl, "forward_qkv"):
            return impl.postprocess_output(impl.forward_qkv(x, meta), meta)
        return impl.postprocess_output(impl.forward(*x.chunk(3, dim=0), None, meta), meta)

    return block


def _leaves(meta, heads, batch=1, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (batch, meta.total_seq_length, heads, 128)
    qkv = [torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=g).requires_grad_(True) for _ in range(3)]
    return qkv, torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=g)


def _run(fn, leaves, dout):
    out = fn(*leaves)
    return (out.detach(), *torch.autograd.grad(out, leaves, dout))


def _same(tag, got, ref, dq_bitwise=False):
    for name, g, r in zip(("O", "dQ", "dK", "dV"), got, ref, strict=True):
        if name == "dQ" and not dq_bitwise and r.any():
            _check(f"{tag} dQ", r, g, _GRAD_TOL)
        else:
            assert torch.equal(g, r), f"{tag}: {name} differs"


def _op_inputs(b=1, n=12, heads=2, dim=128, seed=0, sizes=None):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if sizes is None:
        sizes = [256 if i % 3 else 131 for i in range(n)]
        sizes[-1], sizes[1] = 37, 0
    sizes = torch.tensor(sizes, device="cuda", dtype=torch.int32)
    n = sizes.numel()
    scores = torch.rand(b, heads, n, n, device="cuda", generator=g)
    block_map = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, scores.topk(min(4, n), -1).indices, True)
    qkv = torch.randn(3 * b, n * 256, heads, dim, device="cuda", dtype=torch.bfloat16, generator=g)
    pad = (torch.arange(256, device="cuda") >= sizes[:, None]).flatten()
    dout = torch.randn(b, n * 256, heads, dim, device="cuda", dtype=torch.bfloat16, generator=g)
    return qkv, block_map, sizes, dout.masked_fill(pad[None, :, None, None], 0), pad


def _public(q, k, v, block_map, sizes, **kw):
    from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_256_bshd
    if "pack_tails" in kw and "pack_tails" not in inspect.signature(block_sparse_attn_256_bshd).parameters:
        # tree without the policy input: the seam reads FASTVIDEO_VSA_PACK_TAILS per call
        os.environ["FASTVIDEO_VSA_PACK_TAILS"] = "1" if kw.pop("pack_tails") else "0"
    return block_sparse_attn_256_bshd(q, k, v, block_map, sizes, **kw)


def _proof(sizes):
    untile = torch.arange(sizes.numel() * 256, device="cuda")
    return dict(query_sizes=sizes, query_untile=untile, query_versions=(untile._version, sizes._version))


def _op_grads(qkv, block_map, sizes, dout, dlse=None, compiled=False, **kw):
    """Public training entry; returns (O, dQ, dK, dV)."""
    leaves = [t.detach().clone().requires_grad_(True) for t in qkv.chunk(3, dim=0)]

    def call(q, k, v):
        return _public(q, k, v, block_map, sizes, **kw)

    fn = torch.compile(call, fullgraph=True, dynamic=True) if compiled else call
    if dlse is None:
        out = fn(*leaves)[0]
        grads = torch.autograd.grad(out, leaves, dout)
    else:  # the public wrapper detaches LSE; take it from the op itself
        from fastvideo_kernel import vsa256_ops
        out, lse = vsa256_ops.training_attention(*leaves, block_map, sizes, **kw)
        grads = torch.autograd.grad((out, lse), leaves, (dout, dlse.masked_fill(~torch.isfinite(lse), 0)))
    return (out.detach(), *grads)


# ---------------------------------------------------------------------------------------------------------------------
# F1: public training dispatch, op schemas, option reach
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "fullgraph"])
@pytest.mark.parametrize("pack", [True, False, "auto"], ids=["pack1", "pack0", "auto"])
def test_f1_public_dispatch_reach(reach, pack, compiled):
    """Real H3 layer block, cube: the vsa256 op pair runs (eager and fullgraph dynamic), the metadata pack policy
    (builder pack_tails True / False / "auto") reaches the backward as the op's static bool, query pruning reaches the
    packed backward; each arm equals the pruning-off eager arm with the same policy."""
    if pack is not True:
        _need("policy")
    impl = _impl(8)
    meta = _meta(pack_tails=pack, **_ALL_ON)
    if pack == "auto":
        pack = meta.pack_tails
    assert _pack(meta) is pack
    assert bool((meta.variable_block_sizes <= 128).any())
    leaves, dout = _leaves(meta, 8)
    ref = _run(_layer_block(impl, _meta(pack_tails=_pack(meta), **_ALL_OFF)), leaves, dout)
    _clear(reach)
    block = _layer_block(impl, meta)
    got, ops = _op_calls(_run, torch.compile(block, fullgraph=True, dynamic=True) if compiled else block, leaves, dout)
    assert ops >= 1 and len(reach["fwd"]) == 1  # the compiled profile also lists the op's inner call events
    assert reach["tail"] == ([True] if pack else [])
    assert reach["prepare"] == ([True] if pack else [])  # cube + trusted geometry: pruned main lists
    _same(f"pack={pack} compiled={compiled}", got, ref)


def test_f1_policy_decided_once_in_builder(monkeypatch, reach):
    """pack_tails="auto": the builder reads the environment once; later environment changes do not reach the op."""
    _need("policy")
    meta = _meta(pack_tails="auto", **_ALL_ON)
    policy = meta.pack_tails
    monkeypatch.setenv("FASTVIDEO_VSA_PACK_TAILS", "0" if policy else "1")
    leaves, dout = _leaves(meta, 8, seed=2)
    _run(_layer_block(_impl(8), meta), leaves, dout)
    assert reach["tail"] == ([True] if policy else [])


@pytest.mark.parametrize("pack", [True, "auto"], ids=["policy_off_always_pack", "policy_on_auto"])
def test_f1_pack_policy_flag_reach(reach, pack):
    """Same-head ablation of the pack policy: pack_tails=True (policy OFF = the pre-policy default, always pack)
    vs "auto" (policy ON): the op's static bool follows the metadata; results equal the matching reference."""
    _need("policy")
    impl = _impl(8)
    meta = _meta(pack_tails=pack, **_ALL_ON)
    leaves, dout = _leaves(meta, 8, seed=4)
    _clear(reach)
    got = _run(_layer_block(impl, meta), leaves, dout)
    assert reach["tail"] == ([True] if meta.pack_tails else [])
    print("ablation-diagnostic", json.dumps(dict(pack_tails_request=str(pack), effective_pack_tails=meta.pack_tails)))
    ref = _run(_layer_block(impl, _meta(pack_tails=meta.pack_tails, **_ALL_OFF)), leaves, dout)
    _same(f"pack={pack}", got, ref)


@pytest.mark.parametrize("on", [False, True], ids=["pruning_off", "pruning_on"])
def test_f1_ablation_flags_reach(reach, on):
    """The query_pad_pruning metadata flag switches exactly its option (diagnostic: pruned Q128 children)."""
    impl = _impl(8)
    meta = _meta(pack_tails=True, query_pad_pruning=on)
    leaves, dout = _leaves(meta, 8, seed=1)
    ref = _run(_layer_block(impl, _meta(pack_tails=True, **_ALL_OFF)), leaves, dout)
    _clear(reach)
    got = _run(_layer_block(impl, meta), leaves, dout)
    qs = reach["prepare_sizes"][0]
    pruned = 0 if qs is None else int((qs <= 0).sum() + (qs <= 128).sum())
    print("ablation-diagnostic", json.dumps(dict(query_pad_pruning=on, pruned_q128_children=pruned)))
    assert reach["prepare"] == [on] and (pruned > 0) is on
    _same(f"query_pad_pruning={on}", got, ref)


@pytest.mark.parametrize("proof", [False, True], ids=["no_proof", "proof"])
@pytest.mark.parametrize("pack_tails", [True, False], ids=["pack1", "pack0"])
def test_f1_op_schema_fake_autograd(proof, pack_tails):
    """opcheck (schema, fake tensors, autograd registration) with and without the proof inputs; backward fake arity."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from fastvideo_kernel import vsa256_ops  # noqa: F401
    qkv, block_map, sizes, _, _ = _op_inputs()
    qkv.requires_grad_(True)
    q, k, v = (t.detach().requires_grad_(True) for t in qkv.chunk(3))
    p = _proof(sizes)
    extra = (p["query_sizes"], p["query_untile"], *p["query_versions"]) if proof else (None, None, 0, 0)
    torch.library.opcheck(torch.ops.fastvideo_kernel.vsa256_fwd.default,
                          (q, k, v, block_map, sizes, *extra, pack_tails),
                          test_utils=("test_schema", "test_faketensor", "test_autograd_registration"))
    out, lse = torch.ops.fastvideo_kernel.vsa256_fwd(q, k, v, block_map, sizes, *extra, pack_tails)
    assert out.shape == (qkv.shape[0] // 3, *qkv.shape[1:]) and lse.shape == (out.shape[0], out.shape[2], out.shape[1])
    with FakeTensorMode(allow_non_fake_inputs=True):
        grads = torch.ops.fastvideo_kernel.vsa256_bwd(torch.empty_like(out), q, k, v, out, lse, block_map, sizes,
                                                      *extra, pack_tails, None)
    assert len(grads) == 3
    assert grads[0].shape == q.shape


@pytest.mark.parametrize("pack_tails", [True, False], ids=["pack1", "pack0"])
@pytest.mark.parametrize("layout", ["dim0_chunks", "dim2_unbind"])
def test_f1_strided_views_fake_matches_real(layout, pack_tails):
    """Eligible non-default layouts reach the op with fakes that match the real outputs: dim-0 chunks of a stacked
    [3B, S, H, D] tensor and dim-2 unbind views of a [B, S, 3, H, D] allocation (strided, 16-byte aligned). opcheck's fake-tensor test
    compares fake and real output metadata (shapes, strides, dtypes) for the forward and for the backward op."""
    from fastvideo_kernel import vsa256_ops
    qkv, block_map, sizes, dout, _ = _op_inputs(seed=18)
    if layout == "dim0_chunks":
        q, k, v = qkv.requires_grad_(True).chunk(3, dim=0)
    else:
        packed = torch.stack(list(qkv.chunk(3, dim=0)), dim=2).requires_grad_(True)  # [B, S, 3, H, D]
        q, k, v = packed.unbind(2)
        assert not q.is_contiguous()
    assert vsa256_ops.training_eligible(q, k, v, block_map)
    pack = pack_tails and all(t.is_contiguous() for t in (q, k, v))
    args = (q, k, v, block_map, sizes, None, None, 0, 0, pack)
    torch.library.opcheck(torch.ops.fastvideo_kernel.vsa256_fwd.default, args,
                          test_utils=("test_schema", "test_faketensor", "test_autograd_registration"))
    out, lse = torch.ops.fastvideo_kernel.vsa256_fwd(*args)
    assert out.is_contiguous() and lse.is_contiguous()
    torch.library.opcheck(torch.ops.fastvideo_kernel.vsa256_bwd.default,
                          (dout, q.detach(), k.detach(), v.detach(), out.detach(), lse.detach(), block_map, sizes,
                           None, None, 0, 0, pack, None), test_utils=("test_schema", "test_faketensor"))


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_f1_sac_op_keys_eager_equal_compiled(reach, backend):
    """SAC over the all-on block: identical fastvideo_kernel op keys in eager and fullgraph dynamic (aot_eager and
    Inductor); MUST_SAVE vsa256_fwd with recompute elsewhere keeps O/dK/dV bitwise and every option's reach."""
    from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
    impl = _impl(8)
    meta = _meta(**_ALL_ON)
    block = _layer_block(impl, meta)
    keys = {"eager": set(), "compiled": set()}
    mode = {"now": "eager"}

    def policy(ctx, op, *args, **kwargs):
        keys[mode["now"]].add(str(op))
        return (CheckpointPolicy.MUST_SAVE if op is torch.ops.fastvideo_kernel.vsa256_fwd.default else
                CheckpointPolicy.PREFER_RECOMPUTE)

    def sac(q, k, v):
        return checkpoint(block, q, k, v, use_reentrant=False,
                          context_fn=functools.partial(create_selective_checkpoint_contexts, policy))

    leaves, dout = _leaves(meta, 8, seed=3)
    ref = _run(block, leaves, dout)
    _clear(reach)
    got = _run(sac, leaves, dout)
    _same("sac eager", got, ref)
    assert reach["prepare"] == [True] and len(reach["fwd"]) == 1
    mode["now"] = "compiled"
    _clear(reach)
    got = _run(torch.compile(sac, backend=backend, fullgraph=True, dynamic=True), leaves, dout)
    _same("sac compiled", got, ref)
    assert reach["prepare"] == [True] and len(reach["fwd"]) == 1
    kernel_keys = {m: {k for k in keys[m] if k.startswith("fastvideo_kernel.")} for m in keys}
    assert kernel_keys["eager"] == kernel_keys["compiled"], kernel_keys
    assert {"fastvideo_kernel.vsa256_fwd.default", "fastvideo_kernel.vsa_tile_permute_fwd.default"} <= \
        kernel_keys["eager"], kernel_keys


# ---------------------------------------------------------------------------------------------------------------------
# F2: padding and lifetime
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "fullgraph"])
@pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")])
def test_f2_public_route_invalid_slot_poison(reach, poison, compiled):
    """Public packed route with and without the query proof: a non-finite K/V row reachable only through invalid
    packed slots stays out of dQ/dK/dV on the target rows; pruned == full bitwise (O, dK, dV) on those rows."""
    sizes, routes, bad, target = _poison_case("pad_row0")
    sizes = torch.tensor(sizes, device="cuda", dtype=torch.int32)
    g = torch.Generator(device="cuda").manual_seed(615)
    qkv = torch.randn(3, 1536, 2, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    qkv[1:, bad] = poison
    pad = (torch.arange(256, device="cuda") >= sizes[:, None]).flatten()  # the proof's contract: zero padded dO
    dout = torch.randn(1, 1536, 2, 128, device="cuda", dtype=torch.bfloat16, generator=g).masked_fill(
        pad[None, :, None, None], 0)
    full = _op_grads(qkv, routes, sizes, dout, compiled=compiled)
    _clear(reach)
    pruned = _op_grads(qkv, routes, sizes, dout, compiled=compiled, **_proof(sizes))
    assert reach["tail"] == [True] and reach["prepare"] == [True]
    for name, p_, f_ in zip(("O", "dQ", "dK", "dV"), pruned, full, strict=True):
        p_, f_ = p_[:, target], f_[:, target]
        assert torch.isfinite(p_).all() and torch.isfinite(f_).all(), f"{name} non-finite on target rows"
        if name == "dQ":
            _check("poison dQ", f_, p_, _GRAD_TOL)
        else:
            assert torch.equal(p_, f_), name


@pytest.mark.parametrize("case", ["edges", "odd", "empty", "overflow"])
def test_f2_query_sizes_edges_fit_overflow(reach, case):
    """Proof-carrying pruning vs the full backward over query sizes 0/1/128/129/256, odd/empty parent lists and a plan
    that overflows the bounded tail capacity: O/dK/dV bitwise, dQ within tolerance and exactly zero on padded rows."""
    sizes = dict(edges=[0, 1, 128, 129, 256, 255, 127, 256], odd=[129, 0, 256, 1, 200], empty=[0] * 6,
                 overflow=[127] * 3 + [129] * 3 + [255] * 6)[case]
    qkv, block_map, sizes, dout, pad = _op_inputs(sizes=sizes, seed=4)
    full = _op_grads(qkv, block_map, sizes, dout)
    _clear(reach)
    pruned = _op_grads(qkv, block_map, sizes, dout, **_proof(sizes))
    assert reach["prepare"] == [True]
    _same(f"prune {case}", pruned, full)
    assert torch.count_nonzero(pruned[1][:, pad]) == 0


def _h3_small(meta_kw=None):
    impl = _impl(2)
    meta = _meta(_SMALL, **(meta_kw or {}))
    return impl, meta


def _tiled(impl, meta, seed=0):
    """Tiled Q/K/V leaves (tiling happens before, outside any compiled region; gradients are taken here)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        return [impl.tile(torch.randn(1, meta.total_seq_length, 2, 128, device="cuda", dtype=torch.bfloat16,
                                      generator=g), meta).clone().requires_grad_(True) for _ in range(3)]


def _fresh(tiled):
    return [t.detach().clone().requires_grad_(True) for t in tiled]


def _padded_slot(meta):
    tile = int(torch.nonzero(meta.variable_block_sizes <= 128)[0])
    return tile * meta.tile_elems + meta.tile_elems - 1


def _attend(impl, meta, compiled, between=None):
    def fn(q, k, v):
        out = impl.forward(q, k, v, None, meta)
        if between is not None:
            between(meta)
        return impl.postprocess_output(out, meta)

    return torch.compile(fn, fullgraph=True, dynamic=True) if compiled else fn


def _grads(fn, tiled):
    out = fn(*tiled)
    return [out.detach(), *torch.autograd.grad(out.float().pow(2).sum(), tiled)]


@pytest.fixture
def geometry_cache():
    from fastvideo.attention.backends.video_sparse_attn_h3 import _h3_tile_geometry
    yield
    _h3_tile_geometry.cache_clear()  # tests below mutate the cached geometry tensors


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "fullgraph"])
def test_f2_replaced_map_before_postprocess_uses_forward_map(reach, compiled):
    """A pruned forward's postprocess uses the pinned forward map even after the metadata field is replaced by a
    same-shape map that selects a padded row (the pin is carried through traced caller code in fullgraph)."""
    impl, _ = _h3_small()
    results = []
    for swap in (False, True):
        _, meta = _h3_small()
        tiled = _tiled(impl, meta)

        def replace(m):
            swapped = m.untile_combined_index.clone()
            swapped[0] = _padded_slot(m)
            m.untile_combined_index = swapped

        _clear(reach)
        results.append(_grads(_attend(impl, meta, compiled, replace if swap else None), tiled))
        torch._dynamo.reset()
        assert reach["prepare"] == [True]
    _same(f"replaced map compiled={compiled}", results[1], results[0])


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "fullgraph"])
def test_f2_map_mutated_before_postprocess(reach, geometry_cache, compiled):
    """In-place mutation of the trusted map between forward and postprocess is rejected: eager by postprocess's
    version check (9147 contract); inside one fullgraph region by autograd's saved-tensor check on the map the vsa256
    op saved (raised while AOT traces the joint graph; different message, same rejection)."""
    impl, meta = _h3_small()
    tiled = _tiled(impl, meta)
    slot = _padded_slot(meta)

    def mutate(m):
        m.untile_combined_index[0] = slot

    if compiled:
        with pytest.raises(Exception, match="modified by an inplace operation"):
            _attend(impl, meta, True, mutate)(*tiled)
        return
    out = impl.forward(*tiled, None, meta)
    mutate(meta)
    with pytest.raises(RuntimeError, match="modified in place"):
        impl.postprocess_output(out, meta)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "fullgraph"])
@pytest.mark.parametrize("target", ["untile", "sizes"])
def test_f2_proof_tensor_mutated_before_forward_is_not_trusted(reach, geometry_cache, compiled, target):
    """A trusted map or sizes tensor changed in place (value-preserving) after the builder: no pruning in either mode
    (eager checks at forward, compiled inside the backward op); results equal the unpruned run."""
    impl, meta = _h3_small()
    tiled = _tiled(impl, meta)
    getattr(meta, "untile_combined_index" if target == "untile" else "variable_block_sizes").add_(0)
    _clear(reach)
    got = _grads(_attend(impl, meta, compiled), tiled)
    assert reach["prepare"] == [False]
    torch._dynamo.reset()
    meta._query_pad_state = None
    ref = _grads(_attend(impl, meta, False), _fresh(tiled))
    _same("mutated before forward", got, ref)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "fullgraph"])
def test_f2_map_mutated_after_postprocess_is_rejected(geometry_cache, compiled):
    """Mutating a saved proof tensor after postprocess, before backward, fails the backward (saved-tensor check)."""
    impl, meta = _h3_small()
    tiled = _tiled(impl, meta)
    out = _attend(impl, meta, compiled)(*tiled)
    meta.untile_combined_index.add_(0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.float().pow(2).sum().backward()


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "fullgraph"])
def test_f2_two_outstanding_forwards(reach, compiled):
    """Two pruned forwards outstanding before either backward (reverse order) each keep their own saved state."""
    impl, meta = _h3_small()
    fn = _attend(impl, meta, compiled)
    runs = [_tiled(impl, meta, seed=s) for s in (5, 6)]
    seq = [_grads(fn, _fresh(tiled)) for tiled in runs]
    _clear(reach)
    outs = [fn(*tiled) for tiled in runs]
    grads = [torch.autograd.grad(out.float().pow(2).sum(), tiled) for tiled, out in reversed(list(zip(runs, outs)))]
    assert reach["prepare"] == [True, True]
    for out, g, ref in zip(outs, reversed(grads), seq, strict=True):
        _same("outstanding", [out.detach(), *g], ref)


# ---------------------------------------------------------------------------------------------------------------------
# F3: gradient contracts
# ---------------------------------------------------------------------------------------------------------------------


def test_f3_nonzero_dlse_forces_full_lists(reach):
    qkv, block_map, sizes, dout, _ = _op_inputs(seed=7)
    dlse = torch.randn(1, 2, qkv.shape[1], device="cuda") * .1
    full = _op_grads(qkv, block_map, sizes, dout, dlse=dlse)
    _clear(reach)
    got = _op_grads(qkv, block_map, sizes, dout, dlse=dlse, **_proof(sizes))
    assert reach["prepare"] == [False]
    _same("dlse", got, full)


@pytest.mark.parametrize("variant", ["no_proof", "stale_untile", "no_untile"])
def test_f3_padded_dout_without_guarantee_keeps_baseline(reach, variant):
    """Arbitrary nonzero dO on padded rows: without a valid proof the backward stays the full baseline
    (TailTraining without query sizes)."""
    from fastvideo_kernel.vsa_tail_backward import TailTraining
    qkv, block_map, sizes, _, _ = _op_inputs(seed=8)
    dout = torch.randn(1, qkv.shape[1], 2, 128, device="cuda", dtype=torch.bfloat16)
    kw = {}
    if variant == "stale_untile":
        kw = _proof(sizes)
        kw["query_untile"].add_(0)
    elif variant == "no_untile":
        kw = dict(query_sizes=sizes)
    got = _op_grads(qkv, block_map, sizes, dout, **kw)
    assert reach["prepare"] == [False]
    leaves = [t.detach().clone().requires_grad_(True) for t in qkv.chunk(3)]
    out = TailTraining.apply(*leaves, block_map, sizes)[0]
    ref = (out.detach(), *torch.autograd.grad(out, leaves, dout))
    _same(f"baseline {variant}", got, ref)


@pytest.mark.parametrize("gate", [False, True], ids=["no_gate", "gate"])
@pytest.mark.parametrize("layout", ["cube", "chunk256"])
def test_f3_gate_and_chunk_controls(reach, layout, gate):
    """Cube prunes with and without the coarse gate (the gate's pooled branch may give padded rows nonzero total dQ;
    only the fine branch's padded dO must be zero); chunk256 has no query-padding proof and never prunes."""
    impl = _impl(2)
    results = []
    for trusted in (True, False):
        meta = _meta(_SMALL, layout=layout)
        if not trusted:
            meta._query_pad_state = None
        tiled = _tiled(impl, meta)
        g = torch.Generator(device="cuda").manual_seed(9)
        gate_t = torch.randn(tiled[0].shape, device="cuda", dtype=torch.bfloat16, generator=g) * .1 if gate else None
        _clear(reach)
        results.append(_grads(lambda q, k, v: impl.postprocess_output(impl.forward(q, k, v, gate_t, meta), meta),
                              tiled))
        assert reach["prepare"] == [trusted and layout == "cube"]
    _same(f"{layout} gate={gate}", results[0], results[1])


def _dense_fp32_dq(q, k, v, routes, sizes, dout):
    b_, s_, h_, d_ = q.shape
    kv_valid = (torch.arange(256, device="cuda")[None, :] < sizes.clamp(0, 256)[:, None]).flatten()
    dq = torch.zeros(b_, s_, h_, d_, device="cuda", dtype=torch.float32)
    rows = torch.arange(s_, device="cuda") // 256
    for b in range(b_):
        for h in range(h_):
            qf, kf, vf, dof = (t[b, :, h].float() for t in (q, k, v, dout))
            allow = routes[b, h].repeat_interleave(256, 1)[rows] & kv_valid[None, :]
            p = torch.softmax((qf @ kf.T * d_**-0.5).masked_fill(~allow, float("-inf")), -1).nan_to_num(0.0)
            ds = p * (dof @ vf.T - (dof * (p @ vf)).sum(-1, keepdim=True))
            dq[b, :, h] = ds @ kf * d_**-0.5
    return dq


def test_f3_empty_query_dq_fp32_reference(reach):
    """Ruling 64 on the integrated route: pruned dQ has the same error against a dense FP32 reference as the full
    backward (ratio 1 within the frozen 6e-6 band scaled to this small case: 1e-3), with O/dK/dV bitwise and padded
    dQ rows exactly zero. The seven-row 1.25x identical-code spread rule stays FAILED as disclosed; not re-run here."""
    qkv, block_map, sizes, dout, pad = _op_inputs(n=10, seed=10, sizes=[256, 131, 128, 0, 1, 129, 256, 64, 200, 128])
    ref = _dense_fp32_dq(*qkv.chunk(3), block_map, sizes, dout)
    live = ~pad
    full = _op_grads(qkv, block_map, sizes, dout)
    _clear(reach)
    pruned = _op_grads(qkv, block_map, sizes, dout, **_proof(sizes))
    assert reach["prepare"] == [True]
    err_full = (full[1].float() - ref)[:, live].abs().mean().item()
    err_pruned = (pruned[1].float() - ref)[:, live].abs().mean().item()
    print(f"ruling64 FP32-reference dQ error: full={err_full:.6e} pruned={err_pruned:.6e} "
          f"ratio={err_pruned / err_full:.6f}")
    assert abs(err_pruned / err_full - 1) <= 1e-3
    assert torch.count_nonzero(pruned[1][:, pad]) == 0
    for name, g, r in zip(("O", "dK", "dV"), (pruned[0], pruned[2], pruned[3]), (full[0], full[2], full[3]),
                          strict=True):
        assert torch.equal(g, r), name


# ---------------------------------------------------------------------------------------------------------------------
# F4: compile, shape and layout
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("layout,merge", [("cube", False), ("chunk256", True)], ids=["cube", "chunk256_merged"])
def test_f4_h3_block_fullgraph_dynamic(reach, layout, merge, batch):
    """Full H3 block (tile op, pool/top-k, attention op, untile) compiles fullgraph dynamic with every option on;
    compiled == eager (O, dK, dV bitwise) at batch 1 and 2."""
    impl = _impl(8)
    meta = _meta(layout=layout, merge=merge, **_ALL_ON)
    leaves, dout = _leaves(meta, 8, batch=batch, seed=11)
    block = _layer_block(impl, meta)
    ref = _run(block, leaves, dout)
    _clear(reach)
    got = _run(torch.compile(block, fullgraph=True, dynamic=True), leaves, dout)
    assert len(reach["fwd"]) == 1
    assert reach["prepare"] == [layout == "cube"]
    _same(f"{layout} b{batch}", got, ref)


def test_f4_cuda_graph_replay(monkeypatch, reach):
    """Inductor CUDA graphs (mode="reduce-overhead", fullgraph) over the all-on H3 block, forward + backward: no
    cudagraph skip is logged, and replays with new input values equal eager. (The proof's version checks run when the
    op bodies run, i.e. at capture, not on replay: disclosed.)"""
    from torch._dynamo.utils import counters
    from torch._inductor import cudagraph_trees
    replays = []
    replay = torch.cuda.CUDAGraph.replay
    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", lambda self: replays.append(1) or replay(self))
    impl = _impl(8)
    meta = _meta(**_ALL_ON)
    block = _layer_block(impl, meta)
    counters.clear()
    compiled = torch.compile(block, mode="reduce-overhead", fullgraph=True, dynamic=False)
    eager_maps, replays_per_step, captures_per_step = [], [], []
    for step in range(5):
        leaves, dout = _leaves(meta, 8, seed=20 + step)
        _clear(reach)
        reach["maps"] = []
        ref = _run(block, leaves, dout)  # eager: records the block map these new input values select
        eager_maps.append(reach["maps"][0])
        reach["maps"] = None
        before = len(replays)
        got = _run(compiled, leaves, dout)
        replays_per_step.append(len(replays) - before)
        manager = cudagraph_trees.get_manager(create_if_none_exists=False)
        captures_per_step.append(0 if manager is None else int(repr(manager.graph_counter)[6:-1]))  # count(N): graphs captured so far
        _same(f"cudagraph step {step}", got, ref)
    # Replay with changed inputs: from step 2 on, the step only replays (no re-record) while the selected block map
    # differs from every earlier step, and the replayed result still equals eager on the new inputs. The block map is
    # recomputed on device inside the captured vsa_h3_block_map kernels; host-side Python (op-body version checks for
    # the tile permutation and the query-padding proof, the postprocess pin check, eligibility/pack policy) does NOT
    # re-run on replay.
    # Exact capture contract: forward + backward graphs captured once (by the end of step 1), never re-captured later;
    # every later step replays exactly those two graphs.
    print(f"cudagraph per-step replays={replays_per_step} captured-graph counter={captures_per_step}")
    assert captures_per_step == [0, 2, 2, 2, 2], captures_per_step  # step 0 warm-up, step 1 records fwd + bwd
    assert replays_per_step == [0, 2, 2, 2, 2], replays_per_step
    assert all(not torch.equal(eager_maps[i], eager_maps[j]) for i in range(5) for j in range(i)), "maps did not change"
    assert counters["inductor"]["cudagraph_skips"] == 0, dict(counters["inductor"])
    print(f"cudagraph trees: recorded graphs={captures_per_step[-1]}, replays={len(replays)}")


def test_f4_changed_same_shape_maps_compiled():
    """One compiled public call, two different block maps/sizes of the same shape: each equals its eager run
    (no stale plan or geometry captured)."""
    qkv, map1, sizes1, dout, _ = _op_inputs(seed=12)
    _, map2, _, _, _ = _op_inputs(seed=13)
    sizes2 = sizes1.flip(0).contiguous()
    leaf = qkv.detach().clone().requires_grad_(True)

    def call(x, m, s):
        q, k, v = x.chunk(3, dim=0)
        return _public(q, k, v, m, s)[0]

    compiled = torch.compile(call, fullgraph=True, dynamic=True)
    for m, s in ((map1, sizes1), (map2, sizes2), (map1, sizes1)):
        ref = _run(lambda x: call(x, m, s), [leaf], dout)
        got = _run(lambda x: compiled(x, m, s), [leaf], dout)
        assert torch.equal(got[0], ref[0])
        g, r = got[1], ref[1]
        _check("changed-map dQ", r[:1], g[:1], _GRAD_TOL)
        assert torch.equal(g[1:], r[1:])


@pytest.mark.parametrize("pack", ["1", "0"], ids=["pack1", "pack0"])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("batch", [1, 2])
def test_f4_batch_dim_routes(monkeypatch, reach, batch, dim, pack):
    """Compiled == eager (O, dK, dV bitwise), batch 1/2, D64/D128, both pack settings, with the query proof."""
    monkeypatch.setenv("FASTVIDEO_VSA_PACK_TAILS", pack)
    qkv, block_map, sizes, dout, _ = _op_inputs(b=batch, dim=dim, seed=14)
    eager = _op_grads(qkv, block_map, sizes, dout, **_proof(sizes))
    assert reach["prepare"] == ([True] if pack == "1" else [])
    compiled = _op_grads(qkv, block_map, sizes, dout, compiled=True, **_proof(sizes))
    _same("compiled vs eager", compiled, eager)


def test_f4_tile64_does_not_reach_the_training_op(reach):
    """tile_size=64 keeps its native 64-token route; the Q256 training op and its options are not involved."""
    impl = _impl(2)
    meta = _meta(_SMALL, tile_size=64)
    tiled = _tiled(impl, meta)
    out, *grads = _grads(lambda q, k, v: impl.postprocess_output(impl.forward(q, k, v, None, meta), meta), tiled)
    assert reach["fwd"] == [] and all(torch.isfinite(g).all() for g in grads)


# ---------------------------------------------------------------------------------------------------------------------
# F5: cross-stack and scheduling controls
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("layout,merge", [("cube", False), ("chunk256", True)], ids=["cube", "chunk256_merged"])
def test_f5_no_grad_keeps_its_route(reach, layout, merge):
    """No-grad BF16 forward never reaches the training op, and matches the training forward to bf16 tolerance
    (different kernel route)."""
    impl = _impl(8)
    meta = _meta(layout=layout, merge=merge, **_ALL_ON)
    leaves, _ = _leaves(meta, 8, seed=16)
    block = _layer_block(impl, meta)
    train = block(*leaves).detach()
    _clear(reach)
    with torch.no_grad():
        x = impl.preprocess_qkv(torch.cat(leaves, dim=0), meta).clone()
        chunked, ops = _op_calls(lambda: impl.postprocess_output(impl.forward(*x.chunk(3, dim=0), None, meta), meta))
    assert ops == 0
    _check("no-grad vs training forward", train, chunked, _GRAD_TOL)


def test_f5_vc_env_bypasses_training_op(monkeypatch):
    """FASTVIDEO_VSA_VC=1 keeps the public wrapper off the training op (fused-VC routing unchanged)."""
    import importlib
    wrapper = importlib.import_module("fastvideo_kernel.block_sparse_attn_256")
    monkeypatch.setenv("FASTVIDEO_VSA_VC", "1")
    seen = []

    def record(q, k, v, routes, sizes, **_):
        seen.append(routes.shape[-1])
        return torch.empty_like(q), torch.empty(1, 2, 512, device="cuda")

    monkeypatch.setattr(adapter, "block_sparse_attn_cute_fwd_bshd", record)
    monkeypatch.setattr(wrapper.vsa256_ops, "training_attention", lambda *a, **k: pytest.fail("training op reached"))
    q, k, v = (torch.randn(1, 512, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    wrapper.block_sparse_attn_256_bshd(q, k, v, torch.ones(1, 2, 2, 2, device="cuda", dtype=torch.bool),
                                       torch.tensor([131, 0], device="cuda", dtype=torch.int32))
    assert seen == [4]  # generic 128-expanded route


# ---------------------------------------------------------------------------------------------------------------------
# F4 (conductor 31941): h3mh's exact per-block compile mode on the real H3 block across real-shape documents
# ---------------------------------------------------------------------------------------------------------------------

_SPEC_DIR = Path("/mnt/home/hyunsungl/space/vsa-vc-kernel-0926/init/h3-real-shape")
_SPECS = {"h3-real-shape-spec-s085k32.json": "5fab7d309cc1a21d03ca72ba1ebd46d6a58396a0869dbaf422d3846af4584533",
          "h3-real-shape-spec.json": "f1c85686b55281e9cba8fcd09f8e4721875ae0d99a0d0452e3d093ff95a78b26",
          "h3-real-shape-spec-s050.json": "cd24cc2fcf0c1e606fab861c5fae9bed73cb385ef8f538abaaf164c922b7ad3e"}


def _errors(got, ref):
    out = {}
    for name, g, r in zip(("O", "dQ", "dK", "dV"), got, ref, strict=True):
        d = (g.float() - r.float()).abs()
        out[f"{name}_max_abs"], out[f"{name}_mean_abs"] = float(d.max()), float(d.mean())
    return out


def _real_docs(count, spec=None):
    """``count`` distinct document geometries PER frozen spec (``spec`` None: all specs, concatenated), drawn round
    robin across the spec's packs, each with that spec's own operating point (attention.vsa.sparsity and optional
    topk_cap: s085k32 v4 5fab7d30 = (0.85, 32) primary, f1c85686 = (0.75, None), cd24cc2f = (0.5, None)).
    Entries: (label, (sparsity, topk_cap), doc)."""
    import hashlib
    docs = []
    for name, digest in _SPECS.items():
        if spec is not None and name != spec:
            continue
        raw = (_SPEC_DIR / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest, name
        data = json.loads(raw)
        sparsity = float(data["attention"]["vsa"]["sparsity"])
        cap = data["attention"]["vsa"].get("topk_cap")
        cap = None if cap is None else int(cap)
        packs = [list(enumerate(pack["docs"])) for pack in data["packs"]]
        seen, picked = set(), []
        for rank in range(max(len(p) for p in packs)):
            for pi, pack in enumerate(packs):
                if rank >= len(pack) or len(picked) == count:
                    continue
                di, doc = pack[rank]
                key = (tuple(doc["latent_grid_THW"]), tuple(n for _, n in doc["segments"]))
                if key not in seen and doc["rows"] <= 32768:
                    seen.add(key)
                    picked.append((f"{name}:s{sparsity}k{cap}:pack{pi}:doc{di}", (sparsity, cap), doc))
        docs += picked
    return docs


@pytest.mark.parametrize("spec", [*_SPECS, None, "pack1_witness"],
                         ids=["s085k32", "s075", "s050", "all_alternating", "pack1"])
@pytest.mark.parametrize("layout,merge", [("chunk256", True), ("cube", False)], ids=["chunk256_merged", "cube"])
def test_f4_h3mh_compile_mode_sac_real_shapes(monkeypatch, reach, layout, merge, spec):
    """h3mh block compile (smol_gen/h3vid/runtime/compile_helpers.py apply_block_compile @33f00f40b):
    torch.compile(checkpointed(block), backend="inductor", mode="default", dynamic=True, fullgraph=True) with
    fail_on_recompile_limit_hit=True, use_duck_shape=False, triton.mix_order_reduction=False; SAC inside the compiled
    region, MUST_SAVE on vsa256_fwd, PREFER_RECOMPUTE elsewhere. Per operating point (one frozen spec and its own
    sparsity) a fresh compiled callable over 12 distinct real-shape geometries drawn across packs, and both operating
    points alternating through one callable (24 entries): no graph break, one compiled graph (no recompile), and
    end-to-end parity with each side on its own block map: identical maps (0 flipped blocks), O/dK/dV bitwise, dQ
    within tolerance; per-document errors and flip counts are printed as h3mh-parity rows."""
    import torch._inductor.config as inductor_config
    import torch.fx.experimental._config as fx_config
    from torch._dynamo.utils import counters
    from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
    from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAMetadataBuilder
    # h3mh saves its attention op (whose outputs include the routes); here: the attention op and the block-map op
    must_save = {torch.ops.fastvideo_kernel.vsa256_fwd.default, torch.ops.fastvideo_kernel.vsa_h3_block_map.default}

    def policy(ctx, op, *args, **kwargs):
        return CheckpointPolicy.MUST_SAVE if op in must_save else CheckpointPolicy.PREFER_RECOMPUTE

    # 56 heads as in h3mh (spec "attention.heads"): every real document then takes the invocation-owned tile
    # permutation op; below 2**25 tiled elements H3 keeps the builder-holder scatter, which mutates builder state and
    # is not SAC-compilable (a pre-existing boundary outside this stack).
    impl, builder = _impl(56), MiniMaxH3VSAMetadataBuilder()

    def block(q, k, v, meta):
        x = impl.preprocess_qkv(torch.cat([q, k, v], dim=0), meta)
        return impl.postprocess_output(impl.forward_qkv(x, meta), meta)

    def checkpointed(q, k, v, meta):
        return checkpoint(block, q, k, v, meta, use_reentrant=False,
                          context_fn=functools.partial(create_selective_checkpoint_contexts, policy))

    if spec == "pack1_witness":
        if layout != "cube":
            pytest.skip("the auto-PACK1 witnesses are cube documents")
        # The two frozen cube documents the auto policy packs (pack2/doc4, pack3/doc9; the policy is geometry-only),
        # at the primary s085k32 operating point. pack_tails is a static bool, so PACK1 is its own compiled graph:
        # one graph per static policy value, no recompile within.
        name = "h3-real-shape-spec-s085k32.json"
        by_label = {d[0]: d for d in _real_docs(10**6, name)}
        docs = [by_label[f"{name}:s0.85k32:pack2:doc4"], by_label[f"{name}:s0.85k32:pack3:doc9"]]
    else:
        docs = _real_docs(12, spec)
    if spec is None:  # every operating point alternating through ONE compiled callable (36 entries)
        docs = [d for group in zip(docs[:12], docs[12:24], docs[24:]) for d in group]
    assert len({(tuple(d[2]["latent_grid_THW"]), tuple(n for _, n in d[2]["segments"])) for d in docs}) >= \
        (2 if spec == "pack1_witness" else 10)
    assert len({d[1] for d in docs}) == (len(_SPECS) if spec is None else 1)
    parity, policies = [], set()
    import fastvideo.attention.backends.video_sparse_attn_h3 as h3
    torch._dynamo.reset()
    counters.clear()
    with torch._dynamo.config.patch(fail_on_recompile_limit_hit=True), \
            fx_config.patch(use_duck_shape=False), inductor_config.patch({"triton.mix_order_reduction": False}):
        compiled = torch.compile(checkpointed, backend="inductor", mode="default", dynamic=True, fullgraph=True)
        for label, (sparsity, cap), doc in docs:
            prefix = tuple(n for _, n in doc["segments"][:-1] if n)
            meta = builder.build(current_timestep=0, raw_latent_shape=tuple(doc["latent_grid_THW"]),
                                 patch_size=(1, 1, 1), VSA_sparsity=sparsity, prefix_segments=prefix,
                                 device=torch.device("cuda"), tile_layout=layout, merge_prefix=merge,
                                 topk_cap=cap, **_ALL_ON)
            assert meta.total_seq_length == doc["rows"], label
            policies.add(_pack(meta))
            if cap is not None:  # s085k32 v4: builder k == the spec's exact rule, per document
                n = meta.num_video_tiles
                assert meta.video_topk == max(1, min((3 * n + 19) // 20, int(cap), n)), (label, n, meta.video_topk)
            leaves, dout = _leaves(meta, 56, seed=len(docs))
            assert 3 * leaves[0].numel() >= 2**25
            _clear(reach)
            reach["maps"] = []
            got = _run(lambda q, k, v: compiled(q, k, v, meta), leaves, dout)
            compiled_map = reach["maps"][0]
            eager_map = torch.zeros_like(compiled_map)
            with monkeypatch.context() as m:
                build_mask = h3._build_block_mask
                m.setattr(h3, "_build_block_mask", lambda *a, **k: eager_map.copy_(build_mask(*a, **k)))
                ref = _run(lambda q, k, v: block(q, k, v, meta), leaves, dout)
            flips = int((eager_map != compiled_map).sum())
            parity.append(dict(doc=label, rows=doc["rows"], flipped_blocks=flips,
                               flipped_fraction=flips / compiled_map.numel(), **_errors(got, ref)))
            # End-to-end parity, each side on its own block map: the maps must be identical (selection runs in the
            # opaque vsa_h3_block_map op, outside Inductor fusion) and O/dK/dV bitwise, dQ within tolerance.
            assert flips == 0, parity[-1]
            _same(f"h3mh {layout} {label}", got, ref)
            reach["maps"] = None
            _clear(reach)
            _run(lambda q, k, v: compiled(q, k, v, meta), leaves, dout)  # reach of a clean compiled call
            assert len(reach["fwd"]) == 1 and reach["tail"] == ([True] if _pack(meta) else []), label
            if _features()["fused"]:
                assert reach["stacked"][:1] == [True] and reach["bwd"][0]["fused"], label  # one fused dqkv
            assert reach["prepare"] == ([layout == "cube"] if _pack(meta) else []), label  # compiled pruning
    graphs = counters["stats"]["unique_graphs"]
    breaks = sum(counters["graph_break"].values())
    print(f"h3mh-compile {layout} {spec}: docs={[d[0] for d in docs]} unique_graphs={graphs} graph_breaks={breaks} "
          f"flip_docs={[r['doc'] for r in parity if r['flipped_blocks']]}")
    for r in parity:
        print("h3mh-parity", json.dumps(r))
    # one static policy per corpus: the auto policy packs only the witnesses; without the policy input every doc packs
    assert policies == ({True} if spec == "pack1_witness" or not _features()["policy"] else {False}), policies
    if spec == "pack1_witness":
        assert reach["tail"] and reach["prepare"] == [True], "PACK1 witness must run the pruned packed backward"
    assert breaks == 0 and graphs == 1, (graphs, dict(counters["graph_break"]), dict(counters["frames"]))


@pytest.mark.parametrize("case", ["misaligned_view", "broadcast_map", "vc_env"])
def test_f1_cute_attention_ineligible_inputs_skip_the_op(monkeypatch, reach, case):
    """_cute_attention uses the public wrapper's eligibility predicate: inputs outside it (a strided view that is not
    16-byte aligned, a map broadcast over batch/heads, FASTVIDEO_VSA_VC=1) never reach the op pair."""
    from fastvideo_kernel import vsa256_ops
    qkv, block_map, sizes, dout, _ = _op_inputs(seed=19)
    q, k, v = (t.clone().requires_grad_(True) for t in qkv.chunk(3, dim=0))
    if case == "misaligned_view":  # strided (dim-2 unbind) view whose storage offset breaks 16-byte alignment
        b, n, h, d = q.shape
        flat = torch.randn(3 * q.numel() + 1, device="cuda", dtype=torch.bfloat16)
        q = flat[1:].view(b, n, 3, h, d)[:, :, 0].detach().requires_grad_(True)
        assert not q.is_contiguous() and q.data_ptr() % 16 != 0
    elif case == "broadcast_map":
        block_map = block_map[:, :1]
    else:
        monkeypatch.setenv("FASTVIDEO_VSA_VC", "1")
        assert not vsa256_ops.training_eligible(q, k, v, block_map)
        return  # the VC route itself is the AI tree's (see test_f5_vc_env_bypasses_training_op)
    assert not vsa256_ops.training_eligible(q, k, v, block_map)
    (out, _), ops = _op_calls(adapter._cute_attention, q, k, v, block_map, sizes)
    grads = torch.autograd.grad(out, (q, k, v), dout)
    assert ops == 0 and reach["tail"] == []  # no op body ran; plain Q256 training autograd
    assert torch.isfinite(out).all() and all(torch.isfinite(g).all() for g in grads)


def test_video_topk_integer_rule():
    """The single host-int top-k: s085k32 == min((3n + 19) // 20, 32) for n = 1..300 (the cap binds from n = 207);
    uncapped 0.75 / 0.5 equal the pre-existing compute_topk; 0.85 uncapped is (3n + 19) // 20 (no float overshoot)."""
    from fastvideo.attention.backends.video_sparse_attn import compute_topk
    from fastvideo.attention.backends.video_sparse_attn_h3 import _video_topk
    for n in range(1, 301):
        assert _video_topk(0.85, n, 32) == min((3 * n + 19) // 20, 32), n
        assert _video_topk(0.85, n, None) == (3 * n + 19) // 20, n
        for s in (0.75, 0.5):
            assert _video_topk(s, n, None) == compute_topk(s, n), (s, n)
    assert _video_topk(0.85, 206, 32) == 31 and _video_topk(0.85, 207, 32) == 32 and _video_topk(0.85, 300, 32) == 32



def test_f1_public_guard_misaligned_strided_view_fullgraph(reach):
    """Through the PUBLIC wrapper (the real guard, not a direct op call), strided views of a [B, S, 3, H, D] allocation:
    - fullgraph=True: Dynamo cannot trace the guard's data_ptr() alignment read (and the generic 128-expanded fallback is
      not fullgraph-traceable either), so compilation fails LOUDLY and no vsa256 op body ever runs - a misaligned view is
      never routed into the op;
    - torch.compile without fullgraph: the guard runs at the graph break with exact data_ptr semantics: the 2-byte offset
      view takes the fallback (no op body), the aligned twin takes the op, and each equals its eager result."""
    from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_256_bshd
    qkv, block_map, sizes, dout, _ = _op_inputs(seed=21)
    b, n, h, d = 1, qkv.shape[1], 2, 128
    results = {}
    for offset in (1, 0):
        flat = torch.zeros(3 * b * n * h * d + 8, device="cuda", dtype=torch.bfloat16)
        packed = flat[offset:offset + 3 * b * n * h * d].view(b, n, 3, h, d)
        packed.copy_(torch.stack(list(qkv.chunk(3, dim=0)), dim=2))
        leaf = packed.detach().requires_grad_(True)
        q, k, v = leaf.unbind(2)
        assert not q.is_contiguous() and (q.data_ptr() % 16 != 0) == (offset == 1)

        def call(x):
            return block_sparse_attn_256_bshd(*x.unbind(2), block_map, sizes)[0]

        _clear(reach)
        torch._dynamo.reset()
        with pytest.raises(torch._dynamo.exc.Unsupported):
            torch.compile(call, fullgraph=True, dynamic=True)(leaf)
        assert reach["fwd"] == [], "fullgraph trace reached an op body"
        for mode, fn in (("eager", call), ("compiled", torch.compile(call, dynamic=True))):
            torch._dynamo.reset()
            _clear(reach)
            out, ops = _op_calls(fn, leaf)
            (g, ) = torch.autograd.grad(out, leaf, dout)
            results[(offset, mode)] = (out.detach(), g, [True] * ops)
        torch._dynamo.reset()
    for mode in ("eager", "compiled"):
        assert results[(1, mode)][2] == [], f"misaligned view reached the op ({mode})"
        assert results[(0, mode)][2], f"aligned view did not reach the op ({mode})"
    for off in (0, 1):
        assert torch.equal(results[(off, "compiled")][0], results[(off, "eager")][0]), f"O compiled != eager ({off})"
        _check(f"dQKV compiled vs eager ({off})", results[(off, "eager")][1], results[(off, "compiled")][1], _GRAD_TOL)
    _check("fallback O vs op O", results[(0, "eager")][0], results[(1, "eager")][0], _GRAD_TOL)


def test_f1_topk_cap_change_is_not_reused(reach):
    """The cap is part of the cached top-k key: the same document built with cap 32 and cap 16 gets different k, and
    metadata whose cap field is changed after build is recomputed in eager (no stale reuse of the cap-32 k)."""
    impl = _impl(8)
    m32, m16 = _meta(VSA_sparsity=0.5, topk_cap=32), _meta(VSA_sparsity=0.5, topk_cap=16)
    n = m32.num_video_tiles
    assert m32.video_topk == min((n + 1) // 2, 32) and m16.video_topk == min((n + 1) // 2, 16) != m32.video_topk
    seen = []
    import fastvideo.attention.backends.video_sparse_attn_h3 as h3
    orig = h3._build_block_mask
    try:
        h3._build_block_mask = lambda *a, **kw: seen.append(a[5] if len(a) > 5 else kw.get("k_vid")) or orig(*a, **kw)
        leaves, dout = _leaves(m32, 8, seed=22)
        m32.video_topk_cap = 16  # changed after build: eager must not keep the cached cap-32 k
        _run(_layer_block(impl, m32), leaves, dout)
        _run(_layer_block(impl, m16), leaves, dout)
    finally:
        h3._build_block_mask = orig
    assert seen[0] == seen[-1] == 16, seen


@pytest.mark.parametrize("backend", ["eager", "aot_eager", "inductor"])
def test_f1_sac_must_save_block_map_not_recomputed(backend):
    """SAC with MUST_SAVE {vsa256_fwd, vsa_h3_block_map}: the block-map op body runs once per step (forward only; the
    backward recompute reuses the saved map), while PREFER_RECOMPUTE for it re-runs it in the backward - proving the
    count detects recomputation. Eager and compiled SAC op keys match."""
    from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
    import fastvideo.attention.backends.video_sparse_attn_h3 as h3
    impl = _impl(8)
    meta = _meta(**_ALL_ON)
    block = _layer_block(impl, meta)
    counts, keys = {}, {}
    orig = h3._build_block_mask
    for save_map in (True, False):
        must = {torch.ops.fastvideo_kernel.vsa256_fwd.default}
        if save_map:
            must.add(torch.ops.fastvideo_kernel.vsa_h3_block_map.default)
        seen = set()

        def policy(ctx, op, *args, must=must, seen=seen, **kwargs):
            seen.add(str(op))
            return CheckpointPolicy.MUST_SAVE if op in must else CheckpointPolicy.PREFER_RECOMPUTE

        def sac(q, k, v, policy=policy):
            return checkpoint(block, q, k, v, use_reentrant=False,
                              context_fn=functools.partial(create_selective_checkpoint_contexts, policy))

        fn = sac if backend == "eager" else torch.compile(sac, backend=backend, fullgraph=True, dynamic=True)
        leaves, dout = _leaves(meta, 8, seed=23)
        calls = []
        h3._build_block_mask = lambda *a, **kw: calls.append(1) or orig(*a, **kw)
        try:
            if backend != "eager":  # compile first so trace-time calls are not counted
                _run(fn, leaves, dout)
            calls.clear()
            out = fn(*leaves)
            fwd_calls = len(calls)
            torch.autograd.grad(out, leaves, dout)
            counts[save_map] = (fwd_calls, len(calls))
        finally:
            h3._build_block_mask = orig
        keys[save_map] = {k for k in seen if k.startswith("fastvideo_kernel.")}
        torch._dynamo.reset()
    assert counts[True] == (1, 1), counts  # saved: not recomputed in the backward
    assert counts[False] == (1, 2), counts  # recompute policy: the backward re-runs the op body
    assert "fastvideo_kernel.vsa_h3_block_map.default" in keys[True], keys


def test_f1_h3mh_fullgraph_rejects_noncontiguous_inputs(reach):
    """Disclosed limitation (conductor 32596): under h3mh's fullgraph config (Inductor, dynamic, fullgraph, SAC in-region,
    fail_on_recompile_limit_hit, use_duck_shape=False) the training path requires contiguous q/k/v. A non-contiguous
    (dim-2 unbind) input raises at compile time on the guard's data_ptr() read, and no vsa256 op body runs."""
    import torch.fx.experimental._config as fx_config
    from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
    from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_256_bshd
    qkv, block_map, sizes, _, _ = _op_inputs(seed=24)
    leaf = torch.stack(list(qkv.chunk(3, dim=0)), dim=2).requires_grad_(True)  # [B, S, 3, H, D]
    assert not leaf.unbind(2)[0].is_contiguous()
    must = {torch.ops.fastvideo_kernel.vsa256_fwd.default, torch.ops.fastvideo_kernel.vsa_h3_block_map.default}

    def policy(ctx, op, *args, **kwargs):
        return CheckpointPolicy.MUST_SAVE if op in must else CheckpointPolicy.PREFER_RECOMPUTE

    def block(x):
        return block_sparse_attn_256_bshd(*x.unbind(2), block_map, sizes)[0]

    def checkpointed(x):
        return checkpoint(block, x, use_reentrant=False,
                          context_fn=functools.partial(create_selective_checkpoint_contexts, policy))

    torch._dynamo.reset()
    _clear(reach)
    with torch._dynamo.config.patch(fail_on_recompile_limit_hit=True), fx_config.patch(use_duck_shape=False):
        compiled = torch.compile(checkpointed, backend="inductor", mode="default", dynamic=True, fullgraph=True)
        with pytest.raises(torch._dynamo.exc.Unsupported, match=r"unsupported operand type\(s\) for %"):
            compiled(leaf)
    assert reach["fwd"] == [], "a vsa256 op body ran"



# ---------------------------------------------------------------------------------------------------------------------
# Fused-QKV gradient (conditional input 2dba19be): stacked op route with ONE dqkv allocation
# ---------------------------------------------------------------------------------------------------------------------


def _stack_grads(qkv, block_map, sizes, dout, stacked, compiled=False, **kw):
    """Public entry on the stacked (qkv=) or ordinary route; returns (O, dQ, dK, dV)."""
    leaf = qkv.detach().clone().requires_grad_(True)
    b = qkv.shape[0] // 3

    def call(x):
        q, k, v = x.chunk(3, dim=0)
        return _public(q, k, v, block_map, sizes, qkv=x if stacked else None, **kw)[0]

    fn = torch.compile(call, fullgraph=True, dynamic=True) if compiled else call
    out = fn(leaf)
    (g, ) = torch.autograd.grad(out, leaf, dout)
    return (out.detach(), g[:b], g[b:2 * b], g[2 * b:])


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "fullgraph"])
@pytest.mark.parametrize("fused", [True, False], ids=["fused_on", "fused_off"])
def test_fq_flag_reach_and_allocation(reach, fused, compiled):
    """fused_qkv_grad ON: the op receives the stacked input and its backward writes one dqkv allocation (no
    SplitBackward concat); OFF restores the baseline: three gradient buffers and autograd's chunk concat. Both equal."""
    _need("fused")
    impl = _impl(8)
    meta = _meta(fused_qkv_grad=fused, **_ALL_ON)
    leaves, dout = _leaves(meta, 8, seed=31)
    ref = _run(_layer_block(impl, _meta(fused_qkv_grad=False, **_ALL_ON)), leaves, dout)
    _clear(reach)
    block = _layer_block(impl, meta)
    fn = torch.compile(block, fullgraph=True, dynamic=True) if compiled else block
    out = fn(*leaves)
    concat = _reaches(out.grad_fn, "SplitBackward0") if not compiled else None
    got = (out.detach(), *torch.autograd.grad(out, leaves, dout))
    diag = dict(fused_qkv_grad=fused, compiled=compiled, stacked_input=reach["stacked"][0],
                grad_storages=reach["bwd"][0]["storages"], fused_views=reach["bwd"][0]["fused"],
                split_backward_concat=concat)
    print("ablation-diagnostic", json.dumps(diag))
    assert reach["stacked"][0] is fused and reach["bwd"][0]["fused"] is fused
    assert reach["bwd"][0]["storages"] == (1 if fused else 3)
    if not compiled:
        assert concat is (not fused)
    _same(f"fused={fused} compiled={compiled}", got, ref)


@pytest.mark.parametrize("proof", [False, True], ids=["no_proof", "proof"])
@pytest.mark.parametrize("pack_tails", [True, False], ids=["pack1", "pack0"])
def test_fq_op_schema_stacked(proof, pack_tails):
    """opcheck of the stacked forward; backward fake arity 1 ([dqkv]) vs 3."""
    _need("fused")
    from torch._subclasses.fake_tensor import FakeTensorMode
    qkv, block_map, sizes, _, _ = _op_inputs()
    qkv.requires_grad_(True)
    p = _proof(sizes)
    extra = (p["query_sizes"], p["query_untile"], *p["query_versions"]) if proof else (None, None, 0, 0)
    torch.library.opcheck(torch.ops.fastvideo_kernel.vsa256_fwd.default, (qkv, None, None, block_map, sizes, *extra,
                                                                          pack_tails),
                          test_utils=("test_schema", "test_faketensor", "test_autograd_registration"))
    out, lse = torch.ops.fastvideo_kernel.vsa256_fwd(qkv, None, None, block_map, sizes, *extra, pack_tails)
    assert out.shape == (qkv.shape[0] // 3, *qkv.shape[1:])
    with FakeTensorMode(allow_non_fake_inputs=True):
        grads = torch.ops.fastvideo_kernel.vsa256_bwd(torch.empty_like(out), qkv, None, None, out, lse, block_map, sizes,
                                                      *extra, pack_tails, None)
    assert len(grads) == 1 and grads[0].shape == qkv.shape


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "fullgraph"])
@pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")])
def test_fq_stacked_route_invalid_slot_poison(reach, poison, compiled):
    """Stacked packed route: a non-finite K/V row reachable only through invalid packed slots stays out of dQ/dK/dV
    on the target rows; stacked == ordinary bitwise (O, dK, dV)."""
    _need("fused")
    sizes, routes, bad, target = _poison_case("pad_row0")
    sizes = torch.tensor(sizes, device="cuda", dtype=torch.int32)
    g = torch.Generator(device="cuda").manual_seed(615)
    qkv = torch.randn(3, 1536, 2, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    qkv[1:, bad] = poison
    dout = torch.randn(1, 1536, 2, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    ordinary = _stack_grads(qkv, routes, sizes, dout, stacked=False, pack_tails=True, compiled=compiled)
    _clear(reach)
    stacked = _stack_grads(qkv, routes, sizes, dout, stacked=True, pack_tails=True, compiled=compiled)
    assert reach["tail"] == [True] and reach["bwd"][0]["fused"]
    for name, s_, o_ in zip(("O", "dQ", "dK", "dV"), stacked, ordinary, strict=True):
        s_, o_ = s_[:, target], o_[:, target]
        assert torch.isfinite(s_).all(), f"{name} non-finite on target rows"
        if name == "dQ":
            _check("poison dQ", o_, s_, _GRAD_TOL)
        else:
            assert torch.equal(s_, o_), name


@pytest.mark.parametrize("pack_tails", [True, False], ids=["pack1", "pack0"])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("batch", [1, 2])
def test_fq_batch_dim_routes(reach, batch, dim, pack_tails):
    """Stacked == ordinary (O, dK, dV bitwise) and compiled == eager, B1/2 x D64/128 x both pack settings, with the
    query proof; the stacked backward writes one allocation."""
    _need("fused")
    qkv, block_map, sizes, dout, _ = _op_inputs(b=batch, dim=dim, seed=14)
    ordinary = _stack_grads(qkv, block_map, sizes, dout, stacked=False, pack_tails=pack_tails, **_proof(sizes))
    _clear(reach)
    stacked = _stack_grads(qkv, block_map, sizes, dout, stacked=True, pack_tails=pack_tails, **_proof(sizes))
    # fused-QKV merge contract (worker-4 merge-checklist): with query pruning keep BOTH the query_sizes forwarding and
    # the output views - the stacked backward prunes (packed plan with query sizes) AND writes one dqkv allocation.
    assert reach["stacked"][0] and reach["bwd"][0]["fused"]
    assert reach["prepare"] == ([True] if pack_tails else []), reach["prepare"]
    _same("stacked vs ordinary", stacked, ordinary)
    compiled = _stack_grads(qkv, block_map, sizes, dout, stacked=True, pack_tails=pack_tails, compiled=True,
                            **_proof(sizes))
    _same("compiled vs eager", compiled, stacked)


def test_fq_noncontiguous_stack_takes_gated_fallback(reach):
    """A non-contiguous [3B, S, H, D] stack cannot be the fused input: the ordinary three-input route runs (no fused
    gradient claimed) with the same result."""
    _need("fused")
    qkv, block_map, sizes, dout, _ = _op_inputs(seed=15)
    wide = torch.zeros(3, qkv.shape[1], 4, 128, device="cuda", dtype=torch.bfloat16)
    wide[:, :, :2] = qkv
    leaf = wide.requires_grad_(True)
    strided = leaf[:, :, :2]
    assert not strided.is_contiguous()
    q, k, v = strided.chunk(3, dim=0)
    out = _public(q, k, v, block_map, sizes, pack_tails=True, qkv=strided)[0]
    (g, ) = torch.autograd.grad(out, leaf, dout)
    assert reach["stacked"][0] is False and reach["bwd"][0]["fused"] is False
    ref = _stack_grads(qkv, block_map, sizes, dout, stacked=False, pack_tails=False)
    assert torch.equal(out, ref[0])
    _check("strided dQ", ref[1], g[:1, :, :2], _GRAD_TOL)
    assert torch.equal(g[1:, :, :2], torch.cat(ref[2:]))



@pytest.mark.parametrize("heads", [2, 8], ids=["holder_tile_path", "permutation_op_path"])
def test_fq_tile_path_eligibility(reach, heads):
    """The fused route on both H3 tile paths (merge contract: the fused path relies on the tile-permutation
    eligibility guard, 3 * numel >= 2**25; at h3mh's H56 every real document passes it, 46/46). Below the guard (H2,
    eager) H3 tiles into the builder-owned holder buffer; above it (H8) through the invocation-owned permutation op.
    Both give the stacked single-allocation backward with results equal to fused_qkv_grad OFF."""
    _need("fused")
    impl = _impl(heads)
    meta = _meta(**_ALL_ON)
    leaves, dout = _leaves(meta, heads, seed=33)
    assert (3 * leaves[0].numel() >= 2**25) is (heads == 8)
    ref = _run(_layer_block(impl, _meta(fused_qkv_grad=False, **_ALL_ON)), leaves, dout)
    _clear(reach)
    got = _run(_layer_block(impl, meta), leaves, dout)
    assert reach["stacked"][0] and reach["bwd"][0]["fused"] and reach["bwd"][0]["storages"] == 1
    _same(f"tile path heads={heads}", got, ref)
