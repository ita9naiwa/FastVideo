"""Opt-in tail packing preserves gradients across device-plan/Graph transitions."""
import inspect

import pytest
import torch

from fastvideo_kernel import block_sparse_attn_cute_fwd as adapter
from .test_vsa256_backward import _check, _GRAD_TOL


@pytest.mark.parametrize('dim', [64, 128])
def test_tail_backward_graph(monkeypatch, dim):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip('SM100 GPU required')
    import inspect
    pytest.importorskip("flash_attn.cute.interface")
    if "_workspace" not in inspect.signature(adapter._load_fa4_cute()[3]).parameters:
        pytest.skip("FA4 backward workspace support required")
    monkeypatch.setenv('FASTVIDEO_VSA_PACK_TAILS', '1')
    torch.manual_seed(451 + dim)
    q = torch.randn(2, 1024, 3, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k, v = [torch.randn(2, 1536, 3, dim, device='cuda', dtype=torch.bfloat16,
                       requires_grad=True) for _ in range(2)]
    dout = torch.randn_like(q)
    dlse = torch.randn(2, 3, 1024, device='cuda') * .1
    cases = ([256, 131, 128, 0, 132, 256], [127, 1, 128, 0, 256, 0],
             [67, 67, 128, 256, 0, 128], [256]*6, [0]*6, [256, 131, 67, 0, 255, 32])
    sizes = torch.tensor(cases[0], device='cuda', dtype=torch.int32)
    routes = torch.rand(2, 3, 4, 6, device='cuda') > .4
    routes[:, :, 0] = False

    def call(optimized, lse_only=False):
        fn = adapter._cute_attention if optimized else adapter._CuteAttentionQ256Training.apply
        out, lse = fn(q, k, v, routes, sizes)
        grad_lse = dlse.masked_fill(~torch.isfinite(lse), 0)
        grads = (torch.autograd.grad(lse, (q, k, v), grad_lse) if lse_only else
                 torch.autograd.grad((out, lse), (q, k, v), (dout, grad_lse)))
        return grads

    for lse_only in (False, True):
        for ref, got in zip(call(False, lse_only), call(True, lse_only)):
            _check('gradient', ref, got, _GRAD_TOL)
    # Fresh leaves avoid default-stream AccumulateGrad nodes during capture.
    q, k, v = (t.detach().requires_grad_(True) for t in (q, k, v))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call(True)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = call(True)
    for iteration in range(12):
        with torch.no_grad():
            sizes.copy_(torch.tensor(cases[iteration % len(cases)], device='cuda', dtype=torch.int32))
            for tensor in (q, k, v, dout, dlse):
                tensor.normal_()
            routes.copy_(torch.rand_like(routes, dtype=torch.float32) > .4)
            routes[:, :, 0] = False
            for tensor in actual:
                tensor.fill_(float('nan'))
        graph.replay()
        for ref, got in zip(call(False), actual):
            _check('replayed gradient', ref, got, _GRAD_TOL)
            if not any(cases[iteration % len(cases)]):
                assert torch.count_nonzero(got) == 0


@pytest.mark.parametrize("map_shape", [(1, 3), (2, 1), (1, 1)])
def test_tail_broadcast_routes_use_native_fallback(monkeypatch, map_shape):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 GPU required")
    pytest.importorskip("flash_attn.cute.interface")
    from fastvideo_kernel.vsa_tail_backward import TailTraining
    monkeypatch.setenv("FASTVIDEO_VSA_PACK_TAILS", "1")

    def unexpected(*args):
        raise AssertionError("broadcast routes entered the tail packing path")

    monkeypatch.setattr(TailTraining, "apply", unexpected)
    torch.manual_seed(512)
    q, k, v = [torch.randn(2, 512, 3, 64, device="cuda", dtype=torch.bfloat16,
                           requires_grad=True) for _ in range(3)]
    routes = torch.ones(*map_shape, 2, 2, device="cuda", dtype=torch.bool)
    routes[..., 0, 1] = False
    sizes = torch.tensor([256, 131], device="cuda", dtype=torch.int32)
    dout = torch.randn_like(q)
    out = adapter._cute_attention(q, k, v, routes, sizes)[0]
    ref = adapter._CuteAttentionQ256Training.apply(
        q, k, v, routes.expand(2, 3, 2, 2).contiguous(), sizes)[0]
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    for expected, got in zip(torch.autograd.grad(ref, (q, k, v), dout),
                             torch.autograd.grad(out, (q, k, v), dout)):
        _check("broadcast gradient", expected, got, _GRAD_TOL)


def _poison_case(placement):
    """Return (sizes, routes, poisoned K/V rows, target rows) for a 6-parent B1 call whose tails fit."""
    gen = torch.Generator(device='cuda').manual_seed(613)
    routes = torch.rand(1, 2, 6, 6, device='cuda', generator=gen) > .4
    if placement == 'pad_row0':  # sizes[0] == 0: row 0 is padding
        return [0, 131, 256, 140, 150, 256], routes, slice(0, 256), slice(256, None)
    if placement == 'unreachable_row0':  # parent 0 has no tail and is never selected
        routes[..., 0] = False
        return [256, 131, 256, 140, 150, 128], routes, slice(0, 256), slice(0, None)
    # Two documents packed block-diagonally: X = parents 0-1, target Y = parents 2-5.
    routes[..., :2, 2:] = False
    routes[..., 2:, :2] = False
    if placement == 'other_doc_row0':  # X has no tails, so row 0 is reachable only via invalid slots
        return [256, 256, 131, 256, 140, 150], routes, slice(0, 256), slice(512, None)
    assert placement == 'other_doc_valid_tail'  # X's valid tail tokens share the packed tile with Y's tails
    return [131, 256, 131, 256, 140, 150], routes, slice(128, 131), slice(512, None)


@pytest.mark.parametrize('poison', [float('nan'), float('inf'), float('-inf'), torch.finfo(torch.bfloat16).max])
@pytest.mark.parametrize('placement', [
    'pad_row0', 'unreachable_row0', 'other_doc_row0',
    pytest.param('other_doc_valid_tail', marks=pytest.mark.xfail(
        strict=True, reason='known limitation: the global tail plan co-packs valid tails of different '
        'parents/documents, so dO.V^T of a non-finite valid token reaches dS = 0 * NaN; needs a per-document plan')),
])
@pytest.mark.parametrize('entry', ['autograd_function', 'custom_op'])
def test_tail_backward_invalid_slot_poison(placement, poison, entry):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip('SM100 GPU required')
    pytest.importorskip("flash_attn.cute.interface")
    from fastvideo_kernel.vsa_tail_backward import TailTraining, _prepare
    if entry == 'custom_op':  # the fullgraph seam (vsa256_fwd/bwd) must share the fixed tail backward
        from fastvideo_kernel import vsa256_ops  # noqa: F401
        TailTraining = type('TailOp', (), {'apply': staticmethod(
            lambda q, k, v, routes, sizes: torch.ops.fastvideo_kernel.vsa256_fwd(q, k, v, routes, sizes, True))})
    if "_workspace" not in inspect.signature(adapter._load_fa4_cute()[3]).parameters:
        pytest.skip("FA4 backward workspace support required")
    sizes, routes, bad, target = _poison_case(placement)
    sizes = torch.tensor(sizes, device='cuda', dtype=torch.int32)
    valid = _prepare(routes, sizes)[2][2]
    assert valid.any() and not valid.all(), 'case must pack tails and leave invalid slots'
    torch.manual_seed(614)
    q, k, v, dout = [torch.randn(1, 1536, 2, 128, device='cuda', dtype=torch.bfloat16) for _ in range(4)]
    k_bad, v_bad = k.clone(), v.clone()
    k_bad[:, bad] = poison
    v_bad[:, bad] = poison

    def call(fn, k, v):
        leaves = [t.detach().requires_grad_(True) for t in (q, k, v)]
        out, lse = fn(*leaves, routes, sizes)
        return (out, lse, *torch.autograd.grad(out, leaves, dout))

    clean = call(TailTraining.apply, k, v)
    ref = call(adapter._CuteAttentionQ256Training.apply, k_bad, v_bad)
    got = call(TailTraining.apply, k_bad, v_bad)
    for name, c, r, g in zip(('O', 'LSE', 'dQ', 'dK', 'dV'), clean, ref, got):
        c, r, g = (x[:, :, target] if name == 'LSE' else x[:, target] for x in (c, r, g))
        if name == 'dQ':
            _check(name, c, g, _GRAD_TOL)
        else:
            assert torch.equal(c, g), f'{name} changed by poison outside the target rows'
        if name in ('dQ', 'dK', 'dV'):
            _check(f'{name} vs PACK_TAILS=0', r, g, _GRAD_TOL)

    # CUDA-graph replay on the poisoned inputs matches eager.
    leaves = [t.detach().requires_grad_(True) for t in (q, k_bad, v_bad)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            torch.autograd.grad(TailTraining.apply(*leaves, routes, sizes)[0], leaves, dout)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        replayed = torch.autograd.grad(TailTraining.apply(*leaves, routes, sizes)[0], leaves, dout)
    graph.replay()
    torch.cuda.synchronize()
    for name, e, g in zip(('dQ', 'dK', 'dV'), got[2:], replayed):
        _check(f'replayed {name}', e[:, target], g[:, target], _GRAD_TOL)


@pytest.mark.parametrize('poison', [float('nan'), float('inf')])
def test_public_route_invalid_slot_poison_eager_and_fullgraph(monkeypatch, poison):
    """Public training entry (block_sparse_attn_256_bshd -> vsa256_fwd/bwd custom ops), PACK_TAILS on: a non-finite row
    reachable only through invalid packed slots (sizes[0] == 0 padding) must not reach dQ/dK/dV, in eager and under
    torch.compile(fullgraph=True); packed == non-packed on the target rows, compiled == eager (dQ uses atomics)."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip('SM100 GPU required')
    pytest.importorskip("flash_attn.cute.interface")
    if "_workspace" not in inspect.signature(adapter._load_fa4_cute()[3]).parameters:
        pytest.skip("FA4 backward workspace support required")
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    monkeypatch.delenv("FASTVIDEO_VSA_VC", raising=False)
    from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_256_bshd
    sizes, routes, bad, target = _poison_case('pad_row0')
    sizes = torch.tensor(sizes, device='cuda', dtype=torch.int32)
    torch.manual_seed(615)
    q, k, v, dout = [torch.randn(1, 1536, 2, 128, device='cuda', dtype=torch.bfloat16) for _ in range(4)]
    k[:, bad] = poison
    v[:, bad] = poison

    def call(fn, pack):
        monkeypatch.setenv("FASTVIDEO_VSA_PACK_TAILS", "1" if pack else "0")
        leaves = [t.detach().requires_grad_(True) for t in (q, k, v)]
        out = fn(*leaves, routes, sizes)[0]
        return (out, *torch.autograd.grad(out, leaves, dout))

    unpacked = call(block_sparse_attn_256_bshd, False)
    packed = call(block_sparse_attn_256_bshd, True)
    torch._dynamo.reset()
    compiled = call(torch.compile(block_sparse_attn_256_bshd, fullgraph=True, dynamic=True), True)
    torch._dynamo.reset()
    for name, u, p, c in zip(('O', 'dQ', 'dK', 'dV'), unpacked, packed, compiled, strict=True):
        u, p, c = u[:, target], p[:, target], c[:, target]
        assert torch.isfinite(p).all() and torch.isfinite(c).all(), f'{name} non-finite on the target rows'
        if name == 'O':
            assert torch.equal(p, u) and torch.equal(c, p)
        else:
            _check(f'{name} packed vs unpacked', u, p, _GRAD_TOL)
            if name == 'dQ':
                _check('dQ compiled vs eager', p, c, _GRAD_TOL)
            else:
                assert torch.equal(c, p), f'{name} compiled != eager'


def test_tail_query_sizes_skip_padded_children(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 GPU required")
    pytest.importorskip("flash_attn.cute.interface")
    from fastvideo_kernel import vsa_tail_backward as tail
    monkeypatch.setenv("FASTVIDEO_VSA_PACK_TAILS", "1")
    torch.manual_seed(613)
    sizes = torch.tensor([256, 129, 128, 1, 0, 200], device="cuda", dtype=torch.int32)
    routes = torch.rand(2, 3, 6, 6, device="cuda") > .4
    q, k, v = [torch.randn(2, 1536, 3, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    # Main lists keep exactly the Q128 children that hold a valid query row.
    live = torch.stack((sizes > 0, sizes > 128), -1).flatten()
    base, cand = tail._prepare(routes, sizes)[0], tail._prepare(routes, sizes, sizes)[0]
    assert cand.block_size == (128, 128)

    def child_counts(sparse, kind, factor):
        idx, cnt = getattr(sparse, f"{kind}_block_idx"), getattr(sparse, f"{kind}_block_cnt")
        counts = torch.zeros(*cnt.shape, 12, dtype=torch.int32, device="cuda")
        for j in range(idx.shape[-1]):
            for c in range(factor):
                child = (idx[..., j] * factor + c).clamp(0, 11).long()[..., None]
                counts.scatter_add_(-1, child, (j < cnt)[..., None].int())
        return counts

    for kind in ("full", "mask"):
        assert torch.equal(child_counts(cand, kind, 1), child_counts(base, kind, 2) * live)

    pad = (torch.arange(256, device="cuda") >= sizes[:, None]).flatten()
    dout = torch.randn_like(q).masked_fill(pad[None, :, None, None], 0)
    dlse = torch.randn(2, 3, 1536, device="cuda") * .1

    def grads(query_sizes, with_lse):
        leaves = [t.detach().requires_grad_(True) for t in (q, k, v)]
        out, lse = tail.TailTraining.apply(*leaves, routes, sizes, query_sizes)
        if with_lse:
            return torch.autograd.grad((out, lse), leaves, (dout, dlse.masked_fill(~torch.isfinite(lse), 0)))
        return torch.autograd.grad(out, leaves, dout)

    for with_lse in (False, True):
        ref, got = grads(None, with_lse), grads(sizes, with_lse)
        _check("query-size gradient", ref[0], got[0], _GRAD_TOL)
        torch.testing.assert_close(got[1], ref[1], rtol=0, atol=0)
        torch.testing.assert_close(got[2], ref[2], rtol=0, atol=0)
    assert torch.count_nonzero(grads(sizes, False)[0][:, pad]) == 0
