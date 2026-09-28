"""VSA-256 FA4 CuTe forward/backward parity for BHSD and BSHD APIs.

Covers the shapes the CuTe backward actually sees in production: the gated
compression branch (`compress_attn_weight`), partially filled Q tiles,
and q_len != kv_len. Also pins the inference fast path, which must skip the
KV-owned backward metadata without changing the forward result.
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

from fastvideo_kernel import video_sparse_attn, video_sparse_attn_bshd

from .test_vsa256_triton import _metrics, _torch_vsa256_reference

_BLOCK = 256
_BLOCK_SIZE_3D = (4, 8, 8)  # prod == 256

# Measured on GB200 (sm_100) with bf16 inputs: grads land around 1e-4 avg_abs
# and <=0.11 max_rel across every case below, so these leave ~10x headroom
# without being loose enough to hide a real regression.
_OUT_TOL = (1e-3, 0.2)
_GRAD_TOL = (1e-3, 0.25)


@pytest.mark.parametrize("block,factor", [(128, 1), (128, 2), (256, 1), (256, 2)])
def test_vsa_backward_shared_indices_preserve_active_lists(block, factor):
    from fastvideo_kernel import block_sparse_attn_cute_fwd as adapter

    routes = [[True, True, False, True], [True, False, True, True], [False, True, True, False]]
    sizes = [0, block, block - 1, 1]
    forward, backward = adapter._build_sparse_tensors(
        torch.tensor(routes, device="cuda", dtype=torch.bool)[None, None],
        torch.tensor(sizes, device="cuda", dtype=torch.int32),
        q_len=3 * block - 7, q_block_size=block, kv_block_size=block,
        need_forward=False, need_backward=True, force_q_sparse_block_size=block * factor,
    )
    assert forward is None
    assert backward.full_block_idx.data_ptr() == backward.mask_block_idx.data_ptr()
    for kv, size in enumerate(sizes):
        selected = [start // factor for start in range(0, 3, factor)
                    if any(routes[q][kv] for q in range(start, min(start + factor, 3)))]
        for counts, indices, active in (
            (backward.full_block_cnt, backward.full_block_idx, size == block),
            (backward.mask_block_cnt, backward.mask_block_idx, 0 < size < block),
        ):
            expected = selected if active else []
            assert counts.dtype == torch.int32
            assert counts[0, 0, kv].item() == len(expected)
            assert indices[0, 0, kv, :len(expected)].tolist() == expected


@pytest.fixture(autouse=True)
def _require_cute_backend(monkeypatch):
    pytest.importorskip(
        "flash_attn.cute.block_sparsity",
        reason="optional FA4 CuTe build (flash_attn.cute) not installed",
    )
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    monkeypatch.delenv("FASTVIDEO_VSA_TRITON", raising=False)
    monkeypatch.delenv("FASTVIDEO_KERNEL_VSA_FORCE_TRITON", raising=False)


def _zero_pad_tail(x: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Zero the padded tail of every 256-token tile of a [B, H, S, D] tensor.

    VSA callers scatter into a zeroed tile buffer, so padded slots are zero;
    both the kernel and the reference rely on that.
    """
    bsz, heads, _, dim = x.shape
    blocks = var.numel()
    token_idx = torch.arange(_BLOCK, device=x.device, dtype=torch.int32)
    valid = (token_idx.view(1, -1) < var.view(-1, 1)).view(1, 1, blocks, _BLOCK, 1)
    valid = valid.expand(bsz, heads, blocks, _BLOCK, dim).reshape_as(x)
    return x * valid.to(x.dtype)


def _make_inputs(
    q_blocks: int,
    kv_blocks: int,
    kv_var: torch.Tensor,
    q_var: torch.Tensor,
    heads: int = 2,
    dim: int = 128,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    sq, skv = q_blocks * _BLOCK, kv_blocks * _BLOCK
    q = torch.randn(1, heads, sq, dim, device=device, dtype=dtype)
    k = torch.randn(1, heads, skv, dim, device=device, dtype=dtype)
    v = torch.randn(1, heads, skv, dim, device=device, dtype=dtype)
    grad_out = torch.randn_like(q)
    return _zero_pad_tail(q, q_var), _zero_pad_tail(k, kv_var), _zero_pad_tail(v, kv_var), grad_out


def _check(tag: str, ref: torch.Tensor, got: torch.Tensor, tol: Tuple[float, float]) -> None:
    assert torch.isfinite(got).all().item(), f"{tag}: non-finite values"
    avg_abs, max_rel = _metrics(ref, got)
    print(f"  {tag}: avg_abs={avg_abs:.6e}, max_rel={max_rel:.6e}")
    assert avg_abs < tol[0], f"{tag}: avg_abs {avg_abs:.3e} >= {tol[0]:.3e}"
    assert max_rel < tol[1], f"{tag}: max_rel {max_rel:.3e} >= {tol[1]:.3e}"


def _run_bhsd(q, k, v, kv_var, q_var, topk, gate=None):
    qg, kg, vg = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out = video_sparse_attn(qg, kg, vg, kv_var, q_var, topk, block_size=_BLOCK_SIZE_3D, compress_attn_weight=gate)
    return out, (qg, kg, vg)


def _run_bshd(q, k, v, kv_var, q_var, topk, gate=None):
    qg, kg, vg = (t.transpose(1, 2).contiguous().requires_grad_(True) for t in (q, k, v))
    gate_bshd = None if gate is None else gate.transpose(1, 2).contiguous()
    out = video_sparse_attn_bshd(qg,
                                 kg,
                                 vg,
                                 kv_var,
                                 q_var,
                                 topk,
                                 block_size=_BLOCK_SIZE_3D,
                                 compress_attn_weight=gate_bshd)
    return out.transpose(1, 2), (qg, kg, vg)


def _reference(q, k, v, q_var, kv_var, topk, gate=None):
    qr, kr, vr = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out = _torch_vsa256_reference(qr, kr, vr, q_var, kv_var, topk, compress_attn_weight=gate)
    return out, (qr, kr, vr)


def _compare(tag, layout, q, k, v, kv_var, q_var, topk, grad_out, gate=None):
    runner = _run_bhsd if layout == "bhsd" else _run_bshd
    out, (qg, kg, vg) = runner(q, k, v, kv_var, q_var, topk, gate=gate)
    (out * grad_out).sum().backward()
    grads = [g.grad if g.grad.dim() == 4 and layout == "bhsd" else g.grad for g in (qg, kg, vg)]
    if layout == "bshd":
        grads = [g.transpose(1, 2) for g in grads]

    out_ref, refs = _reference(q, k, v, q_var, kv_var, topk, gate=gate)
    (out_ref * grad_out).sum().backward()

    print(f"[{tag}-{layout}]")
    _check("out", out_ref, out, _OUT_TOL)
    for name, ref, got in zip(("dq", "dk", "dv"), refs, grads):
        _check(name, ref.grad, got, _GRAD_TOL)


@pytest.mark.cuda
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_vsa256_cute_forward_backward_vs_torch_ref(layout: str) -> None:
    kv_var = torch.tensor([256, 173, 79, 256], dtype=torch.int32, device="cuda")
    q_var = torch.full((3, ), _BLOCK, dtype=torch.int32, device="cuda")
    q, k, v, grad_out = _make_inputs(3, 4, kv_var, q_var)
    _compare("vsa256-cute", layout, q, k, v, kv_var, q_var, 2, grad_out)


@pytest.mark.cuda
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_vsa256_cute_backward_with_compress_gate(layout: str) -> None:
    """The gated compression branch is what Wan and MiniMax-H3 actually run.

    It is also the branch that composes the sparse output with the compression
    output, so it is the one that breaks if that composition mutates FA4's
    saved output in place.
    """
    kv_var = torch.tensor([256, 200, 256, 91], dtype=torch.int32, device="cuda")
    q_var = torch.full((3, ), _BLOCK, dtype=torch.int32, device="cuda")
    q, k, v, grad_out = _make_inputs(3, 4, kv_var, q_var, seed=7)
    gate = torch.randn_like(q) * 0.1
    _compare("vsa256-cute-gated", layout, q, k, v, kv_var, q_var, 2, grad_out, gate=gate)


@pytest.mark.cuda
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_vsa256_cute_backward_partial_q_blocks(layout: str) -> None:
    """Q tiles that are not full: only the compression divisor depends on it,
    but it is the one axis the existing coverage held constant."""
    kv_var = torch.tensor([256, 128, 256], dtype=torch.int32, device="cuda")
    q_var = torch.tensor([256, 61, 199], dtype=torch.int32, device="cuda")
    q, k, v, grad_out = _make_inputs(3, 3, kv_var, q_var, seed=11)
    _compare("vsa256-cute-partial-q", layout, q, k, v, kv_var, q_var, 2, grad_out)


@pytest.mark.cuda
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_vsa256_cute_backward_cross_q_kv(layout: str) -> None:
    """q_len != kv_len: forward has coverage, backward did not."""
    kv_var = torch.tensor([256, 143, 256, 256, 88], dtype=torch.int32, device="cuda")
    q_var = torch.full((2, ), _BLOCK, dtype=torch.int32, device="cuda")
    q, k, v, grad_out = _make_inputs(2, 5, kv_var, q_var, seed=13)
    _compare("vsa256-cute-cross", layout, q, k, v, kv_var, q_var, 3, grad_out)


@pytest.mark.cuda
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
@pytest.mark.parametrize("kv_blocks,tails", [(8, False), (12, False), (12, True)], ids=["qk-equal", "cross", "cross-tails"])
def test_vsa256_nograd_forward_vs_torch_ref(monkeypatch, layout: str, kv_blocks: int, tails: bool) -> None:
    """Inference route (no grad) of the public BHSD/BSHD entries, heads 8, 8 Q blocks, topk 2 (Q == KV, Q != KV, random
    tail KV sizes): the CuTe and the Triton (route-A 256->64) backends each vs the torch reference, and vs each other."""
    monkeypatch.delenv("FASTVIDEO_VSA_VC", raising=False)  # the BF16 no-grad route; VC=1 would take the FP8 path
    torch.manual_seed(0)
    kv_var = (torch.randint(16, _BLOCK + 1, (kv_blocks, ), dtype=torch.int32, device="cuda") if tails else torch.full(
        (kv_blocks, ), _BLOCK, dtype=torch.int32, device="cuda"))
    q_var = torch.full((8, ), _BLOCK, dtype=torch.int32, device="cuda")
    q, k, v, _ = _make_inputs(8, kv_blocks, kv_var, q_var, heads=8, seed=0)
    runner = _run_bhsd if layout == "bhsd" else _run_bshd
    outs = {}
    with torch.no_grad():
        ref = _torch_vsa256_reference(q, k, v, q_var, kv_var, 2)
        for backend in ("cute", "triton"):
            if backend == "triton":
                monkeypatch.setenv("FASTVIDEO_VSA_TRITON", "1")
            outs[backend], _ = runner(q, k, v, kv_var, q_var, 2)
            _check(f"nograd-{backend}-{layout}", ref, outs[backend], _OUT_TOL)
    _check(f"nograd-cute-vs-triton-{layout}", outs["triton"], outs["cute"], _OUT_TOL)


@pytest.mark.cuda
def test_vsa256_cute_inference_matches_training_forward() -> None:
    """Training may use native256 while inference retains expanded128.
    Both implement the same operator, with the existing output tolerance."""
    kv_var = torch.tensor([256, 173, 79, 256], dtype=torch.int32, device="cuda")
    q_var = torch.full((3, ), _BLOCK, dtype=torch.int32, device="cuda")
    q, k, v, _ = _make_inputs(3, 4, kv_var, q_var, seed=5)

    with torch.no_grad():
        out_infer = video_sparse_attn_bshd(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            kv_var,
            q_var,
            2,
            block_size=_BLOCK_SIZE_3D,
            compress_attn_weight=None,
        )

    out_train, _ = _run_bshd(q, k, v, kv_var, q_var, 2)
    _check("inference-training", out_infer, out_train.transpose(1, 2).detach(), _OUT_TOL)


@pytest.mark.cuda
def test_vsa256_cute_lse_is_bhs() -> None:
    """The aux return is [B, H, S] on both entrypoints, matching the Triton
    path's contract."""
    from fastvideo_kernel.block_sparse_attn_256 import (block_sparse_attn_256, block_sparse_attn_256_bshd)

    device = torch.device("cuda")
    heads, dim, q_blocks, kv_blocks = 2, 128, 3, 4
    sq, skv = q_blocks * _BLOCK, kv_blocks * _BLOCK
    q = torch.randn(1, heads, sq, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, heads, skv, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, heads, skv, dim, device=device, dtype=torch.bfloat16)
    vbs = torch.full((kv_blocks, ), _BLOCK, dtype=torch.int32, device=device)
    mask = torch.zeros(1, heads, q_blocks, kv_blocks, dtype=torch.bool, device=device)
    mask[..., :2] = True

    _, lse_bhsd = block_sparse_attn_256(q, k, v, mask, vbs)
    assert lse_bhsd.shape == (1, heads, sq), lse_bhsd.shape

    _, lse_bshd = block_sparse_attn_256_bshd(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        mask,
        vbs,
    )
    assert lse_bshd.shape == (1, heads, sq), lse_bshd.shape


@pytest.mark.parametrize("dim", [64, 128])
def test_vsa256_vector_training_lse_and_graph(dim):
    """Vector forward must retain scalar backward and differentiable LSE."""
    from fastvideo_kernel import block_sparse_attn_cute_fwd as adapter
    from flash_attn.cute.interface import flash_attn_func

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 vector training path")
    if getattr(adapter._build_vbs_vector_mask_mod(256), "__vec_size__", None) != 128:
        pytest.skip("FA4 vector masks unavailable")
    torch.manual_seed(91 + dim)
    q = torch.randn(1, 1024, 2, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k, v = [torch.randn(1, 1536, 2, dim, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True) for _ in range(2)]
    sizes = torch.tensor([256, 131, 67, 0, 255, 32], device="cuda", dtype=torch.int32)
    routes = torch.ones(1, 2, 4, 6, device="cuda", dtype=torch.bool)
    routes[:, :, 0] = False
    dout = torch.randn_like(q)
    dlse = torch.randn(1, 2, 1024, device="cuda")
    dlse[:, :, :256] = 0

    def call(vector, lse_only=False):
        if vector:
            out, lse = adapter._cute_attention(q, k, v, routes, sizes)
        else:
            forward, backward = adapter._build_sparse_tensors(
                routes, sizes, q_len=q.shape[1], q_block_size=256,
                kv_block_size=256, need_backward=True)
            out, lse = flash_attn_func(
                q, k, v, mask_mod=adapter._build_vbs_mask_mod(256), aux_tensors=[sizes],
                block_sparse_tensors=forward, block_sparse_tensors_bwd=backward, return_lse=True)
        grads = (torch.autograd.grad(lse, (q, k, v), dlse) if lse_only else
                 torch.autograd.grad((out, lse), (q, k, v), (dout, dlse)))
        return out, lse, *grads

    for lse_only in (False, True):
        reference, actual = call(False, lse_only), call(True, lse_only)
        for name, expected, got in zip(("dq", "dk", "dv"), reference[2:], actual[2:]):
            _check(name, expected, got, _GRAD_TOL)
    # Eager references retain AccumulateGrad nodes from the default stream.
    # Capture uses fresh leaves and warms them on the capture stream.
    q, k, v = (tensor.detach().requires_grad_(True) for tensor in (q, k, v))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call(True)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = call(True)
    for iteration in range(12):
        with torch.no_grad():
            for tensor in (q, k, v, dout, dlse):
                tensor.normal_()
            dlse[:, :, :256] = 0
            sizes[1] = 67 if iteration % 2 else 131
            sizes[2] = 0 if iteration % 3 == 0 else 67
            routes.copy_(torch.rand_like(routes, dtype=torch.float32) > .45)
            routes[:, :, 0] = False
            routes[:, :, 1:, 0] = True
            for tensor in captured:
                tensor.fill_(float("nan"))
        reference = call(False)
        graph.replay()
        torch.cuda.synchronize()
        finite = torch.isfinite(reference[1])
        assert torch.equal(finite, torch.isfinite(captured[1]))
        torch.testing.assert_close(reference[1][finite], captured[1][finite], atol=1e-5, rtol=1e-5)
        for index, name in ((0, "out"), (2, "dq"), (3, "dk"), (4, "dv")):
            _check(name, reference[index], captured[index], _GRAD_TOL)
