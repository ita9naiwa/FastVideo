import functools
import json
import math
import os

import pytest
import torch
from torch.nn import functional as F

from fastvideo_kernel import block_sparse_attn_cute_fwd as adapter


def reference(q, k, v, mask, sizes, block):
    allowed = mask.repeat_interleave(block, -2).repeat_interleave(block, -1)
    valid = torch.arange(k.shape[1], device=k.device) % block < sizes.repeat_interleave(block)
    scores = q.float().transpose(1, 2) @ k.float().transpose(1, 2).transpose(-1, -2)
    scores = scores * q.shape[-1] ** -0.5
    prob = scores.masked_fill(~(allowed & valid), -torch.inf).softmax(-1).nan_to_num()
    return (prob @ v.float().transpose(1, 2)).transpose(1, 2)


def metrics(actual, expected):
    x, y = actual.float().flatten(), expected.float().flatten()
    return {"relative_l2": ((x-y).norm()/y.norm().clamp_min(1e-8)).item(),
            "cosine": F.cosine_similarity(x, y, dim=0).item(),
            "max_abs": (x-y).abs().max().item()}


def expcast_reference(values, kwargs, sizes):
    q, k, v = (x.float().transpose(1, 2) for x in values[:3])
    sparse = kwargs["block_sparse_tensors"]
    qb, kb = sparse.block_size
    out = torch.zeros_like(q)
    factor = torch.tensor(q.shape[-1]**-0.5, device=q.device) * math.log2(math.e)
    factor = factor * (kwargs["q_descale"] * kwargs["k_descale"])
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            for row in range((q.shape[2]+qb-1)//qb):
                query = q[b, h, row*qb:(row+1)*qb]
                maximum = torch.full(query.shape[:1], -torch.inf, device=q.device)
                numerator, denominator = torch.zeros_like(query), torch.zeros_like(maximum)
                # Match FA4's partial-list then full-list order, each traversed in reverse.
                for prefix in ("mask", "full"):
                    count = int(getattr(sparse, prefix+"_block_cnt")[b, h, row])
                    blocks = getattr(sparse, prefix+"_block_idx")[b, h, row, :count].tolist()
                    for logical in reversed(blocks):
                        for child in reversed(range(kb//128)):
                            start = logical*kb+child*128
                            raw = query @ k[b, h, start:start+128].T
                            slots = torch.arange(child*128, (child+1)*128, device=q.device)
                            raw = raw.masked_fill(slots[None] >= sizes[logical], -torch.inf)
                            new_max = torch.maximum(maximum, raw.amax(-1))
                            safe_max = new_max.masked_fill(new_max.isneginf(), 0)
                            alpha = torch.exp2((maximum-safe_max)*factor[b, h])
                            factor8 = factor[b, h]*8
                            bias = (torch.tensor(119.65, device=q.device).double()-safe_max.double()*factor8.double()).float()
                            codes = (raw.double()*factor8.double()+bias[:, None].double()).float()
                            probabilities = codes.round().clamp(0, 120).to(torch.uint8).view(torch.float8_e4m3fn).float()
                            numerator = numerator*alpha[:, None] + probabilities @ v[b, h, start:start+128]
                            denominator = denominator*alpha + probabilities.sum(-1)
                            maximum = new_max
                out[b, h, row*qb:(row+1)*qb] = numerator/denominator[:, None].clamp_min(1e-30)*kwargs["vc_vscale"][b, h]
    return out.transpose(1, 2)


def inputs(block):
    q = torch.randn(1, block*4, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, block*6, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.zeros(1, 2, 4, 6, dtype=torch.bool, device="cuda")
    mask[:, :, 0, 1] = True  # First selected block is partial and excludes physical block zero.
    mask[:, :, 1, [0, 2]] = True
    mask[:, :, 2, [3, 5]] = True  # A second packed document; final Q block is empty.
    sizes = torch.tensor([block, block-17, block//2+3, block, 0, 13], device="cuda", dtype=torch.int32)
    return q, k, v, mask, sizes


def run(block):
    args = inputs(block)
    q, k, v, mask, sizes = args
    call = adapter.block_sparse_attn_cute_fwd_bshd
    os.environ.pop("FASTVIDEO_VSA_VC", None)
    baseline, _ = call(*args)
    expected = reference(*args, block)
    torch.testing.assert_close(baseline.float(), expected, atol=0.015, rtol=0.03)
    os.environ["FASTVIDEO_VSA_VC"] = "1"
    seen = []
    captured_call = []
    from flash_attn.cute import interface
    original_fwd = interface._flash_attn_fwd

    def traced(fn):
        @functools.wraps(fn)
        def wrapped(*values, **kwargs):
            captured_call.append((values, kwargs))
            seen.append((values[0].dtype, kwargs.get("vc_expcast"), kwargs.get("block_sparse_tensors") is not None,
                         kwargs.get("mask_mod") is not None or kwargs.get("vc_vbs128") is True))
            return fn(*values, **kwargs)
        return wrapped

    interface._flash_attn_fwd = traced(original_fwd)
    try:
        actual, lse = call(*args)
        assert seen and all(x == (torch.float8_e4m3fn, True, True, True) for x in seen), seen
    finally:
        interface._flash_attn_fwd = original_fwd
    assert actual.dtype == torch.bfloat16 and torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[:, 3*block:]) == 0
    physical_block = captured_call[-1][1]["block_sparse_tensors"].block_size[1]
    children = torch.arange(block // physical_block, device=sizes.device) * physical_block
    oracle_sizes = (sizes[:, None] - children).clamp(0, physical_block).flatten()
    assert torch.equal(oracle_sizes, captured_call[-1][1]["aux_tensors"][0])
    quantized_oracle = expcast_reference(*captured_call[-1], oracle_sizes)
    fp8_inputs, call_kwargs = captured_call[-1]
    exponent = quantized_oracle.abs().clamp_min(torch.finfo(torch.bfloat16).tiny).log2().floor()
    ulp = exponent.exp2()*torch.finfo(torch.bfloat16).eps
    value_bound = (fp8_inputs[2].float().abs().amax(1)*call_kwargs["vc_vscale"]).amax(-1)
    bound = ulp + 8e-6*(1+value_bound[:, None, :, None])
    bound_ratio = ((actual.float()-quantized_oracle).abs()/bound).amax().item()
    assert bound_ratio <= 1, ("ExpCast oracle BF16 ULP bound", bound_ratio)
    error = metrics(actual, expected)
    # Quantized attention is approximate: <=10% relative L2 and >=0.99 cosine are acceptance limits.
    assert error["relative_l2"] < 0.10 and error["cosine"] > 0.99, error
    changed_k, changed_v = k.clone(), v.clone()
    # Sign changes preserve amax-derived scales, isolating forbidden-token leakage from quantization scale drift.
    changed_v[:, 3*block:] *= -1
    changed, _ = call(q, changed_k, changed_v, mask, sizes)
    torch.testing.assert_close(changed[:, :2*block], actual[:, :2*block], atol=0, rtol=0)
    slots = torch.arange(k.shape[1], device=k.device) % block
    invalid = slots >= sizes.repeat_interleave(block)
    changed_k, changed_v = k.clone(), v.clone()
    changed_v[:, invalid] *= -1
    changed, _ = call(q, changed_k, changed_v, mask, sizes)
    torch.testing.assert_close(changed, actual, atol=0, rtol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, _ = call(*args)
    for _ in range(3):
        graph.replay()
        torch.testing.assert_close(captured, actual, atol=0, rtol=0)
        eager, _ = call(*args)
        torch.testing.assert_close(eager, actual, atol=0, rtol=0)
    with torch.enable_grad():
        try:
            call(q.clone().requires_grad_(), k, v, mask, sizes)
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            assert "infer" in str(exc).lower() or "grad" in str(exc).lower(), str(exc)
        else:
            raise AssertionError("VC must reject autograd instead of silently detaching")
    print(json.dumps({"block": block, "shape_q": list(q.shape), "shape_kv": list(k.shape),
                      "error_vs_fp32": error, "bf16_error_vs_fp32": metrics(baseline, expected),
                      "error_vs_expcast_oracle": metrics(actual, quantized_oracle),
                      "expcast_ulp_bound_ratio": bound_ratio,
                      "dispatch": str(seen), "empty_rows": "zero", "forbidden_value_isolation": "exact",
                      "padding_value_invariance": "exact", "graph": "exact", "grad_guard": "passed"}), flush=True)



@pytest.mark.parametrize("block", [128, 256])
@torch.no_grad()
def test_vc_sparse_attention_contract(block, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("VC sparse attention requires a CUDA GPU")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("VC sparse attention requires Blackwell SM10x")
    if not os.environ.get("FASTVIDEO_VSA_VC_ROOT"):
        pytest.fail("Set FASTVIDEO_VSA_VC_ROOT to the VC-enabled FA4 checkout")
    monkeypatch.setenv("FASTVIDEO_VSA_CUTEDSL", "1")
    monkeypatch.setenv("FASTVIDEO_VSA_VC", "0")
    torch.manual_seed(20260917)
    run(block)
