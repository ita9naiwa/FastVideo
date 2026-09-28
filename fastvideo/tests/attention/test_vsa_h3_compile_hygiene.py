# SPDX-License-Identifier: Apache-2.0
"""Compile hygiene of the SP=1 H3 VSA layer path (layer._forward_separate_qkv) over gated/ungated, training fwd+bwd and
no-grad inference, the BF16 and the VC FP8 route (no-grad only), tile 128 and 256, under (a) Inductor fullgraph + dynamic
(+ SAC when training, h3mh's config) and (b) mode="reduce-overhead" (CUDA graphs).

Four real documents per cell, compiled cold (before any eager call, as the loader) and replayed warm: 0 graph breaks,
0 cudagraph skips, one graph except the documented n_tiles % 8 recompile of the gated training graph (117h: Inductor
rewrites the BF16 coarse attention to SDPA, which guards n_tiles % 8), compiled close to eager (ungated no-grad: bitwise).
"""
import functools
import logging
import os

import pytest
import torch
import torch.fx.experimental._config as fx_config
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from fastvideo.attention import layer
from fastvideo.attention.backends import video_sparse_attn_h3 as h3

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                                reason="needs SM10x (CuTe VSA ops)")

# s085k32 spec documents (latent T, H, W; prefix segments) plus one with n_tiles % 8 == 0 per tile (t128: 96, t256: 56).
DOCS = [((47, 7, 12), (87, 1, 84, 514)), ((52, 15, 26), (476, 1, 390, 572)), ((37, 12, 31), (224, 1, 0, 402))]
DOC8 = {128: ((31, 12, 31), (224, 1, 0, 402)), 256: ((36, 12, 31), (224, 1, 0, 402))}
CFG = {128: dict(VSA_sparsity=0.75, topk_cap=64), 256: dict(VSA_sparsity=0.85, topk_cap=32)}  # h3mh t128 / s085k32
HEADS = 4
CELLS = [(tile, gated, train, route) for tile in (128, 256) for gated in (False, True) for train in (True, False)
         for route in ("bf16", "vc") if not (train and route == "vc")]


class _Recompiles(logging.Handler):

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.reasons = []

    def emit(self, record):
        if "triggered by the following guard failure" in record.getMessage():
            self.reasons.append(record.getMessage())


def _meta(tile, doc):
    (t, hh, w), prefix = doc
    return h3.MiniMaxH3VSAMetadataBuilder().build(current_timestep=0,
                                                  raw_latent_shape=(t, 2 * hh, 2 * w),
                                                  patch_size=(1, 2, 2),
                                                  prefix_segments=prefix,
                                                  device=torch.device("cuda"),
                                                  tile_size=tile,
                                                  tile_layout=f"chunk{tile}",
                                                  merge_prefix=True,
                                                  **CFG[tile])


@pytest.mark.parametrize("mode", ["fullgraph_dynamic", "reduce-overhead"])
@pytest.mark.parametrize(
    "tile,gated,train,route",
    CELLS,
    ids=[f"t{t}-{'gated' if g else 'ungated'}-{'train' if tr else 'nograd'}-{r}" for t, g, tr, r in CELLS])
def test_h3_compile_hygiene(monkeypatch, tile, gated, train, route, mode):
    if route == "vc" and not os.environ.get("FASTVIDEO_VSA_VC_ROOT"):
        pytest.skip("VC route needs FASTVIDEO_VSA_VC_ROOT (VC-enabled FA4 checkout)")
    monkeypatch.setenv("FASTVIDEO_VSA_VC", "1" if route == "vc" else "0")
    monkeypatch.delenv("FASTVIDEO_H3_VSA_PROBE", raising=False)
    monkeypatch.setattr(torch._dynamo.config, "fail_on_recompile_limit_hit", True)
    monkeypatch.setattr(fx_config, "use_duck_shape", False)
    impl = h3.MiniMaxH3VSAImpl(num_heads=HEADS,
                               head_size=128,
                               causal=False,
                               softmax_scale=128**-0.5,
                               prefix="blocks.0.attn")
    impl.layer_idx = 0
    if train:
        impl.prepare_for_compile(torch.device("cuda"))
    else:
        impl.prepare_for_regional_compile(torch.device("cuda"))
        assert impl._regional_compile_nograd_route == route

    def block(q, k, v, g, md):
        return layer._forward_separate_qkv(impl, q, k, v, q.shape[1], None, md, g)[0]

    ops = torch.ops.fastvideo_kernel
    must_save = {ops.vsa256_fwd.default, ops.vsa_train_fwd.default, ops.vsa_h3_block_map.default}
    policy = lambda ctx, op, *a, **k: CheckpointPolicy.MUST_SAVE if op in must_save else CheckpointPolicy.PREFER_RECOMPUTE

    def sac(*x):
        return checkpoint(block,
                          *x,
                          use_reentrant=False,
                          context_fn=functools.partial(create_selective_checkpoint_contexts, policy))

    if mode == "fullgraph_dynamic":
        compiled = torch.compile(sac if train else block, backend="inductor", dynamic=True, fullgraph=True)
    else:
        compiled = torch.compile(block, mode="reduce-overhead", dynamic=True, fullgraph=True)

    def step(fn, md, seed, is_compiled):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        x = [
            torch.randn(1, md.total_seq_length, HEADS, 128, device="cuda", dtype=torch.bfloat16, generator=gen)
            for _ in range(5 if gated else 4)
        ]
        up = x.pop()
        x = [t.requires_grad_(train) for t in x] + ([] if gated else [None])
        if is_compiled and mode == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()
        with torch.set_grad_enabled(train):
            out = fn(*x, md)
            res = (out, *torch.autograd.grad(out, [t for t in x if t is not None], up)) if train else (out, )
        return [r.detach().clone() for r in res]

    mds = [_meta(tile, d) for d in DOCS + [DOC8[tile]]]
    counters = torch._dynamo.utils.counters
    torch._dynamo.reset()
    counters.clear()
    torch._logging.set_logs(recompiles=True)  # clears handlers and stops propagation at torch._dynamo
    handler, dynamo_log = _Recompiles(), logging.getLogger("torch._dynamo")
    dynamo_log.addHandler(handler)
    try:
        got = [step(compiled, md, 10 + i, True) for _ in range(2) for i, md in enumerate(mds)]  # cold, then warm
        graphs, breaks = counters["stats"]["unique_graphs"], sum(counters["graph_break"].values())
        skips = counters["inductor"]["cudagraph_skips"]
    finally:
        dynamo_log.removeHandler(handler)
        torch._logging.set_logs()
        torch._dynamo.reset()
    assert breaks == 0, dict(counters["graph_break"])
    assert skips == 0
    assert all("% 8" in r for r in handler.reasons), handler.reasons
    assert graphs == 1 + len(handler.reasons) and graphs <= (2 if gated and train else 1), handler.reasons

    eager = [step(block, md, 10 + i, False) for i, md in enumerate(mds)]
    for j, outs in enumerate(got):
        for c, e in zip(outs, eager[j % len(mds)], strict=True):
            if not gated and not train:
                assert torch.equal(c, e), j
            else:  # ungated dq: nondeterministic even eager vs eager; gated: BF16 coarse chain, 1-2 ulp (117f)
                rel = ((c.float() - e.float()).norm() / e.float().norm()).item()
                assert rel < 2e-2, (j, rel)
