"""VBS prefix declaration and mutable-input CUDA Graph gradient parity."""

import pytest
import torch

from fastvideo_kernel import block_sparse_attn_cute_fwd as adapter


@pytest.mark.parametrize("block", [128, 256])
@pytest.mark.parametrize("dim", [64, 128])
def test_vbs_prefix_mask_graph_gradients(block, dim, monkeypatch):
    pytest.importorskip("flash_attn.cute.block_sparsity")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Requires SM100-family CUDA device")
    torch.manual_seed(707)
    q = torch.randn(2, 777, 3, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 1024, 3, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    sizes = torch.full((1024 // block,), block, device="cuda", dtype=torch.int32)
    routes = torch.ones(2, 3, (777 + block - 1) // block, 1024 // block, device="cuda", dtype=torch.bool)
    dout, dlse = torch.randn_like(q), torch.randn(2, 3, 777, device="cuda")
    function = adapter._CuteAttentionQ128 if block == 128 else adapter._CuteAttentionQ256Training
    mask = adapter._build_vbs_mask_mod(128)
    assert mask.__vbs_kv_block_size__ == 128
    assert getattr(adapter._build_vbs_mask_mod(256), "__vbs_kv_block_size__", 0) == 0

    def run():
        out, lse = function.apply(q, k, v, routes, sizes)
        assert lse.requires_grad == (block == 256)
        return torch.autograd.grad((out, lse) if block == 256 else (out,),
                                   (q, k, v), (dout, dlse) if block == 256 else (dout,))

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for iteration in range(12):
        with torch.no_grad():
            for tensor in (q, k, v, dout, dlse):
                tensor.normal_()
            sizes.random_(0, block + 1)
            if iteration == 0:
                sizes.zero_()
            if iteration == 1:
                sizes.fill_(block)
            routes.copy_(torch.rand_like(routes, dtype=torch.float32) > 0.5)
            routes[:, :, 0] = False
        with monkeypatch.context() as context:
            context.setattr(mask, "__vbs_kv_block_size__", 0)
            reference = run()
        graph.replay()
        torch.cuda.synchronize()
        for expected, actual in zip(reference, captured):
            error = (expected.float() - actual.float()).abs()
            assert actual.isfinite().all()
            assert error.mean() < 1e-3
            assert error.max() / (expected.float().abs().mean() + 1e-6) < 0.25
