"""Direct sparse routes must preserve map traversal under changing Graph inputs."""
import pytest
import torch

from fastvideo_kernel import block_sparse_attn_cute_fwd as adapter


@pytest.mark.parametrize("block", [128, 256])
@pytest.mark.parametrize("prefix,topk", [(0, 1), (2, 6)])
@torch.inference_mode()
def test_vc_direct_routes_graph(block, prefix, topk):
    from flash_attn.cute.vc_preprocess import prepare

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("B300 VC kernel test")
    parents, start = 12, 13
    torch.manual_seed(91)
    q, k, v = [torch.randn(1, parents * block, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    p = prepare(q, k, v, smooth=False, bshd=True)
    sizes = torch.tensor([0] * 8 + [67, min(131, block), block, block], device="cuda", dtype=torch.int32)
    selected = torch.empty((1, 2, parents, topk), device="cuda", dtype=torch.int64)
    block_map = torch.empty((1, 2, parents, parents), device="cuda", dtype=torch.bool)

    def change_routes():
        scores = torch.randn(1, 2, parents, parents - prefix, device="cuda")
        selected.copy_(scores.topk(topk, -1).indices + prefix + start)
        selected[:, :, 0, :] = torch.arange(prefix, prefix + topk, device="cuda") + start
        block_map.zero_()
        block_map[..., :prefix] = True
        block_map.scatter_(-1, selected - start, True)

    def call(direct):
        if direct:
            return adapter.block_sparse_attn_vc_routes_fwd_bshd(
                p, selected, sizes, block, prefix, start, return_lse=False,
            )[0]
        return adapter.block_sparse_attn_vc_prepared_fwd_bshd(p, block_map, sizes, return_lse=False)[0]

    change_routes()
    assert torch.equal(call(False), call(True))
    with pytest.raises(ValueError):
        adapter.block_sparse_attn_vc_routes_fwd_bshd(p, selected.to(torch.int32), sizes, block, prefix, start)
    graphs, outputs = [], []
    for direct in (False, True):
        for _ in range(3):
            call(direct)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = call(direct)
        graphs.append(graph)
        outputs.append(output)
    for step in range(10):
        change_routes()
        fresh = prepare(q.roll(step + 1, 1) * (step % 7 != 0), k.roll(2 * step + 1, 1),
                        v.roll(3 * step + 1, 1) * (-1 if step % 2 else 1), smooth=False, bshd=True)
        for key in ("q", "k", "v", "qs", "ks", "vs"):
            p[key].copy_(fresh[key])
        for index in ((0, 1) if step % 2 == 0 else (1, 0)):
            outputs[index].fill_(float("nan"))
            graphs[index].replay()
        assert torch.equal(*outputs)
        for direct, output in enumerate(outputs):
            assert torch.equal(output, call(bool(direct)))
            assert torch.isfinite(output).all()
            assert (output[:, :block] == 0).all()
