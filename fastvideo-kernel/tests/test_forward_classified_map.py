"""Forward classification must preserve stable full/partial lists and Graph updates."""
import pytest
import torch

from fastvideo_kernel import block_sparse_attn_cute_fwd as adapter


@pytest.mark.parametrize("block,n", [(128, 1), (128, 33), (128, 300), (256, 33), (256, 4097)])
def test_classified_forward_map(block, n):
    pytest.importorskip("flash_attn.cute.block_sparsity")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    maps = torch.rand(1, 2, 3, n * 2, device="cuda") > .5
    maps = maps[..., ::2]
    sizes = torch.empty(n * 2, device="cuda", dtype=torch.int32)[::2]

    def run():
        return adapter._build_sparse_tensors(
            maps, sizes, q_len=3 * block, q_block_size=block,
            kv_block_size=block, need_backward=False, force_q_sparse_block_size=block,
        )[0]

    def check(result):
        columns = torch.arange(n, device="cuda").expand(maps.shape)
        for indices, counts, valid in (
            (result.full_block_idx, result.full_block_cnt, sizes == block),
            (result.mask_block_idx, result.mask_block_cnt, (sizes > 0) & (sizes < block)),
        ):
            mask = maps & valid
            ordered = torch.where(mask, columns, n).sort(-1).values
            expected = torch.where(ordered < n, ordered, -1).int()
            assert torch.equal(indices, expected)
            assert torch.equal(counts, mask.sum(-1).int())

    sizes.fill_(block)
    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for mode in ("empty", "full", "mixed"):
        maps.logical_not_()
        sizes.copy_(torch.arange(n, device="cuda") % (block + 2))
        if mode != "mixed":
            sizes.fill_(0 if mode == "empty" else block)
        for field in ("full_block_idx", "full_block_cnt", "mask_block_idx", "mask_block_cnt"):
            getattr(captured, field).fill_(-777)
        graph.replay()
        check(captured)
        check(run())
