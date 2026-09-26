"""The KV-owned transpose is readable through its original tensor strides."""
import pytest
import torch

from fastvideo_kernel import block_sparse_attn_cute_fwd as adapter


@pytest.mark.parametrize("q_blocks,factor", [(1, 1), (17, 2), (4097, 1)])
def test_backward_map_strides(q_blocks, factor):
    pytest.importorskip("flash_attn.cute.block_sparsity")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    maps = torch.rand(2, 2, q_blocks, 5, device="cuda") > .7
    sizes = torch.tensor([0, 128, 127, 1, 128], device="cuda", dtype=torch.int32)

    def run():
        return adapter._build_sparse_tensors(
            maps, sizes, q_len=q_blocks * 128, q_block_size=128, kv_block_size=128,
            need_backward=True, need_forward=False, force_q_sparse_block_size=128 * factor,
        )[1]

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for _ in range(3):
        maps.logical_not_()
        sizes.copy_(sizes.roll(1))
        graph.replay()
        padded = torch.nn.functional.pad(maps, (0, 0, 0, (-q_blocks) % factor))
        grouped = padded.reshape(2, 2, -1, factor, 5).any(3).transpose(2, 3)
        n = grouped.shape[-1]
        columns = torch.arange(n, device="cuda").expand(grouped.shape)
        ordered = torch.where(grouped, columns, n).sort(-1).values
        expected = torch.where(ordered < n, ordered, -1).int()
        assert torch.equal(actual.full_block_idx, expected)
        assert torch.equal(actual.mask_block_idx, expected)
        counts = grouped.sum(-1).int()
        assert torch.equal(actual.full_block_cnt, counts * (sizes == 128))
        assert torch.equal(actual.mask_block_cnt, counts * ((sizes > 0) & (sizes < 128)))
