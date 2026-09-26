# SPDX-License-Identifier: Apache-2.0
"""_TilePermutation zeroes only the pad rows (fixed-count compaction + index_fill_): bitwise equal to the full masked fill,
no host synchronization, CUDA-graph replay with changed permutations of the same shape, unchanged inverse gradient."""
import pytest
import torch

from fastvideo.attention.backends.video_sparse_attn import _gather_tile_rows, _TilePermutation

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def _reference(x, partition, nonpad, padded_length):
    source = partition.new_zeros(padded_length).index_copy_(0, nonpad, partition)
    padding = torch.ones(padded_length, device=x.device, dtype=torch.bool).index_fill_(0, nonpad, False)
    return _gather_tile_rows(x, source).masked_fill_(padding[None, :, None, None], 0)


@pytest.mark.parametrize("batch,rows,padded", [(1, 256, 256), (3, 700, 1024), (4, 1, 256), (2, 5000, 5120)])
def test_pad_rows_bitwise_equal_to_masked_fill(batch, rows, padded):
    torch.manual_seed(batch * rows)
    x = torch.randn(batch, rows, 2, 128, device="cuda", dtype=torch.bfloat16)
    x[:, 0, 0, 0] = float("nan")  # a valid NaN (row 0 is also the gather source of every pad slot) must survive
    partition = torch.randperm(rows, device="cuda")
    nonpad = torch.randperm(padded, device="cuda")[:rows]
    out = _TilePermutation.apply(x, partition, nonpad, padded)
    ref = _reference(x, partition, nonpad, padded)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


def test_pad_rows_no_host_sync_and_graph_replay():
    torch.manual_seed(0)
    rows, padded = 700, 1024
    x = torch.randn(3, rows, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    partition = torch.randperm(rows, device="cuda")
    nonpad = torch.randperm(padded, device="cuda")[:rows]
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = _TilePermutation.apply(x, partition, nonpad, padded)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    g = torch.randn_like(out)
    (gx, ) = torch.autograd.grad(out, x, g)
    inverse = partition.new_empty(rows).index_copy_(0, partition, nonpad)
    assert torch.equal(gx, g[:, inverse])
    with torch.no_grad():
        xs = x.detach().clone()
        for _ in range(2):
            _TilePermutation.apply(xs, partition, nonpad, padded)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _TilePermutation.apply(xs, partition, nonpad, padded)
        for step in range(3):
            partition.copy_(torch.randperm(rows, device="cuda"))
            nonpad.copy_(torch.randperm(padded, device="cuda")[:rows])
            xs.normal_()
            graph.replay()
            assert torch.equal(captured.view(torch.int16), _reference(xs, partition, nonpad, padded).view(torch.int16))
