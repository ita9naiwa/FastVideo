"""Stable sparse-list compaction, including changed graph contents."""
import pytest
import torch

from fastvideo_kernel.triton_kernels.index import map_to_index


def reference(mask):
    n = mask.shape[-1]
    columns = torch.arange(n, device=mask.device).expand(mask.shape)
    indices = torch.where(mask.bool(), columns, n).sort(-1).values
    return torch.where(indices < n, indices, -1).int(), mask.bool().sum(-1).int()


@pytest.mark.parametrize('n', [0, 1, 127, 128, 129, 300, 4096, 4097])
@pytest.mark.parametrize('transpose', [False, True])
def test_compaction(n, transpose):
    mask = torch.rand(2, 3, 7, n, device='cuda') > .8
    if transpose:
        mask = mask.transpose(-1, -2).contiguous().transpose(-1, -2)
    for kind in ('random', 'empty', 'full'):
        if kind == 'empty':
            mask.zero_()
        elif kind == 'full':
            mask.fill_(True)
        for actual, expected in zip(map_to_index(mask), reference(mask)):
            assert torch.equal(actual, expected)


def test_numeric_fallback():
    mask = torch.tensor([[[[0., -1., 2., 0.]]]], device='cuda')
    for actual, expected in zip(map_to_index(mask), reference(mask)):
        assert torch.equal(actual, expected)


def test_changed_graph_contents():
    mask = torch.zeros(2, 3, 7, 300, device='cuda', dtype=torch.bool)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            map_to_index(mask)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = map_to_index(mask)
    for density in (0., .05, .5, 1., .1, 0.):
        mask.copy_(torch.rand(mask.shape, device='cuda') < density)
        graph.replay()
        for actual, expected in zip(captured, reference(mask)):
            assert torch.equal(actual, expected)
