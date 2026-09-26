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


def _classified_reference(mask, sizes, kv_block):
    full = mask.bool() & (sizes == kv_block)
    part = mask.bool() & (sizes > 0) & (sizes < kv_block)
    return (*reference(full), *reference(part))


@pytest.mark.parametrize('kv_block', [128, 256])
def test_classified_indices_one_compile_per_bucket(kv_block):
    """Outputs match the reference for any (H, Q, N, strides), and shapes never force a recompile."""
    from fastvideo_kernel.triton_kernels import index
    kernel = index._classified_map_to_index_kernel
    shapes = [(1, 1, 1, 1), (1, 2, 5, 17), (2, 3, 7, 100), (1, 56, 9, 128), (1, 4, 11, 129), (2, 8, 3, 200),
              (1, 56, 13, 256), (1, 5, 2, 300)]
    for b, h, q, n in shapes:
        mask = torch.rand(b, h, q, n, device='cuda') > .6
        for transposed in (False, True):
            m = mask.transpose(-1, -2).contiguous().transpose(-1, -2) if transposed else mask
            sizes = torch.randint(0, kv_block + 1, (n,), device='cuda', dtype=torch.int32)
            got = index.map_to_classified_indices(m, sizes, kv_block)
            for g, r in zip(got, _classified_reference(m, sizes, kv_block), strict=True):
                assert torch.equal(g, r), (b, h, q, n, transposed)
    cache = getattr(kernel, 'device_caches', None)
    if cache is not None:
        # BLOCK buckets 128 / 256 / 512 (n = 300) for this KV_BLOCK; no per-(H, Q, N, stride) variants.
        compiled = cache[torch.cuda.current_device()][0]
        per_block = [k for k in compiled if f"('constexpr', {kv_block}), ('constexpr'" in str(k)]  # this KV_BLOCK
        assert len(per_block) <= 3, len(per_block)
