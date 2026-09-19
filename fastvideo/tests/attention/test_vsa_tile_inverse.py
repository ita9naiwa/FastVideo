"""Builder permutation tiling preserves native values, derivatives and fallback."""
import pytest
import torch
from fastvideo.attention.backends import video_sparse_attn as m


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_inverse_tile_values_gradients_and_index_versions(dtype):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    x = torch.randn(2, 33, 3, 16, device='cuda', dtype=dtype, requires_grad=True)
    src = torch.randperm(33, device='cuda')
    dst = torch.randperm(48, device='cuda')[:33]
    reference = m.scatter_into_tile_buf(x, (2, 48, 3, 16), dst, None, src)
    actual = m._TilePermutation.apply(x, src, dst, 48)
    dy = torch.randn_like(actual)
    ga, gb = [torch.autograd.grad(t, x, dy, retain_graph=True)[0] for t in (reference, actual)]
    torch.testing.assert_close(reference, actual, atol=0, rtol=0)
    torch.testing.assert_close(ga, gb, atol=0, rtol=0)
    src.add_(0)
    for out in (reference, actual):
        with pytest.raises(RuntimeError, match='modified by an inplace operation'):
            torch.autograd.grad(out, x, dy)


def test_inverse_tile_builder_mutation_and_foreign_metadata_fallback(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    md = m.VideoSparseAttentionMetadataBuilder().build(
        0, (32, 32, 32), (1, 1, 1), .9, torch.device('cuda'), cache_tile_buf=False)
    x = torch.randn(1, 32768, 8, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    impl = object.__new__(m.VideoSparseAttentionImpl)
    actual = impl.tile(x, md)
    assert type(actual.grad_fn).__name__ == '_TilePermutationBackward'
    reference = m.scatter_into_tile_buf(x, tuple(actual.shape), md.non_pad_index, None, md.tile_partition_indices)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    def forbidden(*args):
        raise AssertionError('changed/foreign metadata used inverse path')
    monkeypatch.setattr(m._TilePermutation, 'apply', forbidden)
    md.tile_partition_indices = md.tile_partition_indices.clone()
    torch.testing.assert_close(impl.tile(x, md), reference, atol=0, rtol=0)
    md._tile_index_state = None
    torch.testing.assert_close(impl.tile(x, md), reference, atol=0, rtol=0)


def test_inverse_tile_inference_builder_keeps_fallback():
    for fn in (m.get_tile_partition_indices, m.get_reverse_tile_partition_indices,
               m.construct_variable_block_sizes, m.get_non_pad_index):
        fn.cache_clear()
    with torch.inference_mode():
        md = m.VideoSparseAttentionMetadataBuilder().build(
            0, (7, 7, 7), (1, 1, 1), .9, torch.device('cpu'), cache_tile_buf=False)
    assert md._tile_index_state is None


def test_inverse_tile_keeps_input_value_pruning_and_higher_order():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    x = torch.randn(2, 33, 3, 16, device='cuda', requires_grad=True)
    src = torch.randperm(33, device='cuda')
    dst = torch.randperm(48, device='cuda')[:33]
    functions = (lambda: m.scatter_into_tile_buf(x, (2, 48, 3, 16), dst, None, src),
                 lambda: m._TilePermutation.apply(x, src, dst, 48))
    outputs = [fn() for fn in functions]
    with torch.no_grad():
        x.add_(1)
    gradients = [torch.autograd.grad(out.sum(), x)[0] for out in outputs]
    torch.testing.assert_close(*gradients, atol=0, rtol=0)
    values = []
    for fn in functions:
        first = torch.autograd.grad(fn().square().sum(), x, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), x)[0]
        values.append((first, second))
    for a, b in zip(*values):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_inverse_tile_builder_rejects_mutated_cached_indices():
    for fn in (m.get_tile_partition_indices, m.get_reverse_tile_partition_indices,
               m.construct_variable_block_sizes, m.get_non_pad_index):
        fn.cache_clear()
    builder = m.VideoSparseAttentionMetadataBuilder()
    args = (0, (5, 7, 9), (1, 1, 1), .9, torch.device('cpu'))
    first = builder.build(*args, cache_tile_buf=False)
    assert first.tile_partition_indices._version == first.non_pad_index._version == 0
    assert first._tile_index_state is not None
    first.tile_partition_indices.add_(0)
    again = builder.build(*args, cache_tile_buf=False)
    assert again.tile_partition_indices is first.tile_partition_indices
    assert again._tile_index_state is None


@pytest.mark.parametrize('batch,offset,dim', [(1, 0, 16), (2, 8, 16), (2, 1, 16), (2, 0, 3)])
def test_tile_row_gather_preserves_alignment_and_stride_fallback(batch, offset, dim):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    storage = torch.randn(batch * 33 * 3 * dim + offset, device='cuda', dtype=torch.bfloat16)
    x = storage[offset:].view(batch, 33, 3, dim)
    index = torch.randperm(33, device='cuda')
    for tensor in (x, x[..., ::2]):
        expected = tensor.index_select(1, index)
        actual = m._gather_tile_rows(tensor, index)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
