import math

import pytest

import torch

from fastvideo.attention.backends import video_sparse_attn as vsa_module
from fastvideo.attention.backends.video_sparse_attn import (
    VSA_TILE_SIZE,
    VideoSparseAttentionImpl,
    VideoSparseAttentionMetadataBuilder,
    _compute_cur_topk,
)


def _build_metadata(cache_tile_buf: bool, raw_latent_shape=(4, 4, 4), VSA_sparsity=0.5):
    return VideoSparseAttentionMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=raw_latent_shape,
        patch_size=(1, 1, 1),
        VSA_sparsity=VSA_sparsity,
        device=torch.device("cpu"),
        cache_tile_buf=cache_tile_buf,
    )


def test_vsa_tile_does_not_cache_training_scratch_when_disabled():
    metadata = _build_metadata(cache_tile_buf=False)
    impl = object.__new__(VideoSparseAttentionImpl)
    x = torch.ones(1, 64, 2, 2)

    tiled = impl.tile(x, metadata)

    assert tiled.shape == x.shape
    assert metadata.tile_buf is None


def test_vsa_tile_caches_scratch_by_default():
    metadata = _build_metadata(cache_tile_buf=True)
    impl = object.__new__(VideoSparseAttentionImpl)
    x = torch.ones(1, 64, 2, 2)

    tiled = impl.tile(x, metadata)

    assert metadata.tile_buf is tiled


def test_vsa_tile_cached_and_uncached_produce_identical_values():
    # The cache flag must be a pure performance knob: enabling it (inference /
    # memory-rich training) and disabling it (#1423 OOM-safe training) must
    # yield bit-identical tilings. Use a padded shape (dit T=5 -> padded T=8)
    # so the zero-pad-position scatter is actually exercised.
    impl = object.__new__(VideoSparseAttentionImpl)
    md_cached = _build_metadata(cache_tile_buf=True, raw_latent_shape=(5, 4, 4))
    md_uncached = _build_metadata(cache_tile_buf=False, raw_latent_shape=(5, 4, 4))

    total_seq_length = md_cached.total_seq_length
    x = torch.randn(2, total_seq_length, 3, 4)

    cached = impl.tile(x, md_cached).clone()
    uncached = impl.tile(x, md_uncached)

    assert cached.shape == uncached.shape
    assert torch.equal(cached, uncached)
    # The cached path stashes the buffer; the uncached path does not.
    assert md_cached.tile_buf is not None
    assert md_uncached.tile_buf is None


def test_vsa_forward_cur_topk_uses_padded_kv_block_count(monkeypatch):
    impl = object.__new__(VideoSparseAttentionImpl)
    metadata = _build_metadata(cache_tile_buf=True, raw_latent_shape=(5, 32, 32), VSA_sparsity=0.75)
    block_elements = math.prod(VSA_TILE_SIZE)
    padded_seq_len = metadata.variable_block_sizes.numel() * block_elements
    expected_topk = math.ceil((1 - metadata.VSA_sparsity) * metadata.variable_block_sizes.numel())
    unpadded_topk = math.ceil((1 - metadata.VSA_sparsity) * (metadata.total_seq_length / block_elements))
    captured = {}

    def fake_video_sparse_attn(
        query,
        key,
        value,
        variable_block_sizes,
        q_variable_block_sizes,
        topk,
        block_size,
        compress_attn_weight,
    ):
        captured["topk"] = topk
        captured["block_size"] = block_size
        assert torch.equal(variable_block_sizes, metadata.variable_block_sizes)
        assert torch.equal(q_variable_block_sizes, metadata.variable_block_sizes)
        return query

    monkeypatch.setattr(vsa_module, "video_sparse_attn", fake_video_sparse_attn)

    query = torch.ones(1, padded_seq_len, 1, 1)
    output = impl.forward(query, query, query, query, metadata)

    assert unpadded_topk < expected_topk
    assert captured["topk"] == expected_topk
    assert captured["block_size"] == VSA_TILE_SIZE
    assert output.shape == query.shape


def test_vsa_cur_topk_clamps_to_valid_block_range():
    metadata = _build_metadata(cache_tile_buf=True, raw_latent_shape=(5, 32, 32), VSA_sparsity=1.0)
    num_kv_blocks = metadata.variable_block_sizes.numel()

    assert _compute_cur_topk(metadata) == 1

    metadata.VSA_sparsity = -0.01
    assert _compute_cur_topk(metadata) == num_kv_blocks


@pytest.mark.parametrize("block", [64, 128])
@pytest.mark.parametrize("kernel_backend", ["triton", "cutedsl"])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_vsa128_view_dispatch_preserves_fallbacks(monkeypatch, block, kernel_backend, cache, grad_enabled):
    kernel = pytest.importorskip("fastvideo_kernel.block_sparse_attn_256")
    monkeypatch.setattr(kernel, "_resolve_backend", lambda: kernel_backend)
    tile = (4, 4, 4) if block == 64 else (4, 8, 4)
    monkeypatch.setattr(vsa_module, "VSA_TILE_SIZE", tile)
    metadata = _build_metadata(cache, raw_latent_shape=(4, 8, 16))
    n = metadata.variable_block_sizes.numel() * block
    x = torch.randn(1, n, 3, 64)
    seen = []

    def fake(q, k, v, sizes, qsizes, topk, block_size, compress_attn_weight):
        seen.append((q.is_contiguous(), q.data_ptr() == x.data_ptr()))
        return q

    monkeypatch.setattr(vsa_module, "video_sparse_attn", fake)
    impl = object.__new__(VideoSparseAttentionImpl)
    with torch.set_grad_enabled(grad_enabled):
        impl.forward(x, x, x, x, metadata)
    views = block == 128 and kernel_backend == "cutedsl" and not (grad_enabled and cache)
    assert seen == [(not views, views)]


@pytest.mark.parametrize("dim", [64, 128])
def test_vsa128_backend_views_match_cached_training_snapshots(monkeypatch, dim):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM10x required")
    kernel = pytest.importorskip("fastvideo_kernel.block_sparse_attn_256")
    monkeypatch.setattr(kernel, "_resolve_backend", lambda: "cutedsl")
    monkeypatch.setenv("FASTVIDEO_VSA_PACK_TAILS", "0")
    monkeypatch.setattr(vsa_module, "VSA_TILE_SIZE", (4, 8, 4))
    torch.manual_seed(934)
    metadata = _build_metadata(False, raw_latent_shape=(4, 8, 32), VSA_sparsity=.75)
    sizes = torch.tensor([128, 67, 17, 127] * 2, device="cuda", dtype=torch.int32)
    metadata.variable_block_sizes = sizes
    packed = torch.randn(8, 1024, 3, dim, device="cuda", dtype=torch.bfloat16)
    valid = (torch.arange(1024, device="cuda") % 128 < sizes[torch.arange(1024, device="cuda") // 128])
    packed.mul_(valid[None, :, None, None])
    packed[6:].mul_(.1)
    packed.requires_grad_()
    q, k, v, gate = packed.chunk(4)
    dy = torch.randn_like(gate)
    impl = object.__new__(VideoSparseAttentionImpl)
    got = impl.forward(q, k, v, gate, metadata)
    metadata.cache_tile_buf = True
    ref = impl.forward(q, k, v, gate, metadata)
    got_grad = torch.autograd.grad(got, packed, dy)[0]
    ref_grad = torch.autograd.grad(ref, packed, dy)[0]
    for actual, expected in ((got, ref), (got_grad, ref_grad)):
        diff = (actual.float() - expected.float()).abs()
        assert torch.isfinite(actual).all()
        assert diff.mean() < 1e-3
        assert diff.max() / (expected.float().abs().mean() + 1e-6) < .25
