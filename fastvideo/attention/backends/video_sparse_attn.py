# SPDX-License-Identifier: Apache-2.0
import functools
import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

try:
    from fastvideo_kernel import video_sparse_attn
except ImportError:
    video_sparse_attn = None
try:
    from fastvideo_kernel import video_sparse_attn_bshd
except ImportError:
    video_sparse_attn_bshd = None

from typing import Any

from fastvideo.attention.backends.abstract import (AttentionBackend, AttentionImpl, AttentionMetadata,
                                                   AttentionMetadataBuilder)
from fastvideo.distributed import get_sp_group
from fastvideo.logger import init_logger

logger = init_logger(__name__)
# VSA tile shape. The tile volume picks the kernel path automatically in
# forward(): (4,4,4)=64 -> existing TK/Triton path (default, unchanged);
# (4,8,8)=256 -> FA4 CuTe block-sparse attention fastpath (Blackwell).
VSA_TILE_SIZE = (4, 4, 4)


@functools.lru_cache(maxsize=10)
def get_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    T, H, W = dit_seq_shape
    ts, hs, ws = tile_size
    indices = torch.arange(T * H * W, device=device, dtype=torch.long).reshape(T, H, W)
    ls = []
    for t in range(math.ceil(T / ts)):
        for h in range(math.ceil(H / hs)):
            for w in range(math.ceil(W / ws)):
                ls.append(indices[t * ts:min(t * ts + ts, T), h * hs:min(h * hs + hs, H),
                                  w * ws:min(w * ws + ws, W)].flatten())
    index = torch.cat(ls, dim=0)
    return index


@functools.lru_cache(maxsize=10)
def get_reverse_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    return torch.argsort(get_tile_partition_indices(dit_seq_shape, tile_size, device))


@functools.lru_cache(maxsize=10)
def construct_variable_block_sizes(
    dit_seq_shape: tuple[int, int, int],
    num_tiles: tuple[int, int, int],
    device: torch.device,
    tile_size: tuple[int, int, int] = VSA_TILE_SIZE,
) -> torch.LongTensor:
    """
    Compute the number of valid (non‑padded) tokens inside every
    (ts_t × ts_h × ts_w) tile after padding ‑‑ flattened in the order
    (t‑tile, h‑tile, w‑tile) that `rearrange` uses.

    Returns
    -------
    torch.LongTensor  # shape: [∏ full_window_size]
    """
    # unpack
    t, h, w = dit_seq_shape
    ts_t, ts_h, ts_w = tile_size
    n_t, n_h, n_w = num_tiles

    def _sizes(dim_len: int, tile: int, n_tiles: int) -> torch.LongTensor:
        """Vector with the size of each tile along one dimension."""
        sizes = torch.full((n_tiles, ), tile, dtype=torch.int, device=device)
        # size of last (possibly partial) tile
        remainder = dim_len - (n_tiles - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    t_sizes = _sizes(t, ts_t, n_t)  # [n_t]
    h_sizes = _sizes(h, ts_h, n_h)  # [n_h]
    w_sizes = _sizes(w, ts_w, n_w)  # [n_w]

    # broadcast‑multiply to get voxels per tile, then flatten
    block_sizes = (
        t_sizes[:, None, None]  # [n_t, 1,   1]
        * h_sizes[None, :, None]  # [1,   n_h, 1]
        * w_sizes[None, None, :]  # [1,   1,   n_w]
    ).reshape(-1)  # [n_t * n_h * n_w]

    return block_sizes


@functools.lru_cache(maxsize=10)
def get_non_pad_index(
    variable_block_sizes: torch.LongTensor,
    max_block_size: int,
):
    n_win = variable_block_sizes.shape[0]
    device = variable_block_sizes.device
    starts_pad = torch.arange(n_win, device=device) * max_block_size
    index_pad = starts_pad[:, None] + torch.arange(max_block_size, device=device)[None, :]
    index_mask = torch.arange(max_block_size, device=device)[None, :] < variable_block_sizes[:, None]
    return index_pad[index_mask]


class VideoSparseAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "VIDEO_SPARSE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["VideoSparseAttentionImpl"]:
        return VideoSparseAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["VideoSparseAttentionMetadata"]:
        return VideoSparseAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["VideoSparseAttentionMetadataBuilder"]:
        return VideoSparseAttentionMetadataBuilder


@dataclass
class VideoSparseAttentionMetadata(AttentionMetadata):
    current_timestep: int
    dit_seq_shape: list[int]
    num_tiles: list[int]
    total_seq_length: int
    tile_partition_indices: torch.LongTensor
    reverse_tile_partition_indices: torch.LongTensor
    variable_block_sizes: torch.LongTensor
    non_pad_index: torch.LongTensor
    # Precomputed fancy index that fuses ``x[:, non_pad_index][:, reverse_tile_partition_indices]``
    # in postprocess_output().  Avoids materializing the intermediate
    # ``[B, len(non_pad_index), H, D]`` tensor on every layer.
    untile_combined_index: torch.LongTensor
    # Per-step shared padded buffer used by tile().  Inference can reuse this
    # across VSA layers, but training disables it so activation checkpointing
    # can release the large tiled QKVG scratch tensor after each attention call.
    tile_buf: torch.Tensor | None = None
    cache_tile_buf: bool = True
    # Only builder-created, unmodified permutation metadata admits inverse tiling.
    _tile_index_state: tuple[torch.Tensor, int, torch.Tensor, int] | None = None


def compute_topk(sparsity: float, num_blocks: int) -> int:
    """Blocks to keep for a sparsity level, clamped to [1, num_blocks].

    ceil((1 - sparsity) * num_blocks) in exact integer arithmetic on the keep fraction in parts per million: the float
    product overshoots at exact multiples (1 - 0.85 = 0.15000000000000002, so 20 blocks gave 4 instead of 3).
    """
    keep_ppm = round((1 - sparsity) * 1_000_000)
    return max(1, min(-(-keep_ppm * num_blocks // 1_000_000), num_blocks))


def _compute_cur_topk(attn_metadata: VideoSparseAttentionMetadata) -> int:
    return compute_topk(attn_metadata.VSA_sparsity, attn_metadata.variable_block_sizes.numel())


def scatter_into_tile_buf(
    x: torch.Tensor,
    target_shape: tuple[int, ...],
    dst_index: torch.Tensor,
    buf: torch.Tensor | None,
    src_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Zero-padded tile scatter shared by the VSA backends.

    Allocates (zeros) when ``buf`` is missing or mismatched; otherwise reuses
    it — pad slots are never written and every non-pad slot is fully
    overwritten per call, so a reused buffer stays valid. Callers own the
    buffer's lifetime (per-metadata for Wan VSA, per-builder for VSA-H3) and
    its aliasing contract: the result is only valid until the next call with
    the same buffer.
    """
    if (buf is None or buf.shape != target_shape or buf.dtype != x.dtype or buf.device != x.device):
        buf = torch.zeros(target_shape, device=x.device, dtype=x.dtype)
    buf[:, dst_index] = x if src_index is None else x[:, src_index]
    return buf


def _tile_row_layout(x: torch.Tensor) -> bool:
    if x.ndim != 4:
        return False
    batch, length, heads, dim = x.shape
    width = heads * dim
    return (batch > 0 and length > 0 and width > 0 and x.stride()[1:] == (width, dim, 1)
            and x.stride(0) >= length * width and x.stride(0) % width == 0 and x.data_ptr() % 16 == 0
            and width * x.element_size() % 16 == 0)


def _gather_tile_rows(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if x.shape[0] > 1 and _tile_row_layout(x):
        batch, length, heads, dim = x.shape
        batch_rows = x.stride(0) // (heads * dim)
        rows = (index[None, :] +
                torch.arange(batch, device=x.device, dtype=index.dtype)[:, None] * batch_rows).flatten()
        # Last row ends at the original last logical element, never after its storage.
        flat = x.as_strided(((batch - 1) * batch_rows + length, heads * dim), (heads * dim, 1))
        return flat.index_select(0, rows).view(batch, index.numel(), heads, dim)
    return x.index_select(1, index)


class _TilePermutation(torch.autograd.Function):
    """Invert builder-bijective rows with distinct nonpad destinations.

    Save original indices for version checks; rebuild the plan each call so
    captured calls support changed valid permutations without a stale cache.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, partition: torch.Tensor, nonpad: torch.Tensor,
                padded_length: int) -> torch.Tensor:
        # Retain original needed-index mutation/version checks.
        ctx.save_for_backward(partition, nonpad)
        return _tile_permute_rows(x, partition, nonpad, padded_length)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        partition, nonpad = ctx.saved_tensors
        return _tile_unpermute_rows(grad, partition, nonpad), None, None, None


def _tile_permute_rows(x: torch.Tensor, partition: torch.Tensor, nonpad: torch.Tensor,
                       padded_length: int) -> torch.Tensor:
    """Shared tile-copy body: fresh [B, padded_length, H, D] with row partition[i] of x in slot nonpad[i], pads zero."""
    source = partition.new_zeros(padded_length)
    source.index_copy_(0, nonpad, partition)
    padding = torch.ones(padded_length, device=x.device, dtype=torch.bool)
    padding.index_fill_(0, nonpad, False)
    out = _gather_tile_rows(x, source)
    # Zero only the pad rows: the bijective partition fixes their count from shapes, so a fixed-size compaction
    # (no device-dependent output size, no host sync) replaces a masked fill over every element.
    pad_count = padded_length - partition.numel()
    if pad_count:
        out.index_fill_(1, torch.nonzero_static(padding, size=pad_count).flatten(), 0)
    return out


def _tile_unpermute_rows(grad: torch.Tensor, partition: torch.Tensor, nonpad: torch.Tensor) -> torch.Tensor:
    """Adjoint of _tile_permute_rows: gather the non-pad slots back into packed row order."""
    inverse = partition.new_empty(partition.numel())
    inverse.index_copy_(0, partition, nonpad)
    return _gather_tile_rows(grad, inverse)


# Same tile copy as an opaque custom-op pair (fake + autograd registrations) so torch.compile(fullgraph=True) and
# activation-checkpoint policies see one op key in eager and compiled mode (torch.ops.fastvideo_kernel.vsa_tile_permute_*).
# The builder-trust check (index tensors unmodified since the metadata recorded their versions) runs inside the op, where
# tensor versions are readable without a data-dependent guard; an untrusted call scatters through the authoritative
# untile index into a fresh buffer (the same result as the no-grad holder path, never the shared holder itself).
@torch.library.custom_op("fastvideo_kernel::vsa_tile_permute_fwd", mutates_args=())
def vsa_tile_permute_fwd(x: torch.Tensor, partition: torch.Tensor, nonpad: torch.Tensor, untile: torch.Tensor,
                         padded_length: int, partition_version: int, nonpad_version: int) -> torch.Tensor:
    if partition._version == partition_version and nonpad._version == nonpad_version:
        return _tile_permute_rows(x, partition, nonpad, padded_length)
    return scatter_into_tile_buf(x, (x.shape[0], padded_length, *x.shape[2:]), untile, None)


@vsa_tile_permute_fwd.register_fake
def _vsa_tile_permute_fwd_fake(x, partition, nonpad, untile, padded_length, partition_version, nonpad_version):
    return x.new_empty((x.shape[0], padded_length, *x.shape[2:]))


@torch.library.custom_op("fastvideo_kernel::vsa_tile_permute_bwd", mutates_args=())
def vsa_tile_permute_bwd(grad: torch.Tensor, partition: torch.Tensor, nonpad: torch.Tensor, untile: torch.Tensor,
                         partition_version: int, nonpad_version: int) -> torch.Tensor:
    # Adjoint of the index the forward consumed: repeat its trust decision (saved tensors cannot change in between).
    if partition._version == partition_version and nonpad._version == nonpad_version:
        return _tile_unpermute_rows(grad, partition, nonpad)
    return _gather_tile_rows(grad, untile)


@vsa_tile_permute_bwd.register_fake
def _vsa_tile_permute_bwd_fake(grad, partition, nonpad, untile, partition_version, nonpad_version):
    return grad.new_empty((grad.shape[0], untile.shape[0], *grad.shape[2:]))


def _tile_permute_setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[1], inputs[2], inputs[3])
    ctx.versions = (inputs[5], inputs[6])


def _tile_permute_backward(ctx, grad):
    partition, nonpad, untile = ctx.saved_tensors
    return vsa_tile_permute_bwd(grad.contiguous(), partition, nonpad, untile, *ctx.versions), None, None, None, None, None, None


vsa_tile_permute_fwd.register_autograd(_tile_permute_backward, setup_context=_tile_permute_setup_context)


# Three-operand form of the tile copy: q, k and v are gathered from their own tensors into three tiled outputs, so callers
# that hold separate q/k/v never materialize torch.cat([q, k, v]) (a full extra write + read of Q/K/V that SAC also recomputes)
# and the backward receives dq/dk/dv tiles separately (no cat of the three gradients either). Each output and gradient equals
# the matching chunk of vsa_tile_permute_fwd applied to the cat.
@torch.library.custom_op("fastvideo_kernel::vsa_tile_permute_qkv_fwd", mutates_args=())
def vsa_tile_permute_qkv_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, partition: torch.Tensor,
                             nonpad: torch.Tensor, untile: torch.Tensor, padded_length: int, partition_version: int,
                             nonpad_version: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outs = tuple(x.new_empty((x.shape[0], padded_length, *x.shape[2:])) for x in (q, k, v))
    if partition._version == partition_version and nonpad._version == nonpad_version:
        source = partition.new_zeros(padded_length)
        source.index_copy_(0, nonpad, partition)
        pad_count = padded_length - partition.numel()
        if pad_count:
            padding = torch.ones(padded_length, device=q.device, dtype=torch.bool)
            padding.index_fill_(0, nonpad, False)
            pad_rows = torch.nonzero_static(padding, size=pad_count).flatten()
        for x, dst in zip((q, k, v), outs, strict=True):
            torch.index_select(x, 1, source, out=dst)
            if pad_count:
                dst.index_fill_(1, pad_rows, 0)
        return outs
    for x, dst in zip((q, k, v), outs, strict=True):
        dst.zero_()
        dst[:, untile] = x
    return outs


@vsa_tile_permute_qkv_fwd.register_fake
def _vsa_tile_permute_qkv_fwd_fake(q, k, v, partition, nonpad, untile, padded_length, partition_version,
                                   nonpad_version):
    return tuple(x.new_empty((x.shape[0], padded_length, *x.shape[2:])) for x in (q, k, v))


@torch.library.custom_op("fastvideo_kernel::vsa_tile_permute_qkv_bwd", mutates_args=())
def vsa_tile_permute_qkv_bwd(gq: torch.Tensor, gk: torch.Tensor, gv: torch.Tensor, partition: torch.Tensor,
                             nonpad: torch.Tensor, untile: torch.Tensor, partition_version: int,
                             nonpad_version: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if partition._version == partition_version and nonpad._version == nonpad_version:
        index = partition.new_empty(partition.numel())
        index.index_copy_(0, partition, nonpad)
    else:
        index = untile
    return tuple(_gather_tile_rows(g, index) for g in (gq, gk, gv))


@vsa_tile_permute_qkv_bwd.register_fake
def _vsa_tile_permute_qkv_bwd_fake(gq, gk, gv, partition, nonpad, untile, partition_version, nonpad_version):
    return tuple(g.new_empty((g.shape[0], untile.shape[0], *g.shape[2:])) for g in (gq, gk, gv))


def _tile_permute_qkv_setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[3], inputs[4], inputs[5])
    ctx.versions = (inputs[7], inputs[8])


def _tile_permute_qkv_backward(ctx, gq, gk, gv):
    partition, nonpad, untile = ctx.saved_tensors
    dq, dk, dv = vsa_tile_permute_qkv_bwd(gq.contiguous(), gk.contiguous(), gv.contiguous(), partition, nonpad, untile,
                                          *ctx.versions)
    return dq, dk, dv, None, None, None, None, None, None


vsa_tile_permute_qkv_fwd.register_autograd(_tile_permute_qkv_backward, setup_context=_tile_permute_qkv_setup_context)


# The gated (compression-branch) training form of vsa_tile_permute_qkv_fwd: the same three tiled outputs plus the fp32
# per-tile row sums [3, B, H, n_tiles, D] of each, from one Triton pass that reads each packed row once. The sums are
# bitwise x.view(B, n, T, H, D).sum(2, dtype=torch.float32) of the tiled output (ATen's order for that reduction on
# CUDA: row t goes to partial t % 16, s_y = ((p[y] + p[y+4]) + p[y+8]) + p[y+12], total (s0 + s2) + (s1 + s3)), so the
# block means and the fine map equal _pool_tiles on the tiled operands. The backward adds each tile's sum gradient
# (rounded to the input dtype, as the pooling backward is) to the gathered row gradient in the same pass, which is the
# autograd sum of the two gradients the tiled operand used to receive.
@triton.jit
def _tile_gather_sum_kernel(x_ptr, out_ptr, sums_ptr, source_ptr, x_batch_stride, n_tiles, H: tl.constexpr,
                            D: tl.constexpr, HB: tl.constexpr, T: tl.constexpr):
    tile, hb, b = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    r = tl.arange(0, 16)
    c = tl.arange(0, HB * D)
    col = hb * HB * D + c
    acc = tl.zeros((16, HB * D), dtype=tl.float32)
    for t0 in range(0, T, 16):
        slot = (tile * T + t0 + r).to(tl.int64)
        src = tl.load(source_ptr + slot)  # packed row, -1 for a pad slot
        rows = tl.load(x_ptr + b * x_batch_stride + src[:, None] * (H * D) + col[None, :],
                       mask=src[:, None] >= 0,
                       other=0.0)
        tl.store(out_ptr + (b * n_tiles * T + slot)[:, None] * (H * D) + col[None, :], rows)
        acc += rows.to(tl.float32)
    # Exact extraction (every other addend is zero), then ATen's combine order.
    acc = tl.reshape(acc, (4, 4, HB * D))  # [t // 4 % 4, t % 4, :]
    a = tl.arange(0, 4)[:, None, None]
    s = tl.sum(tl.where(a == 0, acc, 0.0), 0)
    s = s + tl.sum(tl.where(a == 1, acc, 0.0), 0)
    s = s + tl.sum(tl.where(a == 2, acc, 0.0), 0)
    s = s + tl.sum(tl.where(a == 3, acc, 0.0), 0)
    y = tl.arange(0, 4)[:, None]
    total = (tl.sum(tl.where(y == 0, s, 0.0), 0) + tl.sum(tl.where(y == 2, s, 0.0), 0)) + (
        tl.sum(tl.where(y == 1, s, 0.0), 0) + tl.sum(tl.where(y == 3, s, 0.0), 0))
    head = hb * HB + c // D
    tl.store(sums_ptr + ((b * H + head) * n_tiles + tile) * D + c % D, total)


@triton.jit
def _tile_scatter_sum_grad_kernel(g_ptr, gsum_ptr, out_ptr, index_ptr, rows_total, n_tiles, H: tl.constexpr,
                                  D: tl.constexpr, HB: tl.constexpr, T: tl.constexpr, BR: tl.constexpr):
    blk, hb, b = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    rows = blk.to(tl.int64) * BR + tl.arange(0, BR)
    live = rows < rows_total
    c = tl.arange(0, HB * D)
    col = hb * HB * D + c
    slot = tl.load(index_ptr + rows, mask=live, other=0)
    g = tl.load(g_ptr + (b * n_tiles * T + slot)[:, None] * (H * D) + col[None, :], mask=live[:, None])
    head = hb * HB + c // D
    gs = tl.load(gsum_ptr + ((b * H + head[None, :]) * n_tiles + (slot // T)[:, None]) * D + (c % D)[None, :],
                 mask=live[:, None])
    grad = (g.to(tl.float32) + gs.to(g.dtype).to(tl.float32)).to(g.dtype)
    tl.store(out_ptr + (b * rows_total + rows)[:, None] * (H * D) + col[None, :], grad, mask=live[:, None])


def _heads_per_program(heads: int) -> int:
    return 2 if heads % 2 == 0 else 1


@torch.library.custom_op("fastvideo_kernel::vsa_tile_permute_qkv_sums_fwd", mutates_args=())
def vsa_tile_permute_qkv_sums_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, partition: torch.Tensor,
                                  nonpad: torch.Tensor, untile: torch.Tensor, padded_length: int, tile: int,
                                  partition_version: int,
                                  nonpad_version: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _, heads, dim = q.shape
    n_tiles = padded_length // tile
    outs = tuple(x.new_empty((batch, padded_length, heads, dim)) for x in (q, k, v))
    sums = q.new_empty((3, batch, heads, n_tiles, dim), dtype=torch.float32)
    if partition._version == partition_version and nonpad._version == nonpad_version:
        source = partition.new_full((padded_length, ), -1).index_copy_(0, nonpad, partition)
        hb = _heads_per_program(heads)
        for x, dst, s in zip((q, k, v), outs, sums, strict=True):
            _tile_gather_sum_kernel[(n_tiles, heads // hb, batch)](x, dst, s, source, x.stride(0), n_tiles, heads, dim,
                                                                   hb, tile)
        return (*outs, sums)
    for x, dst, s in zip((q, k, v), outs, sums, strict=True):
        dst.zero_()
        dst[:, untile] = x
        s.copy_(dst.view(batch, n_tiles, tile, heads, dim).sum(dim=2, dtype=torch.float32).permute(0, 2, 1, 3))
    return (*outs, sums)


@vsa_tile_permute_qkv_sums_fwd.register_fake
def _vsa_tile_permute_qkv_sums_fwd_fake(q, k, v, partition, nonpad, untile, padded_length, tile, partition_version,
                                        nonpad_version):
    outs = tuple(x.new_empty((x.shape[0], padded_length, *x.shape[2:])) for x in (q, k, v))
    return (*outs, q.new_empty((3, q.shape[0], q.shape[2], padded_length // tile, q.shape[3]), dtype=torch.float32))


@torch.library.custom_op("fastvideo_kernel::vsa_tile_permute_qkv_sums_bwd", mutates_args=())
def vsa_tile_permute_qkv_sums_bwd(gq: torch.Tensor, gk: torch.Tensor, gv: torch.Tensor, gsums: torch.Tensor,
                                  partition: torch.Tensor, nonpad: torch.Tensor, untile: torch.Tensor, tile: int,
                                  partition_version: int,
                                  nonpad_version: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if partition._version == partition_version and nonpad._version == nonpad_version:
        index = partition.new_empty(partition.numel())
        index.index_copy_(0, partition, nonpad)
    else:
        index = untile
    batch, padded_length, heads, dim = gq.shape
    rows = index.numel()
    hb, br = _heads_per_program(heads), 16
    outs = tuple(g.new_empty((batch, rows, heads, dim)) for g in (gq, gk, gv))
    for g, s, dst in zip((gq, gk, gv), gsums, outs, strict=True):
        _tile_scatter_sum_grad_kernel[(triton.cdiv(rows, br), heads // hb, batch)](g, s, dst, index, rows,
                                                                                   padded_length // tile, heads, dim,
                                                                                   hb, tile, br)
    return outs


@vsa_tile_permute_qkv_sums_bwd.register_fake
def _vsa_tile_permute_qkv_sums_bwd_fake(gq, gk, gv, gsums, partition, nonpad, untile, tile, partition_version,
                                        nonpad_version):
    return tuple(g.new_empty((g.shape[0], untile.shape[0], *g.shape[2:])) for g in (gq, gk, gv))


def _tile_permute_qkv_sums_setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[3], inputs[4], inputs[5])
    ctx.tile, ctx.versions = inputs[7], (inputs[8], inputs[9])


def _tile_permute_qkv_sums_backward(ctx, gq, gk, gv, gsums):
    partition, nonpad, untile = ctx.saved_tensors
    dq, dk, dv = vsa_tile_permute_qkv_sums_bwd(gq.contiguous(), gk.contiguous(), gv.contiguous(), gsums.contiguous(),
                                               partition, nonpad, untile, ctx.tile, *ctx.versions)
    return dq, dk, dv, None, None, None, None, None, None, None


vsa_tile_permute_qkv_sums_fwd.register_autograd(_tile_permute_qkv_sums_backward,
                                                setup_context=_tile_permute_qkv_sums_setup_context)


class VideoSparseAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(  # type: ignore
        self,
        current_timestep: int,
        raw_latent_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        VSA_sparsity: float,
        device: torch.device,
        cache_tile_buf: bool = True,
        **kwargs: dict[str, Any],
    ) -> VideoSparseAttentionMetadata:
        patch_size = patch_size
        dit_seq_shape = (raw_latent_shape[0] // patch_size[0], raw_latent_shape[1] // patch_size[1],
                         raw_latent_shape[2] // patch_size[2])

        num_tiles = (math.ceil(dit_seq_shape[0] / VSA_TILE_SIZE[0]), math.ceil(dit_seq_shape[1] / VSA_TILE_SIZE[1]),
                     math.ceil(dit_seq_shape[2] / VSA_TILE_SIZE[2]))
        total_seq_length = math.prod(dit_seq_shape)

        tile_partition_indices = get_tile_partition_indices(dit_seq_shape, VSA_TILE_SIZE, device)
        reverse_tile_partition_indices = get_reverse_tile_partition_indices(dit_seq_shape, VSA_TILE_SIZE, device)
        variable_block_sizes = construct_variable_block_sizes(dit_seq_shape, num_tiles, device)
        non_pad_index = get_non_pad_index(variable_block_sizes, math.prod(VSA_TILE_SIZE))
        untile_combined_index = non_pad_index[reverse_tile_partition_indices]

        metadata = VideoSparseAttentionMetadata(
            current_timestep=current_timestep,
            dit_seq_shape=dit_seq_shape,  # type: ignore
            VSA_sparsity=VSA_sparsity,  # type: ignore
            num_tiles=num_tiles,  # type: ignore
            total_seq_length=total_seq_length,  # type: ignore
            tile_partition_indices=tile_partition_indices,  # type: ignore
            reverse_tile_partition_indices=reverse_tile_partition_indices,
            variable_block_sizes=variable_block_sizes,
            non_pad_index=non_pad_index,
            untile_combined_index=untile_combined_index,
            cache_tile_buf=cache_tile_buf)
        # Cached index tensors may have been modified through older metadata.
        if (not tile_partition_indices.is_inference() and not non_pad_index.is_inference()
                and tile_partition_indices._version == 0 and non_pad_index._version == 0):
            metadata._tile_index_state = (tile_partition_indices, tile_partition_indices._version, non_pad_index,
                                          non_pad_index._version)
        return metadata


class VideoSparseAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.prefix = prefix
        sp_group = get_sp_group()
        self.sp_size = sp_group.world_size

    def tile(self, x: torch.Tensor, attn_metadata: VideoSparseAttentionMetadata) -> torch.Tensor:
        """Tile ``x`` into ``attn_metadata.tile_buf`` and return it.

        When caching is enabled, the returned tensor aliases the per-metadata
        buffer and is only valid until the next ``tile()`` / ``preprocess_qkv`` call on the
        same ``attn_metadata``.  Callers must consume (or copy) the
        result before invoking another VSA layer with the same metadata.
        Training normally disables caching because attention can retain
        QKV views until backward. The 128-token CuTe path retains copies
        when caching and gradient recording are both enabled.
        """
        num_tiles = attn_metadata.num_tiles
        t_padded_size = num_tiles[0] * VSA_TILE_SIZE[0]
        h_padded_size = num_tiles[1] * VSA_TILE_SIZE[1]
        w_padded_size = num_tiles[2] * VSA_TILE_SIZE[2]
        target_shape = (x.shape[0], t_padded_size * h_padded_size * w_padded_size, x.shape[-2], x.shape[-1])

        if not attn_metadata.cache_tile_buf:
            state = getattr(attn_metadata, "_tile_index_state", None)
            if (state is not None and torch.is_grad_enabled() and x.requires_grad and x.ndim == 4 and x.is_cuda
                    and x.dtype == torch.bfloat16 and (x.is_contiguous() or _tile_row_layout(x))
                    and state[0].device == x.device and state[2].device == x.device and x.numel() >= 2**25
                    and x.shape[1] == attn_metadata.total_seq_length == state[0].numel() == state[2].numel()
                    and state[0] is attn_metadata.tile_partition_indices and state[2] is attn_metadata.non_pad_index
                    and state[0]._version == state[1] and state[2]._version == state[3]):
                return _TilePermutation.apply(x, state[0], state[2], target_shape[1])
            return scatter_into_tile_buf(x, target_shape, attn_metadata.non_pad_index, None,
                                         attn_metadata.tile_partition_indices)

        # Buffer scoped to the per-step metadata (lazily allocated on the
        # first VSA layer's call within a denoising step), which keeps reuse
        # safe across concurrent requests.
        buf = scatter_into_tile_buf(x, target_shape, attn_metadata.non_pad_index, attn_metadata.tile_buf,
                                    attn_metadata.tile_partition_indices)
        attn_metadata.tile_buf = buf
        return buf

    def untile(self, x: torch.Tensor, untile_combined_index: torch.LongTensor) -> torch.Tensor:
        # Single fancy index using precomputed combined indices; avoids
        # the intermediate ``[B, len(non_pad_index), H, D]`` tensor that
        # the two-step ``x[:, non_pad_index][:, reverse_tile_partition_indices]``
        # would allocate on every layer.
        return x[:, untile_combined_index]

    def preprocess_qkv(
        self,
        qkv: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        """Tile QKV; aliasing contract: see ``tile()``."""
        return self.tile(qkv, attn_metadata)

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        return self.untile(output, attn_metadata.untile_combined_index)

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        block_elements = math.prod(VSA_TILE_SIZE)
        cur_topk = _compute_cur_topk(attn_metadata)

        # 256-element tiles auto-route to the FA4 CuTe BSHD fastpath, which
        # consumes [B, S, H, D] directly -- skip the transpose round-trip.
        if block_elements == 256 and video_sparse_attn_bshd is not None:
            return video_sparse_attn_bshd(query,
                                          key,
                                          value,
                                          attn_metadata.variable_block_sizes,
                                          attn_metadata.variable_block_sizes,
                                          cur_topk,
                                          block_size=VSA_TILE_SIZE,
                                          compress_attn_weight=gate_compress)

        if video_sparse_attn is None:
            raise NotImplementedError("video_sparse_attn is not installed")
        # The 128-token CuTe path can consume transpose views. Cached training
        # buffers, 64-token kernels and Triton retain their snapshots.
        use_views = False
        if block_elements == 128 and not (torch.is_grad_enabled() and attn_metadata.cache_tile_buf):
            try:
                from fastvideo_kernel.block_sparse_attn_256 import _resolve_backend
            except ImportError:
                pass
            else:
                use_views = _resolve_backend() == "cutedsl"
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        gate_compress = gate_compress.transpose(1, 2)
        if not use_views:
            query, key, value, gate_compress = (x.contiguous() for x in (query, key, value, gate_compress))
        return video_sparse_attn(query,
                                 key,
                                 value,
                                 attn_metadata.variable_block_sizes,
                                 attn_metadata.variable_block_sizes,
                                 cur_topk,
                                 block_size=VSA_TILE_SIZE,
                                 compress_attn_weight=gate_compress).transpose(1, 2)
