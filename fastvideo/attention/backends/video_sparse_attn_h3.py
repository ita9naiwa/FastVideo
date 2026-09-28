# SPDX-License-Identifier: Apache-2.0
"""VSA for MiniMax H3's packed mixed-modality self-attention.

H3 runs one joint bidirectional attention over
``[text | condition keyframes | audio | generated video]``, so this
backend differs from the Wan-tuned ``video_sparse_attn``:

- Tiles are ``[segment-pure prefix chunks] + [3D video tiles]``; prefix
  tiles never straddle segment boundaries. The tile size is selectable at
  metadata build time: 256 tokens ``(4,8,8)`` (default), 128 tokens
  ``(4,4,8)`` or 64 tokens ``(4,4,4)`` (see ``VSA_H3_TILE_SHAPES``).
- Selection is pure Python on pooled tile scores; the block-sparse kernel
  consumes an explicit bool mask, so no kernel changes are needed.
- The compression branch is gated by ``to_gate_compress``, which the base
  H3 checkpoint does not carry: the loader zero-initializes it, so
  untrained inference is exactly pure sparse and finetuning can learn the
  gate. VSA-distilled students (e.g. FastVideo-Minimax-H3-Preview) ship
  trained gates, which load and activate the branch.
- Non-video *queries* are always dense. Non-video *keys* are either
  always-selected for every query ("exempt", default) or compete in
  top-k under a FLOP-matched budget ("compete") — the ablation axis,
  switched per request via ``generate_video(..., vsa_mode=...)``
  (default: exempt). Per-request scheduling knobs
  (``vsa_dense_first_n_steps``, ``vsa_dense_layers``) let mixed schedules
  run the diffuse steps/layers dense while pushing the rest harder.

At tile 256 this targets sm10.x through the FA4 CuTe 256-tile path
(``FASTVIDEO_VSA_CUTEDSL=1``); the Triton 256→64 expansion is the
fallback and keeps identical mask semantics. At tile 128 grad-tracking
CuTe calls run the FA4 Q128/KV128 training op pair (``vsa_train_fwd/bwd``, tile=128);
everything else takes ``block_sparse_attn_128_bshd``. At tile 64 the block map is
already at the kernels' native 64-token granularity, so both forward and
backward run the Triton block-sparse kernels directly (no expansion,
``FASTVIDEO_VSA_CUTEDSL`` does not apply). A third, opt-in route exists
for the tile-64 FORWARD only: ``FASTVIDEO_VSA_SM100A=1`` sends no-grad
forwards through the data-center Blackwell CUDA block-sparse kernel
(``fastvideo_kernel.block_sparse_attn_sm100a``, upstream PR #1719 plus
our per-q-tile ``q2k_num`` fix) when the extension is built, the device
is sm_100 or sm_103, and the geometry qualifies. The CUDA kernel assigns
adjacent pairs of query tiles to CTAs, so an odd logical tile count receives one
internal, zero-valid partner tile for the no-grad call only. Score search,
the trained mask, gate-compress, and the returned packed sequence remain on
the original logical tiles. Grad-tracking forwards and every backward stay
on Triton unchanged. If the env is set but a precondition fails, the route
logs one warning and falls back.
"""

import functools
import math
import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import torch

try:
    from fastvideo_kernel.block_sparse_attn import block_sparse_attn as block_sparse_attn_64_bhsd
    from fastvideo_kernel import vsa256_ops
    from fastvideo_kernel.block_sparse_attn_256 import (_resolve_backend, block_sparse_attn_128_bshd,
                                                        block_sparse_attn_256_bshd)
    from fastvideo_kernel.triton_kernels.index import map_to_index
except ImportError:
    block_sparse_attn_64_bhsd = None
    block_sparse_attn_128_bshd = None
    block_sparse_attn_256_bshd = None
    map_to_index = None

try:
    # Optional: only present in fastvideo_kernel builds that carry the
    # sm_100a/sm_103a CUDA block-sparse forward (upstream PR #1719). The module itself imports
    # fine without the compiled symbols (`_HAS_VSA_SM100A` is then False and
    # `is_supported` says no), so this only guards *module* availability.
    from fastvideo_kernel import block_sparse_attn_sm100a as _sm100a
except ImportError:
    _sm100a = None

from fastvideo.attention.backends.abstract import (AttentionBackend, AttentionImpl, AttentionMetadata,
                                                   AttentionMetadataBuilder, layer_idx_from_prefix)
from fastvideo.attention.backends.video_sparse_attn import (_gather_tile_rows, _tile_row_layout, compute_topk,
                                                            construct_variable_block_sizes, get_non_pad_index,
                                                            get_tile_partition_indices, scatter_into_tile_buf)
from fastvideo.attention.backends.video_sparse_attn_h3_probe import probe_enabled, record_probe
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Opt-in switch for the data-center Blackwell CUDA forward on the tile-64 no-grad path.
VSA_SM100A_ENV = "FASTVIDEO_VSA_SM100A"

VSA_H3_TILE_SIZE = (4, 8, 8)  # 256 elements -> FA4 CuTe fastpath on sm10.x (default)
_TILE_ELEMS = math.prod(VSA_H3_TILE_SIZE)
# Selectable tile geometries, keyed by element count (= the build-time
# ``tile_size``). 64 runs the native 64-token Triton block-sparse kernels for
# forward AND backward — the block map is already at kernel granularity, so no
# 256->64 mask expansion is involved and FASTVIDEO_VSA_CUTEDSL does not apply.
VSA_H3_TILE_SHAPES: dict[int, tuple[int, int, int]] = {
    _TILE_ELEMS: VSA_H3_TILE_SIZE,
    # the 256 cube halved along h (conductor ruling 85a); a refinement of the 256 partition
    128: (4, 4, 8),
    64: (4, 4, 4),
}
# Consecutive-chunk layouts and the tile size each one is defined for (see _h3_tile_geometry). Builder layout names (so
# tile-parameterised callers need no string switch): "cube" | "cube128" | "cube256" and "chunk" | "chunk128" | "chunk256",
# the chunk names optionally suffixed "-merged-prefix" (= merge_prefix=True). A size in the name must match tile_size.
_H3_CHUNK_LAYOUTS = {"chunk256": 256, "chunk128": 128}


def token_tile_and_valid(variable_block_sizes: torch.Tensor,
                         tile_elems: int = _TILE_ELEMS) -> tuple[torch.Tensor, torch.Tensor]:
    """Per padded-token tile id and pad-validity mask.

    The single encoding of the padding contract, shared by the probe and the
    test oracle so they cannot drift from the backend's tile geometry.
    ``tile_elems`` must match the metadata the sizes came from
    (``MiniMaxH3VSAMetadata.tile_elems``).
    """
    device = variable_block_sizes.device
    token_tile = torch.arange(variable_block_sizes.numel(), device=device).repeat_interleave(tile_elems)
    token_valid = (torch.arange(tile_elems, device=device)[None, :] < variable_block_sizes[:, None]).reshape(-1)
    return token_tile, token_valid


def _validate_h3_tile_geometry(
    prefix_segments: tuple[int, ...],
    dit_seq_shape: tuple[int, int, int],
    variable_block_sizes: torch.Tensor,
    untile_combined_index: torch.Tensor,
    tile_elems: int = _TILE_ELEMS,
) -> None:
    """Fail synchronously on out-of-bounds tile geometry.

    Invariants the block-sparse kernel trusts without checking:
    every tile's valid size is in (0, tile_elems]; the sizes sum to the
    packed sequence length; and ``untile_combined_index`` maps each packed
    row to exactly one non-pad slot of the padded tile buffer. A violation
    would surface only as an async device fault at some later kernel or
    collective (e.g. an FSDP all-gather), which is unattributable — so raise
    here, once per cached geometry, with the numbers in hand.
    """
    total = sum(prefix_segments) + math.prod(dit_seq_shape)
    n_pad = variable_block_sizes.numel() * tile_elems
    sizes_min = int(variable_block_sizes.min())
    sizes_max = int(variable_block_sizes.max())
    sizes_sum = int(variable_block_sizes.sum())
    if sizes_min < 1 or sizes_max > tile_elems or sizes_sum != total:
        raise ValueError(f"VSA-H3 tile sizes out of bounds for prefix={prefix_segments}, video={dit_seq_shape}, "
                         f"tile_elems={tile_elems}: min={sizes_min}, max={sizes_max}, sum={sizes_sum}, "
                         f"expected sum={total}.")
    if untile_combined_index.numel() != total:
        raise ValueError(f"VSA-H3 untile index has {untile_combined_index.numel()} entries for a packed "
                         f"sequence of {total} rows (prefix={prefix_segments}, video={dit_seq_shape}).")
    idx_min = int(untile_combined_index.min())
    idx_max = int(untile_combined_index.max())
    if idx_min < 0 or idx_max >= n_pad:
        # Range first: the pad-slot gather below would itself index out of
        # bounds (the very async fault this guard exists to preempt).
        raise ValueError(f"VSA-H3 untile index is not an injective map into non-pad slots: range "
                         f"[{idx_min}, {idx_max}] vs padded length {n_pad} "
                         f"(prefix={prefix_segments}, video={dit_seq_shape}).")
    in_tile_offset = untile_combined_index % tile_elems
    maps_into_pad = bool((in_tile_offset >= variable_block_sizes[untile_combined_index // tile_elems]).any())
    if maps_into_pad or int(torch.unique(untile_combined_index).numel()) != total:
        raise ValueError(f"VSA-H3 untile index is not an injective map into non-pad slots: "
                         f"pad-slot hit={maps_into_pad} "
                         f"(prefix={prefix_segments}, video={dit_seq_shape}).")


@functools.lru_cache(maxsize=10)
def _h3_tile_geometry(
    prefix_segments: tuple[int, ...],
    dit_seq_shape: tuple[int, int, int],
    device: torch.device,
    tile_shape: tuple[int, int, int] = VSA_H3_TILE_SIZE,
    tile_layout: str = "cube",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Tile the packed sequence: prefix chunks, then video tiles.

    ``tile_layout="cube"`` (default): segment-pure prefix chunks and one tile
    per ``tile_shape`` cube of the video grid. ``"chunk256"``: the same
    cube-ordered video tokens cut into consecutive full tiles (only the last
    video tile is partial). ``"chunk256-merged-prefix"``: additionally chunk
    the concatenated prefix as one sequence instead of per segment.
    ``"chunk128"`` / ``"chunk128-merged-prefix"``: the same at 128 tokens.

    Returns (tile_partition_indices, variable_block_sizes,
    untile_combined_index, num_prefix_tiles, num_video_tiles).
    """
    tile_elems = math.prod(tile_shape)
    chunk_layout = tile_layout.removesuffix("-merged-prefix")
    if tile_layout != "cube" and _H3_CHUNK_LAYOUTS.get(chunk_layout) != tile_elems:
        raise ValueError(f"unknown VSA-H3 tile_layout {tile_layout!r} for {tile_elems}-token tiles")
    prefix_len = sum(prefix_segments)

    def chunks(n: int) -> list[int]:
        full, rem = divmod(n, tile_elems)
        return [tile_elems] * full + ([rem] if rem else [])

    if tile_layout.endswith("-merged-prefix"):
        prefix_sizes = chunks(prefix_len)
    else:
        prefix_sizes = [size for segment in prefix_segments for size in chunks(segment)]
    num_prefix_tiles = len(prefix_sizes)

    if tile_layout != "cube":
        video_sizes = torch.tensor(chunks(math.prod(dit_seq_shape)), dtype=torch.long, device=device)
    else:
        ts_t, ts_h, ts_w = tile_shape
        t, h, w = dit_seq_shape
        num_tiles = (math.ceil(t / ts_t), math.ceil(h / ts_h), math.ceil(w / ts_w))
        video_sizes = construct_variable_block_sizes(dit_seq_shape, num_tiles, device, tile_shape)
    num_video_tiles = int(video_sizes.numel())

    video_indices = get_tile_partition_indices(dit_seq_shape, tile_shape, device) + prefix_len
    tile_partition_indices = torch.cat([
        torch.arange(prefix_len, device=device, dtype=torch.long),
        video_indices,
    ])
    # cat promotes the int32 helper output to int64 alongside the prefix sizes
    variable_block_sizes = torch.cat([
        torch.tensor(prefix_sizes, dtype=torch.long, device=device),
        video_sizes,
    ])

    # get_non_pad_index is lru-cached on tensor identity; variable_block_sizes
    # is itself cached by this function, so the identity stays stable.
    non_pad_index = get_non_pad_index(variable_block_sizes, tile_elems)

    untile_combined_index = non_pad_index[torch.argsort(tile_partition_indices)]
    # One-time (lru-cached) synchronous bounds check; see _validate_h3_tile_geometry.
    _validate_h3_tile_geometry(prefix_segments, dit_seq_shape, variable_block_sizes, untile_combined_index, tile_elems)
    return (tile_partition_indices, variable_block_sizes, untile_combined_index, num_prefix_tiles, num_video_tiles)


class MiniMaxH3VSABackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "VIDEO_SPARSE_ATTN_H3"

    @staticmethod
    def get_impl_cls() -> type["MiniMaxH3VSAImpl"]:
        return MiniMaxH3VSAImpl

    @staticmethod
    def get_metadata_cls() -> type["MiniMaxH3VSAMetadata"]:
        return MiniMaxH3VSAMetadata

    @staticmethod
    def get_builder_cls() -> type["MiniMaxH3VSAMetadataBuilder"]:
        return MiniMaxH3VSAMetadataBuilder


@functools.lru_cache(maxsize=32)
def _pack_tails_policy(variable_block_sizes: torch.Tensor, tile_elems: int, min_partial: float) -> bool:
    """Training tail packing pays a second backward launch plus plan build; it only wins when many parents are partial
    and the 128-residual rows fit TailTraining's bounded plan. Cached on the (geometry-cached) sizes tensor identity,
    so the one host sync happens once per geometry, outside any compiled region."""
    if tile_elems != 256:
        return False
    sizes = variable_block_sizes.clamp(0, 256).tolist()
    residual_rows = sum(size % 128 for size in sizes)
    capacity = math.ceil(max(128, len(sizes) * 8) / 128) * 128  # vsa_tail_backward._prepare
    partial = sum(size < 256 for size in sizes) / len(sizes)
    return residual_rows <= capacity and partial >= min_partial


@functools.cache
def _alias_guard_hint(value: bool) -> torch.Tensor:
    """One shared 0-d CPU bool hint tensor per value. Marked a static, unguarded input: compiled graphs take it as an
    input (no guard, so no recompile between True and False), and CUDA-graph trees use it in place instead of copying
    a CPU tensor into the CUDA pool (a non-static CPU input fails their pool check)."""
    hint = torch.tensor(bool(value), device="cpu")
    torch._dynamo.mark_static_address(hint, guard=False)
    return hint


class _MiniMaxH3VSATileBufferHolder:
    """Builder-owned no-grad tile scratch and its active geometry."""

    def __init__(self) -> None:
        self.buffer: torch.Tensor | None = None
        self.untile_geometry: torch.Tensor | None = None


@dataclass
class MiniMaxH3VSAMetadata(AttentionMetadata):
    total_seq_length: int
    num_prefix_tiles: int
    num_video_tiles: int
    exempt: bool
    variable_block_sizes: torch.Tensor
    untile_combined_index: torch.Tensor
    # Device-side copy of ``dense_layers``. Regional fullgraph capture uses
    # this tensor with each implementation's tensor-valued layer index so the
    # shared block code does not specialize once per Python ``layer_idx``.
    dense_layers_tensor: torch.Tensor
    # tokens per tile (256, 128 or 64); selects the tile geometry AND the kernel
    # route in forward() (256 -> VSA-256 CuTe/Triton, 128 -> VSA-128 CuTe/Triton, 64 -> native Triton)
    tile_elems: int = _TILE_ELEMS
    # layers forced dense regardless of sparsity (probe-guided opt-outs)
    dense_layers: tuple[int, ...] = ()
    # Builder-owned padded tile buffer. It records the geometry that last
    # populated the allocation so a same-shaped geometry change can clear
    # stale pad rows once while steady-state denoising reuses the buffer.
    tile_buf_holder: _MiniMaxH3VSATileBufferHolder | None = None
    # "cube" (one tile per tile-shape video cube) or "chunk256"/"chunk128" (consecutive
    # tile-size chunks of the same cube-ordered tokens); see _h3_tile_geometry
    tile_layout: str = "cube"
    # Row permutation of the same geometry (partition order + padded non-pad slots). Training tiles
    # through the vsa_tile_permute op pair (gather + inverse-gather backward, invocation-owned output)
    # instead of the index_put scatter into the builder buffer; see MiniMaxH3VSAImpl.tile.
    tile_partition_indices: torch.Tensor | None = None
    non_pad_index: torch.Tensor | None = None
    # compute_topk(VSA_sparsity, num_video_tiles) as a host int, so compiled forwards do not trace float math on the sparsity
    video_topk: int | None = None
    # Optional cap on video top-k (ruling 71: k_vid = min(ceil((1 - s) * n_video), cap)); folded into video_topk.
    video_topk_cap: int | None = None
    # Runtime switch (static per build, default on; same-head ablations): trusted cube geometry lets the packed backward
    # skip wholly padded Q128 children. Further integrated options (pack_tails policy, fused_qkv_grad) are
    # added next to it when their inputs are accepted.
    query_pad_pruning: bool = True
    # forward_qkv keeps [3B, S, H, D] as one autograd input (one fused gradient allocation); OFF = the ordinary
    # three-input route with three gradient buffers and autograd's chunk-backward concat (the pre-component path).
    fused_qkv_grad: bool = True
    # Training backward tail packing for tile-256 (static per step; see _pack_tails_policy).
    pack_tails: bool = True
    # FA4 persistent-grid alias-guard hint: 0-d CPU bool tensor, True iff the document has dense prefix tiles (its
    # prefix query rows are what pile onto a few CTAs on an aliased grid). A tensor so compiled graphs take it as input.
    alias_guard_hint: torch.Tensor | None = None


class MiniMaxH3VSAMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self) -> None:
        self._tile_buf_holder = _MiniMaxH3VSATileBufferHolder()

    def prepare(self) -> None:
        pass

    @staticmethod
    def _pack_tails(requested: bool | str, sizes: torch.Tensor, tile_elems: int, min_partial: float) -> bool:
        """"auto": an explicit FASTVIDEO_VSA_PACK_TAILS forces it, otherwise the partial-parent policy decides."""
        if requested != "auto":
            return bool(requested)
        env = os.environ.get("FASTVIDEO_VSA_PACK_TAILS")
        return env == "1" if env is not None else _pack_tails_policy(sizes, tile_elems, min_partial)

    @staticmethod
    def _layout(name: str, merge_prefix: bool | None, tile_size: int, exempt: bool) -> tuple[str, bool]:
        """Canonical (tile_layout, merge_prefix) for a builder layout name: "cube" or "chunk<tile_size>"."""
        base = name.removesuffix("-merged-prefix")
        if base in ("cube", "cube128", "cube256") and base == name:
            kind, size = "cube", base[4:]
        elif base in ("chunk", *_H3_CHUNK_LAYOUTS):
            kind, size = "chunk", base[5:]
        else:
            raise ValueError(
                "VSA-H3 tile_layout must be 'cube', 'cube128', 'cube256', 'chunk', 'chunk128' or 'chunk256' "
                f"(chunk names optionally with '-merged-prefix'), got {name!r}")
        if size and int(size) != tile_size:
            raise ValueError(f"VSA-H3 tile_layout={name!r} requires tile_size={size}")
        if kind == "chunk" and f"chunk{tile_size}" not in _H3_CHUNK_LAYOUTS:
            raise ValueError(f"VSA-H3 tile_layout={name!r}: 'chunk' supports tile_size 128 or 256, got {tile_size!r}")
        if base != name:
            if merge_prefix is False:
                raise ValueError(f"VSA-H3 tile_layout={name!r} implies merge_prefix=True, got merge_prefix=False")
            merge_prefix = True
        # Merged prefix tiles mix modalities; only exempt mode (prefix always
        # selected) keeps them selection-neutral.
        if merge_prefix and (kind == "cube" or not exempt):
            raise ValueError("VSA-H3 merge_prefix requires a chunk tile_layout and exempt=True")
        return ("cube" if kind == "cube" else f"chunk{tile_size}"), bool(merge_prefix)

    def build(  # type: ignore
        self,
        current_timestep: int,
        raw_latent_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        VSA_sparsity: float,
        prefix_segments: tuple[int, ...],
        device: torch.device,
        exempt: bool = True,
        dense_layers: tuple[int, ...] = (),
        tile_size: int = _TILE_ELEMS,
        tile_layout: str = "cube",
        merge_prefix: bool | None = None,
        topk_cap: int | None = None,
        query_pad_pruning: bool = True,
        fused_qkv_grad: bool = True,
        pack_tails: bool | str = "auto",
        pack_tails_min_partial: float = 0.15,
        **kwargs: dict[str, Any],
    ) -> MiniMaxH3VSAMetadata:
        tile_shape = VSA_H3_TILE_SHAPES.get(int(tile_size))
        if tile_shape is None:
            raise ValueError(f"VSA-H3 tile_size must be one of {sorted(VSA_H3_TILE_SHAPES)}, got {tile_size!r}")
        tile_layout, merge_prefix = self._layout(tile_layout, merge_prefix, int(tile_size), exempt)
        dit_seq_shape = (raw_latent_shape[0] // patch_size[0], raw_latent_shape[1] // patch_size[1],
                         raw_latent_shape[2] // patch_size[2])
        prefix_segments = tuple(int(s) for s in prefix_segments if s > 0)
        total_seq_length = sum(prefix_segments) + math.prod(dit_seq_shape)

        (tile_partition_indices, variable_block_sizes, untile_combined_index, num_prefix_tiles,
         num_video_tiles) = _h3_tile_geometry(prefix_segments, dit_seq_shape, device, tile_shape,
                                              f"{tile_layout}-merged-prefix" if merge_prefix else tile_layout)

        dense_layers = tuple(int(layer) for layer in dense_layers)
        metadata = MiniMaxH3VSAMetadata(
            current_timestep=current_timestep,
            VSA_sparsity=VSA_sparsity,
            total_seq_length=total_seq_length,
            num_prefix_tiles=num_prefix_tiles,
            num_video_tiles=num_video_tiles,
            exempt=exempt,
            variable_block_sizes=variable_block_sizes,
            untile_combined_index=untile_combined_index,
            tile_elems=int(tile_size),
            dense_layers=dense_layers,
            dense_layers_tensor=torch.tensor(dense_layers, device=device, dtype=torch.int64),
            tile_buf_holder=self._tile_buf_holder,
            tile_layout=tile_layout,
            tile_partition_indices=tile_partition_indices,
            non_pad_index=get_non_pad_index(variable_block_sizes, int(tile_size)),  # cached on sizes identity
            video_topk=_video_topk(VSA_sparsity, num_video_tiles, topk_cap),
            video_topk_cap=topk_cap,
            query_pad_pruning=bool(query_pad_pruning),
            fused_qkv_grad=bool(fused_qkv_grad),
            pack_tails=self._pack_tails(pack_tails, variable_block_sizes, int(tile_size), pack_tails_min_partial),
            alias_guard_hint=_alias_guard_hint(num_prefix_tiles > 0),
        )
        # Trust the cached index tensors only while unmodified (same contract as the generic VSA backend).
        if (not metadata.tile_partition_indices.is_inference() and not metadata.non_pad_index.is_inference()
                and metadata.tile_partition_indices._version == 0 and metadata.non_pad_index._version == 0):
            metadata._tile_index_state = (metadata.tile_partition_indices, 0, metadata.non_pad_index, 0)
            # untile is what postprocess_output indexes with, so its exact tensor is certified too; the inverse plan
            # (source slot per padded row, pad rows) is built once here, not per layer.
            if not untile_combined_index.is_inference() and untile_combined_index._version == 0:
                partition, nonpad = metadata.tile_partition_indices, metadata.non_pad_index
                padded = variable_block_sizes.numel() * int(tile_size)
                source = partition.new_zeros(padded).index_copy_(0, nonpad, partition)
                padding = torch.ones(padded, device=partition.device, dtype=torch.bool).index_fill_(0, nonpad, False)
                pad_rows = torch.nonzero_static(padding, size=padded - partition.numel()).flatten()
                metadata._untile_grad_state = (untile_combined_index, 0, source, source._version, pad_rows,
                                               pad_rows._version)
        # _h3_tile_geometry validated that this untile index reads only non-pad rows of these sizes,
        # so postprocess_output leaves padded query rows with zero output gradient. Trust the pair
        # only while both tensors are the unmodified cached objects.
        untile, sizes = metadata.untile_combined_index, metadata.variable_block_sizes
        if (tile_layout == "cube" and not untile.is_inference() and not sizes.is_inference() and untile._version == 0
                and sizes._version == 0):
            metadata._query_pad_state = (untile, 0, sizes, 0)
        return metadata


def _pool_tiles(x: torch.Tensor, variable_block_sizes: torch.Tensor, tile_elems: int = _TILE_ELEMS) -> torch.Tensor:
    """fp32 mean over each tile_elems-token tile. x: [B, S_pad, H, D] -> [B, H, n_tiles, D].

    Pad positions in the tile buffer are guaranteed zero (zeros-init, never
    written), so a plain sum with fp32 accumulation needs no validity mask
    and no materialized fp32 temp; dividing by the true tile size makes it
    the masked mean exactly.
    """
    batch, seq_len, heads, dim = x.shape
    n_tiles = seq_len // tile_elems
    pooled = x.view(batch, n_tiles, tile_elems, heads, dim).sum(dim=2, dtype=torch.float32)
    pooled = pooled / variable_block_sizes.view(1, -1, 1, 1)
    return pooled.permute(0, 2, 1, 3)


def _video_topk(sparsity: float, num_video_tiles: int, cap: int | None) -> int:
    """The single host-int video top-k: ceil((1 - sparsity) * n) in exact integer arithmetic on the decimal sparsity
    (0.85 -> (3n + 19) // 20; float ceil overshoots at n = 20m), clamped to [1, n], optionally capped (ruling 71:
    s085k32 = min((3n + 19) // 20, 32)). Equals compute_topk for binary-exact sparsities such as 0.75 and 0.5."""
    keep = 1 - Fraction(repr(float(sparsity)))
    k = max(1, min(-(-keep.numerator * num_video_tiles // keep.denominator), num_video_tiles))
    return k if cap is None else max(1, min(k, int(cap)))


def _build_block_mask(
    scores: torch.Tensor,
    num_prefix_tiles: int,
    num_video_tiles: int,
    VSA_sparsity: float,
    exempt: bool,
    k_vid: int | None = None,
) -> torch.Tensor:
    """scores: [B, H, n_tiles, n_tiles] -> bool mask, same shape. k_vid: precomputed compute_topk(VSA_sparsity, num_video_tiles)."""
    n_tiles = scores.shape[-1]
    if k_vid is None:
        k_vid = compute_topk(VSA_sparsity, num_video_tiles)
    if k_vid == num_video_tiles:
        return torch.ones_like(scores, dtype=torch.bool)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if exempt or num_prefix_tiles == 0:
        video_cols = scores[..., num_prefix_tiles:]
        idx = video_cols.topk(k_vid, dim=-1).indices + num_prefix_tiles
        mask.scatter_(-1, idx, True)
        mask[..., :num_prefix_tiles] = True
    else:
        k_total = min(k_vid + num_prefix_tiles, n_tiles)
        idx = scores.topk(k_total, dim=-1).indices
        mask.scatter_(-1, idx, True)
    mask[:, :, :num_prefix_tiles, :] = True
    return mask


# Block selection as an opaque op (worker-1, h3-training-stack-integration 70cae9be; same implementation in the seam and the
# integration): pooled block scores and the top-k run as ordinary eager kernels inside the body in eager AND compiled mode,
# so Inductor never reassociates the fp32 score reduction and a top-k near-tie (observed: 2-ulp gap, cube f1c85686
# pack4/doc0, also on the fullgraph seam alone) cannot select a different map than eager. Selection only (bool output,
# detached inputs); the gate branch keeps its own differentiable scores.
@torch.library.custom_op("fastvideo_kernel::vsa_h3_block_map", mutates_args=())
def vsa_h3_block_map(query: torch.Tensor, key: torch.Tensor, variable_block_sizes: torch.Tensor, tile_elems: int,
                     num_prefix_tiles: int, num_video_tiles: int, k_vid: int, exempt: bool) -> torch.Tensor:
    scores = torch.matmul(_pool_tiles(query, variable_block_sizes, tile_elems),
                          _pool_tiles(key, variable_block_sizes, tile_elems).transpose(-2, -1)) / (query.shape[-1]**0.5)
    return _build_block_mask(scores, num_prefix_tiles, num_video_tiles, 1.0, exempt, k_vid)


@vsa_h3_block_map.register_fake
def _vsa_h3_block_map_fake(query, key, variable_block_sizes, tile_elems, num_prefix_tiles, num_video_tiles, k_vid,
                           exempt):
    n_tiles = query.shape[1] // tile_elems
    return query.new_empty((query.shape[0], query.shape[2], n_tiles, n_tiles), dtype=torch.bool)


# Sibling of vsa_h3_block_map for the fused VC route, whose producer already returns the FP32 tile pools. Top-k from the
# host metadata INSIDE the body (video tiles = all tiles after the prefix), so no per-document k reaches the graph.
@torch.library.custom_op("fastvideo_kernel::vsa_h3_block_map_from_pools", mutates_args=())
def vsa_h3_block_map_from_pools(pool_q: torch.Tensor, pool_k: torch.Tensor, num_prefix_tiles: int, sparsity: float,
                                topk_cap: int | None, exempt: bool) -> torch.Tensor:
    num_video_tiles = pool_q.shape[2] - num_prefix_tiles
    scores = torch.matmul(pool_q, pool_k.transpose(-2, -1)) / (pool_q.shape[-1]**0.5)
    return _build_block_mask(scores, num_prefix_tiles, num_video_tiles, sparsity, exempt,
                             _video_topk(sparsity, num_video_tiles, topk_cap))


@vsa_h3_block_map_from_pools.register_fake
def _vsa_h3_block_map_from_pools_fake(pool_q, pool_k, num_prefix_tiles, sparsity, topk_cap, exempt):
    b, h, n, _ = pool_q.shape
    return pool_q.new_empty((b, h, n, n), dtype=torch.bool)


def _versions_match(state: tuple) -> bool:
    """(tensor, recorded _version, tensor, recorded _version, ...) all unmodified."""
    return all(t._version == v for t, v in zip(state[::2], state[1::2], strict=True))


# Training untile as an opaque op pair (same op keys eager and compiled). Forward is today's ``output[:, untile]``. The
# backward gathers the live-row gradient through the builder's inverse plan and zeroes only the pad rows, instead of
# IndexBackward0 (zero fill + sort + index_put accumulate). It re-checks the recorded versions of every index it relies on
# (Dynamo cannot branch on ``_version``) and otherwise computes exactly IndexBackward0's adjoint of the consumed map.
@torch.library.custom_op("fastvideo_kernel::vsa_h3_untile_fwd", mutates_args=())
def vsa_h3_untile_fwd(output: torch.Tensor, untile: torch.Tensor, source: torch.Tensor, pad_rows: torch.Tensor,
                      partition: torch.Tensor, nonpad: torch.Tensor, versions: list[int]) -> torch.Tensor:
    return output[:, untile]


@vsa_h3_untile_fwd.register_fake
def _vsa_h3_untile_fwd_fake(output, untile, source, pad_rows, partition, nonpad, versions):
    return output.new_empty((output.shape[0], untile.shape[0], *output.shape[2:]))


@torch.library.custom_op("fastvideo_kernel::vsa_h3_untile_bwd", mutates_args=())
def vsa_h3_untile_bwd(grad: torch.Tensor, untile: torch.Tensor, source: torch.Tensor, pad_rows: torch.Tensor,
                      partition: torch.Tensor, nonpad: torch.Tensor, versions: list[int]) -> torch.Tensor:
    if _versions_match(
        (untile, versions[0], source, versions[1], pad_rows, versions[2], partition, versions[3], nonpad, versions[4])):
        return _gather_tile_rows(grad, source).index_fill_(1, pad_rows, 0)
    zeros = grad.new_zeros((grad.shape[0], source.shape[0], *grad.shape[2:]))
    return torch.ops.aten.index_put_.default(zeros, [None, untile], grad, True)  # IndexBackward0's computation


@vsa_h3_untile_bwd.register_fake
def _vsa_h3_untile_bwd_fake(grad, untile, source, pad_rows, partition, nonpad, versions):
    return grad.new_empty((grad.shape[0], source.shape[0], *grad.shape[2:]))


def _untile_setup_context(ctx, inputs, output):
    # Saved for backward and for autograd's saved-tensor mutation check.
    ctx.save_for_backward(*inputs[1:6])
    ctx.versions = inputs[6]


def _untile_backward(ctx, grad):
    return (vsa_h3_untile_bwd(grad.contiguous(), *ctx.saved_tensors, ctx.versions), None, None, None, None, None, None)


vsa_h3_untile_fwd.register_autograd(_untile_backward, setup_context=_untile_setup_context)


def vsa_h3_untile(output: torch.Tensor, attn_metadata: MiniMaxH3VSAMetadata) -> torch.Tensor:
    """``output[:, untile_combined_index]`` for [B, S_pad, ...] attention output (same result as
    ``output.index_select(1, attn_metadata.untile_combined_index)``). Public so wrappers outside the backend (h3mh's
    per-doc loop) take the same gather backward as ``postprocess_output``."""
    # A forward that let the backward skip padded query rows pins the untile map it trusted; untile with exactly
    # that map (the effective map) so padded rows keep zero dO.
    pinned = getattr(output, "_vsa_h3_query_pad_untile", None)
    if pinned is None:
        untile = attn_metadata.untile_combined_index
    else:
        untile, version = pinned
        # Eager: reject an in-place change since the builder validated the map. Compiled: Dynamo cannot branch on
        # _version; a change after the forward fails autograd's saved-tensor check on the map the vsa256 op saved,
        # and its backward re-checks the recorded version (full backward on mismatch).
        if not torch.compiler.is_compiling() and untile._version != version:
            raise RuntimeError("VSA-H3 untile_combined_index was modified in place between forward and "
                               "postprocess_output; its backward assumed the original padded-row geometry.")
    # Training with builder-certified partition/nonpad (tile()'s state) AND a certificate for exactly the effective
    # map: opaque op pair with the gather backward (pad rows get exact zero dO, as the pin requires). Eager also
    # checks the recorded versions here; compiled graphs pass them as explicit values and the backward op re-checks
    # them. Anything else indexes the effective map (IndexBackward0).
    tile_state = getattr(attn_metadata, "_tile_index_state", None)
    state = getattr(attn_metadata, "_untile_grad_state", None)
    # Compiled no-grad calls take the same op for its forward alone: its aten gather beats Inductor's generated one.
    trains = torch.is_grad_enabled() and output.requires_grad
    if (state is not None and tile_state is not None
            and (trains or (torch.compiler.is_compiling() and not torch.is_grad_enabled())) and state[0] is untile
            and tile_state[0] is attn_metadata.tile_partition_indices and tile_state[2] is attn_metadata.non_pad_index
            and state[2].device == output.device and output.shape[1] == state[2].numel()
            and (torch.compiler.is_compiling() or (_versions_match(tile_state) and _versions_match(state)))):
        return torch.ops.fastvideo_kernel.vsa_h3_untile_fwd(
            output, untile, state[2], state[4], tile_state[0], tile_state[2],
            [state[1], state[3], state[5], tile_state[1], tile_state[3]])
    return output[:, untile]


def _add_compress(out: torch.Tensor, scores: torch.Tensor, v_pooled: torch.Tensor, gate: torch.Tensor, n_tiles: int,
                  tile: int) -> torch.Tensor:
    """Wan-style compression branch: dense attention over pooled tiles, broadcast to each tile's rows, scaled by the tiled
    gate. Out-of-place: on the CuTe backend ``out`` is the tensor FA4's autograd node saved for its backward, so an
    in-place add would bump its version counter and backward dies with "one of the variables needed for gradient
    computation has been modified"."""
    out_c = torch.matmul(torch.softmax(scores, dim=-1), v_pooled)  # [B, H, n_tiles, D]
    out_c = out_c.permute(0, 2, 1, 3).to(out.dtype)  # [B, n_tiles, H, D]
    batch, seq_len, heads, dim = out.shape
    return (out.view(batch, n_tiles, tile, heads, dim) +
            out_c.unsqueeze(2) * gate.view(batch, n_tiles, tile, heads, dim)).view(batch, seq_len, heads, dim)


def _cute_backend() -> str:
    """fastvideo_kernel's tile-128/256 backend; imported at call time (fastvideo_kernel is optional)."""
    from fastvideo_kernel.block_sparse_attn_256 import _resolve_backend as resolve_backend
    return resolve_backend()


def _trusted_tile_state(attn_metadata: MiniMaxH3VSAMetadata, rows: int) -> tuple | None:
    """The builder-certified ``_tile_index_state`` if it still names this metadata's index tensors and covers ``rows``."""
    state = getattr(attn_metadata, "_tile_index_state", None)
    if (state is not None and state[0] is attn_metadata.tile_partition_indices
            and state[2] is attn_metadata.non_pad_index and rows == state[0].numel()):
        return state
    return None


def _sm100a_unavailable_reason(sm100a_mod: Any, query_bhsd: torch.Tensor, variable_block_sizes: torch.Tensor,
                               grad_mode: bool) -> str | None:
    """Why the opt-in data-center Blackwell route cannot run here, or None if it can.

    Pure decision logic, split out so the routing is unit-testable without a
    GPU or the compiled extension (tests substitute ``sm100a_mod``). Order
    matters only for the message: the cheapest, most actionable reason first.
    """
    if sm100a_mod is None:
        return "fastvideo_kernel.block_sparse_attn_sm100a is not installed"
    if grad_mode:
        return "inputs require grad and the sm_100a/sm_103a kernel is forward-only; grad paths keep Triton"
    if not sm100a_mod.is_supported(query_bhsd, variable_block_sizes):
        return ("block_sparse_attn_sm100a.is_supported returned False (needs an sm_100 or sm_103 device, a built "
                "extension, bf16, head_dim 128, an even tile count, and integer tile sizes)")
    return None


class MiniMaxH3VSAImpl(AttentionImpl):

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
        self.layer_idx = layer_idx_from_prefix(prefix, default=-1)
        self.head_size = head_size
        # Generic torch.compile must not specialize the shared VSA forward on
        # the Python ``layer_idx`` value of each of H3's 50 blocks. This
        # tensor is prepared after weights load and drives only the compiled
        # dense-layer decision; it does not opt the module into sm_100a.
        self._compile_layer_idx: torch.Tensor | None = None
        # None means the regional-compile preparation hook has not run.  The
        # eager path deliberately ignores this cache and preserves its
        # request-time env/probe/fallback behavior; only Dynamo capture reads
        # the prepared, static route.
        self._regional_compile_sm100a_enabled: bool | None = None
        # Compiled tile-256 inference route resolved by prepare_for_regional_compile: "bf16" (the opaque CuTe no-grad
        # op, vsa256_ops.vsa256_nograd_fwd), "vc" (the opaque fused VC ops, FASTVIDEO_VSA_VC=1) or None (unavailable;
        # such runs stay eager).
        self._regional_compile_nograd_route: str | None = None

    def prepare_for_compile(self, device: torch.device) -> None:
        """Tensorize per-layer state shared by every torch.compile route."""
        self._compile_layer_idx = torch.tensor(self.layer_idx, device=device, dtype=torch.int64)

    def prepare_for_regional_compile(self, device: torch.device) -> str | None:
        """Resolve the inference-only compiled routes before fullgraph capture: tile-64 sm_100a and tile-256 CuTe.

        The ordinary eager route probes the environment, extension, device,
        and tensor contract at every call so it can warn and fall back.  Those
        Python/device-capability checks are not safe inside a regional
        ``fullgraph=True`` block.  Probe one representative tile-64 input on
        the loaded model's device now, then let ``forward`` specialize on the
        resulting plain bool while Dynamo is compiling.
        """
        if self._compile_layer_idx is None:
            self.prepare_for_compile(device)
        requested = os.environ.get(VSA_SM100A_ENV, "0") == "1"
        enabled = False
        reason = None if requested else f"{VSA_SM100A_ENV}=1 is required for compile-safe VSA-H3 attention"
        if requested:
            if _sm100a is None:
                reason = "fastvideo_kernel.block_sparse_attn_sm100a is not installed"
            elif not callable(getattr(_sm100a, "block_sparse_attn_sm100a_from_mask", None)):
                reason = ("the installed fastvideo_kernel.block_sparse_attn_sm100a has no native "
                          "block_sparse_attn_sm100a_from_mask entry (needs fastvideo_kernel >= f9e3680f, #1748)")
            else:
                # Two 64-token blocks exercise the exact sm_100a inference
                # specialization while keeping the one-time probe tiny.  The
                # kernel predicate checks extension presence, CUDA capability,
                # dtype/layout, head size, block size, and even block count
                # without reading metadata tensor contents.
                probe_query = torch.empty((1, 1, 128, self.head_size), device=device, dtype=torch.bfloat16)
                probe_block_sizes = torch.full((2, ), 64, device=device, dtype=torch.int32)
                reason = _sm100a_unavailable_reason(
                    _sm100a,
                    probe_query,
                    probe_block_sizes,
                    grad_mode=False,
                )
                enabled = reason is None

        self._regional_compile_sm100a_enabled = enabled
        self._regional_compile_nograd_route = self._resolve_cute256_route(device)
        if not enabled and self._regional_compile_nograd_route is not None:
            reason = None  # tile-256 CuTe runs compile; the loader pairs FASTVIDEO_VSA_SM100A=1 with tile 64
        elif not enabled and os.environ.get("FASTVIDEO_VSA_VC", "0") == "1":
            requested = True
            reason = (
                "FASTVIDEO_VSA_VC=1 but the compiled VC route is unavailable: it needs head size 128, probe recording "
                "off, the CuTe backend on SM10x and the VC modules importable from FASTVIDEO_VSA_VC_ROOT")
        if enabled:
            logger.info_once("VSA-H3 regional compile mask route: native fastvideo-kernel mask entry")
        if requested and reason is not None:
            logger.warning_once(f"VSA-H3 regional compile is unavailable and will stay eager: {reason}")
        return reason

    def _resolve_cute256_route(self, device: torch.device) -> str | None:
        """"bf16" when tile-128/256 no-grad calls reach the opaque CuTe ops (vsa_nograd_fwd / vsa256_nograd_fwd): CuTe
        backend, SM10x, VC off. "vc" (tiles 128 and 256) with
        FASTVIDEO_VSA_VC=1 when the fused VC ops can run: head 128, probe off, VC-enabled FA4 checkout importable."""
        if block_sparse_attn_256_bshd is None or device.type != "cuda":
            return None
        if _cute_backend() != "cutedsl" or torch.cuda.get_device_capability(device)[0] != 10:
            return None
        vc = os.environ.get("FASTVIDEO_VSA_VC", "0") == "1"
        # "vc" covers tiles 128 and 256 (_vc_fused_route; the vc_h3_* ops take the tile as a parameter).
        if vc and (self.head_size != 128 or probe_enabled() is not None):
            return None
        try:
            from fastvideo_kernel.block_sparse_attn_cute_fwd import _load_fa4_cute, _load_vc_module
            _load_fa4_cute()
            for name in ("vc_vsa_preprocess", "interface") if vc else ():
                _load_vc_module(name, os.environ.get("FASTVIDEO_VSA_VC_ROOT"))
        except (ImportError, RuntimeError):
            return None
        return "vc" if vc else "bf16"

    def tile(self, x: torch.Tensor, attn_metadata: MiniMaxH3VSAMetadata) -> torch.Tensor:
        """Scatter rows into the padded tile buffer (pad positions stay zero).

        Calls on builder-trusted index state (``_tile_index_state``) with BF16
        CUDA rows return an invocation-owned tensor from the opaque
        ``vsa_tile_permute_fwd`` op: eligible training calls (autograd retains
        it), eager no-grad calls and compiled tile-128/256 no-grad calls. Other
        eager no-grad calls (untrusted metadata, non-BF16, non-CUDA, the tile-64
        sm100a pair) return the builder-owned buffer; callers must consume it
        before the next ``tile()``. Odd tile-64 no-grad sm100a
        requests carry one additional all-zero tile internally; metadata and
        all observable outputs retain the logical geometry.
        """
        if x.shape[1] != attn_metadata.total_seq_length:
            raise ValueError(f"VSA-H3 metadata was built for sequence length {attn_metadata.total_seq_length}, "
                             f"got {x.shape[1]}. A non-packed sequence (e.g. the token refiner) is "
                             "routed to the VSA-H3 backend; exclude it from the supported backends.")
        n_tiles = attn_metadata.variable_block_sizes.numel()
        grad_mode = torch.is_grad_enabled() and x.requires_grad
        compiling = torch.compiler.is_compiling()
        regional_compiling = compiling and self._regional_compile_sm100a_enabled is True
        if regional_compiling:
            sm100a_requested = True
        elif compiling:
            # Training/generic compile keeps the long-standing Triton route.
            sm100a_requested = False
        else:
            sm100a_requested = os.environ.get(VSA_SM100A_ENV, "0") == "1"
        needs_sm100a_pair = (attn_metadata.tile_elems == 64 and n_tiles % 2 != 0 and not grad_mode and sm100a_requested)
        kernel_tiles = n_tiles + int(needs_sm100a_pair)
        target_shape = (x.shape[0], kernel_tiles * attn_metadata.tile_elems, x.shape[-2], x.shape[-1])

        # Builder-certified indices: the opaque vsa_tile_permute_fwd op (re-checks the index versions itself) returns an
        # invocation-owned buffer for training (size threshold: eager training only), eager no-grad BF16 CUDA rows, and
        # compiled tile-128/256 inference (beats Inductor's gather). Otherwise compiled 128/256 inference scatters into a
        # fresh buffer; the holder's identity/version bookkeeping below is eager-only Python state.
        state = _trusted_tile_state(attn_metadata, x.shape[1])
        compiled_nograd = compiling and not grad_mode and attn_metadata.tile_elems in (128, 256)
        eager_eligible = (state is not None and (grad_mode or not compiling) and not needs_sm100a_pair and x.ndim == 4
                          and x.is_cuda and x.dtype == torch.bfloat16 and (x.is_contiguous() or _tile_row_layout(x))
                          and state[0].device == x.device and state[2].device == x.device
                          and (not grad_mode or compiling or x.numel() >= 2**25) and x.shape[1] == state[2].numel())
        if state is not None and (compiled_nograd or eager_eligible):
            return torch.ops.fastvideo_kernel.vsa_tile_permute_fwd(x, state[0], state[2],
                                                                   attn_metadata.untile_combined_index, target_shape[1],
                                                                   state[1], state[3])
        if compiled_nograd:
            return scatter_into_tile_buf(x, target_shape, attn_metadata.untile_combined_index, None)

        # ``untile_combined_index`` maps each packed row to a logical tile
        # slot. Different geometries can share one transport shape; clear a
        # reused allocation once when the mapping identity changes so no old
        # valid row can survive as padding.
        holder = attn_metadata.tile_buf_holder
        if holder is None:
            raise RuntimeError("VSA-H3 metadata has no builder-owned tile buffer holder")
        buffer_matches = (holder.buffer is not None and holder.buffer.shape == target_shape
                          and holder.buffer.dtype == x.dtype and holder.buffer.device == x.device)
        if buffer_matches and holder.untile_geometry is not attn_metadata.untile_combined_index:
            holder.buffer.zero_()
        holder.buffer = scatter_into_tile_buf(x, target_shape, attn_metadata.untile_combined_index, holder.buffer)
        holder.untile_geometry = attn_metadata.untile_combined_index
        if needs_sm100a_pair:
            # A prior even geometry can reuse this allocation and may have
            # written the last tile as logical data.
            holder.buffer[:, n_tiles * attn_metadata.tile_elems:].zero_()
        return holder.buffer

    def _vc_fused_route(self, x: torch.Tensor, attn_metadata: MiniMaxH3VSAMetadata) -> bool:
        """True when the no-grad VC (FP8) route applies and Q/K/V can skip the BF16 tile copy.

        Exactly the calls that the generic path would send to VC attention (FASTVIDEO_VSA_VC=1 on the CuTe
        backend, tile 128 or 256, no grad, BF16 head 128 on SM10x), minus probe recording, which keeps the tiled path.
        preprocess_qkv and forward both use this predicate, so they always agree. Capture reads only the route
        resolved before compile (no env or device query inside the graph).
        """
        if (attn_metadata.tile_elems not in (128, 256) or x.ndim != 4 or not x.is_cuda or x.dtype != torch.bfloat16
                or x.shape[-1] != 128 or (torch.is_grad_enabled() and x.requires_grad)):
            return False
        if torch.compiler.is_compiling():
            return self._regional_compile_nograd_route == "vc"
        if (os.environ.get("FASTVIDEO_VSA_VC", "0") != "1" or not os.environ.get("FASTVIDEO_VSA_VC_ROOT")
                or block_sparse_attn_256_bshd is None):
            return False
        return (_cute_backend() == "cutedsl" and torch.cuda.get_device_capability(x.device)[0] == 10
                and probe_enabled() is None)

    def preprocess_qkv(self, qkv: torch.Tensor, attn_metadata: MiniMaxH3VSAMetadata) -> torch.Tensor:
        if self._vc_fused_route(qkv, attn_metadata):
            # The fused VC producer reads the packed rows directly into the padded FP8 layout.
            return qkv
        return self.tile(qkv, attn_metadata)

    def preprocess_q_k_v(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         attn_metadata: MiniMaxH3VSAMetadata) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """preprocess_qkv for callers holding separate q, k, v: equals preprocess_qkv(torch.cat([q, k, v])).chunk(3).

        Training calls and no-grad calls (eager and compiled) on trusted cube/chunk metadata gather each operand straight
        into the stacked tiled layout (vsa_tile_permute_qkv_fwd, invocation-owned outputs, no autograd state in no-grad),
        so neither the concatenated [3B, S, H, D] input nor a concatenated gradient is ever materialized (in a compiled
        region that cat lowers to an Inductor pointwise kernel). The size threshold applies to eager training only. Every
        other call (untrusted metadata, the tile-64 sm100a pair, the fused VC route in eager or compiled mode, whose
        producer must see packed rows) concatenates and takes preprocess_qkv unchanged.
        """
        n = attn_metadata.total_seq_length
        state = _trusted_tile_state(attn_metadata, n)
        grad_mode = torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v))
        compiling = torch.compiler.is_compiling()
        # No-grad exclusions: the odd tile-64 sm100a pair needs tile()'s extra zero tile; the fused VC route (eager, and
        # compiled since the prepared "vc" route) reads packed rows, so a tiled gather there would be mapped twice.
        nograd = (not grad_mode
                  and not (attn_metadata.tile_elems == 64 and attn_metadata.variable_block_sizes.numel() % 2 != 0)
                  and not self._vc_fused_route(q, attn_metadata))
        if (state is not None and (grad_mode or nograd) and q.ndim == 4 and q.is_cuda and q.dtype == torch.bfloat16
                and all(x.shape == q.shape and x.dtype == q.dtype and x.device == q.device and
                        (x.is_contiguous() or _tile_row_layout(x))
                        for x in (q, k, v)) and q.shape[1] == n == state[2].numel() and state[0].device == q.device
                and state[2].device == q.device and (not grad_mode or compiling or 3 * q.numel() >= 2**25)):
            padded = attn_metadata.variable_block_sizes.numel() * attn_metadata.tile_elems
            return torch.ops.fastvideo_kernel.vsa_tile_permute_qkv_fwd(q, k, v, state[0], state[2],
                                                                       attn_metadata.untile_combined_index, padded,
                                                                       state[1], state[3])
        return self.preprocess_qkv(torch.cat([q, k, v], dim=0), attn_metadata).chunk(3, dim=0)

    def _layer_sparsity(self, attn_metadata: MiniMaxH3VSAMetadata,
                        compiling: bool) -> tuple[float, torch.Tensor | None]:
        """(layer sparsity, force_dense). Probe-guided dense layers run with an all-True mask. Under a prepared capture the
        decision is tensor-valued (force_dense, OR-ed into the mask) so every block instance reuses one graph instead of
        specializing on the Python layer_idx; eager reads layer_idx."""
        if compiling and self._compile_layer_idx is not None:
            return attn_metadata.VSA_sparsity, (attn_metadata.dense_layers_tensor == self._compile_layer_idx).any()
        return (0.0 if self.layer_idx in attn_metadata.dense_layers else attn_metadata.VSA_sparsity), None

    def _vc_fused_forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                          gate_compress: torch.Tensor | None, attn_metadata: MiniMaxH3VSAMetadata) -> torch.Tensor:
        """Fused VC route: one producer pass from packed rows to padded FP8 Q/K/V plus FP32 tile pools.

        Replaces tile() + _pool_tiles + vc_preprocess.prepare of the generic route. Scores, the block mask and
        the VC attention call (on the 128-granularity map, as block_sparse_attn_256_bshd / _128_bshd send no-grad calls)
        are unchanged; the output is in the padded tile layout that postprocess_output expects.
        """
        import fastvideo_kernel.block_sparse_attn_cute_fwd  # noqa: F401 (registers the vc_h3_* ops)
        sizes = attn_metadata.variable_block_sizes
        n_tiles = sizes.numel()
        tile = attn_metadata.tile_elems
        padded = n_tiles * tile
        for name, tensor in (("query", query), ("key", key), ("value", value)):
            if tensor.shape[1] != attn_metadata.total_seq_length:
                raise ValueError(f"VSA-H3 fused VC {name} has length {tensor.shape[1]}, expected the packed length "
                                 f"{attn_metadata.total_seq_length}.")
        # The padded-row maps are rebuilt per call inside the producer op from the authoritative untile index.
        untile = attn_metadata.untile_combined_index
        q8, k8, v8, qs, ks, vs, *pools = torch.ops.fastvideo_kernel.vc_h3_prepare_fused(
            query, key, value, untile, sizes, tile)
        layer_sparsity, force_dense = self._layer_sparsity(attn_metadata, torch.compiler.is_compiling())
        if layer_sparsity > 0.0:
            mask = torch.ops.fastvideo_kernel.vsa_h3_block_map_from_pools(pools[0], pools[1],
                                                                          attn_metadata.num_prefix_tiles,
                                                                          layer_sparsity, attn_metadata.video_topk_cap,
                                                                          attn_metadata.exempt)
        else:
            mask = torch.ones(query.shape[0], query.shape[2], n_tiles, n_tiles, dtype=torch.bool, device=query.device)
        if force_dense is not None:
            mask = mask | force_dense
        out = torch.ops.fastvideo_kernel.vc_h3_attn_prepared(q8, k8, v8, qs, ks, vs, mask, sizes, tile,
                                                             attn_metadata.alias_guard_hint)
        if gate_compress is not None:
            scores = torch.matmul(pools[0], pools[1].transpose(-2, -1)) / (query.shape[-1]**0.5)
            batch, _, heads, dim = out.shape
            gate = gate_compress.new_zeros(batch, padded, heads, dim)
            gate[:, untile] = gate_compress
            out = _add_compress(out, scores, pools[2], gate, n_tiles, tile)
        return out

    def postprocess_output(self, output: torch.Tensor, attn_metadata: MiniMaxH3VSAMetadata) -> torch.Tensor:
        return vsa_h3_untile(output, attn_metadata)

    def forward_qkv(self, qkv: torch.Tensor, attn_metadata: MiniMaxH3VSAMetadata) -> torch.Tensor:
        """Gate-free entry on the batch-stacked tiled ``[3B, S, H, D]`` tensor.

        Same result as ``forward(*qkv.chunk(3), None, ...)``; native tile-256
        CuTe training keeps ``qkv`` as one autograd input so backward writes a
        single fused gradient instead of a ChunkBackward concat.
        """
        return self.forward(*qkv.chunk(3, dim=0),
                            None,
                            attn_metadata,
                            qkv=qkv if attn_metadata.fused_qkv_grad else None)

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor | None,
        attn_metadata: MiniMaxH3VSAMetadata,
        qkv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._vc_fused_route(query, attn_metadata) and query.shape[1] == attn_metadata.total_seq_length:
            return self._vc_fused_forward(query, key, value, gate_compress, attn_metadata)
        compiling = torch.compiler.is_compiling()
        regional_compiling = compiling and self._regional_compile_sm100a_enabled is True

        tile_elems = attn_metadata.tile_elems
        if tile_elems == 64:
            if block_sparse_attn_64_bhsd is None:
                raise NotImplementedError("fastvideo_kernel.block_sparse_attn is not installed")
        elif block_sparse_attn_256_bshd is None:  # the 128 route lives in the same module
            raise NotImplementedError("fastvideo_kernel.block_sparse_attn_256 is not installed")

        # Probe recording performs filesystem writes and host synchronizations,
        # so the loader keeps probe-enabled runs eager. Avoid even reading that
        # environment switch while Dynamo captures a regional full graph.
        # The metadata always describes the trained logical geometry.
        # ``tile()`` may append exactly one transport-only partner for an odd
        # tile-64 sm100a call. Keep score selection and the gate branch on the
        # logical prefix, and reject every other shape before a kernel sees it.
        n_tiles = attn_metadata.variable_block_sizes.numel()
        logical_seq_len = n_tiles * tile_elems
        pair_pad_seq_len = logical_seq_len + tile_elems
        pair_pad_is_valid = tile_elems == 64 and n_tiles % 2 != 0
        allowed_seq_lengths = (logical_seq_len, pair_pad_seq_len) if pair_pad_is_valid else (logical_seq_len, )
        if query.shape[1] not in allowed_seq_lengths:
            expected = (f"the logical length {logical_seq_len} or one sm100a partner tile "
                        f"({pair_pad_seq_len})" if pair_pad_is_valid else f"the logical length {logical_seq_len}")
            raise ValueError(f"VSA-H3 tiled query has length {query.shape[1]}, expected {expected}.")
        has_sm100a_pair = query.shape[1] == pair_pad_seq_len
        for name, tensor in (("key", key), ("value", value)):
            if tensor.shape[1] != query.shape[1]:
                raise ValueError(f"VSA-H3 tiled {name} length {tensor.shape[1]} does not match query "
                                 f"length {query.shape[1]}.")
        if gate_compress is not None and gate_compress.shape[1] != query.shape[1]:
            raise ValueError(f"VSA-H3 tiled gate length {gate_compress.shape[1]} does not match query "
                             f"length {query.shape[1]}.")

        logical_query = query[:, :logical_seq_len]
        logical_key = key[:, :logical_seq_len]
        logical_value = value[:, :logical_seq_len]
        logical_gate = gate_compress[:, :logical_seq_len] if gate_compress is not None else None

        layer_sparsity, force_dense = self._layer_sparsity(attn_metadata, compiling)
        probe_dir = None if compiling else probe_enabled()

        scores = None
        if gate_compress is not None or probe_dir is not None:
            q_pooled = _pool_tiles(logical_query, attn_metadata.variable_block_sizes, tile_elems)
            k_pooled = _pool_tiles(logical_key, attn_metadata.variable_block_sizes, tile_elems)
            scores = torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) / (query.shape[-1]**0.5)
            if probe_dir is not None:
                record_probe(probe_dir, self.layer_idx, logical_query, logical_key, scores, attn_metadata)

        if layer_sparsity > 0.0:
            # video_topk is cached at build (metadata must be rebuilt when VSA_sparsity changes): eager recomputes
            # on a mismatch; compiled graphs use the cached integer.
            k_vid = attn_metadata.video_topk
            if k_vid is None or (not compiling and k_vid != _video_topk(
                    layer_sparsity, attn_metadata.num_video_tiles, attn_metadata.video_topk_cap)):
                k_vid = _video_topk(layer_sparsity, attn_metadata.num_video_tiles, attn_metadata.video_topk_cap)
            mask = torch.ops.fastvideo_kernel.vsa_h3_block_map(
                logical_query.detach(), logical_key.detach(), attn_metadata.variable_block_sizes, tile_elems,
                attn_metadata.num_prefix_tiles, attn_metadata.num_video_tiles, k_vid, attn_metadata.exempt)
        else:  # dense layer: compute_topk(0, n) == n selects every tile
            mask = torch.ones(query.shape[0], query.shape[2], n_tiles, n_tiles, dtype=torch.bool, device=query.device)
        if force_dense is not None:
            # A scalar bool tensor broadcasts over the block map. This exactly
            # preserves the eager dense-layer contract without a Python branch.
            mask = mask | force_dense

        query_sizes = query_untile = None
        query_versions = (0, 0)
        if tile_elems == 64:
            # Native 64-token path: the block map is already at the kernels'
            # granularity. Both 64-token entries take BHSD ([B, H, S_pad, D]);
            # mirror block_sparse_attn_256_bshd's Triton branch and transpose
            # around the call.
            q_bhsd = query.transpose(1, 2).contiguous()
            k_bhsd = key.transpose(1, 2).contiguous()
            v_bhsd = value.transpose(1, 2).contiguous()

            sm100a_mask = mask
            sm100a_variable_block_sizes = attn_metadata.variable_block_sizes
            if has_sm100a_pair:
                # The synthetic tile is neither a logical query nor key. Its
                # all-False row yields q2k_num=0, the all-False column keeps it
                # out of real rows, and vbs=0 masks all of its key slots.
                sm100a_mask = torch.nn.functional.pad(mask, (0, 1, 0, 1), value=False)
                sm100a_variable_block_sizes = torch.nn.functional.pad(
                    attn_metadata.variable_block_sizes,
                    (0, 1),
                    value=0,
                )

            # Opt-in sm_100a CUDA forward (upstream PR #1719 + per-q-tile
            # q2k_num fix). Forward-only: grad-tracking calls stay on Triton
            # so autograd keeps the Triton fwd+bwd pairing untouched. The
            # kernel does return an LSE in Triton's M format, so a future
            # fwd/bwd pairing is possible, but it is not built here.
            grad_mode = torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad)
            use_sm100a = False
            if regional_compiling:
                # The preparation probe established module/device/kernel
                # support.  Keep only static tensor/geometry facts here; no
                # env access, device-capability query, or is_supported call may
                # enter the Dynamo graph.
                if not (not grad_mode and q_bhsd.dtype == torch.bfloat16 and q_bhsd.shape[-1] == 128
                        and sm100a_variable_block_sizes.numel() % 2 == 0):
                    raise RuntimeError(
                        "VSA-H3 regional fullgraph compile requires the prepared sm_100a BF16/head-128 route "
                        "on a supported device; disable inference_torch_compile for this request.")
                use_sm100a = True
            elif not compiling and os.environ.get(VSA_SM100A_ENV, "0") == "1":
                reason = _sm100a_unavailable_reason(_sm100a, q_bhsd, sm100a_variable_block_sizes, grad_mode)
                if reason is None and map_to_index is None:
                    reason = "fastvideo_kernel.triton_kernels.index (map_to_index) is not importable"
                if reason is None:
                    use_sm100a = True
                else:
                    logger.warning_once(f"{VSA_SM100A_ENV}=1 but falling back to the Triton-64 kernels: {reason}")

            if use_sm100a:
                # Regional preparation emits the compile-route receipt before
                # capture. Logging from this branch would itself break a
                # ``fullgraph=True`` forward.
                if not compiling:
                    logger.info_once("MiniMax-H3 VSA tile-64 forward: using the sm100a/sm103a CUDA block-sparse kernel")
                if regional_compiling:
                    # The native mask entry keeps both Triton mask compaction and
                    # the raw pybind launch behind one fake-backed custom-op boundary.
                    out_bhsd, _ = _sm100a.block_sparse_attn_sm100a_from_mask(
                        q_bhsd,
                        k_bhsd,
                        v_bhsd,
                        sm100a_mask,
                        sm100a_variable_block_sizes,
                    )
                else:
                    # Preserve the established eager/index-native route and
                    # compatibility with older kernel wheels. Per-row counts
                    # are non-uniform (prefix queries are dense; video queries
                    # run prefix+top-k), which the fixed kernel supports.
                    q2k_idx, q2k_num = map_to_index(sm100a_mask)
                    out_bhsd, _ = _sm100a.block_sparse_attn_sm100a(
                        q_bhsd,
                        k_bhsd,
                        v_bhsd,
                        q2k_idx,
                        q2k_num,
                        sm100a_variable_block_sizes.to(torch.int32),
                        need_lse=False,
                    )
            else:
                if has_sm100a_pair:
                    q_bhsd = q_bhsd[:, :, :logical_seq_len].contiguous()
                    k_bhsd = k_bhsd[:, :, :logical_seq_len].contiguous()
                    v_bhsd = v_bhsd[:, :, :logical_seq_len].contiguous()
                out_bhsd, _ = block_sparse_attn_64_bhsd(
                    q_bhsd,
                    k_bhsd,
                    v_bhsd,
                    mask,
                    attn_metadata.variable_block_sizes,
                )
            if has_sm100a_pair and use_sm100a:
                out_bhsd = out_bhsd[:, :, :logical_seq_len]
            out = out_bhsd.transpose(1, 2).contiguous()
        elif tile_elems == 128:
            # FA4 Q128/KV128: grad-tracking CuTe calls go through the opaque tile-parameterized training op pair (same
            # op key eager and compiled, fullgraph-safe, SAC-visible); every other call keeps the library's VSA-128
            # route. Both forward launches get the same alias-guard hint as the tile-256 route.
            if _resolve_backend() == "cutedsl" and vsa256_ops.training_eligible(
                    logical_query, logical_key, logical_value, mask, block=128):
                # Stacked route (forward_qkv with fused_qkv_grad), as at tile 256: the contiguous [3B, S, H, D] stack is
                # the single op input and the backward writes one fused dQ/dK/dV allocation (no chunk-backward concat).
                stacked = (qkv is not None and gate_compress is None and query.shape[1] == logical_seq_len
                           and qkv.is_contiguous() and qkv.shape == (3 * query.shape[0], *query.shape[1:]))
                out, _ = torch.ops.fastvideo_kernel.vsa_train_fwd(qkv if stacked else logical_query,
                                                                  None if stacked else logical_key,
                                                                  None if stacked else logical_value, mask,
                                                                  attn_metadata.variable_block_sizes, tile_elems,
                                                                  attn_metadata.alias_guard_hint)
            else:
                out, _ = block_sparse_attn_128_bshd(logical_query,
                                                    logical_key,
                                                    logical_value,
                                                    mask,
                                                    attn_metadata.variable_block_sizes,
                                                    alias_guard=attn_metadata.alias_guard_hint)
        else:
            # Padded query rows get zero dO through postprocess_output's untile gather, which lets
            # the CuTe backward skip wholly padded Q128 children (the LSE output is discarded; the
            # gate branch adds out of place, so it leaves the attention output's padded dO zero).
            # Trust only the builder-validated objects (identity is traceable). Eager also checks
            # their version counters here; Dynamo cannot branch on _version, so compiled calls hand
            # the builder-recorded versions to the opaque vsa256 op, whose backward re-checks them
            # and falls back to the full backward on any in-place change.
            state = getattr(attn_metadata, "_query_pad_state", None) if attn_metadata.query_pad_pruning else None
            if (state is not None and state[0] is attn_metadata.untile_combined_index
                    and state[2] is attn_metadata.variable_block_sizes
                    and (compiling or (state[0]._version == state[1] and state[2]._version == state[3]))):
                query_sizes, query_untile, query_versions = state[2], state[0], (state[1], state[3])
            # Dense prefix rows make the aliased persistent grid pile work onto a few CTAs; uniform (prefix-0) documents
            # would only lose a wave to the shrink. The host-known prefix count decides the FA4 hint.
            out, _ = block_sparse_attn_256_bshd(
                logical_query,
                logical_key,
                logical_value,
                mask,
                attn_metadata.variable_block_sizes,
                query_sizes=query_sizes,
                query_untile=query_untile,
                query_versions=query_versions,
                pack_tails=attn_metadata.pack_tails,
                qkv=qkv if gate_compress is None and query.shape[1] == logical_seq_len else None,
                alias_guard=attn_metadata.alias_guard_hint,
            )

        if logical_gate is not None:
            # The gate is zero-initialized for H3 (no contribution until finetuned; the model layer skips all-zero gates).
            v_pooled = _pool_tiles(logical_value, attn_metadata.variable_block_sizes, tile_elems)
            out = _add_compress(out, scores, v_pooled, logical_gate, n_tiles, tile_elems)
        if query_sizes is not None:
            # Pin the trusted map (and its builder-recorded version) for postprocess_output. The pin is set in
            # traced caller code on a view: Dynamo rejects setattr on a custom op's tuple output element.
            out = out.view_as(out)
            out._vsa_h3_query_pad_untile = (query_untile, query_versions[0])  # type: ignore[attr-defined]
        return out
