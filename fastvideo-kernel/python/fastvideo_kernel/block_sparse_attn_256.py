"""VSA-128/256 block-sparse attention wrappers.

The default 256-block path is Triton: it expands the logical 256-block map
to the existing 64-block Triton kernel via a dense 4x4 expansion per logical
edge ("route A"), and requires no optional dependencies.

The FA4 CuTe block-sparse fastpath (intended for Blackwell sm_100+) is
*opt-in* via ``FASTVIDEO_VSA_CUTEDSL=1``. It routes to
:mod:`fastvideo_kernel.block_sparse_attn_cute_fwd`, which natively operates
on native 256-token blocks for supported BSHD training inputs, with 128-token
expansion retained for other paths. The CuTe kernel
(``flash_attn.cute`` with block-sparsity) is an optional dependency,
imported lazily only when this fastpath is selected.

``FASTVIDEO_VSA_TRITON=1`` (or the legacy
``FASTVIDEO_KERNEL_VSA_FORCE_TRITON=1``) forces Triton explicitly.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

from .block_sparse_attn import block_sparse_attn_triton, _force_triton
from . import vsa256_ops  # noqa: F401  (registers torch.ops.fastvideo_kernel.vsa256_fwd/bwd at import, e.g. for SAC policies)

# NOTE: ``block_sparse_attn_cute_fwd`` is imported lazily inside the CuTe
# branches below. Importing it at module load would pull in the optional
# FA4 CuTe build (``flash_attn.cute``) and make it a hard dependency of the
# default Triton path.

_KV_BLOCK_PHYS = 128  # FA4 CuTe BSA forward uses 128-token KV blocks.
_KV_BLOCK_TRITON = 64  # Existing Triton path uses 64-token KV blocks.


def _resolve_backend() -> str:
    """Pick the backend for the 128/256-block VSA paths.

    Default is Triton (no optional deps). The FA4 CuTe fastpath is opt-in
    via ``FASTVIDEO_VSA_CUTEDSL=1`` and requires the optional FA4 CuTe
    build. ``FASTVIDEO_VSA_TRITON=1`` / the legacy force-triton flag force
    Triton explicitly and take precedence over the CuTe opt-in.
    """
    if _force_triton():
        return "triton"
    if os.environ.get("FASTVIDEO_VSA_CUTEDSL", "0") == "1":
        return "cutedsl"
    return "triton"


def _expand_mask_and_sizes_to_64(
    logical_mask: torch.Tensor,
    logical_kv_sizes: torch.Tensor,
    factor: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand a [B, H, Qb, KVb] map of (64 * factor)-token blocks to 64-token Triton tiles (route A).

    Each logical edge becomes a factor x factor block of edges; each 64-token child's valid count is the
    logical count clamped into the child's window.
    """
    expanded_mask = logical_mask.repeat_interleave(factor, dim=2).repeat_interleave(factor, dim=3)
    sizes_i32 = logical_kv_sizes.to(torch.int32)
    offsets = torch.arange(0, factor * _KV_BLOCK_TRITON, _KV_BLOCK_TRITON, dtype=torch.int32, device=sizes_i32.device)
    expanded_sizes = torch.clamp(sizes_i32[:, None] - offsets[None, :], min=0, max=_KV_BLOCK_TRITON).reshape(-1)
    return expanded_mask, expanded_sizes


def _expand_mask_and_sizes_256_to_128(
    logical_mask_256: torch.Tensor,
    logical_kv_sizes_256: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand a [B, H, Qb256, KVb256] map to [B, H, Qb256, KVb128].

    Each logical 256-token KV block splits into two physical 128-token
    children. Each child inherits the logical mask edge; its valid-token
    count is the logical count clamped into the child's window.
    """
    expanded_mask = logical_mask_256.repeat_interleave(2, dim=3)

    sizes_i32 = logical_kv_sizes_256.to(torch.int32)
    child0 = torch.clamp(sizes_i32, min=0, max=_KV_BLOCK_PHYS)
    child1 = torch.clamp(sizes_i32 - _KV_BLOCK_PHYS, min=0, max=_KV_BLOCK_PHYS)
    expanded_sizes = torch.empty(
        (sizes_i32.numel() * 2, ),
        dtype=torch.int32,
        device=sizes_i32.device,
    )
    expanded_sizes[0::2] = child0
    expanded_sizes[1::2] = child1
    return expanded_mask, expanded_sizes


def _triton_via_route_a(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logical_mask: torch.Tensor,
    logical_kv_sizes: torch.Tensor,
    block: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton fallback for [B, H, S, D] inputs: expand the ``block``-token map to 64-token tiles and run the kernel."""
    from .triton_kernels.index import map_to_index as triton_map_to_index

    mask_64, sizes_64 = _expand_mask_and_sizes_to_64(logical_mask, logical_kv_sizes, block // _KV_BLOCK_TRITON)
    q2k_idx, q2k_num = triton_map_to_index(mask_64.to(torch.bool))
    return block_sparse_attn_triton(q, k, v, q2k_idx, q2k_num, sizes_64)


def _triton_via_route_a_bshd(q, k, v, logical_mask, logical_kv_sizes, block):
    """The Triton fallback for [B, S, H, D] inputs (the kernel takes BHSD)."""
    out_bhsd, aux = _triton_via_route_a(*(t.transpose(1, 2).contiguous() for t in (q, k, v)), logical_mask,
                                        logical_kv_sizes, block)
    return out_bhsd.transpose(1, 2).contiguous(), aux


def _batched(block_map: torch.Tensor) -> torch.Tensor:
    """Accept a single [H, Qb, KVb] map as batch 1."""
    return block_map.unsqueeze(0) if block_map.dim() == 3 else block_map


def _hint_kwargs(alias_guard):
    """Forward the alias-guard hint only when set, so hint-less calls keep their exact previous arguments."""
    return {} if alias_guard is None else {"alias_guard": alias_guard}


def block_sparse_attn_128(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logical_block_map_128: torch.Tensor,
    logical_variable_block_sizes_128: torch.Tensor,
    *,
    alias_guard=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """VSA-128 sparse-branch entrypoint for [B, H, S, D] inputs (``alias_guard`` as block_sparse_attn_256_bshd)."""
    logical_block_map_128 = _batched(logical_block_map_128)
    if _resolve_backend() == "triton":
        return _triton_via_route_a(q, k, v, logical_block_map_128, logical_variable_block_sizes_128, 128)

    from .block_sparse_attn_cute_fwd import block_sparse_attn_cute_fwd
    return block_sparse_attn_cute_fwd(q, k, v, logical_block_map_128, logical_variable_block_sizes_128,
                                      **_hint_kwargs(alias_guard))


def block_sparse_attn_128_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logical_block_map_128: torch.Tensor,
    logical_variable_block_sizes_128: torch.Tensor,
    *,
    alias_guard=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """VSA-128 sparse-branch entrypoint for [B, S, H, D] inputs (``alias_guard`` as block_sparse_attn_256_bshd)."""
    logical_block_map_128 = _batched(logical_block_map_128)
    if _resolve_backend() == "triton":
        return _triton_via_route_a_bshd(q, k, v, logical_block_map_128, logical_variable_block_sizes_128, 128)

    # Native BF16 Q128: the tile-parameterized opaque ops (same op in eager and compiled mode, fullgraph-safe).
    hint = () if alias_guard is None else (
        alias_guard if isinstance(alias_guard, torch.Tensor) else torch.tensor(bool(alias_guard), device="cpu"), )
    if vsa256_ops.training_eligible(q, k, v, logical_block_map_128, block=128):
        out, lse = vsa256_ops.vsa_train_fwd(q, k, v, logical_block_map_128, logical_variable_block_sizes_128, 128, *hint)
        return out, lse.detach()
    if vsa256_ops.nograd_eligible(q, k, v, logical_block_map_128, block=128):
        return vsa256_ops.vsa_nograd_fwd(q, k, v, logical_block_map_128, logical_variable_block_sizes_128, 128, *hint)
    from .block_sparse_attn_cute_fwd import block_sparse_attn_cute_fwd_bshd
    return block_sparse_attn_cute_fwd_bshd(q, k, v, logical_block_map_128, logical_variable_block_sizes_128,
                                           **_hint_kwargs(alias_guard))


def block_sparse_attn_256(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logical_block_map_256: torch.Tensor,
    logical_variable_block_sizes_256: torch.Tensor,
    *,
    alias_guard=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """VSA-256 sparse-branch entrypoint for [B, H, S, D] inputs (``alias_guard`` as block_sparse_attn_256_bshd)."""
    logical_block_map_256 = _batched(logical_block_map_256)
    if _resolve_backend() == "triton":
        return _triton_via_route_a(q, k, v, logical_block_map_256, logical_variable_block_sizes_256, 256)

    mask_128, sizes_128 = _expand_mask_and_sizes_256_to_128(logical_block_map_256, logical_variable_block_sizes_256)
    from .block_sparse_attn_cute_fwd import block_sparse_attn_cute_fwd
    return block_sparse_attn_cute_fwd(q, k, v, mask_128, sizes_128, **_hint_kwargs(alias_guard))


def block_sparse_attn_256_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logical_block_map_256: torch.Tensor,
    logical_variable_block_sizes_256: torch.Tensor,
    query_sizes: torch.Tensor | None = None,
    query_untile: torch.Tensor | None = None,
    query_versions: tuple[int, int] = (0, 0),
    pack_tails: bool | None = None,
    qkv: torch.Tensor | None = None,
    *,
    alias_guard=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """VSA-256 sparse-branch entrypoint for [B, S, H, D] inputs.

    ``alias_guard`` is FA4's persistent-grid alias-guard hint: None (FA4 default), a bool, or a 0-d CPU bool tensor
    (preferred under torch.compile: a graph input, not a specialized constant). The Triton fallback ignores it.

    Default CuTe path consumes BSHD directly; Triton fallback transposes
    to BHSD as the legacy path expects. ``query_sizes``: optional caller
    guarantee that query rows past each tile's valid prefix get zero output
    gradient; honored only together with the trusted ``query_untile`` map and the
    builder-recorded ``query_versions`` of (query_untile, query_sizes), re-checked inside the
    backward op (``vsa256_ops``).
    ``pack_tails`` (training backward tail packing, static per call): None follows
    FASTVIDEO_VSA_PACK_TAILS (direct callers); a bool (the H3 metadata policy) forces it.
    ``qkv`` (optional, the contiguous ``[3B, S, H, D]`` tensor that q/k/v are
    dim-0 chunks of) lets native training return one fused gradient; other routes ignore it.
    """
    logical_block_map_256 = _batched(logical_block_map_256)
    if _resolve_backend() == "triton":
        return _triton_via_route_a_bshd(q, k, v, logical_block_map_256, logical_variable_block_sizes_256, 256)

    from .block_sparse_attn_cute_fwd import block_sparse_attn_cute_fwd_bshd
    if vsa256_ops.training_eligible(q, k, v, logical_block_map_256):
        # BF16 Q256 training: opaque custom-op pair (same op in eager and compiled mode, fullgraph-safe). The pack
        # policy is static per call: packed-tail backward when enabled (default) and the inputs are contiguous.
        out, lse = vsa256_ops.training_attention(q, k, v, logical_block_map_256, logical_variable_block_sizes_256,
                                                 query_sizes=query_sizes, query_untile=query_untile,
                                                 query_versions=query_versions, pack_tails=pack_tails,
                                                 qkv=qkv, alias_guard=alias_guard)
        return out, lse.detach()
    if vsa256_ops.nograd_eligible(q, k, v, logical_block_map_256):
        # BF16 Q256 inference: the same forward behind one opaque op (eager and compiled), fullgraph-safe.
        hint = () if alias_guard is None else (
            alias_guard if isinstance(alias_guard, torch.Tensor) else torch.tensor(bool(alias_guard), device="cpu"), )
        return vsa256_ops.vsa256_nograd_fwd(q, k, v, logical_block_map_256, logical_variable_block_sizes_256, *hint)
    mask_128, sizes_128 = _expand_mask_and_sizes_256_to_128(logical_block_map_256, logical_variable_block_sizes_256)
    return block_sparse_attn_cute_fwd_bshd(q, k, v, mask_128, sizes_128, **_hint_kwargs(alias_guard))
