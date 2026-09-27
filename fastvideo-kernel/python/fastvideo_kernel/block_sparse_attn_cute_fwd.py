"""FA4 CuTe-DSL block-sparse attention adapter.

This module adapts VSA's ``(block_map, variable_block_sizes)`` inputs into
FA4's forward and backward ``BlockSparseTensorsTorch`` representations.
FA4 supplies the forward/backward kernels; this adapter bridges autograd.

Both [B, H, S, D] (BHSD) and [B, S, H, D] (BSHD) entrypoints are provided.
The BSHD variant is preferred from VSA-128/256 callers to avoid layout
round-trips on the hot path.

The FA4 CuTe block-sparse kernel (``flash_attn.cute`` with
``block_sparsity``) is an *optional* dependency: it is imported lazily and
only exercised when the VSA-128/256 CuTe fastpath is explicitly selected
(``FASTVIDEO_VSA_CUTEDSL=1``). The default path is Triton and does not require
it. Also needs ``nvidia-cutlass-dsl`` and ``quack-kernels``.

VSA-256 training tail packing is enabled by default for contiguous BF16 inputs
on SM10x (head dimensions 64/128), with FA4 backward workspace support.
Set ``FASTVIDEO_VSA_PACK_TAILS=0`` to disable it. Short KV tails benefit, while
full blocks and overflowing tail plans pay preparation overhead. A device-side
capacity check preserves the original sparse calculation on overflow.
"""

from __future__ import annotations

import functools
import importlib
import os
from pathlib import Path
from typing import Tuple

import torch

_FA4_IMPORT_HINT = ("VSA-128/256 CuTe fastpath requires a FlashAttention-4 CuTe build that "
                    "provides `flash_attn.cute` with block-sparsity support (plus "
                    "`nvidia-cutlass-dsl` and `quack-kernels`). This is an optional "
                    "dependency; the default path is Triton. Install the FA4 CuTe "
                    "build and set FASTVIDEO_VSA_CUTEDSL=1 to enable the CuTe fastpath.")


@functools.lru_cache(maxsize=1)
def _load_fa4_cute():
    """Lazily import the optional FA4 CuTe block-sparse symbols.

    Raising a clear, actionable error here keeps the optional FA4 CuTe
    build from being a hard import-time dependency of this module (and of
    the default Triton VSA-256 path).
    """
    try:
        from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
        from flash_attn.cute.interface import (
            _flash_attn_bwd,
            _flash_attn_fwd,
            flash_attn_func,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(_FA4_IMPORT_HINT) from exc
    return BlockSparseTensorsTorch, flash_attn_func, _flash_attn_fwd, _flash_attn_bwd


# FA4's physical Q tile size; KV block size comes from the VSA caller.
_FA4_Q_BLOCK_SIZE = 128


class _SingleQStageLength(int):
    """Keep the real length while selecting FA4's one-stage Q128 path.

    On sm_100 FA4 derives ``q_stage`` from ``max_seqlen_q > tile_m``. Its
    kernel supports one 128-token Q stage, but the fixed-length public wrapper
    does not expose that choice. VSA-128 must select it explicitly; otherwise
    adjacent logical Q blocks are merged into a 256-token sparse block.
    """

    def __mul__(self, other):
        return type(self)(int(self) * int(other))

    def __rmul__(self, other):
        return type(self)(int(other) * int(self))

    def __gt__(self, other):
        if int(other) == _FA4_Q_BLOCK_SIZE:
            return False
        return int(self) > int(other)


def _map_to_index(block_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if block_map.dim() == 3:
        block_map = block_map.unsqueeze(0)
    if block_map.dim() != 4:
        raise ValueError(f"block_map must be [B,H,Q,KV] (or [H,Q,KV]), "
                         f"got shape={tuple(block_map.shape)}")
    if block_map.dtype != torch.bool:
        block_map = block_map.to(torch.bool)
    if not block_map.is_cuda:
        raise RuntimeError("block_map must be a CUDA tensor.")
    from fastvideo_kernel.triton_kernels.index import (
        map_to_index as triton_map_to_index, )

    return triton_map_to_index(block_map)


def _choose_q_sparse_block_size(q_len: int, q_tile_size: int = _FA4_Q_BLOCK_SIZE) -> int:
    # FA4 supports a doubled Q sparsity granularity on sm_100+ when q_len > q_tile_size.
    major, _ = torch.cuda.get_device_capability()
    if major >= 10 and q_len > q_tile_size:
        return 2 * q_tile_size
    return q_tile_size


def _aggregate_q_block_map(
    block_map: torch.Tensor,
    q_sparse_block_size: int,
    q_block_size: int,
) -> torch.Tensor:
    factor = q_sparse_block_size // q_block_size
    if factor <= 0 or q_sparse_block_size % q_block_size != 0:
        raise ValueError(f"q_sparse_block_size must be a positive multiple of "
                         f"q_block_size ({q_block_size}), got {q_sparse_block_size}")
    bsz, nhead, q_blocks, kv_blocks = block_map.shape
    q_blocks_sparse = (q_blocks + factor - 1) // factor
    pad_q = q_blocks_sparse * factor - q_blocks
    if pad_q > 0:
        pad = torch.zeros(
            bsz,
            nhead,
            pad_q,
            kv_blocks,
            dtype=torch.bool,
            device=block_map.device,
        )
        block_map = torch.cat([block_map, pad], dim=2)
    block_map = block_map.view(bsz, nhead, q_blocks_sparse, factor, kv_blocks)
    return block_map.any(dim=3)


@functools.lru_cache(maxsize=4)
def _build_vbs_mask_mod(kv_block_size: int):
    """Build a CuTe mask_mod that trims per-KV-block valid tokens.

    aux_tensors[0] must be an int32 tensor of shape [kv_blocks] giving the
    valid token count in [0, kv_block_size] for each KV block.
    """
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils
    from flash_attn.cute.block_sparsity import fast_sampling

    kv_block_size_const = int(kv_block_size)

    @fast_sampling
    @cute.jit
    def _vbs_mask_mod(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        del batch, head, m_idx, seqlen_info
        block_size_ssa = utils.scalar_to_ssa(kv_block_size_const, cutlass.Int32)
        zero_ssa = utils.scalar_to_ssa(0, cutlass.Int32)
        kv_blk = n_idx // block_size_ssa
        kv_off = n_idx % block_size_ssa
        kv_sizes = aux_tensors[0]
        valid = utils.scalar_to_ssa(kv_sizes[kv_blk[0]], cutlass.Int32)
        return (valid > zero_ssa) & (kv_off < valid)

    # Exact contract for optional FA4 backward specialization: aux[0] is a
    # 1D KV128 valid-prefix array, independent of Q, batch, and head.
    if kv_block_size_const == 128:
        _vbs_mask_mod.__vbs_kv_block_size__ = 128
    return _vbs_mask_mod


@functools.lru_cache(maxsize=2)
def _build_vbs_vector_mask_mod(kv_block_size: int):
    """Pack one aligned SM100 KV128 fragment's validity into four masks."""
    import cutlass
    import cutlass.cute as cute
    try:
        from flash_attn.cute.mask import AttentionMask, r2p_bitmask_below
    except ImportError:
        return _build_vbs_mask_mod(kv_block_size)
    if not hasattr(AttentionMask, "apply_mask_mod_sm100_vector"):
        return _build_vbs_mask_mod(kv_block_size)

    @cute.jit
    def _vbs_vector_mask_mod(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
        base = n_idx[0]
        limit = aux_tensors[0][base // kv_block_size] - base % kv_block_size
        packed = cute.make_rmem_tensor(4, cutlass.Uint32)
        for word in cutlass.range_constexpr(4):
            packed[word] = r2p_bitmask_below(limit, word)
        return packed.load()

    _vbs_vector_mask_mod.__vec_size__ = 128
    return _vbs_vector_mask_mod


def _build_vbs_fwd_mask_mod(q, kv_block_size: int, *, need_backward: bool = False):
    if (not need_backward and kv_block_size in (128, 256) and q.shape[-1] in (64, 128)
            and torch.cuda.get_device_capability(q.device)[0] in (10, 11)):
        return _build_vbs_vector_mask_mod(kv_block_size)
    return _build_vbs_mask_mod(kv_block_size)


def _build_sparse_tensors(
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    q_len: int,
    q_block_size: int,
    kv_block_size: int,
    need_backward: bool,
    need_forward: bool = True,
    force_q_sparse_block_size: int | None = None,
) -> Tuple[object | None, object | None]:
    """Build the Q-owned forward and KV-owned backward sparse metadata.

    ``need_backward`` is False on inference-only calls: the backward metadata
    retains a dense ``[B, H, kv_blocks, q_blocks]`` int32 index tensor shared
    by the full and partial lists until backward runs. Building it when
    nothing requires grad is pure overhead.

    Tile-parametric callers pass ``(q_block_size, kv_block_size) = (TILE, TILE)`` for the forward and
    ``(TILE, TILE // 2, force_q_sparse_block_size=TILE)`` for the backward. Two normalizations keep that valid at
    TILE = 128: a KV block finer than FA4's 128-token tile is merged back to 128 (sizes are summed, map columns OR-ed;
    the children of a 128 block must share one keep bit, as in a ``repeat_interleave``-d map, else a device-side assert
    fires); and a 128-row Q block keeps its own sparse list instead of being doubled to 256 rows, which would give each
    Q tile its neighbour's KV selection.
    """
    if kv_block_size < _FA4_Q_BLOCK_SIZE <= q_block_size:  # tile-64 callers keep their own 64-token lists
        factor = _FA4_Q_BLOCK_SIZE // kv_block_size
        if _FA4_Q_BLOCK_SIZE % kv_block_size or block_map.shape[-1] % factor or variable_block_sizes.numel() % factor:
            raise ValueError(f"kv_block_size={kv_block_size} must tile {_FA4_Q_BLOCK_SIZE} and the KV block count")
        children = block_map.unflatten(-1, (-1, factor))
        block_map = children.any(-1)
        # OR-merging is exact only when every child of a 128 block has the same keep bit; the mask mod checks token
        # validity, not selection. Device-side check: no host sync per call.
        torch._assert_async((block_map == children.all(-1)).all(),
                            "kv_block_size < 128 needs equal keep bits within each 128-token KV block")
        variable_block_sizes = variable_block_sizes.view(-1, factor).sum(-1, dtype=variable_block_sizes.dtype)
        kv_block_size = _FA4_Q_BLOCK_SIZE
    if force_q_sparse_block_size is None and q_block_size == _FA4_Q_BLOCK_SIZE:
        force_q_sparse_block_size = _FA4_Q_BLOCK_SIZE
    BlockSparseTensorsTorch, _, _, _ = _load_fa4_cute()
    if force_q_sparse_block_size is None:
        q_sparse_candidate = _choose_q_sparse_block_size(q_len)
        q_sparse_block_size = max(
            q_block_size,
            ((q_sparse_candidate + q_block_size - 1) // q_block_size) * q_block_size,
        )
    else:
        q_sparse_block_size = force_q_sparse_block_size
        if q_sparse_block_size < q_block_size or q_sparse_block_size % q_block_size != 0:
            raise ValueError("force_q_sparse_block_size must be a positive multiple of q_block_size")
    sparse_map = _aggregate_q_block_map(
        block_map,
        q_sparse_block_size=q_sparse_block_size,
        q_block_size=q_block_size,
    )

    def from_maps(full_map: torch.Tensor, mask_map: torch.Tensor) -> object:
        full_block_idx, full_block_cnt = _map_to_index(full_map.contiguous())
        mask_block_idx, mask_block_cnt = _map_to_index(mask_map.contiguous())
        return BlockSparseTensorsTorch(
            full_block_cnt=full_block_cnt.to(torch.int32).contiguous(),
            full_block_idx=full_block_idx.to(torch.int32).contiguous(),
            mask_block_cnt=mask_block_cnt.to(torch.int32).contiguous(),
            mask_block_idx=mask_block_idx.to(torch.int32).contiguous(),
            block_size=(q_sparse_block_size, kv_block_size),
        )

    fuse_forward = (need_forward and 0 < sparse_map.shape[-1] <= 4096
            and variable_block_sizes.ndim == 1
            and variable_block_sizes.numel() == sparse_map.shape[-1]
            and variable_block_sizes.dtype == torch.int32
            and variable_block_sizes.device == sparse_map.device)
    if need_backward or not fuse_forward:
        kv_full = (variable_block_sizes == kv_block_size).view(1, 1, 1, -1)
        kv_partial = ((variable_block_sizes > 0) & (variable_block_sizes < kv_block_size)).view(1, 1, 1, -1)
    if fuse_forward:
        from fastvideo_kernel.triton_kernels.index import map_to_classified_indices
        full_idx, full_count, mask_idx, mask_count = map_to_classified_indices(
            sparse_map, variable_block_sizes, kv_block_size,
        )
        forward_sparse_tensors = BlockSparseTensorsTorch(
            full_block_idx=full_idx, full_block_cnt=full_count,
            mask_block_idx=mask_idx, mask_block_cnt=mask_count,
            block_size=(q_sparse_block_size, kv_block_size),
        )
    else:
        forward_sparse_tensors = from_maps(
            sparse_map & kv_full,
            sparse_map & kv_partial,
        ) if need_forward else None

    if not need_backward:
        return forward_sparse_tensors, None

    # FA4 backward is KV-owned: for each physical KV tile, list the sparse
    # query tiles that selected it. Full and partial KV tiles stay separate
    # so the token-level validity mask only runs for padded tiles.
    # Validity is constant across each KV-owned row: only one list is active.
    shared_idx, shared_count = _map_to_index(sparse_map.transpose(2, 3))
    backward_sparse_tensors = BlockSparseTensorsTorch(
        full_block_cnt=shared_count * kv_full.reshape(1, 1, -1),
        full_block_idx=shared_idx,
        mask_block_cnt=shared_count * kv_partial.reshape(1, 1, -1),
        mask_block_idx=shared_idx,
        block_size=(q_sparse_block_size, kv_block_size),
    )
    return forward_sparse_tensors, backward_sparse_tensors


def _cute_attention_q128_forward(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    need_backward: bool,
    alias_guard=None,
) -> Tuple[torch.Tensor, torch.Tensor, object | None]:
    """Run FA4 with one physical Q stage per logical VSA-128 block."""
    _, _, flash_attn_fwd, _ = _load_fa4_cute()
    forward_sparse_tensors, backward_sparse_tensors = _build_sparse_tensors(
        block_map,
        variable_block_sizes,
        q_len=q_bshd.shape[1],
        q_block_size=_FA4_Q_BLOCK_SIZE,
        kv_block_size=_FA4_Q_BLOCK_SIZE,
        need_backward=need_backward,
        force_q_sparse_block_size=_FA4_Q_BLOCK_SIZE,
    )
    mask_mod = _build_vbs_fwd_mask_mod(q_bshd, _FA4_Q_BLOCK_SIZE, need_backward=need_backward)
    if (need_backward and q_bshd.dtype == torch.bfloat16 and q_bshd.shape[-1] in (64, 128)
            and torch.cuda.get_device_capability(q_bshd.device)[0] == 10):
        mask_mod = _build_vbs_vector_mask_mod(_FA4_Q_BLOCK_SIZE)
    out, lse = flash_attn_fwd(
        q_bshd,
        k_bshd,
        v_bshd,
        tile_mn=(_FA4_Q_BLOCK_SIZE, _FA4_Q_BLOCK_SIZE),
        max_seqlen_q=_SingleQStageLength(q_bshd.shape[1]),
        mask_mod=mask_mod,
        block_sparse_tensors=forward_sparse_tensors,
        aux_tensors=[variable_block_sizes],
        causal=False,
        return_lse=True,
        **_alias_guard_kwargs(flash_attn_fwd, alias_guard),
    )[:2]
    return out, lse, backward_sparse_tensors


class _CuteAttentionQ128(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_bshd, k_bshd, v_bshd, block_map, variable_block_sizes, alias_guard=None):
        out, lse, backward_sparse_tensors = _cute_attention_q128_forward(
            q_bshd,
            k_bshd,
            v_bshd,
            block_map,
            variable_block_sizes,
            need_backward=True,
            alias_guard=alias_guard,
        )
        ctx.save_for_backward(q_bshd, k_bshd, v_bshd, out, lse, variable_block_sizes)
        ctx.backward_sparse_tensors = backward_sparse_tensors
        ctx.mark_non_differentiable(lse)
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(ctx, grad_out, grad_lse):
        del grad_lse
        q_bshd, k_bshd, v_bshd, out, lse, variable_block_sizes = ctx.saved_tensors
        if grad_out is None:
            grad_out = torch.zeros_like(out)
        _, _, _, flash_attn_bwd = _load_fa4_cute()
        dq, dk, dv = flash_attn_bwd(
            q_bshd,
            k_bshd,
            v_bshd,
            out,
            grad_out,
            lse,
            softmax_scale=q_bshd.shape[-1]**-0.5,
            mask_mod=_build_vbs_mask_mod(_FA4_Q_BLOCK_SIZE),
            aux_tensors=[variable_block_sizes],
            block_sparse_tensors=ctx.backward_sparse_tensors,
        )
        return dq, dk, dv, None, None, None


def _cute_attention_q128(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    alias_guard=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    need_backward = torch.is_grad_enabled() and any(t.requires_grad for t in (q_bshd, k_bshd, v_bshd))
    if need_backward:
        return _CuteAttentionQ128.apply(q_bshd, k_bshd, v_bshd, block_map, variable_block_sizes, alias_guard)
    out, lse, _ = _cute_attention_q128_forward(
        q_bshd,
        k_bshd,
        v_bshd,
        block_map,
        variable_block_sizes,
        need_backward=False,
        alias_guard=alias_guard,
    )
    return out, lse


class _CuteAttentionQ256Training(torch.autograd.Function):
    """Vectorize the forward mask and classify backward KV tiles at 128 tokens."""

    @staticmethod
    def forward(ctx, q, k, v, block_map, sizes, alias_guard=None):
        _, _, flash_attn_fwd, _ = _load_fa4_cute()
        forward_sparse, _ = _build_sparse_tensors(
            block_map, sizes, q_len=q.shape[1], q_block_size=256,
            kv_block_size=256, need_backward=False,
        )
        # A partially filled logical block can contain a full physical KV tile.
        # Classifying its children avoids masking that full tile in backward.
        child_sizes = torch.stack((sizes.clamp(0, 128), (sizes - 128).clamp(0, 128)), -1).flatten()
        _, backward_sparse = _build_sparse_tensors(
            block_map.repeat_interleave(2, -1), child_sizes,
            q_len=q.shape[1], q_block_size=256, kv_block_size=128,
            need_backward=True, need_forward=False, force_q_sparse_block_size=256,
        )
        out, lse = flash_attn_fwd(
            q, k, v, mask_mod=_build_vbs_vector_mask_mod(256),
            aux_tensors=[sizes], block_sparse_tensors=forward_sparse, return_lse=True,
            **_alias_guard_kwargs(flash_attn_fwd, alias_guard),
        )[:2]
        ctx.save_for_backward(q, k, v, out, lse, child_sizes)
        ctx.backward_sparse_tensors = backward_sparse
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, out, lse, sizes = ctx.saved_tensors
        if dout is None:
            dout = torch.zeros_like(out)
        _, _, _, flash_attn_bwd = _load_fa4_cute()
        dq, dk, dv = flash_attn_bwd(
            q, k, v, out, dout, lse,
            mask_mod=_build_vbs_mask_mod(128), aux_tensors=[sizes],
            block_sparse_tensors=ctx.backward_sparse_tensors, dlse=dlse,
        )
        return dq, dk, dv, None, None, None


def _cute_attention(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    pack_tails: bool | None = None,
    qkv: torch.Tensor | None = None,
    alias_guard: bool | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run FA4's autograd-enabled block-sparse attention with BSHD inputs.

    Native-256 training shares the public wrapper's single dispatch (``vsa256_ops.training_eligible`` /
    ``training_attention``); query-padding pruning needs the trusted-map proof only the public wrapper carries.
    Keyword-only training options: ``pack_tails`` None follows FASTVIDEO_VSA_PACK_TAILS (default on); a bool forces it.
    ``qkv``: optional contiguous ``[3B, S, H, D]`` tensor whose dim-0 chunks are q/k/v; it then becomes the single
    autograd input (one fused gradient allocation).
    ``alias_guard`` (None = FA4 default) reaches every forward launch below (training, no-grad and VC).
    """
    if os.environ.get("FASTVIDEO_VSA_VC", "0") == "1":
        return _vc_sparse_attention(q_bshd, k_bshd, v_bshd, block_map, variable_block_sizes, alias_guard)
    _, flash_attn_func, _, _ = _load_fa4_cute()
    q_block_size = q_bshd.shape[1] // block_map.shape[2]
    kv_block_size = k_bshd.shape[1] // block_map.shape[3]
    if q_block_size == kv_block_size == _FA4_Q_BLOCK_SIZE:
        return _cute_attention_q128(q_bshd, k_bshd, v_bshd, block_map, variable_block_sizes, alias_guard)
    need_backward = torch.is_grad_enabled() and any(t.requires_grad for t in (q_bshd, k_bshd, v_bshd))
    if (need_backward and q_bshd.shape[1] == block_map.shape[2] * 256
            and k_bshd.shape[1] == block_map.shape[3] * 256
            and q_bshd.dtype == torch.bfloat16 and q_bshd.shape[-1] in (64, 128)
            and torch.cuda.get_device_capability(q_bshd.device)[0] == 10):
        from fastvideo_kernel import vsa256_ops
        if vsa256_ops.training_eligible(q_bshd, k_bshd, v_bshd, block_map):
            return vsa256_ops.training_attention(q_bshd, k_bshd, v_bshd, block_map, variable_block_sizes,
                                                 pack_tails=pack_tails, qkv=qkv, alias_guard=alias_guard)
        # Outside the op's validated domain (same predicate as the public wrapper): plain Q256 training autograd.
        return _CuteAttentionQ256Training.apply(q_bshd, k_bshd, v_bshd, block_map, variable_block_sizes, alias_guard)
    forward_sparse_tensors, backward_sparse_tensors = _build_sparse_tensors(
        block_map,
        variable_block_sizes,
        q_len=q_bshd.shape[1],
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        need_backward=need_backward,
    )
    return flash_attn_func(
        q_bshd,
        k_bshd,
        v_bshd,
        mask_mod=_build_vbs_fwd_mask_mod(q_bshd, kv_block_size, need_backward=need_backward),
        aux_tensors=[variable_block_sizes],
        block_sparse_tensors=forward_sparse_tensors,
        block_sparse_tensors_bwd=backward_sparse_tensors,
        return_lse=True,
        **_alias_guard_kwargs(flash_attn_func, alias_guard),
    )


@functools.lru_cache(maxsize=8)
def _load_vc_module(name: str, root: str | None):
    module = importlib.import_module(f"flash_attn.cute.{name}")
    if not root or Path(module.__file__).resolve().parent != Path(root).resolve() / "flash_attn" / "cute":
        raise RuntimeError("Set FASTVIDEO_VSA_VC_ROOT to the imported VC-enabled FA4 checkout")
    return module


def prepare_vsa_vc_fwd_bshd(q, k, v, source_map, sizes, block_size, query_map, query_tokens, query_offset):
    return _load_vc_module("vc_vsa_preprocess", os.environ.get("FASTVIDEO_VSA_VC_ROOT")).prepare_vsa(
        q, k, v, source_map, sizes, block_size,
        padded_to_query=query_map, query_tokens=query_tokens, query_offset=query_offset,
    )


def _validate_vc_prepared(p, block_map, variable_block_sizes):
    q, k, v = (p[name] for name in ("q", "k", "v"))
    if torch.is_grad_enabled() and any(t.requires_grad for t in p.values()):
        raise ValueError("VSA VC attention is inference-only")
    if block_map.ndim == 3:
        block_map = block_map.unsqueeze(0)
    if (q.ndim != 4 or k.shape != v.shape or q.shape[0] != k.shape[0]
            or q.shape[2:] != k.shape[2:] or block_map.ndim != 4
            or any(t.device != q.device or t.dtype != q.dtype for t in (k, v))
            or min(block_map.shape) <= 0 or block_map.shape[:2] != (q.shape[0], q.shape[2])):
        raise ValueError("prepared Q/K/V and block_map batch/head dimensions must agree")
    if q.shape[1] % block_map.shape[2] or k.shape[1] % block_map.shape[3]:
        raise ValueError("prepared Q/K lengths must be exact multiples of their block counts")
    if (block_map.device != q.device or block_map.dtype != torch.bool
            or variable_block_sizes.device != q.device or variable_block_sizes.dtype != torch.int32
            or variable_block_sizes.shape != (block_map.shape[3],)):
        raise ValueError("block_map must be bool and KV sizes must be an int32 vector on the Q device")
    q_block_size = q.shape[1] // block_map.shape[2]
    kv_block_size = k.shape[1] // block_map.shape[3]
    if q_block_size not in (128, 256) or kv_block_size not in (128, 256):
        raise ValueError("VSA VC attention requires 128- or 256-token logical blocks")
    return block_map, q_block_size, kv_block_size


def _vc_physical_sizes(sizes, block_size):
    """Split logical 256-token parents into adjacent physical 128-token children."""
    if block_size == 128:
        return sizes
    return torch.stack((sizes.clamp(0, 128), (sizes - 128).clamp(0, 128)), dim=-1).flatten()


def _vc_sparse_tensors(block_map, sizes, q_len, q_block_size, kv_block_size):
    if kv_block_size == 256:
        block_map = block_map.repeat_interleave(2, -1)
        sizes = _vc_physical_sizes(sizes, kv_block_size)
    sparse, _ = _build_sparse_tensors(
        block_map, sizes, q_len=q_len, q_block_size=q_block_size,
        kv_block_size=128, need_backward=False, force_q_sparse_block_size=q_block_size,
    )
    return sparse, sizes, 128


@functools.lru_cache(maxsize=4)
def _supports_vc_vbs128(forward):
    import inspect
    try:
        return "vc_vbs128" in inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False


@functools.lru_cache(maxsize=8)
def _supports_alias_guard(forward):
    import inspect
    try:
        return "alias_guard" in inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False


def _alias_guard_kwargs(forward, alias_guard):
    """FA4 persistent-grid alias-guard hint: None (FA4 default), a bool, or a 0-d CPU bool tensor.

    The tensor form is a compiled-graph input, so the hint never specializes a graph. Omitted when unset or when the
    provider predates the hint.
    """
    if alias_guard is None or not _supports_alias_guard(forward):
        return {}
    return {"alias_guard": bool(alias_guard)}


def block_sparse_attn_vc_prepared_fwd_bshd(p, block_map, variable_block_sizes, *, return_lse=True, alias_guard=None):
    # LSE is centered/quantized auxiliary output, not a BF16 partition-merge weight.
    block_map, q_block_size, kv_block_size = _validate_vc_prepared(p, block_map, variable_block_sizes)
    sparse, variable_block_sizes, kv_block_size = _vc_sparse_tensors(
        block_map, variable_block_sizes, p["q"].shape[1], q_block_size, kv_block_size,
    )
    return _vc_prepared_sparse(p, sparse, variable_block_sizes, q_block_size, kv_block_size, return_lse, alias_guard)


def _vc_prepared_sparse(p, sparse, variable_block_sizes, q_block_size, kv_block_size, return_lse, alias_guard=None):
    interface = _load_vc_module("interface", os.environ.get("FASTVIDEO_VSA_VC_ROOT"))
    q, k, v = (p[name] for name in ("q", "k", "v"))
    compact = (q_block_size == 128 and kv_block_size == 128 and q.shape[-1] == v.shape[-1] == 128
               and q.shape[-2] == k.shape[-2] == v.shape[-2]
               and torch.cuda.get_device_capability(q.device) == (10, 3)
               and _supports_vc_vbs128(interface._flash_attn_fwd))
    mask_options = ({"vc_vbs128": True} if compact else {"mask_mod": _build_vbs_fwd_mask_mod(q, kv_block_size)})
    return interface._flash_attn_fwd(
        q, k, v, q_descale=p["qs"], k_descale=p["ks"], vc_vscale=p["vs"], vc_expcast=True,
        tile_mn=(_FA4_Q_BLOCK_SIZE, _FA4_Q_BLOCK_SIZE),
        max_seqlen_q=_SingleQStageLength(q.shape[1]) if q_block_size == 128 else q.shape[1],
        block_sparse_tensors=sparse, aux_tensors=[variable_block_sizes], return_lse=return_lse,
        **mask_options, **_alias_guard_kwargs(interface._flash_attn_fwd, alias_guard),
    )[:2]



def block_sparse_attn_vc_routes_fwd_bshd(
    p, selected, variable_block_sizes, block_size, prefix, document_start=0, *, return_lse=True, alias_guard=None,
):
    """Consume unique top-k parent IDs directly, without a dense block map.

    selected is int64 [B,H,Q_blocks,K], using document-global block IDs.
    Each row must contain unique IDs from this document's non-prefix blocks;
    callers own value validation. Prefix blocks are included automatically.
    Physical-child classification and ascending full/partial traversal match the map API.
    LSE is quantized auxiliary state, not a BF16 partition-merge weight.
    """
    q, k, v = (p[name] for name in ("q", "k", "v"))
    if torch.is_grad_enabled() and any(t.requires_grad for t in p.values()):
        raise ValueError("VSA VC attention is inference-only")
    if (q.ndim != 4 or k.shape != v.shape or q.shape[0] != k.shape[0]
            or q.shape[2:] != k.shape[2:] or selected.ndim != 4
            or selected.shape[:2] != (q.shape[0], q.shape[2])
            or q.shape[1] != selected.shape[2] * block_size
            or k.shape[1] != variable_block_sizes.numel() * block_size
            or selected.device != q.device
            or any(t.device != q.device or t.dtype != q.dtype for t in (k, v))):
        raise ValueError("prepared Q/K/V lengths, devices and route dimensions must agree")
    native = _load_vc_module("vc_vsa_preprocess", os.environ.get("FASTVIDEO_VSA_VC_ROOT"))
    parents = variable_block_sizes.numel()
    # Preserve provider validations that changing units would otherwise hide.
    physical_route = (
        type(block_size) is int and block_size == 256
        and type(prefix) is int and type(document_start) is int
        and selected.dtype == torch.int64 and selected.is_cuda
        and variable_block_sizes.ndim == 1 and variable_block_sizes.dtype == torch.int32
        and variable_block_sizes.device == selected.device
        and 0 <= document_start <= 2**31 - 1 - parents
    )
    if physical_route:
        physical_sizes = _vc_physical_sizes(variable_block_sizes, block_size)
        local = (selected - document_start).clamp(-1, parents)
        child0 = local * 2
        children = torch.stack((child0, child0 + 1), dim=-1).flatten(-2)
        full_idx, full_cnt, mask_idx, mask_cnt = native.prepare_vsa_routes(
            children, physical_sizes, 128, 2 * prefix, 0,
        )
    else:
        full_idx, full_cnt, mask_idx, mask_cnt = native.prepare_vsa_routes(
            selected, variable_block_sizes, block_size, prefix, document_start,
        )
        physical_sizes = _vc_physical_sizes(variable_block_sizes, block_size)
    sparse_type, _, _, _ = _load_fa4_cute()
    sparse = sparse_type(full_block_idx=full_idx, full_block_cnt=full_cnt,
                        mask_block_idx=mask_idx, mask_block_cnt=mask_cnt,
                        block_size=(block_size, 128))
    return _vc_prepared_sparse(p, sparse, physical_sizes, block_size, 128, return_lse, alias_guard)


def _vc_sparse_attention(q, k, v, block_map, variable_block_sizes, alias_guard=None):
    """Opt-in FP8/ExpCast self-attention; retain VSA routing and padded-KV masks.

    No V-Smooth or token regrouping: those would require sparse mean-restoration
    support. Coarse attention and top-k selection remain in their original dtype.
    FASTVIDEO_VSA_VC_ROOT must identify the already imported VC-enabled FA4 checkout.
    Returned LSE describes centered, quantized K and ExpCast normalization; it is
    auxiliary only and must not be used to merge dense BF16 attention partitions.
    """
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise ValueError("VSA VC attention is inference-only")
    if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:] or q.shape[1] > k.shape[1]:
        raise ValueError("VSA VC attention requires matching batch/heads/dim, k/v shapes, and Q length <= KV length")
    vc_preprocess = _load_vc_module("vc_preprocess", os.environ.get("FASTVIDEO_VSA_VC_ROOT"))
    # Native preparation shares a Q/K token count. Zero Q padding leaves its
    # maximum and all K/V statistics unchanged; only real queries reach attention.
    q_padded = q if q.shape[1] == k.shape[1] else torch.nn.functional.pad(
        q, (0, 0, 0, 0, 0, k.shape[1] - q.shape[1]))
    p = vc_preprocess.prepare(q_padded.contiguous(), k.contiguous(), v.contiguous(), smooth=False, bshd=True)
    p["q"] = p["q"][:, :q.shape[1]]
    out, lse = block_sparse_attn_vc_prepared_fwd_bshd(p, block_map, variable_block_sizes, alias_guard=alias_guard)
    return out.to(q.dtype), lse


def block_sparse_attn_cute_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    alias_guard=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Autograd-enabled CuTe block-sparse attention for [B, H, S, D].

    ``alias_guard``: FA4 persistent-grid alias-guard hint (keyword-only). None = the provider default; True / False
    (or a 0-d CPU bool tensor, preferred under torch.compile) force it. H3 callers pass True iff the document has dense
    prefix tiles (``MiniMaxH3VSAMetadata.alias_guard_hint``). Ignored by providers that predate the hint.
    """
    if block_map.dim() == 3:
        block_map = block_map.unsqueeze(0)

    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()
    out_bshd, lse = _cute_attention(
        q_bshd,
        k_bshd,
        v_bshd,
        block_map,
        variable_block_sizes,
        alias_guard=alias_guard,
    )
    out = out_bshd.transpose(1, 2).contiguous()
    # FA4 already returns lse as [B, H, S], matching the Triton path's aux
    # contract, so it needs no transpose. Detach before any further op: the
    # value is informational and callers never backprop through it.
    return out, lse.detach()


def block_sparse_attn_cute_fwd_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    pack_tails: bool | None = None,
    qkv: torch.Tensor | None = None,
    alias_guard=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Autograd-enabled CuTe block-sparse attention for [B, S, H, D].

    ``alias_guard``: FA4 persistent-grid alias-guard hint (keyword-only). None = the provider default; True / False
    (or a 0-d CPU bool tensor, preferred under torch.compile) force it. H3 callers pass True iff the document has dense
    prefix tiles (``MiniMaxH3VSAMetadata.alias_guard_hint``). Ignored by providers that predate the hint.
    """
    if block_map.dim() == 3:
        block_map = block_map.unsqueeze(0)

    out, lse = _cute_attention(
        q,
        k,
        v,
        block_map,
        variable_block_sizes,
        pack_tails=pack_tails,
        qkv=qkv,
        alias_guard=alias_guard,
    )
    # lse is [B, H, S] regardless of the q/k/v layout; see above.
    return out, lse.detach()


def block_sparse_attn_vc_fwd_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    alias_guard=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit inference-only VC entrypoint, independent of backend environment flags (``alias_guard`` as
    block_sparse_attn_cute_fwd_bshd)."""
    if block_map.dim() == 3:
        block_map = block_map.unsqueeze(0)
    out, lse = _vc_sparse_attention(q, k, v, block_map, variable_block_sizes, alias_guard)
    return out, lse.detach()


# Compiled no-grad VC route (H3 fused producer): opaque ops whose bodies are the eager route's calls, used in eager AND
# compiled mode, so eager == compiled by construction. The ctypes/NVRTC producer and the CuTe JIT stay behind the op
# boundary; fakes derive every output shape from input shapes and ``tile`` (the metadata's tile size, a trace-time
# constant). Only tile 256 is implemented; tile 128 raises until the ruling-84 H3 API (one entry, tile=128|256) lands.
# JIT placement. What compiles inside the op bodies on first use (census over the 46 s085k32 docs x 2 layouts):
#  - the NVRTC producer module: once per device;
#  - the FA4 CuTe forward: one compile_cache key for every doc (fixed Q256/KV128 block sparsity, FP8, D, vector mask);
#  - the Triton _classified_map_to_index_kernel (map_to_classified_indices): one compile per (KV_BLOCK, BLOCK) bucket
#    (classified_index_block of the 128-token KV column count; 3 buckets over the spec corpus).
# The attention key below covers the last two (conservatively adding H, D and the device) plus the alias_guard hint,
# which is in FA4's compile_key. CUDA-graph capture of a call whose key was never run eagerly raises before any launch,
# instead of compiling inside the capture. Dynamo tracing sees symbolic shapes, so the fakes can only require the
# device to be warm.
_VC_WARMED: set = set()  # devices whose producer module is loaded
_VC_WARMED_ATTN: set = set()  # attention JIT keys run outside capture


def _vc_warm_key(x: torch.Tensor) -> str:
    return str(x.device)


def _vc_require_tile(tile: int) -> None:
    if tile == 128:
        raise NotImplementedError("VC compiled route: tile 128 lands with the ruling-84 API")
    if tile != 256:
        raise ValueError(f"VC compiled route: tile must be 128 or 256, got {tile}")


def _vc_attn_key(q8: torch.Tensor, block_map: torch.Tensor, alias_guard) -> tuple:
    # alias_guard is part of FA4's compile_key, so each hint value (None/False/True) is its own CuTe JIT entry.
    from fastvideo_kernel.triton_kernels.index import classified_index_block
    hint = None if alias_guard is None else bool(alias_guard)
    # The kernel lists are built on the 128-token KV children: two per 256-token tile.
    return (str(q8.device), int(q8.shape[2]), int(q8.shape[3]), classified_index_block(2 * int(block_map.shape[-1])), hint)


def _vc_cold(what: str) -> RuntimeError:
    return RuntimeError(
        f"VSA-H3 compiled VC route: captured before any eager VC forward for this {what}; run one eager "
        "forward per geometry first so the NVRTC producer, CuTe and Triton kernels are built outside "
        "the graph")


def _vc_require_warm(x: torch.Tensor, capturing: bool) -> None:
    if capturing and _vc_warm_key(x) not in _VC_WARMED:
        raise _vc_cold("device")


@torch.library.custom_op("fastvideo_kernel::vc_h3_prepare_fused", mutates_args=())
def vc_h3_prepare_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    untile: torch.Tensor,
    sizes: torch.Tensor,
    tile: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Packed rows -> padded FP8 Q/K/V, descales, FP32 tile pools (cold producer, ``tile``-token tiles, query offset 0)."""
    _vc_require_tile(tile)
    _vc_require_warm(q, torch.cuda.is_current_stream_capturing())
    padded = sizes.numel() * tile
    padded_to_original = torch.full((padded, ), -1, dtype=torch.int64, device=q.device)
    padded_to_original[untile] = torch.arange(untile.numel(), dtype=torch.int64, device=q.device)
    padded_to_query = torch.arange(padded, dtype=torch.int64, device=q.device)
    p, pools = prepare_vsa_vc_fwd_bshd(q.contiguous(), k.contiguous(), v.contiguous(), padded_to_original, sizes, tile,
                                       padded_to_query, padded, 0)
    if not torch.cuda.is_current_stream_capturing():
        _VC_WARMED.add(_vc_warm_key(q))
    return p["q"], p["k"], p["v"], p["qs"], p["ks"], p["vs"], *pools


@vc_h3_prepare_fused.register_fake
def _vc_h3_prepare_fused_fake(q, k, v, untile, sizes, tile):
    _vc_require_tile(tile)
    _vc_require_warm(q, True)
    b, _, h, d = q.shape
    n = sizes.shape[0]
    fp8 = [q.new_empty((b, n * tile, h, d), dtype=torch.float8_e4m3fn) for _ in range(3)]
    scales = [q.new_empty(shape, dtype=torch.float32) for shape in ((b, h), (b, h), (b, h, d))]
    pools = [q.new_empty((b, h, n, d), dtype=torch.float32) for _ in range(3)]
    return (*fp8, *scales, *pools)


@torch.library.custom_op("fastvideo_kernel::vc_h3_attn_prepared", mutates_args=())
def vc_h3_attn_prepared(q8: torch.Tensor, k8: torch.Tensor, v8: torch.Tensor, qs: torch.Tensor, ks: torch.Tensor,
                        vs: torch.Tensor, block_map: torch.Tensor, sizes: torch.Tensor, tile: int,
                        alias_guard: torch.Tensor | None = None) -> torch.Tensor:
    """VC attention on prepared FP8 Q/K/V with a [B, H, n, n] tile map -> BF16 [B, n*tile, H, D] (padded tile layout).

    ``alias_guard``: FA4 alias-guard hint as a 0-d CPU bool tensor (H3 ``alias_guard_hint``), a graph input like
    vsa256_fwd's, so a per-document value never specializes the compiled graph.
    """
    _vc_require_tile(tile)
    from fastvideo_kernel.block_sparse_attn_256 import _expand_mask_and_sizes_256_to_128
    key = _vc_attn_key(q8, block_map, alias_guard)
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and key not in _VC_WARMED_ATTN:
        raise _vc_cold(f"attention key {key}")
    mask_128, sizes_128 = _expand_mask_and_sizes_256_to_128(block_map, sizes)
    p = {"q": q8, "k": k8, "v": v8, "qs": qs, "ks": ks, "vs": vs}
    out = block_sparse_attn_vc_prepared_fwd_bshd(p, mask_128, sizes_128, alias_guard=alias_guard)[0].to(torch.bfloat16)
    if not capturing:
        _VC_WARMED_ATTN.add(key)
    return out


@vc_h3_attn_prepared.register_fake
def _vc_h3_attn_prepared_fake(q8, k8, v8, qs, ks, vs, block_map, sizes, tile, alias_guard=None):
    _vc_require_tile(tile)
    _vc_require_warm(qs, True)
    return q8.new_empty(q8.shape, dtype=torch.bfloat16)
