"""Autograd-enabled CuTe-DSL block-sparse attention.

Thin wrapper around ``flash_attn.cute`` that adapts
VSA's `(block_map, variable_block_sizes)` inputs into FA4's
``BlockSparseTensorsTorch`` representation and the per-KV-block validity mask.
The forward pass uses FA4's CuTe kernel and the backward pass uses FA4's native
block-sparse backward with the transposed Q-by-KV index.

Both [B, H, S, D] (BHSD) and [B, S, H, D] (BSHD) entrypoints are provided.
The BSHD variant is preferred from VSA-256 callers to avoid layout
round-trips on the hot path.

The FA4 CuTe block-sparse kernel (``flash_attn.cute`` with
``block_sparsity``) is an *optional* dependency: it is imported lazily and
only exercised when the VSA-256 CuTe fastpath is explicitly selected
(``FASTVIDEO_VSA_CUTEDSL=1``). The default VSA-256 path is Triton and does
not require it. Also needs ``nvidia-cutlass-dsl`` and ``quack-kernels``.
"""

from __future__ import annotations

import functools
from typing import Tuple

import torch

_FA4_IMPORT_HINT = (
    "VSA-256 CuTe fastpath requires a FlashAttention-4 CuTe build that "
    "provides `flash_attn.cute` with block-sparsity support (plus "
    "`nvidia-cutlass-dsl` and `quack-kernels`). This is an optional "
    "dependency; the default VSA-256 path is Triton. Install the FA4 CuTe "
    "build and set FASTVIDEO_VSA_CUTEDSL=1 to enable the CuTe fastpath."
)


def _load_fa4_cute():
    """Lazily import the optional FA4 CuTe block-sparse symbols.

    Raising a clear, actionable error here keeps the optional FA4 CuTe
    build from being a hard import-time dependency of this module (and of
    the default Triton VSA-256 path).
    """
    try:
        from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
        from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(_FA4_IMPORT_HINT) from exc
    return BlockSparseTensorsTorch, _flash_attn_fwd, _flash_attn_bwd


# Q-side tile size; kv_block_size comes from the caller's VSA logical KV block.
_M_BLOCK_SIZE_DEFAULT = 128


def _map_to_index(block_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if block_map.dim() == 3:
        block_map = block_map.unsqueeze(0)
    if block_map.dim() != 4:
        raise ValueError(
            f"block_map must be [B,H,Q,KV] (or [H,Q,KV]), "
            f"got shape={tuple(block_map.shape)}"
        )
    if block_map.dtype != torch.bool:
        block_map = block_map.to(torch.bool)
    if not block_map.is_cuda:
        raise RuntimeError("block_map must be a CUDA tensor.")
    from fastvideo_kernel.triton_kernels.index import (
        map_to_index as triton_map_to_index,
    )

    return triton_map_to_index(block_map)


def _choose_q_sparse_block_size(q_len: int, m_block_size: int = _M_BLOCK_SIZE_DEFAULT) -> int:
    # FA4 supports a doubled Q sparsity granularity on sm_100+ when q_len > m_block_size.
    major, _ = torch.cuda.get_device_capability()
    if major >= 10 and q_len > m_block_size:
        return 2 * m_block_size
    return m_block_size


def _aggregate_q_block_map(
    block_map: torch.Tensor,
    q_sparse_block_size: int,
    q_block_size: int,
) -> torch.Tensor:
    factor = q_sparse_block_size // q_block_size
    if factor <= 0 or q_sparse_block_size % q_block_size != 0:
        raise ValueError(
            f"q_sparse_block_size must be a positive multiple of "
            f"q_block_size ({q_block_size}), got {q_sparse_block_size}"
        )
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

    return _vbs_mask_mod


def _make_sparse_tensors(
    mask_block_cnt: torch.Tensor,
    mask_block_idx: torch.Tensor,
    full_block_cnt: torch.Tensor,
    full_block_idx: torch.Tensor,
    q_sparse_block_size: int,
    kv_block_size: int,
):
    BlockSparseTensorsTorch, _, _ = _load_fa4_cute()
    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        block_size=(q_sparse_block_size, kv_block_size),
    )


@torch.library.custom_op(
    "fastvideo_kernel::block_sparse_attn_cute",
    mutates_args=(),
    device_types="cuda",
)
def _block_sparse_attn_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_block_cnt: torch.Tensor,
    mask_block_idx: torch.Tensor,
    full_block_cnt: torch.Tensor,
    full_block_idx: torch.Tensor,
    q_mask_block_cnt: torch.Tensor,
    q_mask_block_idx: torch.Tensor,
    q_full_block_cnt: torch.Tensor,
    q_full_block_idx: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    q_sparse_block_size: int,
    kv_block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run FA4 CuTe forward with BSHD tensors and sparse metadata."""
    _, _flash_attn_fwd, _ = _load_fa4_cute()
    sparse_tensors = _make_sparse_tensors(
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
        q_sparse_block_size,
        kv_block_size,
    )
    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        tile_mn=(_M_BLOCK_SIZE_DEFAULT, kv_block_size),
        mask_mod=_build_vbs_mask_mod(kv_block_size),
        block_sparse_tensors=sparse_tensors,
        aux_tensors=[variable_block_sizes],
        causal=False,
        return_lse=True,
    )[:2]
    return out, lse


@torch.library.register_fake("fastvideo_kernel::block_sparse_attn_cute")
def _block_sparse_attn_cute_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_block_cnt: torch.Tensor,
    mask_block_idx: torch.Tensor,
    full_block_cnt: torch.Tensor,
    full_block_idx: torch.Tensor,
    q_mask_block_cnt: torch.Tensor,
    q_mask_block_idx: torch.Tensor,
    q_full_block_cnt: torch.Tensor,
    q_full_block_idx: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    q_sparse_block_size: int,
    kv_block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del (
        k,
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
        q_mask_block_cnt,
        q_mask_block_idx,
        q_full_block_cnt,
        q_full_block_idx,
        variable_block_sizes,
        q_sparse_block_size,
        kv_block_size,
    )
    out = q.new_empty(q.shape[:-1] + (v.shape[-1],))
    lse = q.new_empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32)
    return out, lse


def _setup_context_cute(ctx, inputs, output) -> None:
    (
        q,
        k,
        v,
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
        q_mask_block_cnt,
        q_mask_block_idx,
        q_full_block_cnt,
        q_full_block_idx,
        variable_block_sizes,
        q_sparse_block_size,
        kv_block_size,
    ) = inputs
    del (
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
    )
    out, lse = output
    ctx.save_for_backward(
        q,
        k,
        v,
        out,
        lse,
        q_mask_block_cnt,
        q_mask_block_idx,
        q_full_block_cnt,
        q_full_block_idx,
        variable_block_sizes,
    )
    ctx.q_sparse_block_size = q_sparse_block_size
    ctx.kv_block_size = kv_block_size


def _backward_cute(ctx, grad_out, grad_lse):
    del grad_lse
    (
        q,
        k,
        v,
        out,
        lse,
        q_mask_block_cnt,
        q_mask_block_idx,
        q_full_block_cnt,
        q_full_block_idx,
        variable_block_sizes,
    ) = ctx.saved_tensors
    if grad_out is None:
        grad_out = torch.zeros_like(out)
    _, _, _flash_attn_bwd = _load_fa4_cute()
    sparse_tensors_bwd = _make_sparse_tensors(
        q_mask_block_cnt,
        q_mask_block_idx,
        q_full_block_cnt,
        q_full_block_idx,
        ctx.q_sparse_block_size,
        ctx.kv_block_size,
    )
    dq, dk, dv = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        grad_out.contiguous(),
        lse,
        softmax_scale=None,
        causal=False,
        softcap=0.0,
        window_size_left=None,
        window_size_right=None,
        deterministic=False,
        mask_mod=_build_vbs_mask_mod(ctx.kv_block_size),
        aux_tensors=[variable_block_sizes],
        block_sparse_tensors=sparse_tensors_bwd,
    )
    return dq, dk, dv, *((None,) * 11)


_block_sparse_attn_cute.register_autograd(
    _backward_cute,
    setup_context=_setup_context_cute,
)


def _cute_forward(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    q_block_size: int,
    kv_block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Internal: FA4 CuTe BSA fwd with BSHD inputs."""
    q_sparse_candidate = _choose_q_sparse_block_size(q_bshd.shape[1])
    q_sparse_block_size = max(
        q_block_size,
        ((q_sparse_candidate + q_block_size - 1) // q_block_size) * q_block_size,
    )
    sparse_map = _aggregate_q_block_map(
        block_map,
        q_sparse_block_size=q_sparse_block_size,
        q_block_size=q_block_size,
    )
    kv_full = (variable_block_sizes == kv_block_size).view(1, 1, 1, -1)
    kv_partial = (
        (variable_block_sizes > 0) & (variable_block_sizes < kv_block_size)
    ).view(1, 1, 1, -1)
    full_map = sparse_map & kv_full
    mask_map = sparse_map & kv_partial

    full_block_idx, full_block_cnt = _map_to_index(full_map)
    mask_block_idx, mask_block_cnt = _map_to_index(mask_map)

    # FA4 backward iterates the transposed sparse map: each KV block stores
    # the query blocks that selected it.
    q_full_block_idx, q_full_block_cnt = _map_to_index(
        full_map.transpose(-2, -1).contiguous()
    )
    q_mask_block_idx, q_mask_block_cnt = _map_to_index(
        mask_map.transpose(-2, -1).contiguous()
    )

    return _block_sparse_attn_cute(
        q_bshd,
        k_bshd,
        v_bshd,
        mask_block_cnt.to(torch.int32).contiguous(),
        mask_block_idx.to(torch.int32).contiguous(),
        full_block_cnt.to(torch.int32).contiguous(),
        full_block_idx.to(torch.int32).contiguous(),
        q_mask_block_cnt.to(torch.int32).contiguous(),
        q_mask_block_idx.to(torch.int32).contiguous(),
        q_full_block_cnt.to(torch.int32).contiguous(),
        q_full_block_idx.to(torch.int32).contiguous(),
        variable_block_sizes.to(torch.int32).contiguous(),
        q_sparse_block_size,
        kv_block_size,
    )


def block_sparse_attn_cute_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Autograd-enabled CuTe block-sparse attention for [B, H, S, D]."""
    if block_map.dim() == 3:
        block_map = block_map.unsqueeze(0)
    q_block_size = q.shape[2] // block_map.shape[2]
    kv_block_size = k.shape[2] // block_map.shape[3]

    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()
    out_bshd, lse_bshd = _cute_forward(
        q_bshd,
        k_bshd,
        v_bshd,
        block_map,
        variable_block_sizes,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
    )
    out = out_bshd.transpose(1, 2).contiguous()
    if lse_bshd is None:
        lse = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        lse = lse_bshd
    return out, lse


def block_sparse_attn_cute_fwd_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_map: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Autograd-enabled CuTe block-sparse attention for [B, S, H, D]."""
    if block_map.dim() == 3:
        block_map = block_map.unsqueeze(0)
    q_block_size = q.shape[1] // block_map.shape[2]
    kv_block_size = k.shape[1] // block_map.shape[3]

    out, lse_bshd = _cute_forward(
        q,
        k,
        v,
        block_map,
        variable_block_sizes,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
    )
    if lse_bshd is None:
        lse = torch.empty(
            (q.shape[0], q.shape[2], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        lse = lse_bshd
    return out, lse
