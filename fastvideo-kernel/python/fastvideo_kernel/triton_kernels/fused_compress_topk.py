"""
Fused Triton kernels for VSA compress (block mean) and topk mask construction.

Replaces the multi-kernel PyTorch pipeline:
  Original compress: .view() -> .float() -> .sum(dim=3) -> / vbs -> .to(bf16)
  Original topk:     torch.topk() -> zeros() -> scatter_()

With single-pass fused kernels:
  fused_block_mean:  read bf16, accumulate fp32, div by vbs, write bf16
  fused_topk_mask:   read scores, find k-th value, write bool mask
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _fused_block_mean_kernel(
    X_ptr,
    Out_ptr,
    VBS_ptr,
    stride_x_b,
    stride_x_h,
    num_heads: tl.constexpr,
    stride_x_seq,
    stride_o_bh,
    stride_o_blk,
    num_blocks,
    BLOCK_ELEMENTS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    EXACT_DIV: tl.constexpr = False,
):
    """Fused block mean: one program computes mean of one block for one (b,h).

    X uses separate batch/head strides, supporting both BHSD and BSHD.
    Out is [B*H, num_blocks, HEAD_DIM] contiguous.
    2D load + parallel tl.sum reduction, accumulates in fp32.
    """
    # Widen before stride products: supported tensors can exceed 2**31 elements.
    block_idx = tl.program_id(0).to(tl.int64)
    bh_idx = tl.program_id(1).to(tl.int64)

    if block_idx >= num_blocks:
        return

    vbs = tl.load(VBS_ptr + block_idx).to(tl.float32)

    x_base = X_ptr + (bh_idx // num_heads) * stride_x_b + (bh_idx % num_heads) * stride_x_h + block_idx * BLOCK_ELEMENTS * stride_x_seq

    row_offsets = tl.arange(0, BLOCK_ELEMENTS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    offsets = row_offsets[:, None] * stride_x_seq + dim_offsets[None, :]
    block_data = tl.load(x_base + offsets).to(tl.float32)
    total = tl.sum(block_data, axis=0)
    acc = tl.div_rn(total, vbs) if EXACT_DIV else total / vbs

    out_base = Out_ptr + bh_idx * stride_o_bh + block_idx * stride_o_blk + dim_offsets
    tl.store(out_base, acc.to(OUTPUT_DTYPE))


@triton.jit
def _fused_block_mean_bwd_kernel(
    GradOut_ptr,
    GradX_ptr,
    VBS_ptr,
    stride_go_bh,
    stride_go_blk,
    stride_gx_b,
    stride_gx_h,
    num_heads: tl.constexpr,
    stride_gx_seq,
    num_blocks,
    BLOCK_ELEMENTS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    EXACT_DIV: tl.constexpr = False,
    FineGrad_ptr=None,
):
    """Backward of block mean: broadcast grad_out / vbs to each token in the block.

    Mirrors the forward kernel: one program per (block, bh).
    GradOut is [B*H, num_blocks, HEAD_DIM].
    GradX  is [B*H, num_blocks*BLOCK_ELEMENTS, HEAD_DIM].
    2D store writes all BLOCK_ELEMENTS rows in parallel.
    """
    block_idx = tl.program_id(0).to(tl.int64)
    bh_idx = tl.program_id(1).to(tl.int64)

    if block_idx >= num_blocks:
        return

    vbs = tl.load(VBS_ptr + block_idx).to(tl.float32)

    dim_offsets = tl.arange(0, HEAD_DIM)
    go_base = GradOut_ptr + bh_idx * stride_go_bh + block_idx * stride_go_blk
    grad = tl.load(go_base + dim_offsets).to(tl.float32)
    grad_val = tl.div_rn(grad, vbs) if EXACT_DIV else grad / vbs

    row_offsets = tl.arange(0, BLOCK_ELEMENTS)
    base = (bh_idx // num_heads) * stride_gx_b + (bh_idx % num_heads) * stride_gx_h + block_idx * BLOCK_ELEMENTS * stride_gx_seq
    gx_base = GradX_ptr + base
    offsets = row_offsets[:, None] * stride_gx_seq + dim_offsets[None, :]
    grad_2d = tl.broadcast_to(grad_val[None, :], [BLOCK_ELEMENTS, HEAD_DIM])
    if FineGrad_ptr is not None:
        # Compression rounds before the ordinary fine/coarse gradient addition.
        grad_2d = grad_2d.to(OUTPUT_DTYPE).to(tl.float32) + tl.load(FineGrad_ptr + base + offsets).to(tl.float32)
    tl.store(gx_base + offsets, grad_2d.to(OUTPUT_DTYPE))


_TORCH_TO_TRITON_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


def _fused_block_mean_bwd(
    grad_output: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    block_elements: int,
) -> torch.Tensor:
    B, H, num_blocks, D = grad_output.shape
    seq_len = num_blocks * block_elements

    grad_x = torch.empty(B, H, seq_len, D, dtype=grad_output.dtype, device=grad_output.device)

    go_flat = grad_output.contiguous().view(B * H, num_blocks, D)
    gx_flat = grad_x.view(B * H, seq_len, D)

    grid = (num_blocks, B * H)

    _fused_block_mean_bwd_kernel[grid](
        go_flat,
        gx_flat,
        variable_block_sizes,
        go_flat.stride(0),
        go_flat.stride(1),
        grad_x.stride(0),
        grad_x.stride(1),
        H,
        grad_x.stride(2),
        num_blocks,
        BLOCK_ELEMENTS=block_elements,
        HEAD_DIM=D,
        OUTPUT_DTYPE=_TORCH_TO_TRITON_DTYPE[grad_output.dtype],
    )

    return grad_x


def _fused_block_mean_fwd(
    x: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    block_elements: int,
) -> torch.Tensor:
    B, H, seq_len, D = x.shape
    num_blocks = seq_len // block_elements
    assert seq_len % block_elements == 0

    if not (block_elements == 128 and x.dtype == torch.bfloat16 and D in (64, 128) and x.stride(-1) == 1
            and x.data_ptr() % 16 == 0 and all(s > 0 and s % 8 == 0 for s in x.stride()[:-1])):
        x = x.contiguous()
    out = torch.empty(B, H, num_blocks, D, dtype=x.dtype, device=x.device)

    out_flat = out.view(B * H, num_blocks, D)

    grid = (num_blocks, B * H)

    _fused_block_mean_kernel[grid](
        x,
        out_flat,
        variable_block_sizes,
        x.stride(0),
        x.stride(1),
        H,
        x.stride(2),
        out_flat.stride(0),
        out_flat.stride(1),
        num_blocks,
        BLOCK_ELEMENTS=block_elements,
        HEAD_DIM=D,
        OUTPUT_DTYPE=_TORCH_TO_TRITON_DTYPE[x.dtype],
    )

    return out


class _FusedBlockMeanAutograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, variable_block_sizes, block_elements):
        ctx.save_for_backward(variable_block_sizes)
        ctx.block_elements = block_elements
        return _fused_block_mean_fwd(x, variable_block_sizes, block_elements)

    @staticmethod
    def backward(ctx, grad_output):
        variable_block_sizes, = ctx.saved_tensors
        block_elements = ctx.block_elements
        return _fused_block_mean_bwd(grad_output, variable_block_sizes, block_elements), None, None


def fused_block_mean(
    x: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    block_elements: int,
) -> torch.Tensor:
    """Compute block-wise mean with fp32 accumulation, fused in one kernel.

    Forward: fused Triton kernel (bf16 read → fp32 accumulate → div → bf16 write).
    Backward: broadcasts grad_output / vbs back to each token position.

    Args:
        x: [B, H, seq_len, D] in bf16
        variable_block_sizes: [num_blocks] number of valid tokens per block
        block_elements: tokens per block (e.g. 64)

    Returns:
        [B, H, num_blocks, D] in bf16
    """
    return _FusedBlockMeanAutograd.apply(x, variable_block_sizes, block_elements)


@triton.jit
def _fused_topk_mask_kernel(
    Scores_ptr,
    Mask_ptr,
    stride_s_bh,
    stride_s_q,
    stride_s_kv,
    stride_m_bh,
    stride_m_q,
    stride_m_kv,
    kv_blocks: tl.constexpr,
    topk: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
):
    """Build topk boolean mask via randomized pivot selection (quickselect-style).

    For each (b,h,q_block) row: find the k-th largest score using iterative
    pivot-based partitioning, then build mask by comparing against threshold.

    Grid: (num_q_blocks, B * H)
    """
    q_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)

    kv_offsets = tl.arange(0, KV_BLOCK_SIZE)
    score_base = Scores_ptr + bh_idx * stride_s_bh + q_idx * stride_s_q
    mask_base = Mask_ptr + bh_idx * stride_m_bh + q_idx * stride_m_q

    valid_mask = kv_offsets < kv_blocks
    scores = tl.load(score_base + kv_offsets * stride_s_kv, mask=valid_mask, other=-float("inf"))
    scores_f32 = scores.to(tl.float32)

    # Binary search for threshold: find value T such that count(scores > T) <= topk
    # and count(scores >= T) >= topk
    # Exclude -inf from lo so the binary search can converge when masked
    # scores are present (mid = (-inf + hi) * 0.5 = -inf would stall).
    finite_mask = valid_mask & (scores_f32 > float("-inf"))
    lo = tl.min(tl.where(finite_mask, scores_f32, float("inf")), axis=0)
    hi = tl.max(tl.where(valid_mask, scores_f32, float("-inf")), axis=0)
    # If all valid scores are -inf, lo > hi; threshold stays at -inf and
    # the tie-breaking logic below selects the first topk positions.
    lo = tl.minimum(lo, hi)

    # 32 iterations of fp32 bisection give range-relative resolution (hi-lo)/2^32.
    # For VSA's softmax-input scores (q_c@k_c/sqrt(d), O(1) magnitude, range < ~10),
    # this resolves to ~2e-9, well below the bf16 ULP at that magnitude (~8e-3),
    # so the threshold converges exactly to the k-th bf16 score value and the
    # > / == comparisons below are exact.
    for _i in range(32):
        mid = (lo + hi) * 0.5
        count_ge = tl.sum(((scores_f32 >= mid) & valid_mask).to(tl.int32), axis=0)
        lo = tl.where(count_ge >= topk, mid, lo)
        hi = tl.where(count_ge >= topk, hi, mid)

    # lo is our threshold: count(scores >= lo) >= topk
    threshold = lo
    above_threshold = scores_f32 > threshold
    at_threshold = scores_f32 == threshold
    n_above = tl.sum(above_threshold.to(tl.int32), axis=0)
    n_needed_at_thresh = topk - n_above

    at_thresh_cumsum = tl.cumsum(at_threshold.to(tl.int32), axis=0)
    at_thresh_selected = at_threshold & (at_thresh_cumsum <= n_needed_at_thresh)

    final_mask = above_threshold | at_thresh_selected

    tl.store(mask_base + kv_offsets * stride_m_kv, final_mask, mask=valid_mask)


# Triton kernel loads the entire kv row into registers via tl.arange(0, KV_BLOCK_SIZE).
# Each row spawns multiple same-sized register arrays (scores_f32, valid_mask,
# above_threshold, at_threshold, cumsum, etc.).  GPU SMs have a fixed register file
# (e.g. 65536 × 32-bit), so the maximum array length per program is bounded.
# 4096 (2^12) is an empirical power-of-2 cap that avoids register spilling on
# mainstream GPUs.  Beyond this the kernel either fails to compile or spills to
# local memory with severe perf regression, so we fall back to torch.topk.
MAX_KV_BLOCK_SIZE = 4096


def _pytorch_topk_mask_fallback(
    scores: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    topk_idx = torch.topk(scores, topk, dim=-1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_idx, True)


def fused_topk_mask(
    scores: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Build topk boolean mask from scores using fused Triton kernel.

    Falls back to PyTorch when kv_blocks exceeds the Triton block size
    limit (MAX_KV_BLOCK_SIZE) to avoid compilation failures.

    Args:
        scores: [B, H, q_blocks, kv_blocks] block-level attention scores
        topk: number of top blocks to select per q-block

    Returns:
        mask: [B, H, q_blocks, kv_blocks] bool tensor with exactly topk True per row
    """
    B, H, q_blocks, kv_blocks = scores.shape
    topk = min(topk, kv_blocks)

    KV_BLOCK_SIZE = triton.next_power_of_2(kv_blocks)
    if KV_BLOCK_SIZE > MAX_KV_BLOCK_SIZE:
        logger.debug(
            "fused_topk_mask: kv_blocks=%d exceeds Triton limit %d, "
            "falling back to PyTorch topk (slower)",
            kv_blocks,
            MAX_KV_BLOCK_SIZE,
        )
        return _pytorch_topk_mask_fallback(scores, topk)

    mask = torch.zeros(B, H, q_blocks, kv_blocks, dtype=torch.bool, device=scores.device)

    scores_flat = scores.contiguous().view(B * H, q_blocks, kv_blocks)
    mask_flat = mask.view(B * H, q_blocks, kv_blocks)

    grid = (q_blocks, B * H)

    _fused_topk_mask_kernel[grid](
        scores_flat,
        mask_flat,
        scores_flat.stride(0),
        scores_flat.stride(1),
        scores_flat.stride(2),
        mask_flat.stride(0),
        mask_flat.stride(1),
        mask_flat.stride(2),
        kv_blocks=kv_blocks,
        topk=topk,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
    )

    return mask


class _FusedBlockMeanBSHD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sizes, block):
        b, seq, h, d = x.shape
        assert x.stride(-1) == 1 and d in (64, 128) and seq % block == 0
        ctx.save_for_backward(sizes)
        ctx.block = block
        # Preserve ATen's reduction order: cancellation can otherwise change
        # discrete top-k routes. Accumulate in FP32 without a full FP32 copy.
        blocks = x.view(b, seq // block, block, h, d)
        # ATen output vec4 requires 8-byte BF16 pointer alignment, including
        # contiguous views with nonzero storage offsets. Other layouts keep
        # the original float() conversion and its reduction order.
        direct_sum = x.is_contiguous() and (
            (x.dtype == torch.bfloat16 and x.data_ptr() % 8 == 0)
            or (x.storage_offset() == 0 and x.data_ptr() % 16 == 0))
        pooled = blocks.sum(dim=2, dtype=torch.float32) if direct_sum else blocks.float().sum(dim=2)
        pooled = (pooled / sizes.view(1, -1, 1, 1)).to(x.dtype)
        return pooled.permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def backward(ctx, grad):
        sizes, = ctx.saved_tensors
        b, h, nb, d = grad.shape
        if torch.is_grad_enabled():
            # Preserve the original expression for higher-order autograd.
            per_block = (grad.float() / sizes.view(1, 1, nb, 1)).to(grad.dtype)
            expanded = per_block.permute(0, 2, 1, 3).unsqueeze(2)
            return expanded.expand(b, nb, ctx.block, h, d).reshape(b, nb * ctx.block, h, d), None, None
        grad = grad.contiguous()
        out = torch.empty(b, nb * ctx.block, h, d, device=grad.device, dtype=grad.dtype)
        _fused_block_mean_bwd_kernel[(nb, b * h)](
            grad, out, sizes, grad.stride(1), grad.stride(2),
            out.stride(0), out.stride(2), h, out.stride(1), nb,
            BLOCK_ELEMENTS=ctx.block, HEAD_DIM=d,
            OUTPUT_DTYPE=_TORCH_TO_TRITON_DTYPE[grad.dtype], EXACT_DIV=True,
        )
        return out, None, None


def fused_block_mean_bshd(x: torch.Tensor, sizes: torch.Tensor, block: int) -> torch.Tensor:
    """Compress BSHD blocks to BHND; aligned contiguous inputs avoid an FP32 copy.

    As in the original compression expression, padding participates in the sum
    and backward broadcasts to every padded slot. The caller supplies zero
    padding and valid divisors. Unsupported layouts use that original expression.
    """
    b, seq, h, d = x.shape
    if (x.is_cuda and x.dtype in _TORCH_TO_TRITON_DTYPE and d in (64, 128)
            and x.stride(-1) == 1 and sizes.device == x.device
            and sizes.dtype in (torch.int32, torch.int64) and block in (128, 256)):
        return _FusedBlockMeanBSHD.apply(x, sizes.reshape(-1).contiguous(), block)
    pooled = x.view(b, seq // block, block, h, d).float().sum(dim=2)
    pooled = (pooled / sizes.view(1, -1, 1, 1)).to(x.dtype)
    return pooled.permute(0, 2, 1, 3).contiguous()


class _ForkBlockMeanBSHD(_FusedBlockMeanBSHD):
    """Native autograd gathers fine and pooled gradients at one node."""

    @staticmethod
    def forward(ctx, x, sizes, block):
        ctx.set_materialize_grads(False)
        pooled = _FusedBlockMeanBSHD.forward(ctx, x, sizes, block)
        return x.view_as(x), pooled

    @staticmethod
    def backward(ctx, fine, coarse):
        if coarse is None:
            return fine, None, None
        if fine is None or torch.is_grad_enabled() or not fine.is_contiguous():
            expanded, _, _ = _FusedBlockMeanBSHD.backward(ctx, coarse)
            return expanded if fine is None else fine + expanded, None, None
        sizes, = ctx.saved_tensors
        coarse = coarse.contiguous()
        b, h, nb, d = coarse.shape
        out = torch.empty_like(fine)
        _fused_block_mean_bwd_kernel[(nb, b * h)](
            coarse, out, sizes, coarse.stride(1), coarse.stride(2),
            out.stride(0), out.stride(2), h, out.stride(1), nb,
            BLOCK_ELEMENTS=ctx.block, HEAD_DIM=d,
            OUTPUT_DTYPE=_TORCH_TO_TRITON_DTYPE[coarse.dtype], EXACT_DIV=True,
            FineGrad_ptr=fine,
        )
        return out, None, None


def _fork_block_mean_bshd(x, sizes, block):
    # Keep inference, other dtypes/dimensions and unsupported layouts unchanged.
    if (torch.is_grad_enabled() and x.requires_grad and x.is_cuda
            and x.dtype == torch.bfloat16 and x.shape[-1] == 128 and x.is_contiguous()
            and sizes.device == x.device and sizes.is_contiguous()
            and sizes.dtype in (torch.int32, torch.int64) and block in (128, 256)):
        return _ForkBlockMeanBSHD.apply(x, sizes, block)
    return x, fused_block_mean_bshd(x, sizes, block)


@triton.jit
def _weighted_bshd_fwd(C, G, O, N: tl.constexpr, HD: tl.constexpr, BLOCK: tl.constexpr,
                       TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = i < N
    ci = (i // (BLOCK * HD)) * HD + i % HD
    coarse = tl.load(C + ci, valid, 0).to(tl.float32)
    gate = tl.load(G + i, valid, 0).to(tl.float32)
    tl.store(O + i, coarse * gate, valid)


@triton.jit
def _weighted_bshd_bwd(D, C, G, DC, DG, N: tl.constexpr, HD: tl.constexpr, BLOCK: tl.constexpr,
                       NEED_C: tl.constexpr, NEED_G: tl.constexpr, TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = i < N
    value = tl.load(D + i, valid, 0).to(tl.float32)
    if NEED_C:
        gate = tl.load(G + i, valid, 0).to(tl.float32)
        tl.store(DC + i, value * gate, valid)
    if NEED_G:
        ci = (i // (BLOCK * HD)) * HD + i % HD
        coarse = tl.load(C + ci, valid, 0).to(tl.float32)
        tl.store(DG + i, value * coarse, valid)


class _WeightedBSHD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coarse, gate, block):
        ctx.save_for_backward(coarse if ctx.needs_input_grad[1] else None,
                              gate if ctx.needs_input_grad[0] else None)
        ctx.block = block
        out = torch.empty_like(gate)
        _, _, heads, dim = gate.shape
        _weighted_bshd_fwd[(triton.cdiv(gate.numel(), 1024),)](
            coarse, gate, out, N=gate.numel(), HD=heads * dim, BLOCK=block, TILE=1024,
            enable_fp_fusion=False)
        return out

    @staticmethod
    def backward(ctx, dy):
        coarse, gate = ctx.saved_tensors
        batch, seq, heads, dim = dy.shape
        shape = (batch, seq // ctx.block, ctx.block, heads, dim)
        need_c, need_g, _ = ctx.needs_input_grad
        dc = dg = None
        if torch.is_grad_enabled() or not dy.is_contiguous():
            if need_c:
                dc = (dy * gate).view(shape).sum(2)
            if need_g:
                dg = (dy.view(shape) * coarse.unsqueeze(2)).reshape_as(dy)
        else:
            weighted = torch.empty_like(dy) if need_c else None
            dg = torch.empty_like(dy) if need_g else None
            _weighted_bshd_bwd[(triton.cdiv(dy.numel(), 1024),)](
                dy, coarse, gate, weighted, dg, N=dy.numel(), HD=heads * dim,
                BLOCK=ctx.block, NEED_C=need_c, NEED_G=need_g, TILE=1024)
            if need_c:
                # Preserve the BF16 product and the existing native reduction order.
                dc = weighted.view(shape).sum(2)
        return dc, dg, None


def _combine_weighted_bshd(fine, coarse, gate, block, fallback):
    tensors = (fine, coarse) if gate is None else (fine, coarse, gate)
    eligible = (gate is not None and torch.is_grad_enabled()
                and any(x.requires_grad for x in tensors) and block in (128, 256)
                and fine.ndim == 4 and fine.shape[-1] == 128 and fine.shape[1] % block == 0
                # Small inputs favor native launches; cap flattened int32 offsets.
                and 2**25 <= fine.numel() <= 2**31 - 1
                and coarse.shape == (fine.shape[0], fine.shape[1] // block, fine.shape[2], fine.shape[3])
                and all(x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
                        and x.device == fine.device for x in tensors)
                and gate.shape == fine.shape)
    if eligible:
        ranges = [(x.data_ptr(), x.data_ptr() + x.numel() * x.element_size()) for x in tensors]
        eligible = all(a[1] <= b[0] or b[1] <= a[0]
                       for i, a in enumerate(ranges) for b in ranges[i + 1:])
    # Keep addition in native autograd: requesting only fine must prune the
    # weighted branch, including its saved-tensor version checks.
    return fine + _WeightedBSHD.apply(coarse, gate, block) if eligible else fallback(fine, coarse, gate, block)
