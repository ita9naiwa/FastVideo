"""Opaque custom-op seam for BF16 VSA-256 CuTe training (torch.library, fake + autograd registrations).

``torch.ops.fastvideo_kernel.vsa256_fwd`` / ``vsa256_bwd`` wrap the existing Q256 training forward and the two backward
variants (plain 128-child backward, or the packed-tail backward when ``pack_tails``). Metadata enters as tensors (block
map, valid sizes); the pack policy is a static bool chosen by the caller. The same ops run in eager and compiled mode,
so an activation-checkpoint policy sees identical op keys; the FA4 loader caches, CuTe compilation and backward planning
stay inside the opaque bodies. Numerics equal the previous autograd.Function paths: same kernels and arguments.
"""

import torch

from fastvideo_kernel import block_sparse_attn_cute_fwd as adapter


@torch.library.custom_op("fastvideo_kernel::vsa256_fwd", mutates_args=(), device_types="cuda")
def vsa256_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_map: torch.Tensor, sizes: torch.Tensor,
               pack_tails: bool) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, flash_attn_fwd, flash_attn_bwd = adapter._load_fa4_cute()
    if pack_tails:
        from fastvideo_kernel.vsa_tail_backward import _check_workspace_support
        _check_workspace_support(flash_attn_bwd)
    forward_sparse, _ = adapter._build_sparse_tensors(block_map, sizes, q_len=q.shape[1], q_block_size=256,
                                                      kv_block_size=256, need_backward=False)
    out, lse = flash_attn_fwd(q, k, v, mask_mod=adapter._build_vbs_vector_mask_mod(256), aux_tensors=[sizes],
                              block_sparse_tensors=forward_sparse, return_lse=True)[:2]
    return out, lse


@torch.library.register_fake("fastvideo_kernel::vsa256_fwd")
def _vsa256_fwd_fake(q, k, v, block_map, sizes, pack_tails):
    return torch.empty_like(q), q.new_empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32)


@torch.library.custom_op("fastvideo_kernel::vsa256_bwd", mutates_args=(), device_types="cuda")
def vsa256_bwd(dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
               lse: torch.Tensor, block_map: torch.Tensor, sizes: torch.Tensor, pack_tails: bool,
               dlse: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout = dout.contiguous()
    dlse = dlse.contiguous() if dlse is not None else None
    _, _, _, flash_attn_bwd = adapter._load_fa4_cute()
    if pack_tails:
        from fastvideo_kernel.vsa_tail_backward import _prepare, _tail_mask_mod, scatter
        full, child_sizes, (index, parents, valid, tails) = _prepare(block_map, sizes)
        dq, dk, dv, workspace = flash_attn_bwd(q, k, v, out, dout, lse, mask_mod=adapter._build_vbs_mask_mod(128),
                                               aux_tensors=[child_sizes], block_sparse_tensors=full, dlse=dlse,
                                               _return_workspace=True)
        _, packed_dk, packed_dv = flash_attn_bwd(q, k.index_select(1, index), v.index_select(1, index), out, dout, lse,
                                                 mask_mod=_tail_mask_mod(), aux_tensors=[block_map, parents, valid],
                                                 block_sparse_tensors=tails, dlse=dlse, dq=dq, _workspace=workspace)
        scatter(dk, dv, packed_dk, packed_dv, index, valid)
        return dq, dk, dv
    # A partially filled logical block can contain a full physical KV tile: classify 128-token children so backward
    # does not mask that full tile (planning moved here from the old forward; same tensors, built once).
    child_sizes = torch.stack((sizes.clamp(0, 128), (sizes - 128).clamp(0, 128)), -1).flatten()
    _, backward_sparse = adapter._build_sparse_tensors(block_map.repeat_interleave(2, -1), child_sizes,
                                                       q_len=q.shape[1], q_block_size=256, kv_block_size=128,
                                                       need_backward=True, need_forward=False,
                                                       force_q_sparse_block_size=256)
    return flash_attn_bwd(q, k, v, out, dout, lse, mask_mod=adapter._build_vbs_mask_mod(128), aux_tensors=[child_sizes],
                          block_sparse_tensors=backward_sparse, dlse=dlse)


@torch.library.register_fake("fastvideo_kernel::vsa256_bwd")
def _vsa256_bwd_fake(dout, q, k, v, out, lse, block_map, sizes, pack_tails, dlse):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_context(ctx, inputs, output):
    q, k, v, block_map, sizes, pack_tails = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, block_map, sizes)
    ctx.pack_tails = pack_tails


def _backward(ctx, dout, dlse):
    q, k, v, out, lse, block_map, sizes = ctx.saved_tensors
    if dout is None:
        dout = torch.zeros_like(out)
    dq, dk, dv = vsa256_bwd(dout, q, k, v, out, lse, block_map, sizes, ctx.pack_tails, dlse)
    return dq, dk, dv, None, None, None


vsa256_fwd.register_autograd(_backward, setup_context=_setup_context)


# Tile permutation pair: packed rows [B, S, H, D] -> fresh zero-padded tile buffer [B, P, H, D] (row i lands in padded
# slot index[i]), and its adjoint (gather the same slots). Training calls must not reuse a shared scratch buffer, so the
# forward always returns an invocation-owned tensor.
@torch.library.custom_op("fastvideo_kernel::vsa_tile_permute_fwd", mutates_args=(), device_types="cuda")
def vsa_tile_permute_fwd(x: torch.Tensor, index: torch.Tensor, padded_len: int) -> torch.Tensor:
    out = x.new_zeros((x.shape[0], padded_len, *x.shape[2:]))
    out[:, index] = x
    return out


@torch.library.register_fake("fastvideo_kernel::vsa_tile_permute_fwd")
def _vsa_tile_permute_fwd_fake(x, index, padded_len):
    return x.new_empty((x.shape[0], padded_len, *x.shape[2:]))


@torch.library.custom_op("fastvideo_kernel::vsa_tile_permute_bwd", mutates_args=(), device_types="cuda")
def vsa_tile_permute_bwd(grad: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return grad.index_select(1, index)


@torch.library.register_fake("fastvideo_kernel::vsa_tile_permute_bwd")
def _vsa_tile_permute_bwd_fake(grad, index):
    return grad.new_empty((grad.shape[0], index.shape[0], *grad.shape[2:]))


def _permute_setup_context(ctx, inputs, output):
    _, index, _ = inputs
    ctx.save_for_backward(index)


def _permute_backward(ctx, grad):
    (index, ) = ctx.saved_tensors
    return vsa_tile_permute_bwd(grad.contiguous(), index), None, None


vsa_tile_permute_fwd.register_autograd(_permute_backward, setup_context=_permute_setup_context)
