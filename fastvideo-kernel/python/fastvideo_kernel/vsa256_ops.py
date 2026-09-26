"""Opaque custom-op seam for BF16 VSA-256 CuTe training (torch.library, fake + autograd registrations).

``torch.ops.fastvideo_kernel.vsa256_fwd`` / ``vsa256_bwd`` wrap the Q256 training forward and the two backward variants
(plain 128-child backward, or the shared packed-tail backward when ``pack_tails``). Metadata enters as tensors (block
map, valid sizes, optional query-padding proof); the pack policy is a static bool chosen by the caller. The same ops run
in eager and compiled mode, so an activation-checkpoint policy sees identical op keys; the FA4 loader caches, CuTe
compilation and backward planning stay inside the opaque bodies.

Query-padding proof: ``query_sizes`` asks the packed-tail backward to skip wholly padded Q128 children. It is honored
only with the caller's trusted ``query_untile`` map and the builder-recorded versions of that map and of
``query_sizes``: the backward body (opaque, so Dynamo never branches on ``_version``) re-reads both version counters
and prunes only if neither tensor was modified in place since the builder validated them (versions only grow, so equal
at backward means unmodified at forward and at the caller's untile too) and dLSE is absent. Otherwise it runs the full
backward. Both tensors are also saved for backward, so autograd's saved-tensor check still rejects mutation after
forward.

Integration note: the conditional stack inputs extend this schema without changing its identities - the prefix split
adds a static ``q_split`` argument (shared split forward helper), the fused-QKV gradient adds the stacked route
(``k = v = None``, one ``dqkv`` output) - see candidate h3-training-stack-integration.
"""

import os

import torch

from fastvideo_kernel import block_sparse_attn_cute_fwd as adapter


@torch.library.custom_op("fastvideo_kernel::vsa256_fwd", mutates_args=(), device_types="cuda")
def vsa256_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_map: torch.Tensor, sizes: torch.Tensor,
               query_sizes: torch.Tensor | None, query_untile: torch.Tensor | None, query_untile_version: int,
               query_sizes_version: int, pack_tails: bool) -> tuple[torch.Tensor, torch.Tensor]:
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
def _vsa256_fwd_fake(q, k, v, block_map, sizes, query_sizes, query_untile, query_untile_version, query_sizes_version,
                     pack_tails):
    return torch.empty_like(q), q.new_empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32)


@torch.library.custom_op("fastvideo_kernel::vsa256_bwd", mutates_args=(), device_types="cuda")
def vsa256_bwd(dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
               lse: torch.Tensor, block_map: torch.Tensor, sizes: torch.Tensor, query_sizes: torch.Tensor | None,
               query_untile: torch.Tensor | None, query_untile_version: int, query_sizes_version: int,
               pack_tails: bool, dlse: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout = dout.contiguous()
    dlse = dlse.contiguous() if dlse is not None else None
    if not (query_sizes is not None and query_untile is not None and query_untile._version == query_untile_version
            and query_sizes._version == query_sizes_version):
        query_sizes = None  # no proof, or a proof tensor changed since the builder validated it: full backward
    _, _, _, flash_attn_bwd = adapter._load_fa4_cute()
    if pack_tails:
        from fastvideo_kernel.vsa_tail_backward import tail_backward
        return tail_backward(dout, q, k, v, out, lse, block_map, sizes, dlse, query_sizes)
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
def _vsa256_bwd_fake(dout, q, k, v, out, lse, block_map, sizes, query_sizes, query_untile, query_untile_version,
                     query_sizes_version, pack_tails, dlse):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_context(ctx, inputs, output):
    q, k, v, block_map, sizes, query_sizes, query_untile, untile_version, sizes_version, pack_tails = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, block_map, sizes, query_sizes, query_untile)
    ctx.query_versions = (untile_version, sizes_version)
    ctx.pack_tails = pack_tails
    ctx.set_materialize_grads(False)  # as the old autograd.Functions: an unused output's grad arrives as None


def _backward(ctx, dout, dlse):
    q, k, v, out, lse, block_map, sizes, query_sizes, query_untile = ctx.saved_tensors
    if dout is None:
        dout = torch.zeros_like(out)
    dq, dk, dv = vsa256_bwd(dout, q, k, v, out, lse, block_map, sizes, query_sizes, query_untile, *ctx.query_versions,
                            ctx.pack_tails, dlse)
    return dq, dk, dv, None, None, None, None, None, None, None


vsa256_fwd.register_autograd(_backward, setup_context=_setup_context)


def training_eligible(q, k, v, block_map):
    """The validated domain of the op pair (native BF16 Q256 training on SM10x), shared by every caller so an input
    outside it (e.g. strided views that are not 16-byte aligned, shape mismatches, FASTVIDEO_VSA_VC=1) always takes the
    caller's fallback and never reaches an op whose fake would not describe it. Under torch.compile the data_ptr()
    alignment read of a non-contiguous input is not traceable (fullgraph raises; disclosed limitation)."""
    return (torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v))
            and q.is_cuda and q.dtype == k.dtype == v.dtype == torch.bfloat16
            and q.ndim == k.ndim == v.ndim == 4
            and q.shape[-1] == k.shape[-1] == v.shape[-1] and q.shape[-1] in (64, 128)
            and q.shape[0] == k.shape[0] == v.shape[0]
            and q.shape[2] == k.shape[2] == v.shape[2]
            and k.shape[1] == v.shape[1]
            and all(t.is_contiguous() or (t.stride(-1) == 1 and t.data_ptr() % 16 == 0
                                         and all(s > 0 and s % 8 == 0 for s in t.stride()[:-1]))
                    for t in (q, k, v))
            and q.shape[1] == block_map.shape[2] * 256
            and k.shape[1] == block_map.shape[3] * 256
            and block_map.shape[:2] == (q.shape[0], q.shape[2])
            and torch.cuda.get_device_capability(q.device)[0] == 10
            and os.environ.get("FASTVIDEO_VSA_VC", "0") != "1")


def training_attention(q, k, v, block_map, sizes, *, pack_tails=None, query_sizes=None, query_untile=None,
                       query_versions=(0, 0)):
    """Single dispatch into the op pair for native BF16 Q256 training (public wrapper and _cute_attention); callers
    check ``training_eligible`` first.

    ``pack_tails`` None follows FASTVIDEO_VSA_PACK_TAILS (default on); the packed-tail backward additionally needs
    matching head/dim, a per-(batch, head) map and contiguous inputs. ``query_sizes`` of the wrong shape, or without
    ``query_untile`` (the trusted untile map; ``query_versions`` are the builder-recorded versions of that map and of
    ``query_sizes``), is dropped: full backward.
    """
    if pack_tails is None:
        pack_tails = os.environ.get("FASTVIDEO_VSA_PACK_TAILS", "1") == "1"
    pack_tails = bool(pack_tails and q.shape[-1] == k.shape[-1] == v.shape[-1]
                      and q.shape[2] == k.shape[2] == v.shape[2]
                      and block_map.shape[:2] == (q.shape[0], q.shape[2])
                      and all(t.is_contiguous() for t in (q, k, v)))
    if query_sizes is None or query_untile is None or query_sizes.shape != (block_map.shape[2], ):
        query_sizes = query_untile = None
    return vsa256_fwd(q, k, v, block_map, sizes, query_sizes, query_untile, *query_versions, pack_tails)
