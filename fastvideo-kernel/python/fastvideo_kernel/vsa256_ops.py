"""Opaque custom-op seam for BF16 VSA-256 CuTe training (torch.library, fake + autograd registrations).

``torch.ops.fastvideo_kernel.vsa256_fwd`` / ``vsa256_bwd`` wrap the Q256 training forward and the two backward variants
(plain 128-child backward, or the shared packed-tail backward when ``pack_tails``). Metadata enters as tensors (block
map, valid sizes, optional query-padding proof); the pack policy is a static bool chosen by the caller. The same ops run
in eager and compiled mode, so an activation-checkpoint policy sees identical op keys; the FA4 loader caches, CuTe
compilation and backward planning stay inside the opaque bodies. The optional FA4 alias-guard hint is a 0-d CPU bool
tensor (forward only), so it is a graph input and never specializes a compiled graph.

Query-padding proof: ``query_sizes`` asks the packed-tail backward to skip wholly padded Q128 children. It is honored
only with the caller's trusted ``query_untile`` map and the builder-recorded versions of that map and of
``query_sizes``: the backward body (opaque, so Dynamo never branches on ``_version``) re-reads both version counters
and prunes only if neither tensor was modified in place since the builder validated them (versions only grow, so equal
at backward means unmodified at forward and at the caller's untile too) and dLSE is absent. Otherwise it runs the full
backward. Both tensors are also saved for backward, so autograd's saved-tensor check still rejects mutation after
forward.

Stacked route (fused-QKV gradient): ``k = v = None`` means ``q`` is the batch-stacked ``[3B, S, H, D]`` Q/K/V tensor. It
is then the single autograd input; the backward allocates ONE fresh ``[3B, S, H, D]`` gradient, lets the kernels write
dQ/dK/dV into its three B-sized views (both pack policies), checks they did, and returns it whole, so no concat follows.
(A mutation-free custom op cannot return three outputs that share storage, hence one output.)

Tile-parameterized family (ruling 84): ``vsa_train_fwd`` / ``vsa_train_bwd`` take ``tile`` (128 or 256). Tile 128 wraps the FA4
Q128/KV128 training kernels (H3 tile 128); tile 256 delegates to vsa256_fwd/bwd with default options, which remain the
full-option tile-256 ops (stacked QKV, query-padding proof, packed tails).

Integration note (history): a prefix-split q_split argument was considered and DROPPED (conductor 33852: not adopted
for chunk256), see candidate h3-training-stack-integration.
"""

import os

import torch

from fastvideo_kernel import block_sparse_attn_cute_fwd as adapter


def _chunks(q, k, v):
    return q.chunk(3, dim=0) if k is None else (q, k, v)


@torch.library.custom_op("fastvideo_kernel::vsa256_fwd", mutates_args=(), device_types="cuda")
def vsa256_fwd(q: torch.Tensor, k: torch.Tensor | None, v: torch.Tensor | None, block_map: torch.Tensor, sizes: torch.Tensor,
               query_sizes: torch.Tensor | None, query_untile: torch.Tensor | None, query_untile_version: int,
               query_sizes_version: int, pack_tails: bool,
               alias_guard: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, flash_attn_fwd, flash_attn_bwd = adapter._load_fa4_cute()
    if pack_tails:
        from fastvideo_kernel.vsa_tail_backward import _check_workspace_support
        _check_workspace_support(flash_attn_bwd)
    q, k, v = _chunks(q, k, v)
    forward_sparse, _ = adapter._build_sparse_tensors(block_map, sizes, q_len=q.shape[1], q_block_size=256,
                                                      kv_block_size=256, need_backward=False)
    out, lse = flash_attn_fwd(q, k, v, mask_mod=adapter._build_vbs_vector_mask_mod(256), aux_tensors=[sizes],
                              block_sparse_tensors=forward_sparse, return_lse=True,
                              **adapter._alias_guard_kwargs(flash_attn_fwd, alias_guard))[:2]
    return out, lse


@torch.library.register_fake("fastvideo_kernel::vsa256_fwd")
def _vsa256_fwd_fake(q, k, v, block_map, sizes, query_sizes, query_untile, query_untile_version, query_sizes_version,
                     pack_tails, alias_guard=None):
    if k is None:  # stacked [3B, S, H, D]: the output covers one B-sized chunk
        batch = q.shape[0] // 3
        return (q.new_empty((batch, *q.shape[1:])),
                q.new_empty((batch, q.shape[2], q.shape[1]), dtype=torch.float32))
    return torch.empty_like(q), q.new_empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32)


@torch.library.custom_op("fastvideo_kernel::vsa256_bwd", mutates_args=(), device_types="cuda")
def vsa256_bwd(dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor | None, v: torch.Tensor | None, out: torch.Tensor,
               lse: torch.Tensor, block_map: torch.Tensor, sizes: torch.Tensor, query_sizes: torch.Tensor | None,
               query_untile: torch.Tensor | None, query_untile_version: int, query_sizes_version: int,
               pack_tails: bool, dlse: torch.Tensor | None) -> list[torch.Tensor]:
    """Returns [dq, dk, dv], or [dqkv] (one allocation) on the stacked route."""
    dout = dout.contiguous()
    dlse = dlse.contiguous() if dlse is not None else None
    if not (query_sizes is not None and query_untile is not None and query_untile._version == query_untile_version
            and query_sizes._version == query_sizes_version):
        query_sizes = None  # no proof, or a proof tensor changed since the builder validated it: full backward
    stacked = k is None
    dqkv = torch.empty_like(q) if stacked else None
    views = dqkv.chunk(3, dim=0) if stacked else (None, None, None)
    q, k, v = _chunks(q, k, v)
    _, _, _, flash_attn_bwd = adapter._load_fa4_cute()
    if pack_tails:
        from fastvideo_kernel.vsa_tail_backward import tail_backward
        return _grads_out(tail_backward(dout, q, k, v, out, lse, block_map, sizes, dlse, query_sizes, grad_views=views),
                          views, dqkv)
    # A partially filled logical block can contain a full physical KV tile: classify 128-token children so backward
    # does not mask that full tile (planning moved here from the old forward; same tensors, built once).
    child_sizes = torch.stack((sizes.clamp(0, 128), (sizes - 128).clamp(0, 128)), -1).flatten()
    _, backward_sparse = adapter._build_sparse_tensors(block_map.repeat_interleave(2, -1), child_sizes,
                                                       q_len=q.shape[1], q_block_size=256, kv_block_size=128,
                                                       need_backward=True, need_forward=False,
                                                       force_q_sparse_block_size=256)
    return _grads_out(flash_attn_bwd(q, k, v, out, dout, lse, mask_mod=adapter._build_vbs_mask_mod(128),
                                     aux_tensors=[child_sizes], block_sparse_tensors=backward_sparse, dlse=dlse,
                                     dq=views[0], dk=views[1], dv=views[2]), views, dqkv)


def _grads_out(grads, views, dqkv):
    if dqkv is None:
        return list(grads[:3])
    if any(g is not view for g, view in zip(grads[:3], views, strict=True)):
        raise RuntimeError("vsa256_bwd: the backward did not write the provided dq/dk/dv views")
    return [dqkv]


@torch.library.register_fake("fastvideo_kernel::vsa256_bwd")
def _vsa256_bwd_fake(dout, q, k, v, out, lse, block_map, sizes, query_sizes, query_untile, query_untile_version,
                     query_sizes_version, pack_tails, dlse):
    if k is None:
        return [torch.empty_like(q)]
    return [torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)]


def _setup_context(ctx, inputs, output):
    q, k, v, block_map, sizes, query_sizes, query_untile, untile_version, sizes_version, pack_tails, _alias_guard = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, block_map, sizes, query_sizes, query_untile)
    ctx.query_versions = (untile_version, sizes_version)
    ctx.pack_tails = pack_tails
    ctx.set_materialize_grads(False)  # as the old autograd.Functions: an unused output's grad arrives as None


def _backward(ctx, dout, dlse):
    q, k, v, out, lse, block_map, sizes, query_sizes, query_untile = ctx.saved_tensors
    if dout is None:
        dout = torch.zeros_like(out)
    grads = vsa256_bwd(dout, q, k, v, out, lse, block_map, sizes, query_sizes, query_untile, *ctx.query_versions,
                       ctx.pack_tails, dlse)
    metadata_grads = (None, ) * 8  # block_map, sizes, query_sizes, query_untile, two versions, pack_tails, alias_guard
    if k is None:
        return grads[0], None, None, *metadata_grads
    return grads[0], grads[1], grads[2], *metadata_grads


vsa256_fwd.register_autograd(_backward, setup_context=_setup_context)


def training_eligible(q, k, v, block_map, block=256):
    """The validated domain of the op pair (native BF16 Q256 training on SM10x; ``block=128``: vsa_train_fwd at tile 128), shared by every caller so an input
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
            and q.shape[1] == block_map.shape[2] * block
            and k.shape[1] == block_map.shape[3] * block
            and block_map.shape[:2] == (q.shape[0], q.shape[2])
            and torch.cuda.get_device_capability(q.device)[0] == 10
            and os.environ.get("FASTVIDEO_VSA_VC", "0") != "1")


def training_attention(q, k, v, block_map, sizes, *, pack_tails=None, query_sizes=None, query_untile=None,
                       query_versions=(0, 0), qkv=None, alias_guard=None):
    """Single dispatch into the op pair for native BF16 Q256 training (public wrapper and _cute_attention); callers
    check ``training_eligible`` first.

    ``pack_tails`` None follows FASTVIDEO_VSA_PACK_TAILS (default on); the packed-tail backward additionally needs
    matching head/dim, a per-(batch, head) map and contiguous inputs. ``query_sizes`` of the wrong shape, or without
    ``query_untile`` (the trusted untile map; ``query_versions`` are the builder-recorded versions of that map and of
    ``query_sizes``), is dropped: full backward. ``qkv``: the contiguous ``[3B, S, H, D]`` tensor whose dim-0 chunks are
    q/k/v (caller contract, e.g. H3 ``forward_qkv``); when usable it replaces q/k/v as the op input, so the fused gradient
    is written into one allocation. Otherwise (gated calls, non-contiguous stacks, unequal shapes) the ordinary route.
    ``alias_guard``: FA4 alias-guard hint (None, bool or 0-d CPU bool tensor) for the forward launch of either route; it
    enters the op as a CPU tensor (graph input) and only when set, so hint-less calls keep the previous op arguments.
    """
    hint = ()
    if alias_guard is not None:
        hint = (alias_guard if isinstance(alias_guard, torch.Tensor) else torch.tensor(bool(alias_guard), device="cpu"), )
    if pack_tails is None:
        pack_tails = os.environ.get("FASTVIDEO_VSA_PACK_TAILS", "1") == "1"
    pack_tails = bool(pack_tails and q.shape[-1] == k.shape[-1] == v.shape[-1]
                      and q.shape[2] == k.shape[2] == v.shape[2]
                      and block_map.shape[:2] == (q.shape[0], q.shape[2])
                      and all(t.is_contiguous() for t in (q, k, v)))
    if query_sizes is None or query_untile is None or query_sizes.shape != (block_map.shape[2], ):
        query_sizes = query_untile = None
    if (qkv is not None and qkv.is_contiguous() and q.shape == k.shape == v.shape
            and qkv.shape == (3 * q.shape[0], *q.shape[1:])):
        return vsa256_fwd(qkv, None, None, block_map, sizes, query_sizes, query_untile, *query_versions, pack_tails,
                          *hint)
    return vsa256_fwd(q, k, v, block_map, sizes, query_sizes, query_untile, *query_versions, pack_tails, *hint)


# Tile-parameterized training pair (ruling 84: one op family, tile a parameter). ``tile`` 128: the FA4 Q128/KV128 forward and
# backward of ``block_sparse_attn_cute_fwd._CuteAttentionQ128`` (identical kernel arguments; the backward lists are built in the
# backward body instead of being stashed on ctx). ``tile`` 256: the vsa256 pair above with its defaults (plain 128-child
# backward, no query-padding proof, no packed tails), so vsa256_fwd/bwd stay the full-option tile-256 member of the family.
# ``alias_guard``: the FA4 alias-guard hint (0-d CPU bool tensor or None) reaches the forward launch at either tile.
@torch.library.custom_op("fastvideo_kernel::vsa_train_fwd", mutates_args=(), device_types="cuda")
def vsa_train_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_map: torch.Tensor, sizes: torch.Tensor,
                  tile: int, alias_guard: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if tile == 256:
        return vsa256_fwd(q, k, v, block_map, sizes, None, None, 0, 0, False, alias_guard)
    if tile != 128:
        raise ValueError(f"vsa_train_fwd supports tile 128 or 256, got {tile}")
    out, lse, _ = adapter._cute_attention_q128_forward(q, k, v, block_map, sizes.to(torch.int32), need_backward=False,
                                                       alias_guard=alias_guard)
    return out, lse


@torch.library.register_fake("fastvideo_kernel::vsa_train_fwd")
def _vsa_train_fwd_fake(q, k, v, block_map, sizes, tile, alias_guard=None):
    return torch.empty_like(q), q.new_empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32)


@torch.library.custom_op("fastvideo_kernel::vsa_train_bwd", mutates_args=(), device_types="cuda")
def vsa_train_bwd(dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
                  lse: torch.Tensor, block_map: torch.Tensor, sizes: torch.Tensor, tile: int) -> list[torch.Tensor]:
    if tile == 256:
        return vsa256_bwd(dout, q, k, v, out, lse, block_map, sizes, None, None, 0, 0, False, None)
    if tile != 128:
        raise ValueError(f"vsa_train_bwd supports tile 128 or 256, got {tile}")
    sizes = sizes.to(torch.int32)
    _, backward_sparse = adapter._build_sparse_tensors(block_map, sizes, q_len=q.shape[1], q_block_size=128,
                                                       kv_block_size=128, need_backward=True, need_forward=False,
                                                       force_q_sparse_block_size=128)
    _, _, _, flash_attn_bwd = adapter._load_fa4_cute()
    return list(flash_attn_bwd(q, k, v, out, dout.contiguous(), lse, softmax_scale=q.shape[-1]**-0.5,
                               mask_mod=adapter._build_vbs_mask_mod(128), aux_tensors=[sizes],
                               block_sparse_tensors=backward_sparse)[:3])


@torch.library.register_fake("fastvideo_kernel::vsa_train_bwd")
def _vsa_train_bwd_fake(dout, q, k, v, out, lse, block_map, sizes, tile):
    return [torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)]


def _setup_context_train(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:3], *output, *inputs[3:5])
    ctx.tile = inputs[5]  # inputs[6]: alias_guard (forward-only hint)
    ctx.set_materialize_grads(False)


def _backward_train(ctx, dout, dlse):  # lse is auxiliary (callers detach it), as in _CuteAttentionQ128
    q, k, v, out, lse, block_map, sizes = ctx.saved_tensors
    if dout is None:
        dout = torch.zeros_like(out)
    return *vsa_train_bwd(dout, q, k, v, out, lse, block_map, sizes, ctx.tile), None, None, None, None


vsa_train_fwd.register_autograd(_backward_train, setup_context=_setup_context_train)
