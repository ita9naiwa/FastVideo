import cutlass
import cutlass.cute as cute
import pytest
import torch
from fastvideo_kernel.block_sparse_attn_cute_fwd import _SingleQStageLength
from flash_attn.cute import interface, utils
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch


@cute.jit
def _row_and_kv_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
    block = n_idx // 128
    valid = utils.scalar_to_ssa(aux_tensors[0][block[0]], cutlass.Int32)
    shift = utils.scalar_to_ssa(aux_tensors[0][3], cutlass.Int32)
    return ((m_idx + shift) % 3 != 0) & (n_idx % 128 < valid)


def _oracle(value, vscale, q_tokens, limits):
    valid = torch.arange(value.shape[1], device=value.device) % 128
    valid = valid < limits[:3].repeat_interleave(128)
    mean = value.float()[:, valid].mean(1) * vscale
    expected = mean[:, None].expand(1, q_tokens, 2, 128).clone()
    rows = (torch.arange(q_tokens, device=value.device) + limits[3]) % 3 == 0
    expected[:, rows] = 0
    return expected, rows


def _check(actual, expected, rows, label):
    reference = expected.float()
    ulp = torch.exp2(torch.floor(torch.log2(reference.abs().clamp_min(torch.finfo(torch.bfloat16).tiny)))) / 128
    ratio = float(((actual.float() - reference).abs() / (ulp + 8e-6 * 9)).amax())
    assert torch.isfinite(actual).all() and ratio <= 1, (label, ratio)
    assert torch.count_nonzero(actual[:, rows]) == 0
    assert torch.count_nonzero(actual[:, ~rows]) > 0


@pytest.mark.parametrize("block,parents", [(128, 4), (256, 4), (256, 150)])
def test_sparse_sum_row_mask_and_dynamic_graph(block, parents, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("VC sparse TC-sum requires B300 SM103")
    original_kernel = interface.FlashAttentionForwardSm100
    records = []

    class _TracedKernel(original_kernel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            records.append(self)

    monkeypatch.setattr(interface, "FlashAttentionForwardSm100", _TracedKernel)
    monkeypatch.setattr(interface._flash_attn_fwd, "compile_cache", {})
    with torch.inference_mode():
        tokens = block * parents
        q = torch.zeros((1, tokens, 2, 128), device="cuda", dtype=torch.float8_e4m3fn)
        k = torch.zeros((1, 384, 2, 128), device="cuda", dtype=q.dtype)
        levels = torch.tensor([1.0, 2.0, 4.0], device="cuda").repeat_interleave(128)
        channel = torch.where(torch.arange(128, device="cuda") % 2 == 0, 1.0, -0.5)
        v = (levels[None, :, None, None] * channel[None, None, None, :]).expand_as(k).to(q.dtype).contiguous()
        scales = torch.ones((1, 2), device="cuda")
        vscale = torch.where(torch.arange(128, device="cuda") % 3 == 0, 2.0, 0.5)
        vscale = vscale.expand(1, 2, 128).contiguous()
        limits = torch.tensor([128, 17, 1, 0], device="cuda", dtype=torch.int32)
        indices = torch.arange(3, device="cuda", dtype=torch.int32).expand(1, 2, parents, 3).contiguous()
        counts = torch.full((1, 2, parents), 3, device="cuda", dtype=torch.int32)
        sparse = BlockSparseTensorsTorch(
            counts, indices, torch.zeros_like(counts), torch.zeros_like(indices), block_size=(block, 128)
        )

        def run():
            return interface._flash_attn_fwd(
                q, k, v, q_descale=scales, k_descale=scales,
                vc_vscale=vscale, vc_expcast=True, tile_mn=(128, 128),
                max_seqlen_q=_SingleQStageLength(tokens) if block == 128 else tokens,
                mask_mod=_row_and_kv_mask, block_sparse_tensors=sparse,
                aux_tensors=[limits], return_lse=False,
            )[0]

        actual = run()
        torch.cuda.synchronize()
        kernel = records[-1]
        assert kernel.vc_tc_sum and kernel.cta_group_size == 1
        assert kernel.q_stage == block // 128
        if parents == 150:
            assert kernel.vc_sparse_stats_overlap
        expected, rows = _oracle(v, vscale, tokens, limits)
        _check(actual, expected, rows, f"block={block},eager")
        for _ in range(2):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        for replay, setting in enumerate(((0, 127, 16, 1), (13, 0, 128, 2), (128, 17, 1, 0))):
            limits.copy_(torch.tensor(setting, device="cuda", dtype=torch.int32))
            v.copy_((-v.float()).to(v.dtype))
            captured.fill_(float("nan"))
            graph.replay()
            expected, rows = _oracle(v, vscale, tokens, limits)
            _check(captured, expected, rows, f"block={block},graph={replay}")
