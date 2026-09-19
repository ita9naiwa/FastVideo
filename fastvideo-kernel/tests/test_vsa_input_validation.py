"""Both public layouts retain the same input validation."""
import pytest
import torch
from fastvideo_kernel import ops


@pytest.mark.parametrize('seq_axis', [1, 2])
@pytest.mark.parametrize('invalid', ['batch', 'heads', 'kv_length', 'alignment', 'kv_sizes', 'q_sizes'])
def test_vsa_layout_validation(seq_axis, invalid):
    q_shape, k_shape, v_shape = [1, 256, 2, 64], [1, 256, 2, 64], [1, 256, 2, 64]
    kv_count = q_count = 2
    if invalid == 'batch':
        k_shape[0] = 2
    elif invalid == 'heads':
        v_shape[2] = 3
    elif invalid == 'kv_length':
        v_shape[1] = 128
    elif invalid == 'alignment':
        q_shape[1] = 255
    elif invalid == 'kv_sizes':
        kv_count = 1
    else:
        q_count = 1
    q, k, v = (torch.empty(s, device='meta') for s in (q_shape, k_shape, v_shape))
    if seq_axis == 2:
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    attention = ops.video_sparse_attn_bshd if seq_axis == 1 else ops.video_sparse_attn
    with pytest.raises(ValueError):
        attention(q, k, v, torch.empty(kv_count, device='meta'),
                  torch.empty(q_count, device='meta'), 1, (4, 8, 4))
