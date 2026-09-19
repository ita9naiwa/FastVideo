from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from fastvideo_kernel.block_sparse_attn_cute_fwd import (
    _validate_vc_prepared,
    _load_vc_module,
    block_sparse_attn_vc_prepared_fwd_bshd,
)


def test_prepared_boundary_rejects_truncated_routes_and_gradients():
    prepared = {
        "q": torch.empty(1, 256, 2, 128),
        "k": torch.empty(1, 256, 2, 128),
        "v": torch.empty(1, 256, 2, 128),
        "qs": torch.ones(1, 2),
        "ks": torch.ones(1, 2),
        "vs": torch.ones(1, 2, 128),
    }
    mask = torch.ones(2, 2, 2, dtype=torch.bool)
    sizes = torch.full((2,), 128, dtype=torch.int32)
    normalized, qblock, kblock = _validate_vc_prepared(prepared, mask, sizes)
    assert normalized.shape == (1, 2, 2, 2) and qblock == kblock == 128
    malformed = dict(prepared, k=torch.empty(1, 257, 2, 128), v=torch.empty(1, 257, 2, 128))
    with pytest.raises(ValueError, match="exact multiples"):
        block_sparse_attn_vc_prepared_fwd_bshd(malformed, mask, sizes)
    with pytest.raises(ValueError, match="int32 vector"):
        block_sparse_attn_vc_prepared_fwd_bshd(prepared, mask, sizes[:1])
    prepared["qs"].requires_grad_()
    with pytest.raises(ValueError, match="inference-only"):
        block_sparse_attn_vc_prepared_fwd_bshd(prepared, mask, sizes)


def test_module_cache_rechecks_a_changed_root():
    root = Path.cwd()
    module = SimpleNamespace(__file__=str(root / "flash_attn/cute/interface.py"))
    _load_vc_module.cache_clear()
    try:
        with patch("importlib.import_module", return_value=module) as load:
            assert _load_vc_module("interface", str(root)) is module
            assert _load_vc_module("interface", str(root)) is module
            assert load.call_count == 1
            with pytest.raises(RuntimeError, match="FASTVIDEO_VSA_VC_ROOT"):
                _load_vc_module("interface", str(root / "wrong"))
    finally:
        _load_vc_module.cache_clear()
