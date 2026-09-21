# Let native FA4 canonicalize backward output gradients

Remove two unconditional adapter-level `dO.contiguous()` calls. Native FA4 still checks pointer alignment, final-dimension contiguity and outer strides; it can directly read aligned row-strided gradients and still copies unsupported layouts. This depends on the validated VC/VSA-capable FA4 implementation's native `maybe_contiguous` contract.

B300/Torch 2.14+cu130 BF16 saved-backward measurements: Q128/D128 row-strided input: 48.477 → 45.891 µs (5.33%); Q256/D128: 49.779 → 47.616 µs (4.35%). Already-contiguous and forced-copy layouts were effectively unchanged. A separate N1024/H2/D128 row-strided call saved 524,288 bytes in incremental allocated peak. Actual model-loss layout frequency and full training performance were not measured.

Four Q128/Q256 × D64/D128 tests passed the repository's existing FP32-reference gates: output average error < 0.001 / max-scaled error < 0.2, gradient average error < 0.001 / max-scaled error < 0.25. Additional original checks covered broadcast, last-strided and misaligned dO plus changing Graph inputs and nonzero Q256 dLSE. Native backward accumulation is not guaranteed bit-exact. The included test reproduces the four independent FP32 reference cases; it does not loosen their tolerances. This branch does not include metadata transpose removal.

## Provenance and verification

This packages the explicitly requested 2026-09-21 experiment as a separate feature. Runtime baseline was FastVideo `7aa84cc2db2025b68d6d5bf03361548e6505d450`. Packaging base is `vc-vsa-source-20260920` at `ead7109a4630bda1688bbb79afc113537d136b85`; its relevant runtime files are byte-identical. The newer upstream/default main is not the measured integration baseline. Original benchmarks were not repeated for packaging.

The repository default remains Triton; these changes affect the opt-in CuTe VSA path. Validation used an existing Torch 2.14 CUDA 13 environment, not a fresh default Torch 2.12 package installation. Model training/E2E and every supported device/layout are not qualified.

Run the focused test in a compatible installed FastVideo/FA4 CUDA environment:

```bash
python -m pytest -q fastvideo-kernel/tests/test_backward_dout_layout.py
```
