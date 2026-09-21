# Read backward sparse maps through their existing strides

Remove the contiguous copy between transposing the Q-owned boolean map and building KV-owned backward indices. Both existing index kernels already accept all four tensor strides. Full/partial classification, index order and the shared-index representation are unchanged. No new kernel is introduced.

Measured on B300/Torch 2.14+cu130: B1/H56/300x300 backward metadata: 62.386 → 59.570 µs (4.51% reduction). This avoids a 5,040,000-byte transpose temporary by tensor-size calculation, not a measured whole-model peak-memory saving. Do not add this reduction to separately measured backward timings.

Original validation covered 18 changed-input CUDA Graph metadata cases, including the 4097-column legacy fallback, actual backward consumers and four small FP32-reference cases at the repository's existing tolerances. The included test checks complete index/count arrays and Graph mutation for normal, aggregated and legacy shapes. This branch does not include dO copy removal.

## Provenance and verification

This packages the explicitly requested 2026-09-21 experiment as a separate feature. Runtime baseline was FastVideo `7aa84cc2db2025b68d6d5bf03361548e6505d450`. Packaging base is `vc-vsa-source-20260920` at `ead7109a4630bda1688bbb79afc113537d136b85`; its relevant runtime files are byte-identical. The newer upstream/default main is not the measured integration baseline. Original benchmarks were not repeated for packaging.

The repository default remains Triton; these changes affect the opt-in CuTe VSA path. Validation used an existing Torch 2.14 CUDA 13 environment, not a fresh default Torch 2.12 package installation. Model training/E2E and every supported device/layout are not qualified.

Run the focused test in a compatible installed FastVideo/FA4 CUDA environment:

```bash
python -m pytest -q fastvideo-kernel/tests/test_backward_map_strides.py
```
