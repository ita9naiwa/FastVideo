# Fuse CuTe VSA forward map classification and compaction

The forward metadata builder previously materialized separate full/partial boolean maps and scanned each in a separate kernel. Classify the original map and valid-size vector while producing both stable index lists in one Triton launch. Retain the original fallback above 4096 columns and the existing backward metadata path.

Measured on B300/Torch 2.14+cu130: B1/H56/300x300 forward metadata: 71.994 → 47.260 µs (34.36% reduction). Whole BF16 CuTe forward at N38400/H56/D128/K30 changed from 4.373848 → 4.350384 ms (0.54%); this small, workload-specific result is not an end-to-end model claim. Peak allocation/reserved memory were unchanged.

Sixty metadata/reference cases, changed-input Graphs, actual BF16 forward output/LSE equality and memcheck/racecheck/synccheck passed in the original experiment. The included focused tests reproduce stable order, empty/full/mixed sizes, strided inputs, Graph mutation and the 4097-column fallback. Backward arithmetic and VC256's special expansion path are unchanged.

## Provenance and verification

This packages the explicitly requested 2026-09-21 experiment as a separate feature. Runtime baseline was FastVideo `7aa84cc2db2025b68d6d5bf03361548e6505d450`. Packaging base is `vc-vsa-source-20260920` at `ead7109a4630bda1688bbb79afc113537d136b85`; its relevant runtime files are byte-identical. The newer upstream/default main is not the measured integration baseline. Original benchmarks were not repeated for packaging.

The repository default remains Triton; these changes affect the opt-in CuTe VSA path. Validation used an existing Torch 2.14 CUDA 13 environment, not a fresh default Torch 2.12 package installation. Model training/E2E and every supported device/layout are not qualified.

Run the focused test in a compatible installed FastVideo/FA4 CUDA environment:

```bash
python -m pytest -q fastvideo-kernel/tests/test_forward_classified_map.py
```
