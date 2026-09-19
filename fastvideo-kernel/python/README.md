# Python-only FastVideo kernel package

This alternate build packages the existing Python/Triton VSA implementation and
VC adapters without compiling FastVideo's C++/CUDA extensions. The original
`fastvideo-kernel/pyproject.toml` remains the native-extension build.

```sh
uv build --wheel fastvideo-kernel/python --out-dir dist
```

This package targets the H3 Torch 2.14 environment. It uses the same
`fastvideo-kernel` distribution name and `fastvideo_kernel` namespace as the
native build: install exactly one of them. Pin the wheel artifact and SHA256;
do not publish the same-version alternate wheel beside native wheels on an index.
A Python-only wheel tag does not imply CPU attention support: execution still
requires compatible GPU, Torch, and Triton versions.

For the CuTe VSA path, install the matching VC/VSA-enabled FlashAttention-4
package and set `FASTVIDEO_VSA_CUTEDSL=1`. VC additionally requires
`FASTVIDEO_VSA_VC_ROOT` to identify that installed package's root containing
`flash_attn/cute`. These adapters validate this path against the imported module.
The wheel does not provide native TurboDiffusion, QAT, or ThunderKittens operators.
