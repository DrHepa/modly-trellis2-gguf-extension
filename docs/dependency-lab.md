# External dependency lab

This repo does **not** build native wheels or run dependency-lab experiments. The lab is external by design.

## Before enabling refine on Linux ARM64

Minimum evidence should cover a matrix across:

- platform / architecture
- Python version
- Torch version
- CUDA version
- `flex_gemm`
- `cumesh`
- `nvdiffrast`
- `o_voxel`
- `spconv` / `cumm`

## Focus order

1. `flex_gemm` and the texture/refine path
2. `nvdiffrast`
3. Remaining geometry dependencies

## Evidence bundle

For each tested cell, keep:

- `matrix.csv`
- `env.json`
- `result.json`
- `pip-freeze.txt`
- short notes

Refine should stay disabled until that evidence is complete and reviewed.
