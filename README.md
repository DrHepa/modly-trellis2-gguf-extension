# Trellis.2 GGUF — Modly Extension

This clean rebuild currently supports **Generate Mesh only** from the [Aero-Ex/Trellis2-GGUF](https://huggingface.co/Aero-Ex/Trellis2-GGUF) Hugging Face repo.

## Current scope

- Public runtime scope: **Generate Mesh** only
- `Texture Mesh` / refine is **intentionally not exposed** in this clean slice
- Linux ARM64 refine/texturing remains **blocked** until the dependency/runtime gates are satisfied
- No production `grid_sample_3d` fallback is shipped in this rebuild
- Hugging Face asset resolution is **deterministic** and uses explicit expected paths only; there is no broad recursive first-hit lookup

## Generate Mesh

`Generate Mesh` converts one image into a geometry-only GLB.

| Parameter | Description | Default |
|---|---|---|
| `image` | Input image | — |
| `pipeline` | Quality preset: `fast` (512), `balanced` (1024), `high` (1024 cascade), `ultra` (1536 cascade) | `balanced` |
| `quantization` | GGUF quant level (Q4_K_M → Q8_0) | `Q5_K_M` |
| `ss_steps` | Sparse-structure diffusion steps | `25` |
| `slat_steps` | Shape SLaT diffusion steps | `25` |
| `foreground_ratio` | Subject fill ratio (0–1) | `0.85` |
| `seed` | Reproducibility seed (`-1` = random) | `-1` |

**Output:** geometry-only GLB

## Setup reporting

The setup flow writes `<extension dir>/setup-report.json` and prints the same high-level capability status to stdout.

- `status: success` means setup completed without native dependency failures
- `status: partial` means base setup completed but one or more native/runtime blockers still remain
- `capabilities.generate.blockers` and `capabilities.refine.blockers` name the specific missing, unsupported, or import-time blockers
- On Linux ARM64, base setup may still complete while native CUDA wheels remain unavailable; the report is the setup-verified view of what is still blocked

## Why Texture Mesh is deferred

Texturing/refine is deferred on purpose. This rebuild does not claim production support for the native dependency stack required by the texture path, especially on Linux ARM64. Refine stays hidden until the dependency stack is available, runtime blockers are cleared, and the texturing path is reintroduced behind explicit acceptance gates.

Texture Mesh stays hidden in the public product surface until all dependency gates, import-time blockers, and runtime acceptance gates pass.

## Model source

Weights: [Aero-Ex/Trellis2-GGUF](https://huggingface.co/Aero-Ex/Trellis2-GGUF) on Hugging Face.
