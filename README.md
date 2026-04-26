# Trellis.2 GGUF — Modly Extension

This clean rebuild currently supports **Generate Mesh only** from the [Aero-Ex/Trellis2-GGUF](https://huggingface.co/Aero-Ex/Trellis2-GGUF) Hugging Face repo.

## Current scope

- Public runtime scope: **Generate Mesh** only
- `Texture Mesh` / refine is **intentionally not exposed** in this clean slice
- Linux ARM64 refine/texturing remains **blocked** pending external native dependency lab evidence
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

## Why Texture Mesh is deferred

Texturing/refine is deferred on purpose. This rebuild does not claim production support for the native dependency stack required by the texture path, especially on Linux ARM64. Refine stays hidden until external evidence exists for the dependency matrix and until the texturing path is reintroduced behind explicit acceptance gates.

See `docs/dependency-lab.md` for the public dependency-lab expectations.

## Deferred work

- `RefineAdapter` wired to `Trellis2TexturingPipeline`
- `tex_slat_flow_model.forward` cond-sensitivity acceptance gate
- External Linux ARM64 dependency lab execution and evidence review
- Future geometry export cleanup, if follow-up validation shows it is needed

## Contributor validation

Run these lightweight checks before review:

- `PYTHONDONTWRITEBYTECODE=1 python3 -B -m py_compile runtime_support.py validate_clean_hf_runtime_support.py setup.py generator.py validate_clean_hf_rebuild.py`
- `PYTHONDONTWRITEBYTECODE=1 python3 validate_clean_hf_runtime_support.py`
- `PYTHONDONTWRITEBYTECODE=1 python3 validate_clean_hf_rebuild.py`
- `python3 -m json.tool manifest.json >/dev/null`

## Model source

Weights: [Aero-Ex/Trellis2-GGUF](https://huggingface.co/Aero-Ex/Trellis2-GGUF) on Hugging Face.
