from __future__ import annotations

import tempfile
from pathlib import Path

from runtime_support import (
    DependencyPreflight,
    DuplicateAssetError,
    HFAssetResolver,
    MissingAssetError,
    RuntimeEnvironment,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture\n", encoding="utf-8")


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _seed_generate_tree(root: Path) -> None:
    _touch(root / "pipeline.json")
    for rel in ("Vision", "decoders", "encoders", "shape", "refiner"):
        _mkdir(root / rel)


def _seed_refine_tree(root: Path) -> None:
    _seed_generate_tree(root)
    _touch(root / "texturing_pipeline.json")
    _mkdir(root / "texture")


def test_linux_arm64_without_native_deps() -> None:
    env = RuntimeEnvironment(
        system="Linux",
        machine="aarch64",
        python_tag="cp311",
        python_version="3.11.9",
        torch_version="2.7.0",
        cuda_version="12.4",
        gpu_sm="87",
        known_dependencies={name: False for name in ("flex_gemm", "cumesh", "nvdiffrast", "o_voxel", "spconv", "cumm")},
        asset_groups={"generate": False, "refine": False},
        refine_lab_verified=False,
    )
    generate = DependencyPreflight.evaluate("generate", env)
    refine = DependencyPreflight.evaluate("refine", env)
    assert generate.state == "missing", generate
    assert refine.state == "unsupported", refine
    assert any("Linux ARM64" in blocker for blocker in refine.blockers), refine.blockers


def test_linux_x86_64_all_native_deps_available() -> None:
    env = RuntimeEnvironment(
        system="Linux",
        machine="x86_64",
        python_tag="cp311",
        python_version="3.11.9",
        torch_version="2.7.0",
        cuda_version="12.4",
        gpu_sm="89",
        known_dependencies={name: True for name in ("flex_gemm", "cumesh", "nvdiffrast", "o_voxel", "spconv", "cumm")},
        asset_groups={"generate": True, "refine": True},
        refine_lab_verified=True,
    )
    generate = DependencyPreflight.evaluate("generate", env)
    refine = DependencyPreflight.evaluate("refine", env)
    assert generate.allowed and generate.state == "available", generate
    assert refine.allowed and refine.state == "available", refine


def test_asset_resolver_success_and_missing_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _seed_refine_tree(root)
        geometry = HFAssetResolver.resolve_geometry(root)
        refine = HFAssetResolver.resolve_refine(root)
        assert set(geometry.paths) == set(HFAssetResolver.expected_generate_paths())
        assert set(refine.paths) == set(HFAssetResolver.expected_refine_paths())

        missing_root = root / "missing"
        _seed_generate_tree(missing_root)
        try:
            HFAssetResolver.resolve_refine(missing_root)
        except MissingAssetError as exc:
            assert "texturing_pipeline.json" in str(exc), exc
        else:
            raise AssertionError("resolve_refine should fail when refine assets are absent")


def test_no_recursive_first_hit_behavior() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _seed_generate_tree(root)
        (root / "pipeline.json").unlink()
        _touch(root / "archive" / "pipeline.json")
        try:
            HFAssetResolver.resolve_geometry(root)
        except DuplicateAssetError as exc:
            assert "archive/pipeline.json" in str(exc), exc
        else:
            raise AssertionError("resolver must reject decoy recursive matches")


def test_generator_source_generate_only_contract() -> None:
    source = (Path(__file__).parent / "generator.py").read_text(encoding="utf-8")
    manifest = (Path(__file__).parent / "manifest.json").read_text(encoding="utf-8")
    readme = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")
    assert "GenerateAdapter" in source
    assert "_MODEL_MANAGER_PATHS" in source
    assert "HFAssetResolver.resolve_geometry" in source
    assert 'DependencyPreflight.evaluate("generate"' in source
    assert "Texture Mesh / refine is intentionally unavailable" in source
    assert "texture_mesh(" not in source
    assert "urlretrieve(" not in source
    assert "_glob.glob(pattern, recursive=True)" not in source
    assert "rglob(filename)" not in source
    assert "remesh_resolution" not in source
    assert "remesh_resolution" not in manifest
    assert "remesh_resolution" not in readme


def main() -> None:
    tests = [
        test_linux_arm64_without_native_deps,
        test_linux_x86_64_all_native_deps_available,
        test_asset_resolver_success_and_missing_failure,
        test_no_recursive_first_hit_behavior,
        test_generator_source_generate_only_contract,
    ]
    for test in tests:
        test()
    print(f"runtime_support validation passed ({len(tests)} tests)")


if __name__ == "__main__":
    main()
