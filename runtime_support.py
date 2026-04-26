from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


Capability = Literal["generate", "refine"]
State = Literal["available", "missing", "unsupported", "unknown"]

KNOWN_DEPENDENCIES = (
    "flex_gemm",
    "cumesh",
    "nvdiffrast",
    "o_voxel",
    "spconv",
    "cumm",
)


@dataclass(frozen=True)
class RuntimeEnvironment:
    system: str
    machine: str
    python_tag: str
    python_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    gpu_sm: str | int | None = None
    known_dependencies: Mapping[str, Any] = field(default_factory=dict)
    asset_groups: Mapping[str, bool] = field(default_factory=dict)
    refine_lab_verified: bool = False


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    state: State
    detail: str = ""


@dataclass(frozen=True)
class CapabilityReport:
    capability: Capability
    state: State
    allowed: bool
    blockers: tuple[str, ...]
    deps: tuple[DependencyStatus, ...]
    env: RuntimeEnvironment


@dataclass(frozen=True)
class AssetPath:
    relative_path: str
    kind: Literal["file", "dir"]
    description: str = ""


@dataclass(frozen=True)
class GeometryAssets:
    root: Path
    paths: Mapping[str, Path]


@dataclass(frozen=True)
class RefineAssets:
    root: Path
    paths: Mapping[str, Path]


class AssetResolutionError(RuntimeError):
    pass


class MissingAssetError(AssetResolutionError):
    pass


class DuplicateAssetError(AssetResolutionError):
    pass


class HFAssetResolver:
    GENERATE_REQUIRED: tuple[AssetPath, ...] = (
        AssetPath("pipeline.json", "file", "geometry pipeline config"),
        AssetPath("Vision", "dir", "vision encoder directory"),
        AssetPath("decoders", "dir", "decoder weights/config directory"),
        AssetPath("encoders", "dir", "encoder weights/config directory"),
        AssetPath("shape", "dir", "shape model directory"),
        AssetPath("refiner", "dir", "shared refiner directory"),
    )
    REFINE_REQUIRED: tuple[AssetPath, ...] = (
        AssetPath("texturing_pipeline.json", "file", "texturing pipeline config"),
        AssetPath("texture", "dir", "texture model directory"),
    )

    @classmethod
    def expected_generate_paths(cls) -> tuple[str, ...]:
        return tuple(item.relative_path for item in cls.GENERATE_REQUIRED)

    @classmethod
    def expected_refine_paths(cls) -> tuple[str, ...]:
        return tuple(item.relative_path for item in cls.GENERATE_REQUIRED + cls.REFINE_REQUIRED)

    @classmethod
    def resolve_geometry(cls, root: str | Path) -> GeometryAssets:
        base = Path(root)
        return GeometryAssets(root=base, paths=cls._resolve_required(base, cls.GENERATE_REQUIRED))

    @classmethod
    def resolve_refine(cls, root: str | Path) -> RefineAssets:
        base = Path(root)
        required = cls.GENERATE_REQUIRED + cls.REFINE_REQUIRED
        return RefineAssets(root=base, paths=cls._resolve_required(base, required))

    @classmethod
    def _resolve_required(
        cls,
        root: Path,
        required: tuple[AssetPath, ...],
    ) -> dict[str, Path]:
        resolved: dict[str, Path] = {}
        for asset in required:
            resolved[asset.relative_path] = cls._resolve_exact(root, asset)
        return resolved

    @staticmethod
    def _resolve_exact(root: Path, asset: AssetPath) -> Path:
        candidate = root / asset.relative_path
        if candidate.exists():
            if asset.kind == "file" and not candidate.is_file():
                raise MissingAssetError(
                    f"Expected file at '{asset.relative_path}', found non-file entry."
                )
            if asset.kind == "dir" and not candidate.is_dir():
                raise MissingAssetError(
                    f"Expected directory at '{asset.relative_path}', found non-directory entry."
                )
            return candidate

        duplicates = []
        for match in root.rglob(candidate.name):
            try:
                relative = match.relative_to(root).as_posix()
            except ValueError:
                continue
            if relative != asset.relative_path:
                duplicates.append(relative)

        if duplicates:
            if len(duplicates) == 1:
                raise DuplicateAssetError(
                    f"Missing canonical asset '{asset.relative_path}'. Found decoy at '{duplicates[0]}' instead."
                )
            formatted = ", ".join(sorted(duplicates))
            raise DuplicateAssetError(
                f"Missing canonical asset '{asset.relative_path}'. Found ambiguous decoys: {formatted}."
            )

        raise MissingAssetError(f"Missing required asset '{asset.relative_path}'.")


class DependencyPreflight:
    GENERATE_REQUIRED = ("cumesh", "o_voxel", "spconv", "cumm")
    REFINE_REQUIRED = ("cumesh", "o_voxel", "spconv", "cumm", "nvdiffrast", "flex_gemm")

    @classmethod
    def evaluate(cls, capability: Capability, env: RuntimeEnvironment) -> CapabilityReport:
        deps = tuple(cls.classify_all(env).values())
        blockers: list[str] = []

        if capability == "generate":
            blockers.extend(cls._blockers_for_dependencies(env, deps, cls.GENERATE_REQUIRED))
            blockers.extend(cls._asset_blockers(env, capability))
        elif capability == "refine":
            blockers.extend(cls._refine_platform_blockers(env))
            blockers.extend(cls._blockers_for_dependencies(env, deps, cls.REFINE_REQUIRED))
            blockers.extend(cls._asset_blockers(env, capability))
        else:
            raise ValueError(f"Unsupported capability: {capability}")

        state = cls._capability_state(blockers, deps)
        return CapabilityReport(
            capability=capability,
            state=state,
            allowed=state == "available",
            blockers=tuple(blockers),
            deps=deps,
            env=env,
        )

    @classmethod
    def classify_all(cls, env: RuntimeEnvironment) -> dict[str, DependencyStatus]:
        return {name: cls.classify_dependency(name, env) for name in KNOWN_DEPENDENCIES}

    @classmethod
    def classify_dependency(cls, name: str, env: RuntimeEnvironment) -> DependencyStatus:
        provided = env.known_dependencies.get(name)
        if isinstance(provided, DependencyStatus):
            return provided
        if isinstance(provided, bool):
            return DependencyStatus(name=name, state="available" if provided else "missing")
        if isinstance(provided, str) and provided in ("available", "missing", "unsupported", "unknown"):
            return DependencyStatus(name=name, state=provided)
        if isinstance(provided, Mapping):
            state = provided.get("state", "unknown")
            detail = str(provided.get("detail", ""))
            if state in ("available", "missing", "unsupported", "unknown"):
                return DependencyStatus(name=name, state=state, detail=detail)

        system = env.system.lower()
        machine = env.machine.lower()

        if system == "linux" and machine in {"aarch64", "arm64"}:
            return DependencyStatus(
                name=name,
                state="unsupported",
                detail="source-build-only / unverified on Linux ARM64",
            )
        if system == "linux" and machine in {"x86_64", "amd64"}:
            return DependencyStatus(name=name, state="unknown", detail="not verified from fixture data")
        if system == "windows" and machine in {"x86_64", "amd64"}:
            return DependencyStatus(name=name, state="unknown", detail="not verified from fixture data")
        return DependencyStatus(name=name, state="unknown", detail="platform classification not modeled yet")

    @staticmethod
    def _asset_blockers(env: RuntimeEnvironment, capability: Capability) -> list[str]:
        if env.asset_groups.get(capability, False):
            return []
        return [f"{capability} assets are missing or unresolved"]

    @classmethod
    def _blockers_for_dependencies(
        cls,
        env: RuntimeEnvironment,
        deps: tuple[DependencyStatus, ...],
        required: tuple[str, ...],
    ) -> list[str]:
        dep_map = {dep.name: dep for dep in deps}
        blockers: list[str] = []
        for name in required:
            dep = dep_map[name]
            if dep.state == "available":
                continue
            suffix = f" ({dep.detail})" if dep.detail else ""
            blockers.append(f"{name}: {dep.state}{suffix}")
        return blockers

    @staticmethod
    def _refine_platform_blockers(env: RuntimeEnvironment) -> list[str]:
        system = env.system.lower()
        machine = env.machine.lower()
        blockers: list[str] = []
        if system == "linux" and machine in {"aarch64", "arm64"}:
            blockers.append("refine is unsupported on Linux ARM64 until an external dependency lab verifies it")
        elif not env.refine_lab_verified:
            blockers.append("refine is unsupported until native dependencies and external lab evidence are verified")
        return blockers

    @staticmethod
    def _capability_state(
        blockers: list[str],
        deps: tuple[DependencyStatus, ...],
    ) -> State:
        if not blockers:
            return "available"
        joined = " | ".join(blockers)
        if "unsupported" in joined:
            return "unsupported"
        if "missing" in joined:
            return "missing"
        if any(dep.state == "unknown" for dep in deps):
            return "unknown"
        return "unknown"
