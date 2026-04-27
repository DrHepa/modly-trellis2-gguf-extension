from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path, PureWindowsPath
from typing import Any, Literal


Capability = Literal["generate", "refine"]
State = Literal["available", "missing", "unsupported", "unknown"]
WheelResolutionState = Literal["resolved", "missing"]


@dataclass(frozen=True)
class DependencyDefinition:
    name: str
    package_name: str
    package_aliases: tuple[str, ...] = ()
    runtime_critical_for: tuple[Capability, ...] = ()
    import_time_for: tuple[Capability, ...] = ()
    notes: str = ""

    def all_aliases(self) -> tuple[str, ...]:
        aliases = (self.name, self.package_name, *self.package_aliases)
        deduped: list[str] = []
        for alias in aliases:
            normalized = alias.strip()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return tuple(deduped)

    def is_runtime_critical_for(self, capability: Capability) -> bool:
        return capability in self.runtime_critical_for

    def is_import_time_for(self, capability: Capability) -> bool:
        return capability in self.import_time_for

    def is_required_for(self, capability: Capability) -> bool:
        return self.is_runtime_critical_for(capability) or self.is_import_time_for(capability)


@dataclass(frozen=True)
class WheelSource:
    kind: Literal["wheelhouse", "index", "url", "public-index"]
    label: str
    location: str
    priority: int


@dataclass(frozen=True)
class WheelCandidate:
    requirement: str
    distribution: str
    filename: str
    source: WheelSource
    python_tag: str
    platform_tag: str
    version: str = ""
    build_tag: str = ""
    torch_tag: str = ""
    cuda_tag: str = ""
    url: str = ""
    local_path: str = ""

    @property
    def install_target(self) -> str:
        return self.local_path or self.url or self.filename


@dataclass(frozen=True)
class WheelResolutionResult:
    requirement: str
    state: WheelResolutionState
    selected_source: WheelSource | None = None
    selected_candidate: WheelCandidate | None = None
    tried_sources: tuple[WheelSource, ...] = ()
    checked_candidates: tuple[str, ...] = ()
    detail: str = ""


DEPENDENCY_CATALOG: dict[str, DependencyDefinition] = {
    "cumesh": DependencyDefinition(
        name="cumesh",
        package_name="cumesh",
        import_time_for=("generate", "refine"),
        runtime_critical_for=("generate", "refine"),
        notes="Geometry remeshing/export dependency imported at trellis2_image_to_3d module load time.",
    ),
    "o_voxel": DependencyDefinition(
        name="o_voxel",
        package_name="o-voxel",
        package_aliases=("o_voxel", "o-voxel"),
        import_time_for=("generate", "refine"),
        runtime_critical_for=("generate", "refine"),
        notes="Sparse voxel/native geometry dependency imported at trellis2_image_to_3d module load time.",
    ),
    "spconv": DependencyDefinition(
        name="spconv",
        package_name="spconv",
        package_aliases=("spconv", "spconv-cu*"),
        notes="Conditional alternate sparse backend dependency; default Generate uses flex_gemm, so spconv wheels are reported but not treated as default capability blockers.",
    ),
    "cumm": DependencyDefinition(
        name="cumm",
        package_name="cumm",
        package_aliases=("cumm", "cumm-cu*"),
        notes="Conditional alternate sparse backend dependency paired with spconv; default Generate uses flex_gemm, so cumm wheels are reported but not treated as default capability blockers.",
    ),
    "nvdiffrast": DependencyDefinition(
        name="nvdiffrast",
        package_name="nvdiffrast",
        runtime_critical_for=("refine",),
        import_time_for=("generate", "refine"),
        notes="Upstream trellis2_gguf import path may import this before refine is invoked, so missing wheels can block generate at import time.",
    ),
    "flex_gemm": DependencyDefinition(
        name="flex_gemm",
        package_name="flex-gemm",
        package_aliases=("flex_gemm", "flex-gemm"),
        runtime_critical_for=("refine",),
        import_time_for=("generate", "refine"),
        notes="Upstream trellis2_gguf import path may import this before refine is invoked, so missing wheels can block generate at import time.",
    ),
    "comfyui_gguf_support_files": DependencyDefinition(
        name="comfyui_gguf_support_files",
        package_name="ComfyUI-GGUF",
        package_aliases=("ComfyUI-GGUF", "comfyui-gguf", "comfyui_gguf_support_files"),
        runtime_critical_for=("generate", "refine"),
        notes="Setup installs GGUF support files (ops.py/dequant.py/loader.py); runtime still validates the actual files separately.",
    ),
}

KNOWN_DEPENDENCIES = tuple(DEPENDENCY_CATALOG)

COMFYUI_GGUF_REQUIRED_FILES: tuple[str, ...] = ("ops.py", "dequant.py", "loader.py")

_DEFAULT_PUBLIC_WHEEL_INDEX = "https://pozzettiandrea.github.io/cuda-wheels/"


def comfyui_gguf_target_dir(site_packages: str | Path) -> Path | PureWindowsPath:
    if isinstance(site_packages, Path):
        return site_packages.parent / "ComfyUI-GGUF"

    raw = str(site_packages)
    if _looks_like_windows_path(raw):
        return PureWindowsPath(raw).parent / "ComfyUI-GGUF"
    return Path(raw).parent / "ComfyUI-GGUF"


def comfyui_gguf_missing_files(target_dir: str | Path) -> tuple[str, ...]:
    target = Path(target_dir)
    return tuple(name for name in COMFYUI_GGUF_REQUIRED_FILES if not (target / name).exists())


def comfyui_gguf_support_file_report(target_dir: str | Path) -> dict[str, Any]:
    target = Path(target_dir)
    missing = comfyui_gguf_missing_files(target)
    if not missing:
        return {
            "status": "installed",
            "missing": [],
            "detail": f"verified required support files at {target}",
        }

    status = "missing" if len(missing) == len(COMFYUI_GGUF_REQUIRED_FILES) else "partial"
    detail = (
        "missing required support files: "
        + ", ".join(missing)
        + ". Re-run the extension setup/repair path so ComfyUI-GGUF support files are restored."
    )
    return {
        "status": status,
        "missing": list(missing),
        "detail": detail,
    }


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


def normalize_dependency_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def resolve_dependency_definition(name: str) -> DependencyDefinition | None:
    normalized_name = normalize_dependency_name(name)
    direct = DEPENDENCY_CATALOG.get(normalized_name)
    if direct:
        return direct

    for definition in DEPENDENCY_CATALOG.values():
        for alias in definition.all_aliases():
            normalized_alias = normalize_dependency_name(alias)
            if any(char in normalized_alias for char in "*?["):
                if fnmatchcase(normalized_name, normalized_alias):
                    return definition
            elif normalized_name == normalized_alias:
                return definition
    return None


def build_wheel_sources(
    trellis_wheelhouse: str | Path | None = None,
    trellis_extra_wheel_index: str | None = None,
    trellis_extra_wheel_url: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
    public_index_base: str = _DEFAULT_PUBLIC_WHEEL_INDEX,
) -> tuple[WheelSource, ...]:
    env_map = env if env is not None else os.environ
    resolved_wheelhouse = str(trellis_wheelhouse or env_map.get("TRELLIS2_WHEELHOUSE") or "").strip()
    resolved_extra_index = str(trellis_extra_wheel_index or env_map.get("TRELLIS2_EXTRA_WHEEL_INDEX") or "").strip()
    resolved_extra_url = str(trellis_extra_wheel_url or env_map.get("TRELLIS2_EXTRA_WHEEL_URL") or "").strip()

    sources: list[WheelSource] = []
    if resolved_wheelhouse:
        sources.append(
            WheelSource(
                kind="wheelhouse",
                label="local-wheelhouse",
                location=resolved_wheelhouse,
                priority=0,
            )
        )
    if resolved_extra_url:
        sources.append(
            WheelSource(
                kind="url",
                label="configured-extra-url",
                location=resolved_extra_url,
                priority=1,
            )
        )
    if resolved_extra_index:
        sources.append(
            WheelSource(
                kind="index",
                label="configured-extra-index",
                location=resolved_extra_index,
                priority=2,
            )
        )
    sources.append(
        WheelSource(
            kind="public-index",
            label="public-pozzettiandrea",
            location=public_index_base,
            priority=99,
        )
    )
    return tuple(sorted(sources, key=lambda source: (source.priority, source.label)))


def parse_wheel_candidate(filename: str, source: WheelSource, requirement: str | None = None, url: str = "") -> WheelCandidate:
    raw_path = Path(filename).expanduser()
    local_path = ""
    if source.kind == "wheelhouse":
        candidate_path = raw_path if raw_path.is_absolute() else Path(source.location).expanduser() / raw_path
        local_path = str(candidate_path.resolve())

    wheel_name = raw_path.name
    if not wheel_name.endswith(".whl"):
        raise ValueError(f"Unsupported wheel filename: {filename}")

    stem = wheel_name[:-4]
    match = re.match(
        r"^(?P<distribution>.+?)-(?P<version>\d[^-]*?(?:[+_.][^-]+)*)"
        r"(?:-(?P<build_tag>\d[^-]*))?-(?P<python_tag>[^-]+)-(?P<abi_tag>[^-]+)-(?P<platform_tag>[^-]+)$",
        stem,
    )
    if not match:
        raise ValueError(f"Invalid wheel filename: {filename}")

    distribution = match.group("distribution")
    version = match.group("version")
    build_tag = match.group("build_tag") or ""
    python_tag = match.group("python_tag")
    platform_tag = match.group("platform_tag")

    lower_name = wheel_name.lower()
    torch_match = re.search(r"torch(?P<tag>\d+(?:\.\d+)?)", lower_name)
    cuda_match = re.search(r"(?:^|[-_\.])(?:cu|cuda)(?P<tag>\d{2,3})(?:[-_\.]|$)", lower_name)
    resolved_requirement = requirement or distribution

    return WheelCandidate(
        requirement=normalize_dependency_name(resolved_requirement),
        distribution=distribution,
        filename=wheel_name,
        source=source,
        python_tag=python_tag,
        platform_tag=platform_tag,
        version=version,
        build_tag=build_tag,
        torch_tag=torch_match.group("tag") if torch_match else "",
        cuda_tag=cuda_match.group("tag") if cuda_match else "",
        url=url,
        local_path=local_path,
    )


def build_wheel_candidates(
    source: WheelSource,
    filenames: Iterable[str],
    requirement: str | None = None,
    *,
    url_map: Mapping[str, str] | None = None,
) -> tuple[WheelCandidate, ...]:
    candidates: list[WheelCandidate] = []
    resolved_urls = url_map or {}
    for filename in filenames:
        url = resolved_urls.get(filename, "")
        candidates.append(parse_wheel_candidate(filename, source, requirement=requirement, url=url))
    return tuple(sorted(candidates, key=lambda candidate: candidate.filename))


def resolve_wheel_candidate(
    requirement: str,
    env: RuntimeEnvironment,
    sources: Iterable[WheelSource],
    candidates: Iterable[WheelCandidate],
) -> WheelResolutionResult:
    definition = resolve_dependency_definition(requirement)
    canonical_requirement = definition.name if definition else normalize_dependency_name(requirement)
    ordered_sources = tuple(sorted(sources, key=lambda source: (source.priority, source.label)))
    candidate_list = tuple(sorted(candidates, key=lambda candidate: (candidate.source.priority, candidate.filename)))
    checked_candidates = tuple(candidate.filename for candidate in candidate_list)
    rejection_notes: list[str] = []

    for source in ordered_sources:
        source_matches = [candidate for candidate in candidate_list if candidate.source == source]
        compatible: list[WheelCandidate] = []
        for candidate in source_matches:
            match, reason = wheel_candidate_matches_requirement(candidate, canonical_requirement, env)
            if match:
                compatible.append(candidate)
            elif reason:
                rejection_notes.append(f"{candidate.filename}: {reason}")

        if compatible:
            chosen = sorted(
                compatible,
                key=lambda candidate: _wheel_candidate_rank_key(candidate, env),
                reverse=True,
            )[0]
            return WheelResolutionResult(
                requirement=canonical_requirement,
                state="resolved",
                selected_source=source,
                selected_candidate=chosen,
                tried_sources=ordered_sources,
                checked_candidates=checked_candidates,
                detail=f"selected {chosen.filename} from {source.label}",
            )

    detail_parts = [f"no compatible wheel found for {canonical_requirement}"]
    if rejection_notes:
        detail_parts.append("rejections: " + "; ".join(rejection_notes))
    return WheelResolutionResult(
        requirement=canonical_requirement,
        state="missing",
        tried_sources=ordered_sources,
        checked_candidates=checked_candidates,
        detail=" | ".join(detail_parts),
    )


def wheel_candidate_matches_requirement(
    candidate: WheelCandidate,
    requirement: str,
    env: RuntimeEnvironment,
) -> tuple[bool, str]:
    definition = resolve_dependency_definition(requirement)
    requirement_name = definition.name if definition else normalize_dependency_name(requirement)
    if not _distribution_matches_requirement(candidate.distribution, requirement_name, definition):
        return False, f"distribution {candidate.distribution} does not match {requirement_name} aliases"
    if not _python_tag_matches(candidate.python_tag, env.python_tag):
        return False, f"python tag {candidate.python_tag} does not match host {env.python_tag}"
    if not _platform_tag_matches(candidate.platform_tag, env.system, env.machine):
        return False, f"platform tag {candidate.platform_tag} is incompatible with {env.system}/{env.machine}"
    if not _encoded_tag_matches(candidate.torch_tag, env.torch_version, kind="torch"):
        return False, f"torch tag {candidate.torch_tag} does not match host {env.torch_version}"
    if not _encoded_tag_matches(candidate.cuda_tag, env.cuda_version, kind="cuda"):
        return False, f"cuda tag {candidate.cuda_tag} does not match host {env.cuda_version}"
    return True, ""


def discover_local_wheel_candidates(source: WheelSource, requirement: str | None = None) -> tuple[WheelCandidate, ...]:
    if source.kind != "wheelhouse":
        raise ValueError(f"Source {source.label} is not a wheelhouse")
    wheelhouse = Path(source.location).expanduser()
    if not wheelhouse.exists() or not wheelhouse.is_dir():
        return ()
    return build_wheel_candidates(source, [str(path.resolve()) for path in sorted(wheelhouse.glob("*.whl"))], requirement=requirement)


def _distribution_matches_requirement(
    distribution: str,
    requirement_name: str,
    definition: DependencyDefinition | None,
) -> bool:
    normalized_distribution = normalize_dependency_name(distribution)
    if definition is None:
        return normalized_distribution == normalize_dependency_name(requirement_name)

    aliases = definition.all_aliases()
    for alias in aliases:
        normalized_alias = normalize_dependency_name(alias)
        if any(char in normalized_alias for char in "*?["):
            if fnmatchcase(normalized_distribution, normalized_alias):
                return True
        elif normalized_distribution == normalized_alias:
            return True
    return False


def _looks_like_windows_path(value: str) -> bool:
    normalized = str(value or "").strip()
    return bool(re.match(r"^[a-zA-Z]:[\\/]", normalized) or "\\" in normalized)


def _wheel_candidate_rank_key(candidate: WheelCandidate, env: RuntimeEnvironment) -> tuple[object, ...]:
    exact_python = 1 if candidate.python_tag.strip().lower() == env.python_tag.strip().lower() else 0
    explicit_accel_matches = sum(
        1
        for tag, host, kind in (
            (candidate.torch_tag, env.torch_version, "torch"),
            (candidate.cuda_tag, env.cuda_version, "cuda"),
        )
        if tag and host and _encoded_tag_matches(tag, host, kind=kind)
    )
    return (
        exact_python,
        explicit_accel_matches,
        _version_rank_key(candidate.version),
        _build_rank_key(candidate.build_tag),
        tuple(-ord(char) for char in candidate.filename),
    )


def _version_rank_key(version: str) -> tuple[tuple[int, object], ...]:
    return _pep440ish_version_rank_key(version)


def _build_rank_key(build_tag: str) -> tuple[tuple[int, object], ...]:
    return _tokenize_rank_value(build_tag)


def _pep440ish_version_rank_key(version: str) -> tuple[tuple[int, object], ...]:
    normalized = str(version or "").strip().lower()
    if not normalized:
        return ()

    public, _, local = normalized.partition("+")
    release, qualifiers = _split_version_release_and_qualifiers(public)
    stage_rank, stage_number, post_number = _extract_version_qualifier_rank(qualifiers)

    return (
        (1, release),
        (1, stage_rank),
        (1, stage_number),
        (1, post_number),
        (1, _tokenize_rank_value(local)),
    )


def _split_version_release_and_qualifiers(version: str) -> tuple[tuple[int, ...], tuple[str, ...]]:
    match = re.match(r"^v?(?P<release>\d+(?:[._-]\d+)*)(?P<qualifiers>.*)$", version)
    if not match:
        return ((), tuple(re.findall(r"\d+|[a-z]+", version)))

    release_text = match.group("release")
    qualifiers_text = match.group("qualifiers")
    release = tuple(int(part) for part in re.split(r"[._-]", release_text) if part != "")
    while len(release) > 1 and release[-1] == 0:
        release = release[:-1]
    return (release, tuple(re.findall(r"\d+|[a-z]+", qualifiers_text)))


def _extract_version_qualifier_rank(qualifiers: tuple[str, ...]) -> tuple[int, int, int]:
    stage_aliases = {
        "dev": 0,
        "a": 1,
        "alpha": 1,
        "b": 2,
        "beta": 2,
        "c": 3,
        "rc": 3,
        "pre": 3,
        "preview": 3,
    }

    stage_rank = 4
    stage_number = 0
    post_number = 0

    for index, token in enumerate(qualifiers):
        if token == "post":
            post_number = _qualifier_number_at(qualifiers, index + 1)
            continue

        mapped_stage = stage_aliases.get(token)
        if mapped_stage is not None and stage_rank == 4:
            stage_rank = mapped_stage
            stage_number = _qualifier_number_at(qualifiers, index + 1)

    if post_number:
        stage_rank = 5

    return (stage_rank, stage_number, post_number)


def _qualifier_number_at(tokens: tuple[str, ...], index: int) -> int:
    if 0 <= index < len(tokens) and tokens[index].isdigit():
        return int(tokens[index])
    return 0


def _tokenize_rank_value(value: str) -> tuple[tuple[int, object], ...]:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return ()

    tokens = re.findall(r"\d+|[a-z]+", normalized)
    ranked: list[tuple[int, object]] = []
    for token in tokens:
        if token.isdigit():
            ranked.append((1, int(token)))
        else:
            ranked.append((0, token))
    return tuple(ranked)


def _python_tag_matches(candidate_python_tag: str, host_python_tag: str) -> bool:
    normalized_candidate = candidate_python_tag.strip().lower()
    normalized_host = host_python_tag.strip().lower()
    if not normalized_candidate or not normalized_host:
        return True
    if normalized_candidate == normalized_host:
        return True
    if normalized_candidate.startswith("py3") and normalized_host.startswith("cp3"):
        return True
    return False


def _platform_tag_matches(candidate_platform_tag: str, system: str, machine: str) -> bool:
    normalized_platform = candidate_platform_tag.strip().lower()
    normalized_system = system.strip().lower()
    normalized_machine = machine.strip().lower()

    if normalized_platform == "any":
        return True

    host_os, host_arch = _host_platform_signature(normalized_system, normalized_machine)
    candidate_os, candidate_arch = _candidate_platform_signature(normalized_platform)
    if not host_os or not host_arch:
        return False
    if not candidate_os or not candidate_arch:
        return False
    return host_os == candidate_os and host_arch == candidate_arch


def _host_platform_signature(system: str, machine: str) -> tuple[str, str]:
    if system == "linux" and machine in {"aarch64", "arm64"}:
        return ("linux", "aarch64")
    if system == "linux" and machine in {"x86_64", "amd64"}:
        return ("linux", "x86_64")
    if system == "windows" and machine in {"x86_64", "amd64"}:
        return ("windows", "x86_64")
    return ("", "")


def _candidate_platform_signature(platform_tag: str) -> tuple[str, str]:
    normalized_platform = platform_tag.lower()
    if "manylinux" in normalized_platform or "linux" in normalized_platform:
        candidate_os = "linux"
    elif "win" in normalized_platform:
        candidate_os = "windows"
    else:
        candidate_os = ""

    if "aarch64" in normalized_platform or "arm64" in normalized_platform:
        candidate_arch = "aarch64"
    elif "x86_64" in normalized_platform or "amd64" in normalized_platform:
        candidate_arch = "x86_64"
    else:
        candidate_arch = ""

    return (candidate_os, candidate_arch)


def _encoded_tag_matches(candidate_tag: str, host_value: str, *, kind: Literal["torch", "cuda"]) -> bool:
    if not candidate_tag:
        return True
    if not host_value:
        return False
    normalized_candidate = _normalize_encoded_tag(candidate_tag, kind=kind)
    normalized_host = _normalize_encoded_tag(host_value, kind=kind)
    return bool(normalized_candidate and normalized_host and normalized_candidate == normalized_host)


def _normalize_encoded_tag(value: str, *, kind: Literal["torch", "cuda"]) -> str:
    lowered = str(value).strip().lower().replace("+", "")
    if kind == "torch":
        parts = lowered.split(".")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            return f"{int(parts[0])}{int(parts[1])}"
        digits = re.sub(r"[^0-9]", "", lowered)
        return digits[:2] if len(digits) >= 2 else digits
    digits = re.sub(r"[^0-9]", "", lowered)
    return digits


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
    @staticmethod
    def catalog() -> Mapping[str, DependencyDefinition]:
        return DEPENDENCY_CATALOG

    @classmethod
    def definitions_for(cls, capability: Capability) -> tuple[DependencyDefinition, ...]:
        return tuple(
            definition
            for definition in cls.catalog().values()
            if definition.is_required_for(capability)
        )

    @classmethod
    def evaluate(cls, capability: Capability, env: RuntimeEnvironment) -> CapabilityReport:
        deps = tuple(cls.classify_all(env).values())
        blockers: list[str] = []

        if capability == "generate":
            blockers.extend(cls._blockers_for_dependencies(deps, cls.definitions_for(capability), capability))
            blockers.extend(cls._asset_blockers(env, capability))
        elif capability == "refine":
            blockers.extend(cls._refine_platform_blockers(env))
            blockers.extend(cls._blockers_for_dependencies(deps, cls.definitions_for(capability), capability))
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
        definition = cls.catalog().get(name)
        provided = cls._provided_dependency_value(name, env.known_dependencies)
        if isinstance(provided, DependencyStatus):
            return DependencyStatus(name=name, state=provided.state, detail=provided.detail)
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
                detail=definition.notes if definition and definition.notes else "source-build-only / unverified on Linux ARM64",
            )
        if system == "linux" and machine in {"x86_64", "amd64"}:
            return DependencyStatus(name=name, state="unknown", detail="not verified from fixture data")
        if system == "windows" and machine in {"x86_64", "amd64"}:
            return DependencyStatus(name=name, state="unknown", detail="not verified from fixture data")
        return DependencyStatus(name=name, state="unknown", detail="platform classification not modeled yet")

    @classmethod
    def _provided_dependency_value(
        cls,
        name: str,
        provided: Mapping[str, Any],
    ) -> Any:
        if name in provided:
            return provided[name]

        definition = cls.catalog().get(name)
        if not definition:
            return None

        normalized_candidates = {str(key).strip().lower(): value for key, value in provided.items()}
        for alias in definition.all_aliases():
            lowered = alias.lower()
            if lowered in normalized_candidates:
                return normalized_candidates[lowered]

        for key, value in provided.items():
            normalized_key = str(key).strip().lower()
            for alias in definition.all_aliases():
                if any(char in alias for char in "*?["):
                    if fnmatchcase(normalized_key, alias.lower()):
                        return value
                elif normalized_key == alias.lower():
                    return value
        return None

    @staticmethod
    def _asset_blockers(env: RuntimeEnvironment, capability: Capability) -> list[str]:
        if env.asset_groups.get(capability, False):
            return []
        return [f"{capability} assets are missing or unresolved"]

    @classmethod
    def _blockers_for_dependencies(
        cls,
        deps: tuple[DependencyStatus, ...],
        required: tuple[DependencyDefinition, ...],
        capability: Capability,
    ) -> list[str]:
        dep_map = {dep.name: dep for dep in deps}
        blockers: list[str] = []
        for definition in required:
            dep = dep_map[definition.name]
            if dep.state == "available":
                continue

            labels: list[str] = []
            if definition.is_import_time_for(capability):
                labels.append("import-time blocker")
            if definition.is_runtime_critical_for(capability):
                labels.append("runtime-critical")
            if dep.detail:
                labels.append(dep.detail)

            suffix = f" ({'; '.join(labels)})" if labels else ""
            blockers.append(f"{definition.name}: {dep.state}{suffix}")
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
