from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


FLEX_GEMM_ALGO_ENV = "FLEX_GEMM_ALGO"
FLEX_GEMM_DEBUG_ENV = "TRELLIS2_FLEX_GEMM_DEBUG"
FLEX_GEMM_DEBUG_LOG_PATH_ENV = "TRELLIS2_FLEX_GEMM_DEBUG_LOG_PATH"
FLEX_GEMM_EXPLICIT_ALGO = "explicit_gemm"
FLEX_GEMM_DEFAULT_ALGO = "implicit_gemm_splitk"
FLEX_GEMM_IMPLICIT_GEMM_ALGO = "implicit_gemm"
SUPPORTED_FLEX_GEMM_ALGOS = (
    FLEX_GEMM_DEFAULT_ALGO,
    FLEX_GEMM_IMPLICIT_GEMM_ALGO,
    FLEX_GEMM_EXPLICIT_ALGO,
)
TRITON_OVERRIDE_VERSION = "3.6.0"
TORCH_TRITON_PIN = "3.3.0"
GB10_GPU_SM = 121
TRELLIS_PATCH_BACKUP_SUFFIX = ".arm64-runtime.bak"
TRELLIS_CONFIG_RELATIVE_PATH = "trellis2_gguf/modules/sparse/conv/config.py"
TRELLIS_DEBUG_RELATIVE_PATH = "trellis2_gguf/modules/sparse/conv/conv_flex_gemm.py"
TRELLIS_FDG_VAE_RELATIVE_PATH = "trellis2_gguf/models/sc_vaes/fdg_vae.py"
TRELLIS_CONFIG_ALGO_MARKER = 'ALGORITHM = "implicit_gemm_splitk"'
TRELLIS_CONFIG_EXPLICIT_SNIPPET = 'ALGORITHM = "explicit_gemm"'
TRELLIS_DEBUG_MARKER = "# TRELLIS2_FLEX_GEMM_DEBUG_GATE"
TRELLIS_DEBUG_ANCHOR = "def forward("
FDG_VAE_PATCH_NAME = "fdg_vae_mesh_compat"
FDG_VAE_IMPORT_MARKER = "tiled_flexible_dual_grid_to_mesh"
FDG_VAE_UNPATCHED_IMPORT_SNIPPET = """from trellis.representations.mesh.flexicubes.flexicubes import (\n    FlexiCubes,\n    flexible_dual_grid_to_mesh,\n    sparse_cube2verts,\n    tiled_flexible_dual_grid_to_mesh,\n)\n"""
FDG_VAE_PATCHED_IMPORT_SNIPPET = """from trellis.representations.mesh.flexicubes.flexicubes import (\n    FlexiCubes,\n    flexible_dual_grid_to_mesh,\n    sparse_cube2verts,\n)\n"""
FDG_VAE_UNPATCHED_CALL_SNIPPET = """        vertices, faces, reg_loss = tiled_flexible_dual_grid_to_mesh(\n            flexicubes,\n            voxels,\n            scalar_field,\n            cube_idx,\n            resolution,\n        )\n"""
FDG_VAE_PATCHED_CALL_SNIPPET = """        vertices, faces, reg_loss = flexible_dual_grid_to_mesh(\n            flexicubes,\n            voxels,\n            scalar_field,\n            cube_idx,\n            resolution,\n            train=False,\n        )\n"""
VALIDATION_CONSTRAINTS = (
    "no Generate",
    "no model load",
    "no asset download",
)
DEFAULT_IMPORT_API_TIMEOUT_SEC = 20
DEFAULT_GENERATOR_BOOTSTRAP_TIMEOUT_SEC = 20
DEFAULT_FLEX_GEMM_TIMEOUT_SEC = 20
RUN_RUNTIME_VALIDATION_ENV = "TRELLIS2_RUN_RUNTIME_VALIDATION"


def _normalize_system(system: str | None) -> str:
    return str(system or "").strip().lower()


def _normalize_machine(machine: str | None) -> str:
    return str(machine or "").strip().lower()


def _normalize_gpu_sm(gpu_sm: str | int | None) -> int | None:
    if gpu_sm is None or gpu_sm == "":
        return None
    return int(gpu_sm)


def _env_value(env: Mapping[str, str] | None, key: str) -> str:
    if env is None:
        return ""
    return str(env.get(key, "")).strip()


def parse_env_flag(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _normalize_flex_gemm_algo(value: str | None) -> str:
    return str(value or "").strip().lower()


def _build_patch_path(relative_path: str, install_root: str | None) -> str:
    if install_root:
        return str(PurePosixPath(str(install_root).rstrip("/")) / relative_path)
    return relative_path


def _build_backup_path(target_file: str, backup_suffix: str = TRELLIS_PATCH_BACKUP_SUFFIX) -> str:
    return f"{target_file}{backup_suffix}"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _replace_once(value: str, old: str, new: str) -> str | None:
    if value.count(old) != 1:
        return None
    return value.replace(old, new, 1)


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class HostPlatform:
    system: str
    machine: str
    gpu_sm: int | None = None
    python_tag: str = ""
    python_version: str = ""

    @property
    def normalized_system(self) -> str:
        return _normalize_system(self.system)

    @property
    def normalized_machine(self) -> str:
        return _normalize_machine(self.machine)

    @property
    def normalized_gpu_sm(self) -> int | None:
        return _normalize_gpu_sm(self.gpu_sm)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class FlexGemmAlgoDecision:
    selected: str
    source: str
    targeted_host: bool
    status: str = "selected"
    env_override: str = ""
    blocking_reason: str = ""
    supported_algorithms: tuple[str, ...] = SUPPORTED_FLEX_GEMM_ALGOS
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class FlexGemmDebugPolicy:
    enabled: bool
    env_gate: str
    log_path: str | None
    tensor_dumps: bool = False
    value_dumps: bool = False

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class TritonOverridePlan:
    status: str
    applies: bool
    target_version: str | None
    current_version: str = ""
    install_allowed: bool = False
    targeted_host: bool = False
    reason: str = ""
    command: tuple[str, ...] = ()
    torch_version: str = ""
    torch_triton_pin: str = ""
    conflict_note: str = ""
    reinstall_torch: bool = False
    install_scope: str = "python_package_only"

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class RuntimeOverridesReport:
    host: dict[str, Any]
    flex_gemm: dict[str, Any]
    flex_gemm_debug: dict[str, Any]
    triton: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class TrellisFilePatchPlan:
    patch_name: str
    target_file: str
    status: str
    targeted_host: bool
    backup_path: str
    expected_marker: str
    planned_value: str = ""
    current_value: str = ""
    source: str = ""
    replacement_snippet: str = ""
    blocking_reason: str = ""
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class TrellisPatchesReport:
    status: str = "not_run"
    items: tuple[dict[str, Any], ...] = ()
    blocking_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class RuntimeValidationReport:
    status: str = "not_run"
    checks: tuple[dict[str, Any], ...] = ()
    blocking_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class WheelProvenanceReport:
    status: str = "planned"
    required_packages: tuple[str, ...] = ("cumesh", "flex_gemm", "o_voxel", "nvdiffrast")
    entries: tuple[dict[str, Any], ...] = ()
    sources: tuple[dict[str, Any], ...] = ()
    coverage: dict[str, Any] = field(default_factory=lambda: {"required": 4, "available": 0, "missing": ["cumesh", "flex_gemm", "o_voxel", "nvdiffrast"]})
    blocking_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    public_evidence_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class TextPatchPlan:
    patch_name: str
    target_file: str
    relative_path: str
    status: str
    backup_path: str
    backup_suffix: str
    expected_marker: str
    before_hash: str = ""
    after_hash: str = ""
    drift_reason: str = ""
    patched_text: str = ""
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class ValidationCommandPlan:
    label: str
    check_name: str
    timeout_sec: int
    command: tuple[str, ...]
    script_content: str
    env: dict[str, str] = field(default_factory=dict)
    constraints: tuple[str, ...] = VALIDATION_CONSTRAINTS
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class ValidationCommandResult:
    label: str
    check_name: str
    status: str
    timeout_sec: int
    command: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    constraints: tuple[str, ...] = VALIDATION_CONSTRAINTS
    notes: tuple[str, ...] = ()
    returncode: int | None = None
    duration_sec: float = 0.0
    detail: str = ""
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def is_linux_arm64(host: HostPlatform) -> bool:
    return host.normalized_system == "linux" and host.normalized_machine in {"aarch64", "arm64"}


def detect_linux_arm64_blackwell(host: HostPlatform) -> bool:
    return is_linux_arm64(host) and host.normalized_gpu_sm == GB10_GPU_SM


def resolve_flex_gemm_algo(
    env: Mapping[str, str] | None,
    host: HostPlatform,
) -> FlexGemmAlgoDecision:
    override = _env_value(env, FLEX_GEMM_ALGO_ENV)
    targeted_host = detect_linux_arm64_blackwell(host)
    if override:
        normalized_override = _normalize_flex_gemm_algo(override)
        if normalized_override not in SUPPORTED_FLEX_GEMM_ALGOS:
            return FlexGemmAlgoDecision(
                selected="",
                source="env_override",
                targeted_host=targeted_host,
                status="invalid_override",
                env_override=override,
                blocking_reason=(
                    f"Unsupported FLEX_GEMM_ALGO override '{override}'. Supported values: "
                    f"{', '.join(SUPPORTED_FLEX_GEMM_ALGOS)}"
                ),
                evidence={
                    "env_var": FLEX_GEMM_ALGO_ENV,
                    "host": host.to_dict(),
                    "supported_algorithms": list(SUPPORTED_FLEX_GEMM_ALGOS),
                },
            )
        return FlexGemmAlgoDecision(
            selected=normalized_override,
            source="env_override",
            targeted_host=targeted_host,
            status="selected",
            env_override=override,
            evidence={
                "env_var": FLEX_GEMM_ALGO_ENV,
                "host": host.to_dict(),
                "normalized_override": normalized_override,
            },
        )

    if targeted_host:
        return FlexGemmAlgoDecision(
            selected=FLEX_GEMM_EXPLICIT_ALGO,
            source="linux_arm64_blackwell_policy",
            targeted_host=True,
            status="selected",
            evidence={
                "gpu_sm": host.normalized_gpu_sm,
                "host": host.to_dict(),
            },
        )

    return FlexGemmAlgoDecision(
        selected=FLEX_GEMM_DEFAULT_ALGO,
        source="default",
        targeted_host=False,
        status="selected",
        evidence={
            "host": host.to_dict(),
        },
    )


def resolve_flex_gemm_debug_policy(env: Mapping[str, str] | None) -> FlexGemmDebugPolicy:
    env_gate = _env_value(env, FLEX_GEMM_DEBUG_ENV)
    log_path = _env_value(env, FLEX_GEMM_DEBUG_LOG_PATH_ENV) or None
    enabled = parse_env_flag(env_gate)
    return FlexGemmDebugPolicy(
        enabled=enabled,
        env_gate=FLEX_GEMM_DEBUG_ENV,
        log_path=log_path if enabled else None,
    )


def build_flex_gemm_config_patch_plan(
    host: HostPlatform,
    env: Mapping[str, str] | None,
    *,
    installed_config_text: str | None = None,
    install_root: str | None = None,
    backup_suffix: str = TRELLIS_PATCH_BACKUP_SUFFIX,
) -> TrellisFilePatchPlan:
    decision = resolve_flex_gemm_algo(env, host)
    target_file = _build_patch_path(TRELLIS_CONFIG_RELATIVE_PATH, install_root)
    backup_path = _build_backup_path(target_file, backup_suffix)

    if decision.status == "invalid_override":
        return TrellisFilePatchPlan(
            patch_name="flex_gemm_config",
            target_file=target_file,
            status="invalid_override",
            targeted_host=decision.targeted_host,
            backup_path=backup_path,
            expected_marker=TRELLIS_CONFIG_ALGO_MARKER,
            source=decision.source,
            blocking_reason=decision.blocking_reason,
            notes=("Fail closed: explicit env override is unsupported.",),
        )

    if not decision.targeted_host:
        return TrellisFilePatchPlan(
            patch_name="flex_gemm_config",
            target_file=target_file,
            status="not_needed",
            targeted_host=False,
            backup_path=backup_path,
            expected_marker=TRELLIS_CONFIG_ALGO_MARKER,
            planned_value=decision.selected,
            source=decision.source,
            notes=("Non-target host keeps upstream config unchanged.",),
        )

    if installed_config_text is not None:
        if TRELLIS_CONFIG_EXPLICIT_SNIPPET in installed_config_text:
            return TrellisFilePatchPlan(
                patch_name="flex_gemm_config",
                target_file=target_file,
                status="already_applied",
                targeted_host=True,
                backup_path=backup_path,
                expected_marker=TRELLIS_CONFIG_ALGO_MARKER,
                planned_value=decision.selected,
                current_value=FLEX_GEMM_EXPLICIT_ALGO,
                source=decision.source,
                replacement_snippet=TRELLIS_CONFIG_EXPLICIT_SNIPPET,
            )
        if TRELLIS_CONFIG_ALGO_MARKER not in installed_config_text:
            return TrellisFilePatchPlan(
                patch_name="flex_gemm_config",
                target_file=target_file,
                status="drift",
                targeted_host=True,
                backup_path=backup_path,
                expected_marker=TRELLIS_CONFIG_ALGO_MARKER,
                planned_value=decision.selected,
                source=decision.source,
                blocking_reason="Expected FlexGEMM config marker is missing; refusing loose replacement.",
                notes=("Fail closed: upstream config drift detected.",),
            )

    return TrellisFilePatchPlan(
        patch_name="flex_gemm_config",
        target_file=target_file,
        status="planned",
        targeted_host=True,
        backup_path=backup_path,
        expected_marker=TRELLIS_CONFIG_ALGO_MARKER,
        planned_value=decision.selected,
        source=decision.source,
        replacement_snippet=TRELLIS_CONFIG_EXPLICIT_SNIPPET,
        notes=("Planning only: no file writes are performed in Phase 2.",),
    )


def build_flex_gemm_debug_patch_plan(
    env: Mapping[str, str] | None,
    *,
    installed_debug_text: str | None = None,
    install_root: str | None = None,
    backup_suffix: str = TRELLIS_PATCH_BACKUP_SUFFIX,
) -> TrellisFilePatchPlan:
    debug_policy = resolve_flex_gemm_debug_policy(env)
    target_file = _build_patch_path(TRELLIS_DEBUG_RELATIVE_PATH, install_root)
    backup_path = _build_backup_path(target_file, backup_suffix)

    if not debug_policy.enabled:
        return TrellisFilePatchPlan(
            patch_name="flex_gemm_debug_logging",
            target_file=target_file,
            status="not_needed",
            targeted_host=False,
            backup_path=backup_path,
            expected_marker=TRELLIS_DEBUG_ANCHOR,
            notes=("Debug logging is disabled by default and only plans when env-gated.",),
        )

    if installed_debug_text is not None:
        if TRELLIS_DEBUG_MARKER in installed_debug_text:
            return TrellisFilePatchPlan(
                patch_name="flex_gemm_debug_logging",
                target_file=target_file,
                status="already_applied",
                targeted_host=False,
                backup_path=backup_path,
                expected_marker=TRELLIS_DEBUG_ANCHOR,
                planned_value=debug_policy.log_path or "",
                current_value=debug_policy.log_path or "",
                source="env_debug_gate",
                replacement_snippet=TRELLIS_DEBUG_MARKER,
            )
        if TRELLIS_DEBUG_ANCHOR not in installed_debug_text:
            return TrellisFilePatchPlan(
                patch_name="flex_gemm_debug_logging",
                target_file=target_file,
                status="drift",
                targeted_host=False,
                backup_path=backup_path,
                expected_marker=TRELLIS_DEBUG_ANCHOR,
                planned_value=debug_policy.log_path or "",
                source="env_debug_gate",
                blocking_reason="Expected conv_flex_gemm anchor is missing; refusing debug patch plan.",
                notes=("Fail closed: upstream conv_flex_gemm.py drift detected.",),
            )

    return TrellisFilePatchPlan(
        patch_name="flex_gemm_debug_logging",
        target_file=target_file,
        status="planned",
        targeted_host=False,
        backup_path=backup_path,
        expected_marker=TRELLIS_DEBUG_ANCHOR,
        planned_value=debug_policy.log_path or "",
        source="env_debug_gate",
        replacement_snippet=TRELLIS_DEBUG_MARKER,
        notes=("Planning only: debug logging remains env-gated and off by default.",),
    )


def plan_fdg_vae_patch(
    installed_text: str,
    *,
    install_root: str | None = None,
    backup_suffix: str = TRELLIS_PATCH_BACKUP_SUFFIX,
    target_file: str | None = None,
) -> TextPatchPlan:
    resolved_target_file = target_file or _build_patch_path(TRELLIS_FDG_VAE_RELATIVE_PATH, install_root)
    backup_path = _build_backup_path(resolved_target_file, backup_suffix)
    before_hash = _sha256_text(installed_text)

    already_patched = (
        FDG_VAE_IMPORT_MARKER not in installed_text
        and FDG_VAE_PATCHED_IMPORT_SNIPPET in installed_text
        and FDG_VAE_PATCHED_CALL_SNIPPET in installed_text
    )
    if already_patched:
        return TextPatchPlan(
            patch_name=FDG_VAE_PATCH_NAME,
            target_file=resolved_target_file,
            relative_path=TRELLIS_FDG_VAE_RELATIVE_PATH,
            status="already_applied",
            backup_path=backup_path,
            backup_suffix=backup_suffix,
            expected_marker=FDG_VAE_IMPORT_MARKER,
            before_hash=before_hash,
            after_hash=before_hash,
            notes=("Already patched: safe re-entry without further file changes.",),
        )

    if FDG_VAE_IMPORT_MARKER not in installed_text:
        return TextPatchPlan(
            patch_name=FDG_VAE_PATCH_NAME,
            target_file=resolved_target_file,
            relative_path=TRELLIS_FDG_VAE_RELATIVE_PATH,
            status="drift",
            backup_path=backup_path,
            backup_suffix=backup_suffix,
            expected_marker=FDG_VAE_IMPORT_MARKER,
            before_hash=before_hash,
            drift_reason="Expected fdg_vae import marker is missing; refusing compatibility patch.",
            notes=("Fail closed: upstream fdg_vae.py drift detected.",),
        )

    patched_import_text = _replace_once(installed_text, FDG_VAE_UNPATCHED_IMPORT_SNIPPET, FDG_VAE_PATCHED_IMPORT_SNIPPET)
    if patched_import_text is None:
        return TextPatchPlan(
            patch_name=FDG_VAE_PATCH_NAME,
            target_file=resolved_target_file,
            relative_path=TRELLIS_FDG_VAE_RELATIVE_PATH,
            status="drift",
            backup_path=backup_path,
            backup_suffix=backup_suffix,
            expected_marker=FDG_VAE_IMPORT_MARKER,
            before_hash=before_hash,
            drift_reason="fdg_vae import block does not match the approved patch window.",
            notes=("Fail closed: import drift prevents loose replacement.",),
        )

    patched_text = _replace_once(patched_import_text, FDG_VAE_UNPATCHED_CALL_SNIPPET, FDG_VAE_PATCHED_CALL_SNIPPET)
    if patched_text is None:
        return TextPatchPlan(
            patch_name=FDG_VAE_PATCH_NAME,
            target_file=resolved_target_file,
            relative_path=TRELLIS_FDG_VAE_RELATIVE_PATH,
            status="drift",
            backup_path=backup_path,
            backup_suffix=backup_suffix,
            expected_marker=FDG_VAE_IMPORT_MARKER,
            before_hash=before_hash,
            drift_reason="fdg_vae mesh conversion call does not match the approved patch window.",
            notes=("Fail closed: conversion drift prevents loose replacement.",),
        )

    return TextPatchPlan(
        patch_name=FDG_VAE_PATCH_NAME,
        target_file=resolved_target_file,
        relative_path=TRELLIS_FDG_VAE_RELATIVE_PATH,
        status="planned",
        backup_path=backup_path,
        backup_suffix=backup_suffix,
        expected_marker=FDG_VAE_IMPORT_MARKER,
        before_hash=before_hash,
        after_hash=_sha256_text(patched_text),
        patched_text=patched_text,
        notes=("Planning only: downstream compatibility patch removes the missing import and forces train=False.",),
    )


def apply_fdg_vae_patch_file(
    target_file: str | Path,
    *,
    backup_suffix: str = TRELLIS_PATCH_BACKUP_SUFFIX,
) -> TextPatchPlan:
    target_path = Path(target_file)
    installed_text = target_path.read_text(encoding="utf-8")
    plan = plan_fdg_vae_patch(
        installed_text,
        backup_suffix=backup_suffix,
        target_file=str(target_path),
    )
    if plan.status != "planned":
        return plan

    Path(plan.backup_path).write_text(installed_text, encoding="utf-8")
    target_path.write_text(plan.patched_text, encoding="utf-8")
    return TextPatchPlan(
        patch_name=plan.patch_name,
        target_file=plan.target_file,
        relative_path=plan.relative_path,
        status="applied",
        backup_path=plan.backup_path,
        backup_suffix=plan.backup_suffix,
        expected_marker=plan.expected_marker,
        before_hash=plan.before_hash,
        after_hash=plan.after_hash,
        patched_text=plan.patched_text,
        notes=plan.notes + ("Applied via explicit file helper.",),
    )


def _build_validation_command(script_content: str, python_exe: str) -> tuple[str, ...]:
    return (python_exe, "-c", script_content)


def should_run_runtime_validation(env: Mapping[str, str] | None) -> bool:
    return parse_env_flag(_env_value(env, RUN_RUNTIME_VALIDATION_ENV))


def _plan_result(
    plan: ValidationCommandPlan,
    *,
    status: str,
    detail: str,
    duration_sec: float = 0.0,
    returncode: int | None = None,
    stdout: str = "",
    stderr: str = "",
) -> ValidationCommandResult:
    return ValidationCommandResult(
        label=plan.label,
        check_name=plan.check_name,
        status=status,
        timeout_sec=plan.timeout_sec,
        command=plan.command,
        env=dict(plan.env),
        constraints=plan.constraints,
        notes=plan.notes,
        returncode=returncode,
        duration_sec=round(max(duration_sec, 0.0), 6),
        detail=detail,
        stdout=stdout,
        stderr=stderr,
    )


def run_no_generate_validation(
    selected_algo: str,
    *,
    python_exe: str = "python3",
    extension_root: str | None = None,
    env: Mapping[str, str] | None = None,
    import_timeout_sec: int = DEFAULT_IMPORT_API_TIMEOUT_SEC,
    generator_timeout_sec: int = DEFAULT_GENERATOR_BOOTSTRAP_TIMEOUT_SEC,
    flex_gemm_timeout_sec: int = DEFAULT_FLEX_GEMM_TIMEOUT_SEC,
    enable_flex_gemm_debug: bool = False,
    flex_gemm_debug_log_path: str | None = None,
    runner=subprocess.run,
    time_fn=time.monotonic,
) -> RuntimeValidationReport:
    plans = build_no_generate_validation_plans(
        selected_algo,
        python_exe=python_exe,
        extension_root=extension_root,
        import_timeout_sec=import_timeout_sec,
        generator_timeout_sec=generator_timeout_sec,
        flex_gemm_timeout_sec=flex_gemm_timeout_sec,
        enable_flex_gemm_debug=enable_flex_gemm_debug,
        flex_gemm_debug_log_path=flex_gemm_debug_log_path,
    )

    if not should_run_runtime_validation(env):
        return RuntimeValidationReport(
            status="not_run",
            checks=tuple(
                _plan_result(
                    plan,
                    status="not_run",
                    detail=(
                        f"Runtime validation execution is gated by {RUN_RUNTIME_VALIDATION_ENV}=1; "
                        "this setup invocation kept the checks as planned only."
                    ),
                ).to_dict()
                for plan in plans
            ),
            warnings=(
                f"Runtime validation was not executed because {RUN_RUNTIME_VALIDATION_ENV} is not enabled.",
            ),
        )

    results: list[dict[str, Any]] = []
    blocking_reasons: list[str] = []
    merged_env_base = dict(os.environ)
    if env is not None:
        merged_env_base.update({str(key): str(value) for key, value in env.items()})

    aggregate_status = "passed"
    for plan in plans:
        started = time_fn()
        try:
            completed = runner(
                list(plan.command),
                capture_output=True,
                text=True,
                timeout=plan.timeout_sec,
                check=False,
                env={**merged_env_base, **plan.env},
            )
            duration_sec = time_fn() - started
            stdout = str(getattr(completed, "stdout", "") or "").strip()
            stderr = str(getattr(completed, "stderr", "") or "").strip()
            returncode = int(getattr(completed, "returncode", 0))
            if returncode == 0:
                result = _plan_result(plan, status="passed", detail="Validation check completed successfully.", duration_sec=duration_sec, returncode=returncode, stdout=stdout, stderr=stderr)
            else:
                aggregate_status = "failed"
                detail = f"Validation check exited with status {returncode}."
                blocking_reasons.append(f"runtime_validation:{plan.check_name}: {detail}")
                result = _plan_result(plan, status="failed", detail=detail, duration_sec=duration_sec, returncode=returncode, stdout=stdout, stderr=stderr)
        except subprocess.TimeoutExpired as exc:
            aggregate_status = "failed"
            duration_sec = time_fn() - started
            stdout = str((exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout) or "").strip()
            stderr = str((exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr) or "").strip()
            detail = f"Validation check timed out after {plan.timeout_sec}s."
            blocking_reasons.append(f"runtime_validation:{plan.check_name}: {detail}")
            result = _plan_result(plan, status="timeout", detail=detail, duration_sec=duration_sec, stdout=stdout, stderr=stderr)
        results.append(result.to_dict())

    return RuntimeValidationReport(
        status=aggregate_status,
        checks=tuple(results),
        blocking_reasons=tuple(blocking_reasons),
        warnings=(),
    )


def build_import_api_validation_plan(
    *,
    python_exe: str = "python3",
    timeout_sec: int = DEFAULT_IMPORT_API_TIMEOUT_SEC,
) -> ValidationCommandPlan:
    script_content = "\n".join(
        [
            "# no Generate",
            "# no model load",
            "# no asset download",
            "import importlib",
            "import contextlib",
            "import io",
            "import json",
            'pkg = importlib.import_module("trellis2_gguf")',
            'api = importlib.import_module("trellis2_gguf.modules.sparse.conv.config")',
            'print(json.dumps({"package": getattr(pkg, "__name__", ""), "api_module": getattr(api, "__name__", "")}))',
        ]
    )
    return ValidationCommandPlan(
        label="Import/API smoke",
        check_name="import_api_smoke",
        timeout_sec=timeout_sec,
        command=_build_validation_command(script_content, python_exe),
        script_content=script_content,
        notes=("Import-only validation plan; execution remains opt-in.",),
    )


def build_generator_bootstrap_validation_plan(
    *,
    python_exe: str = "python3",
    extension_root: str | None = None,
    timeout_sec: int = DEFAULT_GENERATOR_BOOTSTRAP_TIMEOUT_SEC,
) -> ValidationCommandPlan:
    extension_root_expr = repr(str(extension_root or "").strip())
    script_content = "\n".join(
        [
            "# no Generate",
            "# no model load",
            "# no asset download",
            "import contextlib",
            "import importlib",
            "import io",
            "import json",
            "import pathlib",
            "import sys",
            "import types",
            f"extension_root = {extension_root_expr}",
            "if extension_root:",
            "    sys.path.insert(0, extension_root)",
            "def _stub_module(name, **attrs):",
            "    module = sys.modules.get(name) or types.ModuleType(name)",
            "    for key, value in attrs.items():",
            "        setattr(module, key, value)",
            "    sys.modules[name] = module",
            "    return module",
            "class _BaseGenerator:",
            "    pass",
            "class _GenerationCancelled(Exception):",
            "    pass",
            "generators = _stub_module(\"services.generators\")",
            "base = _stub_module(\"services.generators.base\", BaseGenerator=_BaseGenerator, smooth_progress=lambda *args, **kwargs: None, GenerationCancelled=_GenerationCancelled)",
            "_stub_module(\"services\", generators=generators)",
            "setattr(generators, \"base\", base)",
            "class _CudaStub:",
            "    @staticmethod",
            "    def is_available():",
            "        return True",
            "    @staticmethod",
            "    def get_device_capability():",
            "        return (12, 1)",
            "_stub_module(\"torch\", cuda=_CudaStub())",
            'generator = importlib.import_module("generator")',
            'instance = generator.Trellis2GGUFGenerator.__new__(generator.Trellis2GGUFGenerator)',
            'instance.model_dir = pathlib.Path("/tmp/modly-trellis2-generate")',
            'calls = {"ensure_comfyui_gguf": 0, "resolve_generate_assets": 0}',
            'def _ensure_comfyui_gguf():',
            '    calls["ensure_comfyui_gguf"] += 1',
            'def _resolve_generate_assets():',
            '    calls["resolve_generate_assets"] += 1',
            '    return types.SimpleNamespace(root=pathlib.Path("/tmp/modly-trellis2-generate"), paths={})',
            'instance._ensure_comfyui_gguf = _ensure_comfyui_gguf',
            'instance._resolve_generate_assets = _resolve_generate_assets',
            'instance._auto_download = lambda: (_ for _ in ()).throw(RuntimeError("unexpected auto download"))',
            '_stub_module("trellis2_gguf")',
            '_stub_module("trellis2_gguf.pipelines", Trellis2ImageTo3DPipeline=object)',
            'bootstrap = getattr(type(instance), "_ensure_trellis2_gguf", None)',
            'captured_stdout = io.StringIO()',
            'with contextlib.redirect_stdout(captured_stdout):',
            '    instance._ensure_trellis2_gguf()',
            'runtime_bootstrap = getattr(instance, "_runtime_bootstrap", {})',
            'print(json.dumps({"module": getattr(generator, "__name__", ""), "extension_root_seeded": bool(extension_root), "bootstrap_callable": callable(bootstrap), "bootstrap_exercised": bool(runtime_bootstrap), "services_stubbed": "services.generators.base" in sys.modules, "policy_source": runtime_bootstrap.get("decision", {}).get("source", ""), "selected_algo": runtime_bootstrap.get("decision", {}).get("selected", ""), "policy_applied": runtime_bootstrap.get("applied"), "ensure_comfyui_gguf_calls": calls["ensure_comfyui_gguf"], "resolve_generate_assets_calls": calls["resolve_generate_assets"], "folder_models_dir": getattr(sys.modules.get("folder_paths"), "models_dir", ""), "trellis_manager_stubbed": "trellis2_model_manager" in sys.modules, "captured_stdout": captured_stdout.getvalue().strip()}))',
        ]
    )
    return ValidationCommandPlan(
        label="Generator bootstrap smoke",
        check_name="generator_bootstrap_smoke",
        timeout_sec=timeout_sec,
        command=_build_validation_command(script_content, python_exe),
        script_content=script_content,
        notes=("Bootstrap validation plan self-seeds extension import path and safely exercises _ensure_trellis2_gguf without Generate/model/download work.",),
    )


def build_flex_gemm_micro_smoke_plan(
    selected_algo: str,
    *,
    python_exe: str = "python3",
    timeout_sec: int = DEFAULT_FLEX_GEMM_TIMEOUT_SEC,
    enable_debug: bool = False,
    debug_log_path: str | None = None,
) -> ValidationCommandPlan:
    env = {FLEX_GEMM_ALGO_ENV: selected_algo}
    if enable_debug:
        env[FLEX_GEMM_DEBUG_ENV] = "1"
        if debug_log_path:
            env[FLEX_GEMM_DEBUG_LOG_PATH_ENV] = debug_log_path
    script_content = "\n".join(
        [
            "# no Generate",
            "# no model load",
            "# no asset download",
            "import importlib",
            "import json",
            "import os",
            'module = importlib.import_module("flex_gemm")',
            'print(json.dumps({"module": getattr(module, "__name__", ""), "selected_algo": os.environ.get("FLEX_GEMM_ALGO", ""), "debug_enabled": os.environ.get("TRELLIS2_FLEX_GEMM_DEBUG", "")}))',
        ]
    )
    notes = ["Import-only FlexGEMM probe; no kernels or assets are touched."]
    if enable_debug:
        notes.append("Debug env metadata is included only because it was explicitly requested.")
    return ValidationCommandPlan(
        label="FlexGEMM micro-smoke",
        check_name="flex_gemm_micro_smoke",
        timeout_sec=timeout_sec,
        command=_build_validation_command(script_content, python_exe),
        script_content=script_content,
        env=env,
        notes=tuple(notes),
    )


def build_no_generate_validation_plans(
    selected_algo: str,
    *,
    python_exe: str = "python3",
    extension_root: str | None = None,
    import_timeout_sec: int = DEFAULT_IMPORT_API_TIMEOUT_SEC,
    generator_timeout_sec: int = DEFAULT_GENERATOR_BOOTSTRAP_TIMEOUT_SEC,
    flex_gemm_timeout_sec: int = DEFAULT_FLEX_GEMM_TIMEOUT_SEC,
    enable_flex_gemm_debug: bool = False,
    flex_gemm_debug_log_path: str | None = None,
) -> tuple[ValidationCommandPlan, ...]:
    return (
        build_import_api_validation_plan(python_exe=python_exe, timeout_sec=import_timeout_sec),
        build_generator_bootstrap_validation_plan(
            python_exe=python_exe,
            extension_root=extension_root,
            timeout_sec=generator_timeout_sec,
        ),
        build_flex_gemm_micro_smoke_plan(
            selected_algo,
            python_exe=python_exe,
            timeout_sec=flex_gemm_timeout_sec,
            enable_debug=enable_flex_gemm_debug,
            debug_log_path=flex_gemm_debug_log_path,
        ),
    )


def plan_triton_override(
    host: HostPlatform,
    *,
    current_triton_version: str = "",
    torch_version: str = "",
    torch_triton_pin: str = TORCH_TRITON_PIN,
) -> TritonOverridePlan:
    targeted_host = detect_linux_arm64_blackwell(host)
    conflict_note = ""
    if targeted_host and torch_triton_pin and torch_triton_pin != TRITON_OVERRIDE_VERSION:
        conflict_note = (
            f"Torch metadata may still pin triton=={torch_triton_pin}; "
            f"runtime policy targets triton=={TRITON_OVERRIDE_VERSION} without reinstalling torch"
        )
    command = ("python3", "-m", "pip", "install", f"triton=={TRITON_OVERRIDE_VERSION}")

    if not targeted_host:
        return TritonOverridePlan(
            status="not_applicable",
            applies=False,
            target_version=None,
            current_version=current_triton_version,
            install_allowed=False,
            targeted_host=False,
            reason="non_target_host",
            command=(),
            torch_version=torch_version,
            torch_triton_pin=torch_triton_pin,
            conflict_note="",
            reinstall_torch=False,
        )

    if current_triton_version == TRITON_OVERRIDE_VERSION:
        return TritonOverridePlan(
            status="already_satisfied",
            applies=False,
            target_version=TRITON_OVERRIDE_VERSION,
            current_version=current_triton_version,
            install_allowed=False,
            targeted_host=True,
            reason="target_version_already_present",
            command=(),
            torch_version=torch_version,
            torch_triton_pin=torch_triton_pin,
            conflict_note=conflict_note,
            reinstall_torch=False,
        )

    if conflict_note:
        return TritonOverridePlan(
            status="conflict_reported",
            applies=True,
            target_version=TRITON_OVERRIDE_VERSION,
            current_version=current_triton_version,
            install_allowed=False,
            targeted_host=True,
            reason="linux_arm64_blackwell_policy",
            command=command,
            torch_version=torch_version,
            torch_triton_pin=torch_triton_pin,
            conflict_note=conflict_note,
            reinstall_torch=False,
        )

    return TritonOverridePlan(
        status="planned",
        applies=True,
        target_version=TRITON_OVERRIDE_VERSION,
        current_version=current_triton_version,
        install_allowed=False,
        targeted_host=True,
        reason="linux_arm64_blackwell_policy",
        command=command,
        torch_version=torch_version,
        torch_triton_pin=torch_triton_pin,
        conflict_note="",
        reinstall_torch=False,
    )


def build_runtime_overrides_report(
    host: HostPlatform,
    env: Mapping[str, str] | None,
    *,
    current_triton_version: str = "",
    torch_version: str = "",
    torch_triton_pin: str = TORCH_TRITON_PIN,
) -> RuntimeOverridesReport:
    return RuntimeOverridesReport(
        host=host.to_dict(),
        flex_gemm=resolve_flex_gemm_algo(env, host).to_dict(),
        flex_gemm_debug=resolve_flex_gemm_debug_policy(env).to_dict(),
        triton=plan_triton_override(
            host,
            current_triton_version=current_triton_version,
            torch_version=torch_version,
            torch_triton_pin=torch_triton_pin,
        ).to_dict(),
    )


def build_report_fragments(
    host: HostPlatform,
    env: Mapping[str, str] | None,
    *,
    current_triton_version: str = "",
    torch_version: str = "",
    torch_triton_pin: str = TORCH_TRITON_PIN,
) -> dict[str, Any]:
    fragments = {
        "runtime_overrides": build_runtime_overrides_report(
            host,
            env,
            current_triton_version=current_triton_version,
            torch_version=torch_version,
            torch_triton_pin=torch_triton_pin,
        ).to_dict(),
        "trellis_patches": TrellisPatchesReport().to_dict(),
        "runtime_validation": RuntimeValidationReport().to_dict(),
        "wheel_provenance": WheelProvenanceReport().to_dict(),
    }
    json.dumps(fragments, sort_keys=True)
    return fragments
