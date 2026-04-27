"""Trellis.2 GGUF extension setup script.

Creates an isolated venv and installs the geometry-first runtime slice.
The initial clean rebuild keeps Generate Mesh public and treats texturing /
refine-only native pieces as optional follow-up work on supported platforms.

Called by Modly at extension install time with:
    python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":124}'
"""

import io
import json
import os
import platform
import re
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from runtime_support import (
    COMFYUI_GGUF_REQUIRED_FILES,
    DependencyPreflight,
    KNOWN_DEPENDENCIES,
    RuntimeEnvironment,
    WheelCandidate,
    WheelResolutionResult,
    WheelSource,
    build_wheel_sources,
    comfyui_gguf_support_file_report,
    comfyui_gguf_target_dir,
    discover_local_wheel_candidates,
    parse_wheel_candidate,
    resolve_wheel_candidate,
)

_COMFYUI_TRELLIS2_ZIP = "https://github.com/Aero-Ex/ComfyUI-Trellis2-GGUF/archive/refs/heads/main.zip"
_WHEELS_INDEX_BASE    = "https://pozzettiandrea.github.io/cuda-wheels/"

# Geometry-first slice: keep required wheels limited to the generate path.
# Refine-only wheels stay optional and are never claimed on unsupported tags.
_CUDA_WHEELS_REQUIRED = ["cumesh", "o-voxel"]
_CUDA_WHEELS_OPTIONAL = ["flex-gemm", "nvdiffrast"]
_CUDA_WHEELS = _CUDA_WHEELS_REQUIRED + _CUDA_WHEELS_OPTIONAL

# Standard Python packages
_PY_PACKAGES = [
    "Pillow",
    "numpy",
    "scipy",
    "trimesh",
    "pymeshlab",
    "meshlib",
    "opencv-python-headless",
    "gguf",
    "sdnq",
    "rectpack",
    "requests",
    "huggingface_hub",
    "transformers==5.2.0",
    "accelerate",
    "einops",
    "easydict",
]

_PY_OPTIONAL_PACKAGES = [
    "open3d",
]

_SETUP_REPORT_FILENAME = "setup-report.json"


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SetupContext:
    python_exe: str
    ext_dir: Path
    venv_dir: Path
    gpu_sm: int
    cuda_version: int
    system: str
    machine: str
    python_tag: str
    python_version: str
    wheel_sources: tuple[WheelSource, ...]
    platform_tag: str | None

def _pip(venv: Path, *args: str) -> None:
    is_win  = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def _python(venv: Path) -> Path:
    is_win = platform.system() == "Windows"
    return venv / ("Scripts/python.exe" if is_win else "bin/python")


def _site_packages(venv: Path) -> Path:
    exe = _python(venv)
    out = subprocess.check_output(
        [str(exe), "-c",
         "import site; print([p for p in site.getsitepackages() if 'site-packages' in p][0])"],
        text=True,
    ).strip()
    return Path(out)


def _get_torch_version(venv: Path) -> str:
    """Return the installed PyTorch version string, e.g. '2.7.0'."""
    exe = _python(venv)
    try:
        out = subprocess.check_output(
            [str(exe), "-c", "import torch; print(torch.__version__.split('+')[0])"],
            text=True,
        ).strip()
        return out
    except Exception:
        return ""


def _python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _is_linux_arm64(system: str | None = None, machine: str | None = None) -> bool:
    return (system or platform.system()).lower() == "linux" and (machine or platform.machine()).lower() in {"aarch64", "arm64"}


def _optional_python_package_policy(
    package: str,
    *,
    system: str | None = None,
    machine: str | None = None,
) -> dict[str, str]:
    if package == "open3d" and _is_linux_arm64(system, machine):
        return {
            "name": package,
            "action": "skip",
            "status": "skipped",
            "detail": "skipped on Linux ARM64/aarch64 because no compatible open3d wheel is expected for this setup path",
        }
    return {
        "name": package,
        "action": "install",
        "status": "pending",
        "detail": "optional package will be installed best-effort",
    }


def _rembg_package_policy(
    gpu_sm: int,
    *,
    system: str | None = None,
    machine: str | None = None,
) -> dict[str, object]:
    if _is_linux_arm64(system, machine):
        return {
            "mode": "cpu",
            "packages": ["rembg", "onnxruntime"],
            "summary": "rembg + onnxruntime",
            "detail": (
                "Linux ARM64/aarch64 uses CPU-compatible rembg packages; "
                "rembg[gpu] and onnxruntime-gpu are not selected on this platform"
            ),
        }

    if gpu_sm >= 70:
        return {
            "mode": "gpu",
            "packages": ["rembg[gpu]"],
            "summary": "rembg[gpu]",
            "detail": "high-GPU platform uses rembg[gpu] existing behavior",
        }

    return {
        "mode": "cpu",
        "packages": ["rembg", "onnxruntime"],
        "summary": "rembg + onnxruntime",
        "detail": "low-GPU path uses CPU rembg packages",
    }


def _install_optional_python_packages(
    context: SetupContext,
    *,
    pip_runner=_pip,
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []

    for package in _PY_OPTIONAL_PACKAGES:
        policy = _optional_python_package_policy(package, system=context.system, machine=context.machine)
        result = {
            "name": package,
            "status": policy["status"],
            "detail": policy["detail"],
        }

        if policy["action"] == "skip":
            print(f"[setup] NOTE: optional Python dependency {package} skipped. {policy['detail']}")
            results.append(result)
            continue

        print(f"[setup] Installing optional Python dependency {package} …")
        try:
            pip_runner(context.venv_dir, "install", package)
            result["status"] = "installed"
            result["detail"] = "installed successfully"
            print(f"[setup] Optional Python dependency {package} installed.")
        except subprocess.CalledProcessError as exc:
            result["status"] = "failed"
            result["detail"] = f"install failed: {exc}"
            print(f"[setup] WARNING: optional Python dependency {package} failed to install ({exc}).")

        results.append(result)

    return results


def _redact_url(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""

    try:
        parts = urlsplit(raw)
    except Exception:
        return raw

    if not parts.scheme or not parts.netloc:
        return raw

    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    netloc = host
    if parts.username or parts.password:
        netloc = f"***:***@{host}" if host else "***:***"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _redact_source_location(source: WheelSource) -> str:
    if source.kind in {"index", "url", "public-index"}:
        return _redact_url(source.location)
    return source.location


def _serialize_wheel_sources(sources: tuple[WheelSource, ...]) -> list[dict[str, object]]:
    return [
        {
            "label": source.label,
            "kind": source.kind,
            "priority": source.priority,
            "location": _redact_source_location(source),
        }
        for source in sources
    ]


def _serialize_capability_report(report) -> dict[str, object]:
    return {
        "state": report.state,
        "allowed": report.allowed,
        "blockers": list(report.blockers),
        "deps": [
            {"name": dep.name, "state": dep.state, "detail": dep.detail}
            for dep in report.deps
        ],
    }


def _select_torch_packages(gpu_sm: int, cuda_version: int) -> tuple[list[str], str, str]:
    if gpu_sm >= 100 or cuda_version >= 128:
        return (
            ["torch==2.7.0", "torchvision==0.22.0"],
            "https://download.pytorch.org/whl/cu128",
            f"GPU SM {gpu_sm}, CUDA {cuda_version} -> PyTorch 2.7 + CUDA 12.8 (Blackwell)",
        )
    if gpu_sm == 0 or gpu_sm >= 70:
        return (
            ["torch==2.6.0", "torchvision==0.21.0"],
            "https://download.pytorch.org/whl/cu126",
            f"GPU SM {gpu_sm} -> PyTorch 2.6 + CUDA 12.6",
        )
    return (
        ["torch==2.5.1", "torchvision==0.20.1"],
        "https://download.pytorch.org/whl/cu118",
        f"GPU SM {gpu_sm} (legacy) -> PyTorch 2.5 + CUDA 11.8",
    )


def _build_setup_context(
    python_exe: str,
    ext_dir: Path,
    gpu_sm: int,
    cuda_version: int = 0,
    *,
    trellis_wheelhouse: str | None = None,
    trellis_extra_wheel_index: str | None = None,
    trellis_extra_wheel_url: str | None = None,
    system: str | None = None,
    machine: str | None = None,
) -> SetupContext:
    resolved_system = system or platform.system()
    resolved_machine = machine or platform.machine()
    wheel_sources = build_wheel_sources(
        trellis_wheelhouse=trellis_wheelhouse,
        trellis_extra_wheel_index=trellis_extra_wheel_index,
        trellis_extra_wheel_url=trellis_extra_wheel_url,
        public_index_base=_WHEELS_INDEX_BASE,
    )
    return SetupContext(
        python_exe=python_exe,
        ext_dir=ext_dir,
        venv_dir=ext_dir / "venv",
        gpu_sm=gpu_sm,
        cuda_version=cuda_version,
        system=resolved_system,
        machine=resolved_machine,
        python_tag=_python_tag(),
        python_version=platform.python_version(),
        wheel_sources=wheel_sources,
        platform_tag=_wheel_platform_tag(resolved_system, resolved_machine),
    )


def _wheel_platform_tag(system: str | None = None, machine: str | None = None) -> str | None:
    system_name = (system or platform.system()).lower()
    machine_name = (machine or platform.machine()).lower()

    if system_name == "windows" and machine_name in {"amd64", "x86_64"}:
        return "win_amd64"
    if system_name == "linux" and machine_name in {"amd64", "x86_64"}:
        return "linux_x86_64"
    return None


def _setup_runtime_environment(
    gpu_sm: int,
    cuda_version: int = 0,
    *,
    system: str | None = None,
    machine: str | None = None,
) -> RuntimeEnvironment:
    system_name = system or platform.system()
    machine_name = machine or platform.machine()

    return RuntimeEnvironment(
        system=system_name,
        machine=machine_name,
        python_tag=_python_tag(),
        python_version=platform.python_version(),
        torch_version="",
        cuda_version=str(cuda_version) if cuda_version else "",
        gpu_sm=str(gpu_sm),
        asset_groups={"generate": True, "refine": True},
        refine_lab_verified=False,
    )


def _native_policy_report(env: RuntimeEnvironment) -> dict[str, object]:
    platform_tag = _wheel_platform_tag(env.system, env.machine)
    machine_name = env.machine.lower()
    known_dependencies = {
        name: {"state": "unknown", "detail": "setup has not installed this dependency yet"}
        for name in KNOWN_DEPENDENCIES
    }

    if env.system.lower() == "linux" and machine_name in {"aarch64", "arm64"}:
        detail = "public custom CUDA wheel index does not publish Linux ARM64 tags; source builds are out of scope for this repo"
        known_dependencies = {
            "flex_gemm": {"state": "unsupported", "detail": detail},
            "cumesh": {"state": "unsupported", "detail": detail},
            "nvdiffrast": {"state": "unsupported", "detail": detail},
            "o_voxel": {"state": "unsupported", "detail": detail},
            "spconv": {"state": "unsupported", "detail": "setup.py does not build spconv from source in this repo"},
            "cumm": {"state": "unsupported", "detail": "setup.py does not build cumm from source in this repo"},
            "comfyui_gguf_support_files": {"state": "missing", "detail": "setup has not installed ComfyUI-GGUF support files yet"},
        }

    evaluated_env = RuntimeEnvironment(
        system=env.system,
        machine=env.machine,
        python_tag=env.python_tag,
        python_version=env.python_version,
        torch_version=env.torch_version,
        cuda_version=env.cuda_version,
        gpu_sm=env.gpu_sm,
        asset_groups=env.asset_groups,
        refine_lab_verified=env.refine_lab_verified,
        known_dependencies=known_dependencies,
    )
    return {
        "env": evaluated_env,
        "platform_tag": platform_tag,
        "generate": DependencyPreflight.evaluate("generate", evaluated_env),
        "refine": DependencyPreflight.evaluate("refine", evaluated_env),
    }


def _preflight_setup_policy(
    gpu_sm: int,
    cuda_version: int = 0,
    *,
    system: str | None = None,
    machine: str | None = None,
) -> dict[str, object]:
    report = _native_policy_report(_setup_runtime_environment(gpu_sm, cuda_version, system=system, machine=machine))
    env = report["env"]
    assert isinstance(env, RuntimeEnvironment)

    print(
        "[setup] Native policy context: "
        f"system={env.system} machine={env.machine} python={env.python_tag} "
        f"cuda={env.cuda_version or 'unknown'} gpu_sm={env.gpu_sm or 'unknown'}"
    )

    continue_install_base = True
    native_wheel_query_supported = report["platform_tag"] is not None
    policy_notes: list[str] = []

    if env.system.lower() == "linux" and env.machine.lower() in {"aarch64", "arm64"}:
        generate = report["generate"]
        refine = report["refine"]
        assert hasattr(generate, "blockers") and hasattr(refine, "blockers")
        policy_notes.append(
            "Linux ARM64 continues with base install, but public native-wheel queries stay disabled and source builds remain out of scope."
        )
        policy_notes.append(f"Generate blockers: {'; '.join(generate.blockers)}")
        policy_notes.append(f"Refine blockers: {'; '.join(refine.blockers)}")
    elif report["platform_tag"] is None:
        policy_notes.append(
            f"Unsupported native-wheel host {env.system}/{env.machine}; base install may continue, but native wheel compatibility is unmodeled."
        )

    refine = report["refine"]
    assert hasattr(refine, "blockers")
    if refine.blockers:
        print(f"[setup] NOTE: Refine remains gated in this slice: {'; '.join(refine.blockers)}")

    report["continue_install_base"] = continue_install_base
    report["native_wheel_query_supported"] = native_wheel_query_supported
    report["policy_notes"] = tuple(policy_notes)
    return report


# --------------------------------------------------------------------------- #
# Custom CUDA wheels (pozzettiandrea.github.io)                                #
# --------------------------------------------------------------------------- #

def _find_wheel_url(lib_name: str, python_tag: str, platform_tag: str, torch_ver: str) -> str | None:
    env = RuntimeEnvironment(
        system=platform.system(),
        machine=platform.machine(),
        python_tag=python_tag,
        python_version=platform.python_version(),
        torch_version=torch_ver,
        cuda_version="",
        asset_groups={"generate": True, "refine": True},
    )
    resolution = _resolve_wheel_candidate(lib_name, env)
    if resolution.selected_candidate:
        return resolution.selected_candidate.install_target
    return None


def _resolve_wheel_candidate(lib_name: str, env: RuntimeEnvironment) -> WheelResolutionResult:
    sources = build_wheel_sources(public_index_base=_WHEELS_INDEX_BASE)
    candidates: list[WheelCandidate] = []
    for source in sources:
        candidates.extend(_source_candidates_for_requirement(source, lib_name, env)[0])
    return resolve_wheel_candidate(lib_name, env, sources, candidates)


def _source_candidates_for_requirement(
    source: WheelSource,
    lib_name: str,
    env: RuntimeEnvironment,
) -> tuple[tuple[WheelCandidate, ...], dict[str, object]]:
    unsupported_public_host = source.kind == "public-index" and _wheel_platform_tag(env.system, env.machine) is None
    if unsupported_public_host:
        return (), {
            "source": source.label,
            "kind": source.kind,
            "status": "skipped",
            "location": _redact_source_location(source),
            "detail": f"public query skipped for unsupported host {env.system}/{env.machine}",
        }
    if source.kind == "wheelhouse":
        candidates = discover_local_wheel_candidates(source, requirement=lib_name)
        return candidates, {
            "source": source.label,
            "kind": source.kind,
            "status": "checked",
            "location": _redact_source_location(source),
            "detail": f"discovered {len(candidates)} candidate(s)",
        }
    if source.kind in {"index", "public-index"}:
        candidates = _index_candidates(source, lib_name)
        return candidates, {
            "source": source.label,
            "kind": source.kind,
            "status": "checked",
            "location": _redact_source_location(source),
            "detail": f"resolved {len(candidates)} candidate(s)",
        }
    if source.kind == "url":
        candidates = _url_candidates(source, lib_name)
        return candidates, {
            "source": source.label,
            "kind": source.kind,
            "status": "checked",
            "location": _redact_source_location(source),
            "detail": f"resolved {len(candidates)} candidate(s)",
        }
    return (), {
        "source": source.label,
        "kind": source.kind,
        "status": "skipped",
        "location": _redact_source_location(source),
        "detail": "source kind is not modeled",
    }


def _index_candidates(source: WheelSource, lib_name: str) -> tuple[WheelCandidate, ...]:
    index_url = f"{source.location.rstrip('/')}/{lib_name}/"
    try:
        with urllib.request.urlopen(index_url, timeout=30) as resp:
            html = resp.read().decode("utf-8")
    except Exception as exc:
        print(f"[setup] WARNING: Could not fetch wheel index for {lib_name} from {source.label}: {exc}")
        return ()

    links = re.findall(r'href=["\']([^"\']*\.whl)["\']', html)
    if not links:
        links = re.findall(r'([\w\-\.]+\.whl)', html)

    candidates: list[WheelCandidate] = []
    seen: set[str] = set()
    for link in links:
        filename = link.split("/")[-1]
        if filename in seen:
            continue
        seen.add(filename)
        url = link if link.startswith("http") else f"{index_url}{filename}"
        try:
            candidates.append(parse_wheel_candidate(filename, source, requirement=lib_name, url=url))
        except ValueError:
            continue
    return tuple(sorted(candidates, key=lambda candidate: candidate.filename))


def _url_candidates(source: WheelSource, lib_name: str) -> tuple[WheelCandidate, ...]:
    target = source.location.replace("{lib_name}", lib_name)
    if not target.endswith(".whl"):
        return ()
    filename = target.split("/")[-1]
    try:
        return (parse_wheel_candidate(filename, source, requirement=lib_name, url=target),)
    except ValueError:
        return ()


def _build_native_runtime_environment(context: SetupContext, torch_version: str) -> RuntimeEnvironment:
    return RuntimeEnvironment(
        system=context.system,
        machine=context.machine,
        python_tag=context.python_tag,
        python_version=context.python_version,
        torch_version=torch_version,
        cuda_version=str(context.cuda_version) if context.cuda_version else os.environ.get("TRELLIS2_CUDA_VERSION", ""),
        gpu_sm=str(context.gpu_sm),
        asset_groups={"generate": True, "refine": False},
        refine_lab_verified=False,
    )


def _empty_dependency_result(name: str, detail: str) -> dict[str, object]:
    return {
        "name": name,
        "status": "missing",
        "detail": detail,
        "selected_source": None,
        "install_target": None,
        "source_events": [],
        "tried_sources": [],
        "checked_candidates": [],
    }


def _resolve_native_dependency(
    context: SetupContext,
    env: RuntimeEnvironment,
    lib_name: str,
) -> tuple[WheelResolutionResult, list[dict[str, object]]]:
    candidates: list[WheelCandidate] = []
    source_events: list[dict[str, object]] = []
    for source in context.wheel_sources:
        source_candidates, event = _source_candidates_for_requirement(source, lib_name, env)
        candidates.extend(source_candidates)
        source_events.append(event)
    return resolve_wheel_candidate(lib_name, env, context.wheel_sources, candidates), source_events


def _install_native_wheels(context: SetupContext) -> tuple[dict[str, dict[str, object]], str]:
    """Best-effort native wheel resolution/install with structured reporting."""
    torch_ver = _get_torch_version(context.venv_dir)
    env = _build_native_runtime_environment(context, torch_ver)

    print(
        f"[setup] Phase 2/3: resolving native wheels (python={context.python_tag}, "
        f"host={context.system}/{context.machine}, torch={torch_ver or 'unknown'}) …"
    )

    dependency_results = {
        name: _empty_dependency_result(name, "setup phase did not evaluate this dependency yet")
        for name in KNOWN_DEPENDENCIES
    }

    gguf_report = comfyui_gguf_support_file_report(comfyui_gguf_target_dir(_site_packages(context.venv_dir)))
    gguf_target_dir = comfyui_gguf_target_dir(_site_packages(context.venv_dir))
    gguf_report = comfyui_gguf_support_file_report(gguf_target_dir)
    dependency_results["comfyui_gguf_support_files"] = {
        "name": "comfyui_gguf_support_files",
        "status": gguf_report["status"],
        "detail": gguf_report["detail"],
        "selected_source": "base-install",
        "install_target": str(gguf_target_dir),
        "source_events": [],
        "tried_sources": [],
        "checked_candidates": [],
        "missing_files": gguf_report["missing"],
    }

    for lib in _CUDA_WHEELS:
        resolution, source_events = _resolve_native_dependency(context, env, lib)
        canonical = lib.replace("-", "_")
        result = {
            "name": canonical,
            "status": "missing",
            "detail": resolution.detail,
            "selected_source": resolution.selected_source.label if resolution.selected_source else None,
            "install_target": _redact_url(resolution.selected_candidate.install_target) if resolution.selected_candidate else None,
            "source_events": source_events,
            "tried_sources": [_redact_source_location(source) for source in resolution.tried_sources],
            "checked_candidates": list(resolution.checked_candidates),
        }

        selected = resolution.selected_candidate
        if selected is None:
            if context.platform_tag is None and not any(event["kind"] in {"wheelhouse", "index", "url"} and event["status"] == "checked" for event in source_events if event["source"] != "public-pozzettiandrea"):
                result["status"] = "unsupported"
            elif any(event["status"] == "skipped" for event in source_events):
                result["status"] = "skipped"
            if lib in _CUDA_WHEELS_REQUIRED:
                print(f"[setup] WARNING: required native wheel missing for {lib}. {resolution.detail}")
            else:
                print(f"[setup] NOTE: optional native wheel missing for {lib}. {resolution.detail}")
            dependency_results[canonical] = result
            continue

        print(f"[setup] Installing {lib} from {_redact_url(selected.install_target)} ({selected.source.label}) …")
        try:
            _pip(context.venv_dir, "install", selected.install_target)
            result["status"] = "installed"
            result["detail"] = f"installed from {selected.source.label}"
            print(f"[setup] {lib} installed.")
        except subprocess.CalledProcessError as exc:
            result["status"] = "failed"
            result["detail"] = f"install failed: {exc}"
            print(f"[setup] WARNING: Failed to install {lib} ({exc}).")
        dependency_results[canonical] = result

    for dep in ("spconv", "cumm"):
        dependency_results[dep] = {
            "name": dep,
            "status": "unsupported" if context.system.lower() == "linux" and context.machine.lower() in {"aarch64", "arm64"} else "missing",
            "detail": "setup.py does not install this dependency in the clean rebuild slice",
            "selected_source": None,
            "install_target": None,
            "source_events": [],
            "tried_sources": [],
            "checked_candidates": [],
        }

    return dependency_results, torch_ver


def _known_dependencies_for_report(dependency_results: dict[str, dict[str, object]]) -> dict[str, dict[str, str]]:
    known: dict[str, dict[str, str]] = {}
    for name, result in dependency_results.items():
        status = str(result.get("status", "missing"))
        if status == "installed":
            state = "available"
        elif status == "unsupported":
            state = "unsupported"
        else:
            state = "missing"
        known[name] = {"state": state, "detail": str(result.get("detail", ""))}
    return known


def _build_final_runtime_environment(
    context: SetupContext,
    dependency_results: dict[str, dict[str, object]],
    torch_version: str,
) -> RuntimeEnvironment:
    return RuntimeEnvironment(
        system=context.system,
        machine=context.machine,
        python_tag=context.python_tag,
        python_version=context.python_version,
        torch_version=torch_version,
        cuda_version=str(context.cuda_version) if context.cuda_version else "",
        gpu_sm=str(context.gpu_sm),
        known_dependencies=_known_dependencies_for_report(dependency_results),
        asset_groups={"generate": True, "refine": False},
        refine_lab_verified=False,
    )


def _build_next_actions(
    context: SetupContext,
    dependency_results: dict[str, dict[str, object]],
    generate_report,
    refine_report,
) -> list[str]:
    actions: list[str] = []
    missing_required = [
        name for name in _CUDA_WHEELS_REQUIRED
        if dependency_results[name.replace("-", "_")]["status"] != "installed"
    ]
    if missing_required:
        actions.append(
            "Provide matching native wheels via TRELLIS2_WHEELHOUSE or TRELLIS2_EXTRA_WHEEL_INDEX for: "
            + ", ".join(sorted(missing_required))
        )
    if context.platform_tag is None:
        actions.append(
            "Public pozzettiandrea native-wheel queries were skipped for this host; use a local/private ARM64 wheel source instead."
        )
    if dependency_results["comfyui_gguf_support_files"]["status"] != "installed":
        actions.append(
            "Re-run the extension setup/repair path to restore the required ComfyUI-GGUF support files."
        )
    if refine_report.blockers:
        actions.append("Keep refine/texturing hidden until required dependency and runtime gates pass.")
    if not missing_required and generate_report.allowed:
        actions.append("Base setup completed and generate dependency gates are satisfied for this setup slice.")
    return actions


def _write_setup_report(ext_dir: Path, report: dict[str, object]) -> Path:
    report_path = ext_dir / _SETUP_REPORT_FILENAME
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def _print_setup_summary(report: dict[str, object]) -> None:
    print("[setup] Phase 3/3 summary")
    print(f"[setup]   status: {report['status']}")
    platform_info = report["platform"]
    python_info = report["python"]
    cuda_info = report["cuda"]
    print(
        "[setup]   host: "
        f"{platform_info['system']}/{platform_info['machine']} "
        f"python={python_info['tag']} ({python_info['version']}) gpu_sm={cuda_info['gpu_sm']} cuda={cuda_info['cuda_version'] or 'unknown'}"
    )
    print(f"[setup]   wheel sources: {', '.join(source['location'] for source in report['wheel_sources'])}")
    capabilities = report["capabilities"]
    print(
        f"[setup]   generate={capabilities['generate']['state']} refine={capabilities['refine']['state']}"
    )
    for name, dep in sorted(report["dependencies"].items()):
        print(f"[setup]   dep {name}: {dep['status']} - {dep['detail']}")
    for action in report["next_actions"]:
        print(f"[setup]   next: {action}")


def _phase0_collect_and_plan(
    python_exe: str,
    ext_dir: Path,
    gpu_sm: int,
    cuda_version: int = 0,
    *,
    trellis_wheelhouse: str | None = None,
    trellis_extra_wheel_index: str | None = None,
    trellis_extra_wheel_url: str | None = None,
    system: str | None = None,
    machine: str | None = None,
) -> tuple[SetupContext, dict[str, object]]:
    context = _build_setup_context(
        python_exe,
        ext_dir,
        gpu_sm,
        cuda_version,
        trellis_wheelhouse=trellis_wheelhouse,
        trellis_extra_wheel_index=trellis_extra_wheel_index,
        trellis_extra_wheel_url=trellis_extra_wheel_url,
        system=system,
        machine=machine,
    )
    policy = _preflight_setup_policy(gpu_sm, cuda_version, system=context.system, machine=context.machine)
    print("[setup] Phase 0/3: collected setup context and wheel source configuration")
    return context, policy


def _phase1_install_base(context: SetupContext) -> dict[str, object]:
    print(f"[setup] Phase 1/3: creating venv at {context.venv_dir} …")
    subprocess.run([context.python_exe, "-m", "venv", str(context.venv_dir)], check=True)

    torch_pkgs, torch_index, torch_message = _select_torch_packages(context.gpu_sm, context.cuda_version)
    print(f"[setup] {torch_message}")
    print("[setup] Installing PyTorch …")
    _pip(context.venv_dir, "install", *torch_pkgs, "--index-url", torch_index)

    print("[setup] Installing core Python dependencies …")
    _pip(context.venv_dir, "install", *_PY_PACKAGES)

    optional_python_packages = _install_optional_python_packages(context)
    rembg_policy = _rembg_package_policy(context.gpu_sm, system=context.system, machine=context.machine)

    print(f"[setup] Installing rembg … ({rembg_policy['detail']})")
    _pip(context.venv_dir, "install", *rembg_policy["packages"])

    torch_ver = _get_torch_version(context.venv_dir)
    if torch_ver:
        _install_triton_windows(context.venv_dir, torch_ver, context.gpu_sm)

    _install_trellis2_gguf(context.venv_dir)
    _install_comfyui_gguf(context.venv_dir)

    return {
        "venv": str(context.venv_dir),
        "torch_packages": torch_pkgs,
        "torch_index": _redact_url(torch_index),
        "torch_version": torch_ver,
        "core_packages": list(_PY_PACKAGES),
        "optional_packages": optional_python_packages,
        "rembg": rembg_policy["summary"],
        "rembg_plan": rembg_policy,
    }


def _plan_setup(
    python_exe: str,
    ext_dir: Path,
    gpu_sm: int,
    cuda_version: int = 0,
    *,
    trellis_wheelhouse: str | None = None,
    trellis_extra_wheel_index: str | None = None,
    trellis_extra_wheel_url: str | None = None,
    system: str | None = None,
    machine: str | None = None,
) -> dict[str, object]:
    context, policy = _phase0_collect_and_plan(
        python_exe,
        ext_dir,
        gpu_sm,
        cuda_version,
        trellis_wheelhouse=trellis_wheelhouse,
        trellis_extra_wheel_index=trellis_extra_wheel_index,
        trellis_extra_wheel_url=trellis_extra_wheel_url,
        system=system,
        machine=machine,
    )
    torch_pkgs, torch_index, torch_message = _select_torch_packages(context.gpu_sm, context.cuda_version)
    return {
        "phase": 0,
        "continue_install_base": bool(policy["continue_install_base"]),
        "native_wheel_query_supported": bool(policy["native_wheel_query_supported"]),
        "platform": {
            "system": context.system,
            "machine": context.machine,
            "platform_tag": context.platform_tag,
        },
        "python": {
            "tag": context.python_tag,
            "version": context.python_version,
            "executable": context.python_exe,
        },
        "cuda": {
            "gpu_sm": str(context.gpu_sm),
            "cuda_version": str(context.cuda_version) if context.cuda_version else "",
            "torch_packages": torch_pkgs,
            "torch_index": _redact_url(torch_index),
            "selection_detail": torch_message,
        },
        "wheel_sources": _serialize_wheel_sources(context.wheel_sources),
        "policy_notes": list(policy.get("policy_notes", ())),
        "venv_dir": str(context.venv_dir),
    }


# --------------------------------------------------------------------------- #
# triton-windows (Windows only)                                                #
# --------------------------------------------------------------------------- #

def _install_triton_windows(venv: Path, torch_ver: str, gpu_sm: int = 0) -> None:
    """Install triton-windows matching the PyTorch version.

    Blackwell GPUs (SM 12.x) require triton-windows >= 3.3.1 which bundles
    ptxas 12.8 and adds SM 12.x support (triton-lang PR #8498).
    """
    if platform.system() != "Windows":
        return

    # Version constraints from ComfyUI-Trellis2-GGUF install.py
    tv = tuple(int(x) for x in torch_ver.split(".")[:2])
    if tv >= (2, 10):
        triton_spec = "triton-windows<3.7"
    elif tv >= (2, 9):
        triton_spec = "triton-windows<3.6"
    elif tv >= (2, 8):
        triton_spec = "triton-windows<3.5"
    elif tv >= (2, 7):
        if gpu_sm >= 100:
            # Blackwell: 3.3.1+ bundles ptxas 12.8 with SM 12.x support
            triton_spec = "triton-windows>=3.3.1,<3.4"
        else:
            triton_spec = "triton-windows<3.4"
    else:
        triton_spec = "triton-windows"

    print(f"[setup] Installing {triton_spec} …")
    try:
        _pip(venv, "install", triton_spec)
        print("[setup] triton-windows installed.")
    except subprocess.CalledProcessError as exc:
        print(f"[setup] WARNING: triton-windows install failed ({exc}). Some CUDA kernels may not work.")


# --------------------------------------------------------------------------- #
# trellis2_gguf source                                                         #
# --------------------------------------------------------------------------- #

def _install_trellis2_gguf(venv: Path) -> None:
    """
    Download ComfyUI-Trellis2-GGUF from GitHub and extract only the
    trellis2_gguf package into site-packages.
    """
    sp    = _site_packages(venv)
    dest  = sp / "trellis2_gguf"

    if dest.exists():
        print("[setup] trellis2_gguf already installed, skipping.")
        return

    print("[setup] Downloading ComfyUI-Trellis2-GGUF source from GitHub …")
    try:
        with urllib.request.urlopen(_COMFYUI_TRELLIS2_ZIP, timeout=300) as resp:
            data = resp.read()
    except Exception as exc:
        raise RuntimeError(f"[setup] Could not download trellis2_gguf source: {exc}") from exc

    zip_root = "ComfyUI-Trellis2-GGUF-main/"
    pkg_prefix   = f"{zip_root}trellis2_gguf/"
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # ── trellis2_gguf package ──────────────────────────────────────── #
        for member in zf.namelist():
            if not member.startswith(pkg_prefix):
                continue
            rel    = member[len(zip_root):]          # "trellis2_gguf/..."
            target = sp / rel
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))

    print(f"[setup] trellis2_gguf installed to {sp}.")


def _install_comfyui_gguf(venv: Path) -> None:
    """
    Download ops.py / dequant.py / loader.py from city96/ComfyUI-GGUF into the
    path that trellis2_gguf's GGUF helpers search:
      <site-packages parent>/ComfyUI-GGUF/
    Without these files the GGUF dequant falls back to a CPU implementation.
    """
    sp = _site_packages(venv)
    gguf_dir = comfyui_gguf_target_dir(sp)

    if not comfyui_gguf_support_file_report(gguf_dir)["missing"]:
        print("[setup] ComfyUI-GGUF already installed, skipping.")
        return

    gguf_dir.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/city96/ComfyUI-GGUF/main/"
    print(f"[setup] Installing ComfyUI-GGUF (city96) GGUF ops to {gguf_dir} …")
    for fname in COMFYUI_GGUF_REQUIRED_FILES:
        dest = gguf_dir / fname
        if dest.exists():
            continue
        try:
            with urllib.request.urlopen(base + fname, timeout=60) as resp:
                dest.write_bytes(resp.read())
            print(f"[setup]   downloaded {fname}")
        except Exception as exc:
            print(f"[setup] WARNING: could not download ComfyUI-GGUF/{fname}: {exc}")
    report = comfyui_gguf_support_file_report(gguf_dir)
    if report["status"] == "installed":
        print("[setup] ComfyUI-GGUF installed.")
        return
    print(f"[setup] WARNING: ComfyUI-GGUF support files remain {report['status']}: {report['detail']}")

# --------------------------------------------------------------------------- #
# Main setup                                                                   #
# --------------------------------------------------------------------------- #

def setup(python_exe: str, ext_dir: Path, gpu_sm: int, cuda_version: int = 0) -> None:
    setup_with_config(
        python_exe=python_exe,
        ext_dir=ext_dir,
        gpu_sm=gpu_sm,
        cuda_version=cuda_version,
    )


def setup_with_config(
    python_exe: str,
    ext_dir: Path,
    gpu_sm: int,
    cuda_version: int = 0,
    *,
    trellis_wheelhouse: str | None = None,
    trellis_extra_wheel_index: str | None = None,
    trellis_extra_wheel_url: str | None = None,
) -> dict[str, object]:
    context, policy = _phase0_collect_and_plan(
        python_exe,
        ext_dir,
        gpu_sm,
        cuda_version,
        trellis_wheelhouse=trellis_wheelhouse,
        trellis_extra_wheel_index=trellis_extra_wheel_index,
        trellis_extra_wheel_url=trellis_extra_wheel_url,
    )

    if not policy["continue_install_base"]:
        raise RuntimeError("[setup] Base install policy rejected setup before phase 1.")

    base_phase = _phase1_install_base(context)
    dependency_results, torch_version = _install_native_wheels(context)

    final_env = _build_final_runtime_environment(context, dependency_results, torch_version)
    generate_report = DependencyPreflight.evaluate("generate", final_env)
    refine_report = DependencyPreflight.evaluate("refine", final_env)

    native_statuses = {result["status"] for result in dependency_results.values()}
    status = "success"
    if any(result == "failed" for result in native_statuses):
        status = "partial"
    if any(result in {"missing", "skipped", "unsupported"} for result in native_statuses):
        status = "partial"

    report = {
        "status": status,
        "phases": {
            "phase0": "completed",
            "phase1": "completed",
            "phase2": "completed",
            "phase3": "completed",
        },
        "platform": {
            "system": context.system,
            "machine": context.machine,
            "platform_tag": context.platform_tag,
            "native_wheel_query_supported": bool(policy["native_wheel_query_supported"]),
        },
        "python": {
            "tag": context.python_tag,
            "version": context.python_version,
            "executable": context.python_exe,
        },
        "cuda": {
            "gpu_sm": str(context.gpu_sm),
            "cuda_version": str(context.cuda_version) if context.cuda_version else "",
            "torch_version": torch_version,
            "torch_index": base_phase["torch_index"],
        },
        "base_install": base_phase,
        "wheel_sources": _serialize_wheel_sources(context.wheel_sources),
        "dependencies": dependency_results,
        "capabilities": {
            "generate": _serialize_capability_report(generate_report),
            "refine": _serialize_capability_report(refine_report),
        },
        "policy_notes": list(policy.get("policy_notes", ())),
        "next_actions": _build_next_actions(context, dependency_results, generate_report, refine_report),
    }

    report_path = _write_setup_report(context.ext_dir, report)
    _print_setup_summary(report)
    print(f"[setup] Wrote report to {report_path}")
    print("[setup] Done. Venv ready at:", context.venv_dir)
    return report


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # PowerShell may pass the surrounding single quotes as part of the string
        raw = sys.argv[1].strip("'\"")
        args = json.loads(raw)
        setup_with_config(
            python_exe   = args["python_exe"],
            ext_dir      = Path(args["ext_dir"]),
            gpu_sm       = int(args.get("gpu_sm",       86)),
            cuda_version = int(args.get("cuda_version",  0)),
            trellis_wheelhouse = args.get("trellis_wheelhouse"),
            trellis_extra_wheel_index = args.get("trellis_extra_wheel_index"),
            trellis_extra_wheel_url = args.get("trellis_extra_wheel_url"),
        )
    elif len(sys.argv) >= 4:
        setup_with_config(
            python_exe   = sys.argv[1],
            ext_dir      = Path(sys.argv[2]),
            gpu_sm       = int(sys.argv[3]),
            cuda_version = int(sys.argv[4]) if len(sys.argv) >= 5 else 0,
        )
    else:
        print("Usage (positional): python setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version]")
        print('Usage (JSON)      : python setup.py "{\"python_exe\":\"...\",\"ext_dir\":\"...\",\"gpu_sm\":86,\"cuda_version\":128}"')
        sys.exit(1)
