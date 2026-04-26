"""Trellis.2 GGUF extension setup script.

Creates an isolated venv and installs the geometry-first runtime slice.
The initial clean rebuild keeps Generate Mesh public and treats texturing /
refine-only native pieces as optional follow-up work on supported platforms.

Called by Modly at extension install time with:
    python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":124}'
"""

import io
import json
import platform
import re
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from runtime_support import DependencyPreflight, RuntimeEnvironment

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
    "open3d",
    "rectpack",
    "requests",
    "huggingface_hub",
    "transformers==5.2.0",
    "accelerate",
    "einops",
    "easydict",
]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

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
        for name in ("flex_gemm", "cumesh", "nvdiffrast", "o_voxel", "spconv", "cumm")
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


def _preflight_setup_policy(gpu_sm: int, cuda_version: int = 0) -> dict[str, object]:
    report = _native_policy_report(_setup_runtime_environment(gpu_sm, cuda_version))
    env = report["env"]
    assert isinstance(env, RuntimeEnvironment)

    print(
        "[setup] Native policy context: "
        f"system={env.system} machine={env.machine} python={env.python_tag} "
        f"cuda={env.cuda_version or 'unknown'} gpu_sm={env.gpu_sm or 'unknown'}"
    )

    if env.system.lower() == "linux" and env.machine.lower() in {"aarch64", "arm64"}:
        generate = report["generate"]
        refine = report["refine"]
        assert hasattr(generate, "blockers") and hasattr(refine, "blockers")
        raise RuntimeError(
            "[setup] Linux ARM64 is not supported by the public native-wheel path for this clean rebuild.\n"
            f"[setup] Generate blockers: {'; '.join(generate.blockers)}\n"
            f"[setup] Refine blockers: {'; '.join(refine.blockers)}\n"
            "[setup] This setup will not pretend linux_x86_64 wheels apply to ARM64, and it will not build native dependencies from source."
        )

    if report["platform_tag"] is None:
        raise RuntimeError(
            f"[setup] Unsupported platform for native wheels: {env.system}/{env.machine}. "
            "Only Windows x86_64 and Linux x86_64 are modeled in this setup script."
        )

    refine = report["refine"]
    assert hasattr(refine, "blockers")
    if refine.blockers:
        print(f"[setup] NOTE: Refine remains gated in this slice: {'; '.join(refine.blockers)}")

    return report


# --------------------------------------------------------------------------- #
# Custom CUDA wheels (pozzettiandrea.github.io)                                #
# --------------------------------------------------------------------------- #

def _find_wheel_url(lib_name: str, python_tag: str, platform_tag: str, torch_ver: str) -> str | None:
    """
    Fetch the pozzettiandrea.github.io wheel index for lib_name and return
    the best matching wheel URL for (python_tag, platform_tag, torch_ver).

    Tries the exact torch version first, then falls back to the closest
    lower version available on the index.
    Returns None if the index page is unreachable or no match found.
    """
    index_url = f"{_WHEELS_INDEX_BASE}{lib_name}/"
    try:
        with urllib.request.urlopen(index_url, timeout=30) as resp:
            html = resp.read().decode("utf-8")
    except Exception as exc:
        print(f"[setup] WARNING: Could not fetch wheel index for {lib_name}: {exc}")
        return None

    # Extract all .whl href links
    links = re.findall(r'href=["\']([^"\']*\.whl)["\']', html)
    if not links:
        links = re.findall(r'([\w\-\.]+\.whl)', html)

    def _abs(link: str) -> str:
        if link.startswith("http"):
            return link
        return f"{index_url}{link.split('/')[-1]}"

    # Normalise torch version: "2.7.0" -> "27" (index uses "torch27", not "torch270")
    parts = torch_ver.split(".")
    major, minor = int(parts[0]), int(parts[1])
    tv_tag = f"{major}{minor}"

    def _candidates(maj: int, min_: int) -> list[str]:
        # Index display names use "torch26", GitHub URLs use "torch2.6"
        tags = [f"torch{maj}{min_}", f"torch{maj}.{min_}"]
        return [
            _abs(link) for link in links
            if python_tag in link.split("/")[-1]
            and platform_tag in link.split("/")[-1]
            and any(t in link.split("/")[-1] for t in tags)
        ]

    # Try exact version first, then fall back to lower minor versions
    for m in range(minor, -1, -1):
        matches = _candidates(major, m)
        if matches:
            if m != minor:
                print(f"[setup] NOTE: No wheel for torch{tv_tag} — using torch{major}{m} fallback.")
            return matches[0]

    return None


def _install_cuda_wheels(venv: Path, gpu_sm: int, platform_tag: str) -> None:
    """Download and install custom CUDA wheels from pozzettiandrea.github.io."""
    if platform_tag not in {"win_amd64", "linux_x86_64"}:
        raise RuntimeError(
            f"[setup] Refusing unsupported CUDA wheel platform tag '{platform_tag}'. "
            "This script only knows how to query win_amd64 and linux_x86_64 wheels."
        )
    python_tag = _python_tag()
    torch_ver    = _get_torch_version(venv)

    print(f"[setup] Installing CUDA wheels (python={python_tag}, platform={platform_tag}, torch={torch_ver}) …")

    for lib in _CUDA_WHEELS:
        print(f"[setup] Finding wheel for {lib} …")
        url = _find_wheel_url(lib, python_tag, platform_tag, torch_ver)
        if url is None:
            if lib in _CUDA_WHEELS_REQUIRED:
                print(
                    f"[setup] ERROR: No compatible wheel found for {lib} (torch {torch_ver}).\n"
                    f"[setup]   This is a required dependency — the extension will fail to load.\n"
                    f"[setup]   Check https://pozzettiandrea.github.io/cuda-wheels/{lib}/ for available wheels."
                )
            else:
                print(f"[setup] WARNING: No wheel found for {lib} — texture baking will be unavailable.")
            continue
        print(f"[setup] Installing {lib} from {url} …")
        try:
            _pip(venv, "install", url)
            print(f"[setup] {lib} installed.")
        except subprocess.CalledProcessError as exc:
            if lib in _CUDA_WHEELS_REQUIRED:
                print(f"[setup] ERROR: Failed to install {lib} ({exc}). The extension will not load.")
            else:
                print(f"[setup] WARNING: Failed to install {lib} ({exc}). Texture baking may be unavailable.")


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
    path that trellis2_gguf's _setup_native_gguf() searches:
      <venv>/Lib/ComfyUI-GGUF/
    Without these files the GGUF dequant falls back to a CPU implementation.
    """
    sp       = _site_packages(venv)
    gguf_dir = sp.parent.parent / "Lib" / "ComfyUI-GGUF"   # <venv>/Lib/ComfyUI-GGUF
    _FILES   = ["ops.py", "dequant.py", "loader.py"]

    if all((gguf_dir / f).exists() for f in _FILES):
        print("[setup] ComfyUI-GGUF already installed, skipping.")
        return

    gguf_dir.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/city96/ComfyUI-GGUF/main/"
    print(f"[setup] Installing ComfyUI-GGUF (city96) GGUF ops to {gguf_dir} …")
    for fname in _FILES:
        dest = gguf_dir / fname
        if dest.exists():
            continue
        try:
            with urllib.request.urlopen(base + fname, timeout=60) as resp:
                dest.write_bytes(resp.read())
            print(f"[setup]   downloaded {fname}")
        except Exception as exc:
            print(f"[setup] WARNING: could not download ComfyUI-GGUF/{fname}: {exc}")
    print("[setup] ComfyUI-GGUF installed.")

# --------------------------------------------------------------------------- #
# Main setup                                                                   #
# --------------------------------------------------------------------------- #

def setup(python_exe: str, ext_dir: Path, gpu_sm: int, cuda_version: int = 0) -> None:
    policy = _preflight_setup_policy(gpu_sm, cuda_version)
    platform_tag = policy["platform_tag"]
    assert isinstance(platform_tag, str)

    venv = ext_dir / "venv"

    print(f"[setup] Creating venv at {venv} …")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # ── PyTorch — select build based on GPU architecture / CUDA driver ── #
    if gpu_sm >= 100 or cuda_version >= 128:
        # Blackwell (RTX 50xx, B100…) — SM 12.x kernels require PyTorch 2.7+
        torch_pkgs  = ["torch==2.7.0", "torchvision==0.22.0"]
        torch_index = "https://download.pytorch.org/whl/cu128"
        print(f"[setup] GPU SM {gpu_sm}, CUDA {cuda_version} -> PyTorch 2.7 + CUDA 12.8 (Blackwell)")
    elif gpu_sm == 0 or gpu_sm >= 70:
        # Volta / Turing / Ampere / Ada / Hopper
        torch_pkgs  = ["torch==2.6.0", "torchvision==0.21.0"]
        torch_index = "https://download.pytorch.org/whl/cu126"
        print(f"[setup] GPU SM {gpu_sm} -> PyTorch 2.6 + CUDA 12.6")
    else:
        # Pascal (SM 6.x) — last PyTorch with SM 6.1 support
        torch_pkgs  = ["torch==2.5.1", "torchvision==0.20.1"]
        torch_index = "https://download.pytorch.org/whl/cu118"
        print(f"[setup] GPU SM {gpu_sm} (legacy) -> PyTorch 2.5 + CUDA 11.8")

    print("[setup] Installing PyTorch …")
    _pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    # ── Core Python dependencies ─────────────────────────────────────── #
    print("[setup] Installing core Python dependencies …")
    _pip(venv, "install", *_PY_PACKAGES)

    # ── rembg (background removal) ───────────────────────────────────── #
    print("[setup] Installing rembg …")
    if gpu_sm >= 70:
        _pip(venv, "install", "rembg[gpu]")
    else:
        _pip(venv, "install", "rembg", "onnxruntime")

    # ── Custom CUDA wheels (cumesh, nvdiffrast, flex_gemm, …) ─────────── #
    _install_cuda_wheels(venv, gpu_sm, platform_tag)

    # ── triton-windows ────────────────────────────────────────────────── #
    torch_ver = _get_torch_version(venv)
    if torch_ver:
        _install_triton_windows(venv, torch_ver, gpu_sm)

    # ── trellis2_gguf source ──────────────────────────────────────────── #
    _install_trellis2_gguf(venv)

    # ── ComfyUI-GGUF (city96) — native GGUF dequant on GPU ───────────── #
    _install_comfyui_gguf(venv)

    print("[setup] Done. Venv ready at:", venv)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # PowerShell may pass the surrounding single quotes as part of the string
        raw = sys.argv[1].strip("'\"")
        args = json.loads(raw)
        setup(
            python_exe   = args["python_exe"],
            ext_dir      = Path(args["ext_dir"]),
            gpu_sm       = int(args.get("gpu_sm",       86)),
            cuda_version = int(args.get("cuda_version",  0)),
        )
    elif len(sys.argv) >= 4:
        setup(
            python_exe   = sys.argv[1],
            ext_dir      = Path(sys.argv[2]),
            gpu_sm       = int(sys.argv[3]),
            cuda_version = int(sys.argv[4]) if len(sys.argv) >= 5 else 0,
        )
    else:
        print("Usage (positional): python setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version]")
        print('Usage (JSON)      : python setup.py "{\"python_exe\":\"...\",\"ext_dir\":\"...\",\"gpu_sm\":86,\"cuda_version\":128}"')
        sys.exit(1)
