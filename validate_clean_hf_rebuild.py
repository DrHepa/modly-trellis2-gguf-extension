from __future__ import annotations

import json
import subprocess
from pathlib import Path

import validate_clean_hf_runtime_support


ROOT = Path(__file__).resolve().parent
EXPECTED_GITIGNORE_ENTRIES = (
    "openspec/",
    ".openspec/",
    "docs/sdd/",
    "__pycache__/",
    "*.py[cod]",
)
ALLOWED_IGNORED_PATHS = {
    "docs/sdd/",
    ".openspec/",
    "openspec/",
    "__pycache__/",
}
SUSPICIOUS_PATH_PARTS = (
    "venv/",
    "ComfyUI-GGUF",
    "__pycache__/",
)
SUSPICIOUS_DIAGNOSTIC_TOKENS = (
    "diagnostic",
    "diagnostics",
    "debug",
    "trace",
)


def _read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def _normalize_status_path(line: str) -> str:
    parts = line.split(maxsplit=1)
    return parts[1].strip() if len(parts) == 2 else ""


def _is_suspicious_diagnostic_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith((".png", ".json")) and any(
        token in lowered for token in SUSPICIOUS_DIAGNOSTIC_TOKENS
    )


def validate_manifest_structure() -> None:
    manifest = json.loads(_read_text("manifest.json"))
    nodes = manifest.get("nodes")
    assert isinstance(nodes, list) and nodes, "manifest must define at least one node"
    node_ids = [node.get("id") for node in nodes]
    assert node_ids == ["generate"], f"unexpected public nodes: {node_ids}"

    manifest_text = json.dumps(manifest)
    assert '"refine"' not in manifest_text, "manifest must not expose a refine node"
    assert "Texture Mesh" not in manifest_text, "manifest must not expose texture mesh copy"
    assert "remesh_resolution" not in manifest_text, "manifest must not expose unused remesh resolution"


def validate_generator_public_contract() -> None:
    source = _read_text("generator.py")
    readme = _read_text("README.md")
    assert 'self.model_dir.name == "refine"' in source
    assert "Texture Mesh / refine is intentionally unavailable" in source
    assert 'DependencyPreflight.evaluate("generate"' in source
    assert "urlretrieve(" not in source
    assert "texture_mesh(" not in source
    assert "_MODEL_MANAGER_PATHS" in source
    assert "rglob(filename)" not in source
    assert "remesh_resolution" not in source
    assert "remesh_resolution" not in readme


def validate_gitignore_contract() -> None:
    lines = {line.strip() for line in _read_text(".gitignore").splitlines() if line.strip() and not line.strip().startswith("#")}
    missing = [entry for entry in EXPECTED_GITIGNORE_ENTRIES if entry not in lines]
    assert not missing, f"missing .gitignore entries: {', '.join(missing)}"


def validate_git_privacy_status() -> None:
    status_output = _run_git("status", "--short", "--branch", "--ignored")
    tracked_output = _run_git("ls-files")

    suspicious_status: list[str] = []
    unexpected_ignored: list[str] = []

    for raw_line in status_output.splitlines():
        if not raw_line or raw_line.startswith("##"):
            continue
        path = _normalize_status_path(raw_line)
        if not path:
            continue
        if raw_line.startswith("!!") and path not in ALLOWED_IGNORED_PATHS:
            unexpected_ignored.append(path)
        is_allowed_ignored = raw_line.startswith("!!") and path in ALLOWED_IGNORED_PATHS
        if not is_allowed_ignored and (
            any(part in path for part in SUSPICIOUS_PATH_PARTS) or _is_suspicious_diagnostic_path(path)
        ):
            suspicious_status.append(raw_line)

    suspicious_tracked = [
        path
        for path in tracked_output.splitlines()
        if any(part in path for part in SUSPICIOUS_PATH_PARTS) or _is_suspicious_diagnostic_path(path)
    ]

    assert not unexpected_ignored, f"unexpected ignored paths present: {unexpected_ignored}"
    assert not suspicious_status, f"unexpected suspicious status entries: {suspicious_status}"
    assert not suspicious_tracked, f"unexpected suspicious tracked files: {suspicious_tracked}"


def main() -> None:
    validate_clean_hf_runtime_support.main()
    validate_manifest_structure()
    validate_generator_public_contract()
    validate_gitignore_contract()
    validate_git_privacy_status()
    print("clean HF rebuild validation passed")


if __name__ == "__main__":
    main()
