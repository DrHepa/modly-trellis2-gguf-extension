import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


def _load_setup_module():
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("extension_setup", repo_root / "setup.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


setup_module = _load_setup_module()


class RedactUrlTests(unittest.TestCase):
    def test_redacts_basic_auth_and_strips_query_and_fragment(self):
        raw = "https://user:secret-token@example.com/private/wheels?token=abc123#frag"
        redacted = setup_module._redact_url(raw)

        self.assertEqual(redacted, "https://***:***@example.com/private/wheels")
        self.assertNotIn("user", redacted)
        self.assertNotIn("secret-token", redacted)
        self.assertNotIn("abc123", redacted)
        self.assertNotIn("?", redacted)
        self.assertNotIn("#", redacted)

    def test_strips_query_and_fragment_without_credentials(self):
        raw = "https://example.com/simple/index.html?password=hunter2#section"
        redacted = setup_module._redact_url(raw)

        self.assertEqual(redacted, "https://example.com/simple/index.html")
        self.assertNotIn("hunter2", redacted)
        self.assertNotIn("?", redacted)
        self.assertNotIn("#", redacted)


class OptionalPythonDependencyTests(unittest.TestCase):
    def test_core_packages_do_not_require_open3d(self):
        self.assertNotIn("open3d", setup_module._PY_PACKAGES)
        self.assertEqual(setup_module._PY_OPTIONAL_PACKAGES, ["open3d"])

    def test_linux_arm64_policy_skips_open3d_without_running_pip(self):
        context = setup_module._build_setup_context(
            python_exe=sys.executable,
            ext_dir=Path("/tmp/modly-trellis2-tests"),
            gpu_sm=0,
            system="Linux",
            machine="aarch64",
        )
        pip_calls = []

        def fake_pip_runner(venv, *args):
            pip_calls.append((venv, args))

        results = setup_module._install_optional_python_packages(context, pip_runner=fake_pip_runner)

        self.assertEqual(pip_calls, [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "open3d")
        self.assertEqual(results[0]["status"], "skipped")
        self.assertIn("Linux ARM64/aarch64", results[0]["detail"])

    def test_optional_install_failure_is_reported_without_raising(self):
        context = setup_module._build_setup_context(
            python_exe=sys.executable,
            ext_dir=Path("/tmp/modly-trellis2-tests"),
            gpu_sm=0,
            system="Linux",
            machine="x86_64",
        )

        def failing_pip_runner(_venv, *args):
            raise subprocess.CalledProcessError(returncode=1, cmd=["pip", *args])

        results = setup_module._install_optional_python_packages(context, pip_runner=failing_pip_runner)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "open3d")
        self.assertEqual(results[0]["status"], "failed")
        self.assertIn("install failed", results[0]["detail"])

    def test_phase1_report_includes_optional_package_outcomes(self):
        context = setup_module._build_setup_context(
            python_exe=sys.executable,
            ext_dir=Path("/tmp/modly-trellis2-tests"),
            gpu_sm=0,
            system="Linux",
            machine="aarch64",
        )
        original_run = setup_module.subprocess.run
        original_pip = setup_module._pip
        original_get_torch_version = setup_module._get_torch_version
        original_install_trellis = setup_module._install_trellis2_gguf
        original_install_gguf = setup_module._install_comfyui_gguf

        try:
            setup_module.subprocess.run = lambda *args, **kwargs: None
            setup_module._pip = lambda *args, **kwargs: None
            setup_module._get_torch_version = lambda _venv: ""
            setup_module._install_trellis2_gguf = lambda _venv: None
            setup_module._install_comfyui_gguf = lambda _venv: None

            base_report = setup_module._phase1_install_base(context)
        finally:
            setup_module.subprocess.run = original_run
            setup_module._pip = original_pip
            setup_module._get_torch_version = original_get_torch_version
            setup_module._install_trellis2_gguf = original_install_trellis
            setup_module._install_comfyui_gguf = original_install_gguf

        self.assertIn("optional_packages", base_report)
        self.assertEqual(base_report["optional_packages"][0]["name"], "open3d")
        self.assertEqual(base_report["optional_packages"][0]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
