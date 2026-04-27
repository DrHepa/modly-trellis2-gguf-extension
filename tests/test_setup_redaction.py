import importlib.util
import tempfile
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
            gpu_sm=121,
            system="Linux",
            machine="aarch64",
        )
        original_run = setup_module.subprocess.run
        original_pip = setup_module._pip
        original_get_torch_version = setup_module._get_torch_version
        original_install_trellis = setup_module._install_trellis2_gguf
        original_install_gguf = setup_module._install_comfyui_gguf
        pip_calls = []

        try:
            setup_module.subprocess.run = lambda *args, **kwargs: None

            def fake_pip(*args, **kwargs):
                pip_calls.append((args, kwargs))

            setup_module._pip = fake_pip
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
        self.assertEqual(base_report["rembg"], "rembg + onnxruntime")
        self.assertEqual(base_report["rembg_plan"]["packages"], ["rembg", "onnxruntime"])
        self.assertEqual(base_report["rembg_plan"]["mode"], "cpu")
        self.assertIn("Linux ARM64/aarch64", base_report["rembg_plan"]["detail"])
        self.assertIn(((context.venv_dir, "install", "rembg", "onnxruntime"), {}), pip_calls)

    def test_phase1_propagates_rembg_install_failure(self):
        context = setup_module._build_setup_context(
            python_exe=sys.executable,
            ext_dir=Path("/tmp/modly-trellis2-tests"),
            gpu_sm=121,
            system="Linux",
            machine="aarch64",
        )
        original_run = setup_module.subprocess.run
        original_pip = setup_module._pip

        try:
            setup_module.subprocess.run = lambda *args, **kwargs: None

            def fake_pip(_venv, *args):
                if args == ("install", "rembg", "onnxruntime"):
                    raise subprocess.CalledProcessError(returncode=1, cmd=["pip", *args])

            setup_module._pip = fake_pip

            with self.assertRaises(subprocess.CalledProcessError):
                setup_module._phase1_install_base(context)
        finally:
            setup_module.subprocess.run = original_run
            setup_module._pip = original_pip


class RembgPolicyTests(unittest.TestCase):
    def test_linux_arm64_high_gpu_uses_cpu_plan(self):
        policy = setup_module._rembg_package_policy(121, system="Linux", machine="aarch64")

        self.assertEqual(policy["mode"], "cpu")
        self.assertEqual(policy["packages"], ["rembg", "onnxruntime"])
        self.assertEqual(policy["summary"], "rembg + onnxruntime")
        self.assertNotIn("rembg[gpu]", policy["packages"])
        self.assertIn("Linux ARM64/aarch64", policy["detail"])

    def test_non_arm64_high_gpu_preserves_gpu_plan(self):
        policy = setup_module._rembg_package_policy(121, system="Linux", machine="x86_64")

        self.assertEqual(policy["mode"], "gpu")
        self.assertEqual(policy["packages"], ["rembg[gpu]"])
        self.assertEqual(policy["summary"], "rembg[gpu]")

    def test_low_gpu_uses_cpu_plan(self):
        policy = setup_module._rembg_package_policy(61, system="Linux", machine="x86_64")

        self.assertEqual(policy["mode"], "cpu")
        self.assertEqual(policy["packages"], ["rembg", "onnxruntime"])
        self.assertEqual(policy["summary"], "rembg + onnxruntime")


class GenerateNativeWheelReportingTests(unittest.TestCase):
    def test_default_generate_required_native_wheels_include_all_import_time_blockers(self):
        self.assertEqual(
            setup_module._CUDA_WHEELS_REQUIRED,
            ["cumesh", "o-voxel", "flex-gemm", "nvdiffrast"],
        )

    def test_next_actions_request_all_default_generate_native_wheels(self):
        context = setup_module._build_setup_context(
            python_exe=sys.executable,
            ext_dir=Path("/tmp/modly-trellis2-tests"),
            gpu_sm=121,
            cuda_version=128,
            system="Linux",
            machine="aarch64",
        )
        dependency_results = {
            "cumesh": {"status": "missing", "detail": "missing"},
            "o_voxel": {"status": "missing", "detail": "missing"},
            "flex_gemm": {"status": "missing", "detail": "missing"},
            "nvdiffrast": {"status": "missing", "detail": "missing"},
            "spconv": {"status": "unsupported", "detail": "conditional alternate backend"},
            "cumm": {"status": "unsupported", "detail": "conditional alternate backend"},
            "comfyui_gguf_support_files": {"status": "installed", "detail": "ok"},
        }
        final_env = setup_module._build_final_runtime_environment(context, dependency_results, "2.7.0")
        generate_report = setup_module.DependencyPreflight.evaluate("generate", final_env)
        refine_report = setup_module.DependencyPreflight.evaluate("refine", final_env)

        actions = setup_module._build_next_actions(context, dependency_results, generate_report, refine_report)
        joined = "\n".join(actions)

        self.assertIn("cumesh", joined)
        self.assertIn("o-voxel", joined)
        self.assertIn("flex-gemm", joined)
        self.assertIn("nvdiffrast", joined)
        self.assertNotIn("spconv", joined)
        self.assertNotIn("cumm", joined)

    def test_plan_report_labels_phase0_notes_separately(self):
        report = setup_module._plan_setup(
            python_exe=sys.executable,
            ext_dir=Path("/tmp/modly-trellis2-tests"),
            gpu_sm=121,
            cuda_version=128,
            system="Linux",
            machine="aarch64",
        )

        self.assertIn("policy_notes", report)
        self.assertIn("phase0_policy_notes", report)
        self.assertTrue(any("Linux ARM64 continues with base install" in note for note in report["policy_notes"]))
        self.assertFalse(any("comfyui_gguf_support_files: missing" in note for note in report["policy_notes"]))
        self.assertTrue(any("comfyui_gguf_support_files: missing" in note for note in report["phase0_policy_notes"]))

    def test_final_report_filters_stale_phase0_blockers_from_policy_notes(self):
        context = setup_module._build_setup_context(
            python_exe=sys.executable,
            ext_dir=Path("/tmp/modly-trellis2-tests"),
            gpu_sm=121,
            cuda_version=128,
            system="Linux",
            machine="aarch64",
        )
        policy = {
            "native_wheel_query_supported": False,
            "policy_notes": [
                "Linux ARM64 continues with base install, but public native-wheel queries stay disabled and source builds remain out of scope."
            ],
            "phase0_policy_notes": [
                "Linux ARM64 continues with base install, but public native-wheel queries stay disabled and source builds remain out of scope.",
                "Generate blockers: comfyui_gguf_support_files: missing",
            ],
        }
        dependency_results = {
            "cumesh": {"status": "installed", "detail": "installed"},
            "o_voxel": {"status": "installed", "detail": "installed"},
            "flex_gemm": {"status": "installed", "detail": "installed"},
            "nvdiffrast": {"status": "installed", "detail": "installed"},
            "spconv": {"status": "unsupported", "detail": "conditional alternate backend"},
            "cumm": {"status": "unsupported", "detail": "conditional alternate backend"},
            "comfyui_gguf_support_files": {"status": "installed", "detail": "verified required support files"},
        }
        base_phase = {
            "venv": "/tmp/modly-trellis2-tests/venv",
            "torch_packages": ["torch==2.7.0", "torchvision==0.22.0"],
            "torch_index": "https://download.pytorch.org/whl/cu128",
            "torch_version": "2.7.0",
            "core_packages": [],
            "optional_packages": [],
            "rembg": "rembg + onnxruntime",
            "rembg_plan": {"mode": "cpu", "packages": ["rembg", "onnxruntime"], "summary": "rembg + onnxruntime", "detail": "cpu"},
        }

        report = setup_module._build_setup_report(context, policy, base_phase, dependency_results, "2.7.0")

        self.assertFalse(any("comfyui_gguf_support_files: missing" in note for note in report["policy_notes"]))
        self.assertTrue(any("comfyui_gguf_support_files: missing" in note for note in report["phase0_policy_notes"]))
        self.assertTrue(any("Linux ARM64 continues with base install" in note for note in report["policy_notes"]))

    def test_local_wheelhouse_native_install_uses_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheelhouse = root / "wheelhouse"
            wheelhouse.mkdir()
            for filename in (
                "cumesh-0.0.1-cp312-cp312-linux_aarch64.whl",
                "flex_gemm-1.0.0-cp312-cp312-linux_aarch64.whl",
                "o_voxel-0.0.1-cp312-cp312-linux_aarch64.whl",
                "nvdiffrast-0.4.0-cp312-cp312-linux_aarch64.whl",
            ):
                (wheelhouse / filename).write_text("stub", encoding="utf-8")

            site_packages = root / "venv" / "lib" / "python3.12" / "site-packages"
            gguf_dir = site_packages.parent / "ComfyUI-GGUF"
            gguf_dir.mkdir(parents=True)
            for name in ("ops.py", "dequant.py", "loader.py"):
                (gguf_dir / name).write_text("ok", encoding="utf-8")

            context = setup_module._build_setup_context(
                python_exe=sys.executable,
                ext_dir=root / "ext",
                gpu_sm=121,
                cuda_version=128,
                trellis_wheelhouse=str(wheelhouse),
                system="Linux",
                machine="aarch64",
            )

            original_pip = setup_module._pip
            original_get_torch_version = setup_module._get_torch_version
            original_site_packages = setup_module._site_packages
            pip_calls = []

            try:
                def fake_pip(*args):
                    pip_calls.append(args)

                setup_module._pip = fake_pip
                setup_module._get_torch_version = lambda _venv: "2.7.0"
                setup_module._site_packages = lambda _venv: site_packages

                dependency_results, _torch_version = setup_module._install_native_wheels(context)
            finally:
                setup_module._pip = original_pip
                setup_module._get_torch_version = original_get_torch_version
                setup_module._site_packages = original_site_packages

            expected_paths = {
                str(wheelhouse / "cumesh-0.0.1-cp312-cp312-linux_aarch64.whl"),
                str(wheelhouse / "flex_gemm-1.0.0-cp312-cp312-linux_aarch64.whl"),
                str(wheelhouse / "o_voxel-0.0.1-cp312-cp312-linux_aarch64.whl"),
                str(wheelhouse / "nvdiffrast-0.4.0-cp312-cp312-linux_aarch64.whl"),
            }
            actual_paths = {args[2] for args in pip_calls}

            self.assertEqual({args[1] for args in pip_calls}, {"install"})
            self.assertEqual(actual_paths, expected_paths)
            self.assertEqual(
                dependency_results["cumesh"]["install_target"],
                str(wheelhouse / "cumesh-0.0.1-cp312-cp312-linux_aarch64.whl"),
            )
            self.assertEqual(dependency_results["cumesh"]["selected_source"], "local-wheelhouse")


if __name__ == "__main__":
    unittest.main()
