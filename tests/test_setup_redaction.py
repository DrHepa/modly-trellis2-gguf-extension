import importlib.util
import os
import tempfile
import subprocess
import sys
import unittest
import hashlib
from pathlib import Path

from runtime_patches import (
    FDG_VAE_PATCHED_CALL_SNIPPET,
    FDG_VAE_PATCHED_IMPORT_SNIPPET,
    HostPlatform,
    build_report_fragments,
)


def _load_setup_module():
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("extension_setup", repo_root / "setup.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


setup_module = _load_setup_module()


FDG_VAE_UNPATCHED_TEXT = """from trellis.representations.mesh.flexicubes.flexicubes import (
    FlexiCubes,
    flexible_dual_grid_to_mesh,
    sparse_cube2verts,
    tiled_flexible_dual_grid_to_mesh,
)


class SparseFeatures2Mesh:
    def convert(self, flexicubes, voxels, scalar_field, cube_idx, resolution):
        vertices, faces, reg_loss = tiled_flexible_dual_grid_to_mesh(
            flexicubes,
            voxels,
            scalar_field,
            cube_idx,
            resolution,
        )
        return vertices, faces, reg_loss
"""


def _write_runtime_patch_targets(site_packages: Path, *, fdg_text: str = FDG_VAE_UNPATCHED_TEXT) -> Path:
    config = site_packages / "trellis2_gguf/modules/sparse/conv/config.py"
    debug = site_packages / "trellis2_gguf/modules/sparse/conv/conv_flex_gemm.py"
    fdg = site_packages / "trellis2_gguf/models/sc_vaes/fdg_vae.py"
    config.parent.mkdir(parents=True, exist_ok=True)
    debug.parent.mkdir(parents=True, exist_ok=True)
    fdg.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('ALGORITHM = "implicit_gemm_splitk"\n', encoding="utf-8")
    debug.write_text('def forward(self):\n    return 1\n', encoding="utf-8")
    fdg.write_text(fdg_text, encoding="utf-8")
    return fdg


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
        original_site_packages = setup_module._site_packages
        pip_calls = []

        try:
            setup_module.subprocess.run = lambda *args, **kwargs: None

            def fake_pip(*args, **kwargs):
                pip_calls.append((args, kwargs))

            setup_module._pip = fake_pip
            setup_module._get_torch_version = lambda _venv: ""
            setup_module._install_trellis2_gguf = lambda _venv: None
            setup_module._site_packages = lambda _venv: Path("/tmp/fake-site-packages")
            setup_module._install_comfyui_gguf = lambda _venv: None

            base_report = setup_module._phase1_install_base(context)
        finally:
            setup_module.subprocess.run = original_run
            setup_module._pip = original_pip
            setup_module._get_torch_version = original_get_torch_version
            setup_module._install_trellis2_gguf = original_install_trellis
            setup_module._install_comfyui_gguf = original_install_gguf
            setup_module._site_packages = original_site_packages

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

    def test_setup_report_includes_runtime_sections_and_blocks_unvalidated_target_readiness(self):
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
            "policy_notes": [],
            "phase0_policy_notes": [],
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
            "runtime_formalization": {
                **build_report_fragments(HostPlatform(system="Linux", machine="aarch64", gpu_sm=121), {}, torch_version="2.7.0"),
                "runtime_validation": {
                    "status": "not_run",
                    "checks": [],
                    "blocking_reasons": ["Runtime validation is required for Linux ARM64 Blackwell Generate readiness but was not run."],
                    "warnings": [],
                },
                "trellis_patches": {"status": "planned", "items": [], "blocking_reasons": []},
            },
        }

        report = setup_module._build_setup_report(context, policy, base_phase, dependency_results, "2.7.0")

        self.assertTrue({"runtime_overrides", "trellis_patches", "runtime_validation", "wheel_provenance"}.issubset(report))
        self.assertFalse(report["capabilities"]["generate"]["allowed"])
        self.assertEqual(report["capabilities"]["generate"]["runtime_validation_status"], "not_run")
        self.assertIn("runtime_validation", "\n".join(report["capabilities"]["generate"]["blockers"]))

    def test_apply_triton_override_invokes_target_runner_once_without_torch_reinstall(self):
        context = setup_module._build_setup_context(
            python_exe=sys.executable,
            ext_dir=Path("/tmp/modly-trellis2-tests"),
            gpu_sm=121,
            system="Linux",
            machine="aarch64",
        )
        pip_calls = []

        def fake_pip_runner(venv, *args):
            pip_calls.append((venv, args))

        report = setup_module._apply_triton_override(
            context,
            "2.7.0",
            current_triton_version="3.3.0",
            pip_runner=fake_pip_runner,
        )

        self.assertEqual(report["status"], "applied")
        self.assertTrue(report["runner_invoked"])
        self.assertEqual(report["requested_spec"], "triton==3.6.0")
        self.assertEqual(pip_calls, [(context.venv_dir, ("install", "triton==3.6.0"))])
        self.assertNotIn("torch", " ".join(pip_calls[0][1]).lower())

    def test_apply_triton_override_skips_non_target_host(self):
        context = setup_module._build_setup_context(
            python_exe=sys.executable,
            ext_dir=Path("/tmp/modly-trellis2-tests"),
            gpu_sm=121,
            system="Linux",
            machine="x86_64",
        )
        pip_calls = []

        report = setup_module._apply_triton_override(
            context,
            "2.7.0",
            current_triton_version="3.3.0",
            pip_runner=lambda *args: pip_calls.append(args),
        )

        self.assertEqual(report["status"], "not_target")
        self.assertFalse(report["runner_invoked"])
        self.assertEqual(pip_calls, [])

    def test_apply_triton_override_reports_runner_failure_fail_closed(self):
        context = setup_module._build_setup_context(
            python_exe=sys.executable,
            ext_dir=Path("/tmp/modly-trellis2-tests"),
            gpu_sm=121,
            system="Linux",
            machine="aarch64",
        )

        def failing_pip_runner(_venv, *args):
            raise subprocess.CalledProcessError(returncode=1, cmd=["pip", *args])

        report = setup_module._apply_triton_override(
            context,
            "2.7.0",
            current_triton_version="3.3.0",
            pip_runner=failing_pip_runner,
        )

        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["runner_invoked"])
        self.assertIn("returned non-zero exit status 1", report["error"])

    def test_phase1_runtime_wiring_calls_triton_then_trellis_then_runtime_planning_then_comfyui(self):
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
        original_site_packages = setup_module._site_packages
        original_apply_triton_override = setup_module._apply_triton_override
        original_runtime_formalization = setup_module._build_runtime_formalization_report
        events = []

        try:
            setup_module.subprocess.run = lambda *args, **kwargs: events.append("venv")
            setup_module._pip = lambda *_args, **_kwargs: None
            setup_module._get_torch_version = lambda _venv: "2.7.0"
            setup_module._apply_triton_override = lambda *_args, **_kwargs: events.append("triton") or {"status": "applied"}
            setup_module._install_trellis2_gguf = lambda _venv: events.append("trellis")
            setup_module._site_packages = lambda _venv: Path("/tmp/fake-site-packages")
            def fake_runtime_formalization(*_args, **_kwargs):
                events.append("runtime")
                self.assertTrue(_kwargs["apply_fdg_patch"])
                return {
                    "runtime_overrides": {},
                    "trellis_patches": {"status": "planned", "items": [], "blocking_reasons": []},
                    "runtime_validation": {"status": "not_run", "checks": [], "blocking_reasons": [], "warnings": []},
                    "wheel_provenance": {"status": "not_collected", "required_packages": [], "entries": [], "blocking_reasons": []},
                }

            setup_module._build_runtime_formalization_report = fake_runtime_formalization
            setup_module._install_comfyui_gguf = lambda _venv: events.append("comfyui")

            result = setup_module._phase1_install_base(context)
        finally:
            setup_module.subprocess.run = original_run
            setup_module._pip = original_pip
            setup_module._get_torch_version = original_get_torch_version
            setup_module._install_trellis2_gguf = original_install_trellis
            setup_module._install_comfyui_gguf = original_install_gguf
            setup_module._site_packages = original_site_packages
            setup_module._apply_triton_override = original_apply_triton_override
            setup_module._build_runtime_formalization_report = original_runtime_formalization

        self.assertGreater(events.index("triton"), 0)
        self.assertGreater(events.index("trellis"), events.index("triton"))
        self.assertGreater(events.index("runtime"), events.index("trellis"))
        self.assertGreater(events.index("comfyui"), events.index("runtime"))
        self.assertIn("runtime_formalization", result)

    def test_runtime_formalization_report_includes_executed_triton_override_result(self):
        context = setup_module._build_setup_context(
            python_exe=sys.executable,
            ext_dir=Path("/tmp/modly-trellis2-tests"),
            gpu_sm=121,
            system="Linux",
            machine="aarch64",
        )
        with tempfile.TemporaryDirectory() as tmp:
            site_packages = Path(tmp)
            _write_runtime_patch_targets(site_packages)

            report = setup_module._build_runtime_formalization_report(
                context,
                torch_version="2.7.0",
                site_packages=site_packages,
                env=os.environ.copy(),
                apply_fdg_patch=True,
                triton_override={
                    "status": "applied",
                    "plan_status": "conflict_reported",
                    "target_version": "3.6.0",
                    "current_version": "3.6.0",
                    "requested_spec": "triton==3.6.0",
                    "runner_invoked": True,
                    "conflict_note": "Torch metadata may still pin triton==3.3.0; runtime policy targets triton==3.6.0 without reinstalling torch",
                },
            )

            fdg_patch = report["trellis_patches"]["items"][2]
            patched_text = Path(fdg_patch["target_file"]).read_text(encoding="utf-8")
            backup_exists = Path(fdg_patch["backup_path"]).exists()

        self.assertEqual(set(report), {"runtime_overrides", "trellis_patches", "runtime_validation", "wheel_provenance"})
        self.assertEqual(report["runtime_overrides"]["flex_gemm"]["selected"], "explicit_gemm")
        self.assertEqual(report["runtime_overrides"]["triton"]["status"], "applied")
        self.assertEqual(report["runtime_overrides"]["triton"]["plan_status"], "conflict_reported")
        self.assertEqual(fdg_patch["status"], "applied")
        self.assertTrue(backup_exists)
        self.assertIn(FDG_VAE_PATCHED_IMPORT_SNIPPET, patched_text)
        self.assertIn(FDG_VAE_PATCHED_CALL_SNIPPET, patched_text)
        self.assertEqual(report["runtime_validation"]["status"], "not_run")
        self.assertEqual(len(report["runtime_validation"]["checks"]), 3)

    def test_runtime_formalization_report_marks_fdg_patch_already_applied_on_second_run(self):
        context = setup_module._build_setup_context(
            python_exe=sys.executable,
            ext_dir=Path("/tmp/modly-trellis2-tests"),
            gpu_sm=121,
            system="Linux",
            machine="aarch64",
        )
        with tempfile.TemporaryDirectory() as tmp:
            site_packages = Path(tmp)
            fdg = _write_runtime_patch_targets(site_packages)

            first = setup_module._build_runtime_formalization_report(
                context,
                torch_version="2.7.0",
                site_packages=site_packages,
                env=os.environ.copy(),
                apply_fdg_patch=True,
            )
            second = setup_module._build_runtime_formalization_report(
                context,
                torch_version="2.7.0",
                site_packages=site_packages,
                env=os.environ.copy(),
                apply_fdg_patch=True,
            )

            patched_text = fdg.read_text(encoding="utf-8")

        self.assertEqual(first["trellis_patches"]["items"][2]["status"], "applied")
        self.assertEqual(second["trellis_patches"]["items"][2]["status"], "already_applied")
        self.assertIn(FDG_VAE_PATCHED_IMPORT_SNIPPET, patched_text)
        self.assertEqual(second["trellis_patches"]["items"][2]["after_hash"], second["trellis_patches"]["items"][2]["before_hash"])

    def test_runtime_formalization_report_fails_closed_when_fdg_patch_drifts(self):
        context = setup_module._build_setup_context(
            python_exe=sys.executable,
            ext_dir=Path("/tmp/modly-trellis2-tests"),
            gpu_sm=121,
            system="Linux",
            machine="aarch64",
        )
        drifted_text = FDG_VAE_UNPATCHED_TEXT.replace("            resolution,\n", "            resolution,\n            iso_value,\n", 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site_packages = root / "site-packages"
            _write_runtime_patch_targets(site_packages, fdg_text=drifted_text)
            ext_dir = root / "ext"
            validation_python = ext_dir / "venv" / "bin" / "python"
            validation_python.parent.mkdir(parents=True, exist_ok=True)
            validation_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            context = setup_module._build_setup_context(
                python_exe=sys.executable,
                ext_dir=ext_dir,
                gpu_sm=121,
                cuda_version=128,
                system="Linux",
                machine="aarch64",
            )
            runtime_formalization = setup_module._build_runtime_formalization_report(
                context,
                torch_version="2.7.0",
                site_packages=site_packages,
                env={},
                apply_fdg_patch=True,
            )
            base_phase = {
                "venv": str(context.venv_dir),
                "torch_packages": ["torch==2.7.0", "torchvision==0.22.0"],
                "torch_index": "https://download.pytorch.org/whl/cu128",
                "torch_version": "2.7.0",
                "core_packages": [],
                "optional_packages": [],
                "rembg": "rembg + onnxruntime",
                "rembg_plan": {"mode": "cpu", "packages": ["rembg", "onnxruntime"], "summary": "rembg + onnxruntime", "detail": "cpu"},
                "runtime_formalization": runtime_formalization,
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
            report = setup_module._build_setup_report(
                context,
                {"native_wheel_query_supported": False, "policy_notes": [], "phase0_policy_notes": []},
                base_phase,
                dependency_results,
                "2.7.0",
            )

        self.assertEqual(runtime_formalization["trellis_patches"]["status"], "blocked")
        self.assertEqual(runtime_formalization["trellis_patches"]["items"][2]["status"], "drift")
        self.assertIn("approved patch window", runtime_formalization["trellis_patches"]["blocking_reasons"][0])
        self.assertFalse(report["capabilities"]["generate"]["allowed"])

    def test_runtime_formalization_report_marks_missing_fdg_target_as_blocking(self):
        context = setup_module._build_setup_context(
            python_exe=sys.executable,
            ext_dir=Path("/tmp/modly-trellis2-tests"),
            gpu_sm=121,
            system="Linux",
            machine="aarch64",
        )
        with tempfile.TemporaryDirectory() as tmp:
            site_packages = Path(tmp)
            config = site_packages / "trellis2_gguf/modules/sparse/conv/config.py"
            debug = site_packages / "trellis2_gguf/modules/sparse/conv/conv_flex_gemm.py"
            config.parent.mkdir(parents=True, exist_ok=True)
            debug.parent.mkdir(parents=True, exist_ok=True)
            config.write_text('ALGORITHM = "implicit_gemm_splitk"\n', encoding="utf-8")
            debug.write_text('def forward(self):\n    return 1\n', encoding="utf-8")

            report = setup_module._build_runtime_formalization_report(
                context,
                torch_version="2.7.0",
                site_packages=site_packages,
                env=os.environ.copy(),
                apply_fdg_patch=True,
            )

        fdg_patch = report["trellis_patches"]["items"][2]
        self.assertEqual(report["trellis_patches"]["status"], "blocked")
        self.assertEqual(fdg_patch["status"], "missing")
        self.assertIn("not present", fdg_patch["drift_reason"])

    def test_runtime_formalization_report_collects_public_wheel_provenance_from_local_wheelhouse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheelhouse = root / "secret-wheelhouse"
            wheelhouse.mkdir()
            payload = b"cumesh-wheel"
            filenames = {
                "cumesh-1.0.0+torch2.7.cu128-cp312-none-manylinux_2_17_aarch64.whl": payload,
                "flex_gemm-1.0.0+torch2.7.cu128-cp312-none-manylinux_2_17_aarch64.whl": b"flex",
                "o_voxel-1.0.0+torch2.7.cu128-cp312-none-manylinux_2_17_aarch64.whl": b"ovoxel",
                "nvdiffrast-1.0.0+torch2.7.cu128-cp312-none-manylinux_2_17_aarch64.whl": b"nvdiff",
            }
            for filename, contents in filenames.items():
                (wheelhouse / filename).write_bytes(contents)

            site_packages = root / "site-packages"
            _write_runtime_patch_targets(site_packages)

            context = setup_module._build_setup_context(
                python_exe=sys.executable,
                ext_dir=root / "ext",
                gpu_sm=121,
                cuda_version=128,
                trellis_wheelhouse=str(wheelhouse),
                system="Linux",
                machine="aarch64",
            )

            report = setup_module._build_runtime_formalization_report(
                context,
                torch_version="2.7.0",
                site_packages=site_packages,
                env=os.environ.copy(),
            )

            provenance = report["wheel_provenance"]
            self.assertEqual(provenance["status"], "available")
            self.assertEqual(provenance["sources"][0]["location_hint"], "<wheelhouse:secret-wheelhouse>")
            self.assertNotIn(str(wheelhouse), str(provenance))
            cumesh = next(entry for entry in provenance["entries"] if entry["package"] == "cumesh")
            self.assertEqual(cumesh["sha256"], hashlib.sha256(payload).hexdigest())

    def test_runtime_validation_uses_extension_venv_python_not_host_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = setup_module._build_setup_context(
                python_exe="/host/python",
                ext_dir=root / "ext",
                gpu_sm=121,
                cuda_version=128,
                system="Linux",
                machine="aarch64",
            )
            validation_python = context.venv_dir / "bin" / "python"
            validation_python.parent.mkdir(parents=True, exist_ok=True)
            validation_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

            report = setup_module._build_runtime_validation_report(
                context,
                "explicit_gemm",
                env={},
            )

        self.assertEqual(report["status"], "not_run")
        self.assertEqual(report["checks"][0]["command"][0], str(validation_python))
        self.assertNotEqual(report["checks"][0]["command"][0], context.python_exe)
        self.assertIn(str(context.ext_dir), report["checks"][1]["command"][2])

    def test_runtime_validation_does_not_fallback_to_host_python_when_venv_python_is_missing(self):
        context = setup_module._build_setup_context(
            python_exe="/host/python",
            ext_dir=Path("/tmp/modly-trellis2-tests-missing-venv-python"),
            gpu_sm=121,
            cuda_version=128,
            system="Linux",
            machine="aarch64",
        )
        policy = {
            "native_wheel_query_supported": False,
            "policy_notes": [],
            "phase0_policy_notes": [],
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
            "venv": str(context.venv_dir),
            "torch_packages": ["torch==2.7.0", "torchvision==0.22.0"],
            "torch_index": "https://download.pytorch.org/whl/cu128",
            "torch_version": "2.7.0",
            "core_packages": [],
            "optional_packages": [],
            "rembg": "rembg + onnxruntime",
            "rembg_plan": {"mode": "cpu", "packages": ["rembg", "onnxruntime"], "summary": "rembg + onnxruntime", "detail": "cpu"},
            "runtime_formalization": {
                **build_report_fragments(HostPlatform(system="Linux", machine="aarch64", gpu_sm=121), {}, torch_version="2.7.0"),
            },
        }

        refreshed = setup_module._refresh_runtime_formalization_after_native_deps(
            context,
            base_phase,
            dependency_results,
            "2.7.0",
            env={"TRELLIS2_RUN_RUNTIME_VALIDATION": "1"},
            runner=lambda *args, **kwargs: self.fail("runner should not execute without a venv python"),
        )
        report = setup_module._build_setup_report(context, policy, {**base_phase, "runtime_formalization": refreshed}, dependency_results, "2.7.0")

        self.assertEqual(refreshed["runtime_validation"]["status"], "failed")
        self.assertEqual(refreshed["runtime_validation"]["checks"][0]["command"][0], str(context.venv_dir / "bin" / "python"))
        self.assertNotEqual(refreshed["runtime_validation"]["checks"][0]["command"][0], context.python_exe)
        self.assertIn(str(context.ext_dir), refreshed["runtime_validation"]["checks"][1]["command"][2])
        self.assertIn("venv_python_missing", "\n".join(refreshed["runtime_validation"]["blocking_reasons"]))
        self.assertFalse(report["capabilities"]["generate"]["allowed"])

    def test_refresh_runtime_formalization_runs_validation_only_when_gate_is_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = setup_module._build_setup_context(
                python_exe=sys.executable,
                ext_dir=root / "ext",
                gpu_sm=121,
                cuda_version=128,
                system="Linux",
                machine="aarch64",
            )
            validation_python = context.venv_dir / "bin" / "python"
            validation_python.parent.mkdir(parents=True, exist_ok=True)
            validation_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            base_phase = {
                "runtime_formalization": {
                    "runtime_overrides": {
                        "flex_gemm": {"selected": "explicit_gemm"},
                    },
                    "trellis_patches": {"status": "planned", "items": [], "blocking_reasons": []},
                    "runtime_validation": {"status": "not_run", "checks": [], "blocking_reasons": [], "warnings": []},
                    "wheel_provenance": {"status": "planned", "entries": [], "blocking_reasons": []},
                }
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

            def fake_runner(command, **kwargs):
                return subprocess.CompletedProcess(command, 0, stdout='{"ok":true}', stderr="")

            skipped = setup_module._refresh_runtime_formalization_after_native_deps(
                context,
                base_phase,
                dependency_results,
                "2.7.0",
                env={},
                runner=fake_runner,
            )
            executed = setup_module._refresh_runtime_formalization_after_native_deps(
                context,
                base_phase,
                dependency_results,
                "2.7.0",
                env={"TRELLIS2_RUN_RUNTIME_VALIDATION": "1"},
                runner=fake_runner,
            )

        self.assertEqual(skipped["runtime_validation"]["status"], "not_run")
        self.assertEqual(executed["runtime_validation"]["status"], "passed")
        self.assertEqual([check["status"] for check in executed["runtime_validation"]["checks"]], ["passed", "passed", "passed"])

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


class FutureSetupReportSchemaTests(unittest.TestCase):
    def test_runtime_patch_report_fragments_expose_future_setup_report_keys(self):
        fragments = build_report_fragments(
            HostPlatform(system="Linux", machine="aarch64", gpu_sm=121, python_tag="cp312"),
            {},
            torch_version="2.7.0",
        )

        self.assertEqual(
            set(fragments),
            {"runtime_overrides", "trellis_patches", "runtime_validation", "wheel_provenance"},
        )
        self.assertEqual(fragments["runtime_overrides"]["flex_gemm"]["status"], "selected")
        self.assertEqual(fragments["runtime_overrides"]["triton"]["status"], "conflict_reported")
        self.assertEqual(
            fragments["runtime_overrides"]["triton"]["command"],
            ["python3", "-m", "pip", "install", "triton==3.6.0"],
        )
        self.assertEqual(fragments["trellis_patches"]["status"], "not_run")
        self.assertEqual(fragments["runtime_validation"]["status"], "not_run")


if __name__ == "__main__":
    unittest.main()
