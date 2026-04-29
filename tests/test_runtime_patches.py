import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from runtime_patches import (
    FDG_VAE_IMPORT_MARKER,
    FDG_VAE_PATCHED_CALL_SNIPPET,
    FDG_VAE_PATCHED_IMPORT_SNIPPET,
    FDG_VAE_UNPATCHED_CALL_SNIPPET,
    FDG_VAE_UNPATCHED_IMPORT_SNIPPET,
    FLEX_GEMM_ALGO_ENV,
    FLEX_GEMM_DEBUG_ENV,
    FLEX_GEMM_DEBUG_LOG_PATH_ENV,
    HostPlatform,
    RUN_RUNTIME_VALIDATION_ENV,
    TRELLIS_CONFIG_ALGO_MARKER,
    TRELLIS_CONFIG_EXPLICIT_SNIPPET,
    TRELLIS_CONFIG_RELATIVE_PATH,
    TRELLIS_DEBUG_MARKER,
    TRELLIS_DEBUG_RELATIVE_PATH,
    TRELLIS_FDG_VAE_RELATIVE_PATH,
    TRITON_OVERRIDE_VERSION,
    apply_fdg_vae_patch_file,
    build_flex_gemm_config_patch_plan,
    build_flex_gemm_debug_patch_plan,
    build_flex_gemm_micro_smoke_plan,
    build_generator_bootstrap_validation_plan,
    build_import_api_validation_plan,
    build_no_generate_validation_plans,
    build_report_fragments,
    detect_linux_arm64_blackwell,
    plan_fdg_vae_patch,
    plan_triton_override,
    resolve_flex_gemm_algo,
    resolve_flex_gemm_debug_policy,
    run_no_generate_validation,
    to_jsonable,
)


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

FDG_VAE_PATCHED_TEXT = FDG_VAE_UNPATCHED_TEXT.replace(
    FDG_VAE_UNPATCHED_IMPORT_SNIPPET,
    FDG_VAE_PATCHED_IMPORT_SNIPPET,
    1,
).replace(
    FDG_VAE_UNPATCHED_CALL_SNIPPET,
    FDG_VAE_PATCHED_CALL_SNIPPET,
    1,
)


class HostDetectionTests(unittest.TestCase):
    def test_linux_arm64_blackwell_detection_requires_sm121(self):
        self.assertTrue(detect_linux_arm64_blackwell(HostPlatform(system="Linux", machine="aarch64", gpu_sm=121)))
        self.assertFalse(detect_linux_arm64_blackwell(HostPlatform(system="Linux", machine="aarch64", gpu_sm=120)))
        self.assertFalse(detect_linux_arm64_blackwell(HostPlatform(system="Linux", machine="x86_64", gpu_sm=121)))


class FlexGemmPolicyTests(unittest.TestCase):
    def test_env_override_takes_precedence_over_targeted_host_policy(self):
        host = HostPlatform(system="Linux", machine="aarch64", gpu_sm=121, python_tag="cp312")

        decision = resolve_flex_gemm_algo({FLEX_GEMM_ALGO_ENV: "implicit_gemm"}, host)

        self.assertEqual(decision.selected, "implicit_gemm")
        self.assertEqual(decision.source, "env_override")
        self.assertEqual(decision.status, "selected")
        self.assertTrue(decision.targeted_host)

    def test_invalid_env_override_fails_closed_without_fallback(self):
        host = HostPlatform(system="Linux", machine="aarch64", gpu_sm=121)

        decision = resolve_flex_gemm_algo({FLEX_GEMM_ALGO_ENV: "manual_override"}, host)

        self.assertEqual(decision.status, "invalid_override")
        self.assertEqual(decision.selected, "")
        self.assertIn("Supported values", decision.blocking_reason)
        self.assertIn("manual_override", decision.blocking_reason)

    def test_arm64_blackwell_auto_selects_explicit_gemm(self):
        host = HostPlatform(system="Linux", machine="arm64", gpu_sm="121")

        decision = resolve_flex_gemm_algo({}, host)

        self.assertEqual(decision.selected, "explicit_gemm")
        self.assertEqual(decision.source, "linux_arm64_blackwell_policy")

    def test_non_target_hosts_preserve_implicit_splitk_default(self):
        host = HostPlatform(system="Linux", machine="x86_64", gpu_sm=121)

        decision = resolve_flex_gemm_algo({}, host)

        self.assertEqual(decision.selected, "implicit_gemm_splitk")
        self.assertEqual(decision.source, "default")
        self.assertFalse(decision.targeted_host)


class FlexGemmConfigPatchPlanTests(unittest.TestCase):
    def test_target_host_config_patch_is_planned(self):
        host = HostPlatform(system="Linux", machine="aarch64", gpu_sm=121)

        plan = build_flex_gemm_config_patch_plan(host, {}, install_root="/venv/site-packages")

        self.assertEqual(plan.status, "planned")
        self.assertEqual(plan.target_file, "/venv/site-packages/" + TRELLIS_CONFIG_RELATIVE_PATH)
        self.assertEqual(plan.planned_value, "explicit_gemm")
        self.assertEqual(plan.expected_marker, TRELLIS_CONFIG_ALGO_MARKER)
        self.assertEqual(plan.replacement_snippet, TRELLIS_CONFIG_EXPLICIT_SNIPPET)
        self.assertTrue(plan.backup_path.endswith(".arm64-runtime.bak"))

    def test_non_target_host_config_patch_is_not_needed(self):
        host = HostPlatform(system="Linux", machine="x86_64", gpu_sm=121)

        plan = build_flex_gemm_config_patch_plan(host, {})

        self.assertEqual(plan.status, "not_needed")
        self.assertEqual(plan.planned_value, "implicit_gemm_splitk")

    def test_config_patch_reports_already_applied_when_explicit_gemm_present(self):
        host = HostPlatform(system="Linux", machine="aarch64", gpu_sm=121)

        plan = build_flex_gemm_config_patch_plan(host, {}, installed_config_text=TRELLIS_CONFIG_EXPLICIT_SNIPPET)

        self.assertEqual(plan.status, "already_applied")
        self.assertEqual(plan.current_value, "explicit_gemm")

    def test_config_patch_reports_drift_when_expected_marker_is_missing(self):
        host = HostPlatform(system="Linux", machine="aarch64", gpu_sm=121)

        plan = build_flex_gemm_config_patch_plan(host, {}, installed_config_text='ALGORITHM = "unexpected"')

        self.assertEqual(plan.status, "drift")
        self.assertIn("refusing loose replacement", plan.blocking_reason)

    def test_config_patch_reports_invalid_override(self):
        host = HostPlatform(system="Linux", machine="aarch64", gpu_sm=121)

        plan = build_flex_gemm_config_patch_plan(host, {FLEX_GEMM_ALGO_ENV: "manual_override"})

        self.assertEqual(plan.status, "invalid_override")
        self.assertIn("manual_override", plan.blocking_reason)


class FlexGemmDebugPolicyTests(unittest.TestCase):
    def test_debug_logging_is_disabled_by_default(self):
        policy = resolve_flex_gemm_debug_policy({})

        self.assertFalse(policy.enabled)
        self.assertIsNone(policy.log_path)
        self.assertFalse(policy.tensor_dumps)
        self.assertFalse(policy.value_dumps)

    def test_debug_logging_requires_env_gate(self):
        policy = resolve_flex_gemm_debug_policy(
            {
                FLEX_GEMM_DEBUG_ENV: "true",
                FLEX_GEMM_DEBUG_LOG_PATH_ENV: "/tmp/flex-gemm-debug.jsonl",
            }
        )

        self.assertTrue(policy.enabled)
        self.assertEqual(policy.log_path, "/tmp/flex-gemm-debug.jsonl")
        self.assertFalse(policy.tensor_dumps)
        self.assertFalse(policy.value_dumps)

    def test_debug_patch_plan_is_off_by_default(self):
        plan = build_flex_gemm_debug_patch_plan({})

        self.assertEqual(plan.status, "not_needed")
        self.assertEqual(plan.target_file, TRELLIS_DEBUG_RELATIVE_PATH)

    def test_debug_patch_plan_is_only_planned_when_requested(self):
        plan = build_flex_gemm_debug_patch_plan(
            {
                FLEX_GEMM_DEBUG_ENV: "1",
                FLEX_GEMM_DEBUG_LOG_PATH_ENV: "/tmp/flex.jsonl",
            }
        )

        self.assertEqual(plan.status, "planned")
        self.assertEqual(plan.planned_value, "/tmp/flex.jsonl")
        self.assertEqual(plan.replacement_snippet, TRELLIS_DEBUG_MARKER)


class TritonOverridePlanTests(unittest.TestCase):
    def test_triton_plan_only_applies_to_target_host(self):
        targeted = plan_triton_override(HostPlatform(system="Linux", machine="aarch64", gpu_sm=121))
        non_targeted = plan_triton_override(HostPlatform(system="Linux", machine="x86_64", gpu_sm=121))

        self.assertEqual(targeted.status, "conflict_reported")
        self.assertTrue(targeted.applies)
        self.assertEqual(targeted.target_version, TRITON_OVERRIDE_VERSION)
        self.assertFalse(targeted.install_allowed)
        self.assertEqual(targeted.command, ("python3", "-m", "pip", "install", "triton==3.6.0"))
        self.assertEqual(non_targeted.status, "not_applicable")
        self.assertFalse(non_targeted.applies)
        self.assertIsNone(non_targeted.target_version)

    def test_triton_plan_reports_already_satisfied_target(self):
        plan = plan_triton_override(
            HostPlatform(system="Linux", machine="aarch64", gpu_sm=121),
            current_triton_version="3.6.0",
        )

        self.assertEqual(plan.status, "already_satisfied")
        self.assertEqual(plan.current_version, "3.6.0")
        self.assertEqual(plan.command, ())

    def test_triton_plan_records_torch_pin_conflict_note(self):
        plan = plan_triton_override(
            HostPlatform(system="Linux", machine="aarch64", gpu_sm=121),
            torch_version="2.7.0",
            torch_triton_pin="3.3.0",
        )

        self.assertEqual(plan.status, "conflict_reported")
        self.assertIn("triton==3.3.0", plan.conflict_note)
        self.assertIn("triton==3.6.0", plan.conflict_note)
        self.assertIn("without reinstalling torch", plan.conflict_note)
        self.assertFalse(plan.reinstall_torch)


class FdgVaePatchPlanTests(unittest.TestCase):
    def test_fdg_vae_patch_is_planned_from_unpatched_text(self):
        plan = plan_fdg_vae_patch(FDG_VAE_UNPATCHED_TEXT, install_root="/venv/site-packages")

        self.assertEqual(plan.status, "planned")
        self.assertEqual(plan.relative_path, TRELLIS_FDG_VAE_RELATIVE_PATH)
        self.assertEqual(plan.target_file, "/venv/site-packages/" + TRELLIS_FDG_VAE_RELATIVE_PATH)
        self.assertTrue(plan.backup_path.endswith(".arm64-runtime.bak"))
        self.assertNotEqual(plan.before_hash, plan.after_hash)
        self.assertIn(FDG_VAE_PATCHED_IMPORT_SNIPPET, plan.patched_text)
        self.assertIn(FDG_VAE_PATCHED_CALL_SNIPPET, plan.patched_text)
        self.assertNotIn(FDG_VAE_IMPORT_MARKER, plan.patched_text)

    def test_fdg_vae_patch_is_idempotent_when_already_applied(self):
        plan = plan_fdg_vae_patch(FDG_VAE_PATCHED_TEXT)

        self.assertEqual(plan.status, "already_applied")
        self.assertEqual(plan.before_hash, plan.after_hash)
        self.assertEqual(plan.patched_text, "")

    def test_fdg_vae_patch_fails_closed_on_drift(self):
        drifted = FDG_VAE_UNPATCHED_TEXT.replace("resolution,", "resolution,\n            iso_value,", 1)

        plan = plan_fdg_vae_patch(drifted)

        self.assertEqual(plan.status, "drift")
        self.assertIn("approved patch window", plan.drift_reason)
        self.assertEqual(plan.after_hash, "")

    def test_fdg_vae_patch_file_writes_backup_and_patched_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_file = Path(tmpdir) / "fdg_vae.py"
            target_file.write_text(FDG_VAE_UNPATCHED_TEXT, encoding="utf-8")

            result = apply_fdg_vae_patch_file(target_file)

            self.assertEqual(result.status, "applied")
            self.assertTrue(Path(result.backup_path).exists())
            self.assertEqual(Path(result.backup_path).read_text(encoding="utf-8"), FDG_VAE_UNPATCHED_TEXT)
            self.assertEqual(target_file.read_text(encoding="utf-8"), FDG_VAE_PATCHED_TEXT)


class NoGenerateValidationPlanTests(unittest.TestCase):
    def test_validation_scripts_import_expected_modules_and_avoid_runtime_entrypoints(self):
        repo_root = Path(__file__).resolve().parents[1]
        import_plan = build_import_api_validation_plan()
        generator_plan = build_generator_bootstrap_validation_plan(extension_root=str(repo_root))

        self.assertIn('importlib.import_module("trellis2_gguf")', import_plan.script_content)
        self.assertIn('importlib.import_module("generator")', generator_plan.script_content)
        self.assertIn("sys.path.insert(0, extension_root)", generator_plan.script_content)
        self.assertIn(f"extension_root = {str(repo_root)!r}", generator_plan.script_content)
        self.assertIn('services.generators.base', generator_plan.script_content)
        self.assertIn('Trellis2GGUFGenerator', generator_plan.script_content)
        self.assertIn('instance._ensure_trellis2_gguf()', generator_plan.script_content)
        self.assertNotIn("generate(", import_plan.script_content)
        self.assertNotIn("generate(", generator_plan.script_content)
        self.assertNotIn("load(", import_plan.script_content)
        self.assertNotIn("load(", generator_plan.script_content)
        self.assertNotIn("snapshot_download(", generator_plan.script_content)
        self.assertEqual(import_plan.constraints, ("no Generate", "no model load", "no asset download"))

    def test_generator_bootstrap_smoke_script_executes_in_isolated_python_without_services_package(self):
        repo_root = Path(__file__).resolve().parents[1]
        plan = build_generator_bootstrap_validation_plan(
            python_exe=sys.executable,
            extension_root=str(repo_root),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                list(plan.command),
                cwd=tmpdir,
                capture_output=True,
                text=True,
                check=False,
                env={"PATH": os.environ.get("PATH", "")},
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["module"], "generator")
        self.assertTrue(payload["bootstrap_callable"])
        self.assertTrue(payload["bootstrap_exercised"])
        self.assertTrue(payload["extension_root_seeded"])
        self.assertTrue(payload["services_stubbed"])
        self.assertEqual(payload["policy_source"], "linux_arm64_blackwell_policy")
        self.assertEqual(payload["selected_algo"], "explicit_gemm")
        self.assertTrue(payload["policy_applied"])
        self.assertEqual(payload["ensure_comfyui_gguf_calls"], 1)
        self.assertEqual(payload["resolve_generate_assets_calls"], 1)
        self.assertEqual(payload["folder_models_dir"], "/tmp")
        self.assertTrue(payload["trellis_manager_stubbed"])

    def test_flex_gemm_micro_smoke_includes_selected_algo_and_optional_debug_env(self):
        plan = build_flex_gemm_micro_smoke_plan(
            "explicit_gemm",
            enable_debug=True,
            debug_log_path="/tmp/flex-debug.jsonl",
            timeout_sec=33,
        )

        self.assertEqual(plan.timeout_sec, 33)
        self.assertEqual(plan.env[FLEX_GEMM_ALGO_ENV], "explicit_gemm")
        self.assertEqual(plan.env[FLEX_GEMM_DEBUG_ENV], "1")
        self.assertEqual(plan.env[FLEX_GEMM_DEBUG_LOG_PATH_ENV], "/tmp/flex-debug.jsonl")
        self.assertIn('importlib.import_module("flex_gemm")', plan.script_content)
        self.assertNotIn("generate(", plan.script_content)
        self.assertNotIn("load(", plan.script_content)

    def test_validation_plans_are_json_serializable(self):
        payload = {
            "fdg_patch": to_jsonable(plan_fdg_vae_patch(FDG_VAE_UNPATCHED_TEXT)),
            "checks": to_jsonable(
                build_no_generate_validation_plans(
                    "explicit_gemm",
                    enable_flex_gemm_debug=True,
                    flex_gemm_debug_log_path="/tmp/flex-debug.jsonl",
                )
            ),
        }

        encoded = json.dumps(payload, sort_keys=True)
        decoded = json.loads(encoded)

        self.assertEqual(decoded["fdg_patch"]["status"], "planned")
        self.assertEqual(len(decoded["checks"]), 3)
        self.assertEqual(decoded["checks"][2]["env"]["FLEX_GEMM_ALGO"], "explicit_gemm")

    def test_runtime_validation_defaults_to_not_run_until_gate_is_enabled(self):
        report = run_no_generate_validation("explicit_gemm", env={})

        self.assertEqual(report.status, "not_run")
        self.assertEqual([check["status"] for check in report.to_dict()["checks"]], ["not_run", "not_run", "not_run"])
        self.assertIn(RUN_RUNTIME_VALIDATION_ENV, report.warnings[0])

    def test_runtime_validation_executes_mocked_checks_when_gate_is_enabled(self):
        calls = []

        def fake_runner(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, stdout='{"ok":true}\n', stderr="")

        report = run_no_generate_validation(
            "explicit_gemm",
            env={RUN_RUNTIME_VALIDATION_ENV: "1"},
            runner=fake_runner,
            time_fn=lambda: 10.0,
        )

        self.assertEqual(report.status, "passed")
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(check["status"] == "passed" for check in report.to_dict()["checks"]))
        self.assertEqual(calls[2][1]["env"][FLEX_GEMM_ALGO_ENV], "explicit_gemm")

    def test_runtime_validation_aggregates_failure_and_timeout_states(self):
        calls = {"count": 0}

        def fake_runner(command, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                return subprocess.CompletedProcess(command, 3, stdout="", stderr="boom")
            if calls["count"] == 3:
                raise subprocess.TimeoutExpired(command, timeout=kwargs["timeout"], output="", stderr="late")
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

        report = run_no_generate_validation(
            "explicit_gemm",
            env={RUN_RUNTIME_VALIDATION_ENV: "1"},
            runner=fake_runner,
            time_fn=lambda: float(calls["count"]),
        ).to_dict()

        self.assertEqual(report["status"], "failed")
        self.assertEqual([check["status"] for check in report["checks"]], ["passed", "failed", "timeout"])
        self.assertIn("runtime_validation:generator_bootstrap_smoke", "\n".join(report["blocking_reasons"]))
        self.assertIn("runtime_validation:flex_gemm_micro_smoke", "\n".join(report["blocking_reasons"]))


class ReportFragmentTests(unittest.TestCase):
    def test_report_fragments_are_json_serializable(self):
        host = HostPlatform(system="Linux", machine="aarch64", gpu_sm=121, python_tag="cp312")

        report = build_report_fragments(
            host,
            {FLEX_GEMM_DEBUG_ENV: "1", FLEX_GEMM_DEBUG_LOG_PATH_ENV: "/tmp/flex.jsonl"},
            torch_version="2.7.0",
        )

        encoded = json.dumps(report, sort_keys=True)
        decoded = json.loads(encoded)

        self.assertEqual(
            set(decoded),
            {"runtime_overrides", "trellis_patches", "runtime_validation", "wheel_provenance"},
        )
        self.assertEqual(decoded["runtime_overrides"]["flex_gemm"]["selected"], "explicit_gemm")

    def test_patch_plans_are_json_serializable(self):
        host = HostPlatform(system="Linux", machine="aarch64", gpu_sm=121)
        payload = {
            "config": to_jsonable(build_flex_gemm_config_patch_plan(host, {})),
            "debug": to_jsonable(
                build_flex_gemm_debug_patch_plan(
                    {
                        FLEX_GEMM_DEBUG_ENV: "1",
                        FLEX_GEMM_DEBUG_LOG_PATH_ENV: "/tmp/flex.jsonl",
                    }
                )
            ),
            "triton": to_jsonable(plan_triton_override(host, torch_triton_pin="3.3.0")),
        }

        encoded = json.dumps(payload, sort_keys=True)
        decoded = json.loads(encoded)

        self.assertEqual(decoded["config"]["status"], "planned")
        self.assertEqual(decoded["debug"]["status"], "planned")
        self.assertEqual(decoded["triton"]["target_version"], "3.6.0")


if __name__ == "__main__":
    unittest.main()
