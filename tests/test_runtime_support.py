import tempfile
import hashlib
import unittest
from pathlib import Path, PureWindowsPath

from runtime_support import (
    COMFYUI_GGUF_REQUIRED_FILES,
    DependencyPreflight,
    RuntimeEnvironment,
    WheelSource,
    build_wheel_candidates,
    collect_wheel_provenance,
    comfyui_gguf_support_file_report,
    comfyui_gguf_target_dir,
    evaluate_generate_runtime_readiness,
    resolve_wheel_candidate,
    requires_arm64_generate_runtime_validation,
    wheel_candidate_matches_requirement,
)


class ComfyUIGGUFTests(unittest.TestCase):
    def test_windows_like_site_packages_maps_to_lib_comfyui_gguf(self):
        target = comfyui_gguf_target_dir(r"C:\venv\Lib\site-packages")
        self.assertEqual(target, PureWindowsPath(r"C:\venv\Lib\ComfyUI-GGUF"))

    def test_linux_like_site_packages_maps_to_python_lib_comfyui_gguf(self):
        target = comfyui_gguf_target_dir("/tmp/venv/lib/python3.12/site-packages")
        self.assertEqual(target, Path("/tmp/venv/lib/python3.12/ComfyUI-GGUF"))

    def test_path_input_returns_operable_path(self):
        target = comfyui_gguf_target_dir(Path("/tmp/venv/lib/python3.12/site-packages"))

        self.assertIsInstance(target, Path)
        self.assertTrue(hasattr(target, "mkdir"))
        self.assertEqual(target, Path("/tmp/venv/lib/python3.12/ComfyUI-GGUF"))

    def test_support_file_report_is_installed_only_when_all_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            for name in COMFYUI_GGUF_REQUIRED_FILES:
                (target / name).write_text("ok", encoding="utf-8")

            report = comfyui_gguf_support_file_report(target)

            self.assertEqual(report["status"], "installed")
            self.assertEqual(report["missing"], [])

    def test_support_file_report_lists_missing_filenames_for_partial_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / "ops.py").write_text("ok", encoding="utf-8")

            partial = comfyui_gguf_support_file_report(target)
            self.assertEqual(partial["status"], "partial")
            self.assertEqual(partial["missing"], ["dequant.py", "loader.py"])
            self.assertIn("dequant.py", partial["detail"])
            self.assertIn("loader.py", partial["detail"])

        with tempfile.TemporaryDirectory() as tmp:
            missing = comfyui_gguf_support_file_report(tmp)
            self.assertEqual(missing["status"], "missing")
            self.assertEqual(missing["missing"], list(COMFYUI_GGUF_REQUIRED_FILES))


class WheelResolutionTests(unittest.TestCase):
    def setUp(self):
        self.env_arm64 = RuntimeEnvironment(
            system="Linux",
            machine="aarch64",
            python_tag="cp312",
            python_version="3.12.3",
            torch_version="2.7.0",
            cuda_version="128",
        )
        self.local_source = WheelSource("wheelhouse", "local-wheelhouse", "/wheels", 0)
        self.public_source = WheelSource("public-index", "public-pozzettiandrea", "https://example.invalid", 99)

    def test_rejects_linux_x86_64_on_linux_aarch64(self):
        candidate = build_wheel_candidates(
            self.local_source,
            ["cumesh-1.0.0-cp312-none-manylinux_2_17_x86_64.whl"],
        )[0]

        match, reason = wheel_candidate_matches_requirement(candidate, "cumesh", self.env_arm64)

        self.assertFalse(match)
        self.assertIn("incompatible", reason)

    def test_accepts_matching_aarch64_candidate(self):
        candidate = build_wheel_candidates(
            self.local_source,
            ["cumesh-1.0.0-cp312-none-manylinux_2_17_aarch64.whl"],
        )[0]

        match, reason = wheel_candidate_matches_requirement(candidate, "cumesh", self.env_arm64)

        self.assertTrue(match)
        self.assertEqual(reason, "")

    def test_alias_matching_covers_expected_package_name_variants(self):
        cases = [
            ("flex_gemm", "flex-gemm-1.0.0-cp312-none-manylinux_2_17_aarch64.whl"),
            ("o_voxel", "o-voxel-1.0.0-cp312-none-manylinux_2_17_aarch64.whl"),
            ("spconv", "spconv-cu128-1.0.0-cp312-none-manylinux_2_17_aarch64.whl"),
            ("cumm", "cumm-cu128-1.0.0-cp312-none-manylinux_2_17_aarch64.whl"),
        ]

        for requirement, filename in cases:
            with self.subTest(requirement=requirement, filename=filename):
                candidate = build_wheel_candidates(self.local_source, [filename])[0]
                match, reason = wheel_candidate_matches_requirement(candidate, requirement, self.env_arm64)
                self.assertTrue(match, reason)

    def test_ranking_prefers_newer_candidate_within_same_source(self):
        candidates = build_wheel_candidates(
            self.local_source,
            [
                "cumesh-1.10.0-cp312-none-manylinux_2_17_aarch64.whl",
                "cumesh-1.2.0-cp312-none-manylinux_2_17_aarch64.whl",
            ],
        )

        result = resolve_wheel_candidate("cumesh", self.env_arm64, (self.local_source,), candidates)

        self.assertEqual(result.state, "resolved")
        self.assertEqual(result.selected_candidate.filename, "cumesh-1.10.0-cp312-none-manylinux_2_17_aarch64.whl")

    def test_ranking_prefers_higher_build_for_same_version(self):
        candidates = build_wheel_candidates(
            self.local_source,
            [
                "cumesh-1.0.0-1-cp312-none-manylinux_2_17_aarch64.whl",
                "cumesh-1.0.0-2-cp312-none-manylinux_2_17_aarch64.whl",
            ],
        )

        result = resolve_wheel_candidate("cumesh", self.env_arm64, (self.local_source,), candidates)

        self.assertEqual(result.state, "resolved")
        self.assertEqual(result.selected_candidate.filename, "cumesh-1.0.0-2-cp312-none-manylinux_2_17_aarch64.whl")

    def test_ranking_prefers_final_release_over_prerelease_for_same_release(self):
        candidates = build_wheel_candidates(
            self.local_source,
            [
                "cumesh-1.0.0rc1-cp312-none-manylinux_2_17_aarch64.whl",
                "cumesh-1.0.0-cp312-none-manylinux_2_17_aarch64.whl",
            ],
        )

        result = resolve_wheel_candidate("cumesh", self.env_arm64, (self.local_source,), candidates)

        self.assertEqual(result.state, "resolved")
        self.assertEqual(result.selected_candidate.filename, "cumesh-1.0.0-cp312-none-manylinux_2_17_aarch64.whl")

    def test_ranking_prefers_post_release_over_final_for_same_release(self):
        candidates = build_wheel_candidates(
            self.local_source,
            [
                "cumesh-1.0.0-cp312-none-manylinux_2_17_aarch64.whl",
                "cumesh-1.0.0.post1-cp312-none-manylinux_2_17_aarch64.whl",
            ],
        )

        result = resolve_wheel_candidate("cumesh", self.env_arm64, (self.local_source,), candidates)

        self.assertEqual(result.state, "resolved")
        self.assertEqual(result.selected_candidate.filename, "cumesh-1.0.0.post1-cp312-none-manylinux_2_17_aarch64.whl")

    def test_ranking_prefers_exact_python_and_explicit_cuda_torch_match(self):
        candidates = build_wheel_candidates(
            self.local_source,
            [
                "cumesh-9.9.9-py3-none-manylinux_2_17_aarch64.whl",
                "cumesh-1.0.0+torch2.7.cu128-cp312-none-manylinux_2_17_aarch64.whl",
            ],
        )

        result = resolve_wheel_candidate("cumesh", self.env_arm64, (self.local_source,), candidates)

        self.assertEqual(result.state, "resolved")
        self.assertEqual(
            result.selected_candidate.filename,
            "cumesh-1.0.0+torch2.7.cu128-cp312-none-manylinux_2_17_aarch64.whl",
        )

    def test_source_priority_remains_authoritative(self):
        local_candidates = build_wheel_candidates(
            self.local_source,
            ["cumesh-1.0.0-cp312-none-manylinux_2_17_aarch64.whl"],
        )
        public_candidates = build_wheel_candidates(
            self.public_source,
            ["cumesh-9.9.9-cp312-none-manylinux_2_17_aarch64.whl"],
        )

        result = resolve_wheel_candidate(
            "cumesh",
            self.env_arm64,
            (self.local_source, self.public_source),
            local_candidates + public_candidates,
        )

        self.assertEqual(result.state, "resolved")
        self.assertEqual(result.selected_source, self.local_source)
        self.assertEqual(result.selected_candidate.filename, "cumesh-1.0.0-cp312-none-manylinux_2_17_aarch64.whl")

    def test_local_wheelhouse_install_target_resolves_to_absolute_path(self):
        candidates = build_wheel_candidates(
            self.local_source,
            ["cumesh-1.0.0-cp312-none-manylinux_2_17_aarch64.whl"],
        )

        result = resolve_wheel_candidate("cumesh", self.env_arm64, (self.local_source,), candidates)

        self.assertEqual(result.state, "resolved")
        self.assertEqual(
            result.selected_candidate.install_target,
            "/wheels/cumesh-1.0.0-cp312-none-manylinux_2_17_aarch64.whl",
        )

    def test_remote_candidate_install_target_preserves_url(self):
        candidates = build_wheel_candidates(
            self.public_source,
            ["cumesh-1.0.0-cp312-none-manylinux_2_17_aarch64.whl"],
            url_map={"cumesh-1.0.0-cp312-none-manylinux_2_17_aarch64.whl": "https://example.invalid/cumesh.whl"},
        )

        result = resolve_wheel_candidate("cumesh", self.env_arm64, (self.public_source,), candidates)

        self.assertEqual(result.state, "resolved")
        self.assertEqual(result.selected_candidate.install_target, "https://example.invalid/cumesh.whl")

    def test_ranking_uses_filename_as_deterministic_tie_break(self):
        candidates = build_wheel_candidates(
            self.local_source,
            [
                "cumesh-1.0.0-cp312-none-manylinux_2_17_aarch64.whl",
                "Cumesh-1.0.0-cp312-none-manylinux_2_17_aarch64.whl",
            ],
        )

        result = resolve_wheel_candidate("cumesh", self.env_arm64, (self.local_source,), candidates)

        self.assertEqual(result.state, "resolved")
        self.assertEqual(result.selected_candidate.filename, "Cumesh-1.0.0-cp312-none-manylinux_2_17_aarch64.whl")


class WheelProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.env_arm64 = RuntimeEnvironment(
            system="Linux",
            machine="aarch64",
            python_tag="cp312",
            python_version="3.12.3",
            torch_version="2.7.0",
            cuda_version="128",
            gpu_sm="121",
        )

    def test_collect_wheel_provenance_observes_local_wheelhouse_without_leaking_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheelhouse = Path(tmp) / "private-wheelhouse"
            wheelhouse.mkdir()
            payloads = {
                "cumesh-1.0.0+torch2.7.cu128-cp312-none-manylinux_2_17_aarch64.whl": b"cumesh",
                "flex_gemm-1.0.0+torch2.7.cu128-cp312-none-manylinux_2_17_aarch64.whl": b"flex",
                "o_voxel-1.0.0+torch2.7.cu128-cp312-none-manylinux_2_17_aarch64.whl": b"ovoxel",
                "nvdiffrast-1.0.0+torch2.7.cu128-cp312-none-manylinux_2_17_aarch64.whl": b"nvdiff",
            }
            for filename, payload in payloads.items():
                (wheelhouse / filename).write_bytes(payload)

            report = collect_wheel_provenance(
                ("cumesh", "flex_gemm", "o_voxel", "nvdiffrast"),
                self.env_arm64,
                (WheelSource("wheelhouse", "local-wheelhouse", str(wheelhouse), 0),),
            )

            self.assertEqual(report["status"], "available")
            self.assertEqual(report["coverage"]["available"], 4)
            self.assertEqual(report["sources"][0]["location_hint"], "<wheelhouse:private-wheelhouse>")
            self.assertNotIn(str(wheelhouse), str(report))
            cumesh = next(entry for entry in report["entries"] if entry["package"] == "cumesh")
            self.assertEqual(cumesh["platform_tag"], "manylinux_2_17_aarch64")
            self.assertEqual(cumesh["python_tag"], "cp312")
            self.assertEqual(
                cumesh["sha256"],
                hashlib.sha256(payloads["cumesh-1.0.0+torch2.7.cu128-cp312-none-manylinux_2_17_aarch64.whl"]).hexdigest(),
            )

    def test_collect_wheel_provenance_marks_missing_packages_when_wheelhouse_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheelhouse = Path(tmp) / "wheelhouse"
            wheelhouse.mkdir()
            (wheelhouse / "cumesh-1.0.0-cp312-none-manylinux_2_17_aarch64.whl").write_text("stub", encoding="utf-8")

            report = collect_wheel_provenance(
                ("cumesh", "flex_gemm"),
                self.env_arm64,
                (WheelSource("wheelhouse", "local-wheelhouse", str(wheelhouse), 0),),
            )

            self.assertEqual(report["status"], "observed")
            self.assertIn("wheel_provenance: missing public wheel evidence for flex_gemm", report["blocking_reasons"])
            missing = next(entry for entry in report["entries"] if entry["package"] == "flex_gemm")
            self.assertEqual(missing["status"], "missing")

    def test_collect_wheel_provenance_redacts_remote_source_credentials(self):
        report = collect_wheel_provenance(
            ("cumesh",),
            self.env_arm64,
            (
                WheelSource("url", "private-url", "https://user:secret@example.com/wheels/cumesh.whl?token=abc", 1),
            ),
        )

        self.assertEqual(report["status"], "planned")
        self.assertEqual(report["sources"][0]["location"], "https://***:***@example.com/wheels/cumesh.whl")
        self.assertNotIn("secret", str(report))


class DependencyClassificationTests(unittest.TestCase):
    def test_generate_defaults_block_on_import_time_native_deps(self):
        env = RuntimeEnvironment(
            system="Linux",
            machine="x86_64",
            python_tag="cp312",
            known_dependencies={
                "cumesh": False,
                "o_voxel": False,
                "flex_gemm": False,
                "nvdiffrast": False,
                "spconv": False,
                "cumm": False,
                "comfyui_gguf_support_files": True,
            },
            asset_groups={"generate": True, "refine": False},
        )

        report = DependencyPreflight.evaluate("generate", env)

        blockers = "\n".join(report.blockers)
        for name in ("cumesh", "o_voxel", "flex_gemm", "nvdiffrast"):
            with self.subTest(name=name):
                self.assertIn(name, blockers)
        self.assertNotIn("spconv", blockers)
        self.assertNotIn("cumm", blockers)

    def test_generate_definitions_do_not_require_spconv_or_cumm_by_default(self):
        generate_names = {definition.name for definition in DependencyPreflight.definitions_for("generate")}

        self.assertIn("cumesh", generate_names)
        self.assertIn("o_voxel", generate_names)
        self.assertIn("flex_gemm", generate_names)
        self.assertIn("nvdiffrast", generate_names)
        self.assertNotIn("spconv", generate_names)
        self.assertNotIn("cumm", generate_names)


class RuntimeReadinessTests(unittest.TestCase):
    def test_target_host_requires_runtime_validation_before_generate_is_ready(self):
        env = RuntimeEnvironment(
            system="Linux",
            machine="aarch64",
            python_tag="cp312",
            gpu_sm="121",
            known_dependencies={
                "cumesh": True,
                "o_voxel": True,
                "flex_gemm": True,
                "nvdiffrast": True,
                "spconv": False,
                "cumm": False,
                "comfyui_gguf_support_files": True,
            },
            asset_groups={"generate": True, "refine": False},
        )

        report = DependencyPreflight.evaluate("generate", env)
        readiness = evaluate_generate_runtime_readiness(
            report,
            runtime_validation={"status": "not_run", "blocking_reasons": []},
            trellis_patches={"status": "planned", "blocking_reasons": []},
        )

        self.assertTrue(requires_arm64_generate_runtime_validation(env))
        self.assertEqual(report.state, "available")
        self.assertFalse(readiness["allowed"])
        self.assertEqual(readiness["state"], "unknown")
        self.assertIn("runtime_validation", "\n".join(readiness["blockers"]))

    def test_target_host_ready_only_when_validation_passes_and_no_patch_blockers_exist(self):
        env = RuntimeEnvironment(
            system="Linux",
            machine="aarch64",
            python_tag="cp312",
            gpu_sm="121",
            known_dependencies={
                "cumesh": True,
                "o_voxel": True,
                "flex_gemm": True,
                "nvdiffrast": True,
                "spconv": False,
                "cumm": False,
                "comfyui_gguf_support_files": True,
            },
            asset_groups={"generate": True, "refine": False},
        )

        report = DependencyPreflight.evaluate("generate", env)
        readiness = evaluate_generate_runtime_readiness(
            report,
            runtime_validation={"status": "passed", "blocking_reasons": []},
            trellis_patches={"status": "already_applied", "blocking_reasons": []},
        )

        self.assertTrue(readiness["allowed"])
        self.assertEqual(readiness["state"], "available")

    def test_target_host_failed_or_skipped_validation_blocks_generate_readiness(self):
        env = RuntimeEnvironment(
            system="Linux",
            machine="aarch64",
            python_tag="cp312",
            gpu_sm="121",
            known_dependencies={
                "cumesh": True,
                "o_voxel": True,
                "flex_gemm": True,
                "nvdiffrast": True,
                "spconv": False,
                "cumm": False,
                "comfyui_gguf_support_files": True,
            },
            asset_groups={"generate": True, "refine": False},
        )

        report = DependencyPreflight.evaluate("generate", env)
        failed = evaluate_generate_runtime_readiness(
            report,
            runtime_validation={"status": "failed", "blocking_reasons": ["runtime_validation:flex_gemm_micro_smoke: failed"]},
            trellis_patches={"status": "already_applied", "blocking_reasons": []},
        )
        skipped = evaluate_generate_runtime_readiness(
            report,
            runtime_validation={"status": "skipped", "blocking_reasons": []},
            trellis_patches={"status": "already_applied", "blocking_reasons": []},
        )

        self.assertFalse(failed["allowed"])
        self.assertFalse(skipped["allowed"])
        self.assertIn("runtime_validation:flex_gemm_micro_smoke: failed", failed["blockers"])
        self.assertEqual(skipped["state"], "unknown")

    def test_non_target_host_keeps_existing_generate_readiness_behavior(self):
        env = RuntimeEnvironment(
            system="Linux",
            machine="x86_64",
            python_tag="cp312",
            gpu_sm="121",
            known_dependencies={
                "cumesh": True,
                "o_voxel": True,
                "flex_gemm": True,
                "nvdiffrast": True,
                "spconv": False,
                "cumm": False,
                "comfyui_gguf_support_files": True,
            },
            asset_groups={"generate": True, "refine": False},
        )

        report = DependencyPreflight.evaluate("generate", env)
        readiness = evaluate_generate_runtime_readiness(
            report,
            runtime_validation={"status": "not_run", "blocking_reasons": []},
            trellis_patches={"status": "not_run", "blocking_reasons": []},
        )

        self.assertFalse(requires_arm64_generate_runtime_validation(env))
        self.assertTrue(readiness["allowed"])
        self.assertEqual(readiness["state"], "available")


if __name__ == "__main__":
    unittest.main()
