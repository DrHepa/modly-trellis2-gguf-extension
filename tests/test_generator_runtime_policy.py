import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_generator_module():
    repo_root = Path(__file__).resolve().parents[1]

    services = types.ModuleType("services")
    generators = types.ModuleType("services.generators")
    base = types.ModuleType("services.generators.base")

    class _BaseGenerator:
        pass

    base.BaseGenerator = _BaseGenerator
    base.smooth_progress = lambda *args, **kwargs: None

    class _GenerationCancelled(Exception):
        pass

    base.GenerationCancelled = _GenerationCancelled

    sys.modules.setdefault("services", services)
    sys.modules.setdefault("services.generators", generators)
    sys.modules.setdefault("services.generators.base", base)

    spec = importlib.util.spec_from_file_location("extension_generator", repo_root / "generator.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


generator_module = _load_generator_module()


class GeneratorRuntimePolicyTests(unittest.TestCase):
    def test_target_host_auto_sets_explicit_flex_gemm_before_import(self):
        env = {}

        result = generator_module._apply_shared_flex_gemm_runtime_policy(
            env=env,
            host=generator_module._generator_runtime_host(system="Linux", machine="aarch64", gpu_sm=121),
        )

        self.assertEqual(env["FLEX_GEMM_ALGO"], "explicit_gemm")
        self.assertTrue(result["applied"])
        self.assertEqual(result["decision"]["source"], "linux_arm64_blackwell_policy")

    def test_non_target_host_keeps_environment_unchanged(self):
        env = {}

        result = generator_module._apply_shared_flex_gemm_runtime_policy(
            env=env,
            host=generator_module._generator_runtime_host(system="Linux", machine="x86_64", gpu_sm=121),
        )

        self.assertEqual(env, {})
        self.assertFalse(result["applied"])
        self.assertEqual(result["decision"]["selected"], "implicit_gemm_splitk")

    def test_explicit_override_is_preserved_without_extra_mutation(self):
        env = {"FLEX_GEMM_ALGO": "implicit_gemm"}

        result = generator_module._apply_shared_flex_gemm_runtime_policy(
            env=env,
            host=generator_module._generator_runtime_host(system="Linux", machine="aarch64", gpu_sm=121),
        )

        self.assertEqual(env["FLEX_GEMM_ALGO"], "implicit_gemm")
        self.assertFalse(result["applied"])
        self.assertEqual(result["applied_via"], "preserved_env_override")

    def test_invalid_override_fails_closed_before_heavy_imports(self):
        env = {"FLEX_GEMM_ALGO": "manual_override"}

        with self.assertRaises(RuntimeError):
            generator_module._apply_shared_flex_gemm_runtime_policy(
                env=env,
                host=generator_module._generator_runtime_host(system="Linux", machine="aarch64", gpu_sm=121),
            )


if __name__ == "__main__":
    unittest.main()
