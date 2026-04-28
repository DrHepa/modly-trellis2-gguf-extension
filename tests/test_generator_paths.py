import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_generator_module():
    repo_root = Path(__file__).resolve().parents[1]

    services_module = types.ModuleType("services")
    generators_module = types.ModuleType("services.generators")
    base_module = types.ModuleType("services.generators.base")
    base_module.BaseGenerator = object
    base_module.smooth_progress = lambda *args, **kwargs: None
    base_module.GenerationCancelled = RuntimeError

    sys.modules.setdefault("services", services_module)
    sys.modules.setdefault("services.generators", generators_module)
    sys.modules["services.generators.base"] = base_module

    spec = importlib.util.spec_from_file_location("extension_generator", repo_root / "generator.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


generator_module = _load_generator_module()
Trellis2GGUFGenerator = generator_module.Trellis2GGUFGenerator


class WeightsDirTests(unittest.TestCase):
    def test_node_model_dir_is_preserved_for_generate(self):
        generator = Trellis2GGUFGenerator.__new__(Trellis2GGUFGenerator)
        generator.model_dir = Path("/models/trellis2/generate")

        self.assertEqual(generator._weights_dir, Path("/models/trellis2/generate"))

    def test_root_model_dir_remains_unchanged(self):
        generator = Trellis2GGUFGenerator.__new__(Trellis2GGUFGenerator)
        generator.model_dir = Path("/models/trellis2")

        self.assertEqual(generator._weights_dir, Path("/models/trellis2"))


class EnsureTrellisFolderPathsTests(unittest.TestCase):
    def setUp(self):
        for name in (
            "folder_paths",
            "comfy",
            "comfy.utils",
            "trellis2_model_manager",
            "trellis2_gguf",
            "trellis2_gguf.pipelines",
            "torch",
        ):
            sys.modules.pop(name, None)

    def test_folder_paths_models_dir_stays_parent_of_node_dir(self):
        generator = Trellis2GGUFGenerator.__new__(Trellis2GGUFGenerator)
        generator.model_dir = Path("/models/trellis2/generate")
        generator._ensure_comfyui_gguf = lambda: None
        generator._resolve_generate_assets = lambda: types.SimpleNamespace(
            root=Path("/models/trellis2/generate"),
            paths={},
        )

        sys.modules["torch"] = types.ModuleType("torch")
        trellis_module = types.ModuleType("trellis2_gguf")
        pipelines_module = types.ModuleType("trellis2_gguf.pipelines")
        pipelines_module.Trellis2ImageTo3DPipeline = object
        sys.modules["trellis2_gguf"] = trellis_module
        sys.modules["trellis2_gguf.pipelines"] = pipelines_module

        generator._ensure_trellis2_gguf()

        self.assertEqual(sys.modules["folder_paths"].models_dir, "/models/trellis2")
