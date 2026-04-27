import importlib.util
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


if __name__ == "__main__":
    unittest.main()
