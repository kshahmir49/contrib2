import unittest
from pathlib import Path


class ReleaseScopeTests(unittest.TestCase):
    def test_only_proposed_source(self):
        root = Path(__file__).resolve().parents[1]
        for file in root.rglob("*.py"):
            self.assertNotIn("distributed", file.name.lower())
            self.assertNotIn("hatrpo", file.name.lower())
        self.assertEqual(len(list((root / "cold_start").rglob("proposed.py"))), 3)


if __name__ == "__main__":
    unittest.main()
