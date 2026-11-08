from pathlib import Path
import ast
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ProposedOnlyRelease(unittest.TestCase):
    def test_only_proposed_cold_start_scripts(self):
        scripts = sorted(p.relative_to(ROOT).as_posix()
                         for p in (ROOT/"cold_start").rglob("*.py"))
        self.assertEqual(scripts, [
            "cold_start/humanoid/proposed.py",
            "cold_start/pusher/proposed.py",
            "cold_start/standup/proposed.py"
        ])
        for rel in scripts:
            source = (ROOT/rel).read_text()
            ast.parse(source)
            self.assertIn("class ResidualAdapter", source)
            self.assertIn("return x +", source)

    def test_warm_transfer_supports_anchor_only(self):
        source = (ROOT/"warm_start"/"train_transfer.py").read_text()
        self.assertIn('METHODS = ["anchor_only"]', source)
        self.assertNotIn('METHODS = ["distributed"', source)

    def test_no_baseline_files(self):
        names = [p.name.lower() for p in ROOT.rglob("*") if p.is_file()]
        for word in ("distributed_sac", "hatrpo", "reconstructed"):
            self.assertFalse(any(word in n for n in names), word)

if __name__ == "__main__":
    unittest.main()
