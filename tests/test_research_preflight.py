import os
import stat
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from clip_dinoiser.research_harness.preflight import probe_python_runtime


class ResearchPreflightTests(unittest.TestCase):
    def test_probe_python_runtime_rejects_non_executable_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "python")
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("#!/bin/sh\n")
            os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)

            payload = probe_python_runtime(
                target,
                required_modules=["json"],
                require_cuda=False,
            )

            self.assertFalse(payload["passed"])
            self.assertFalse(payload["is_executable"])


if __name__ == "__main__":
    unittest.main()
