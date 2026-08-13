import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "safe_checkpoint_inspector.py"
REAL_CHECKPOINT = (
    Path(os.environ["V2SAM_TEST_CHECKPOINT"])
    if "V2SAM_TEST_CHECKPOINT" in os.environ
    else ROOT.parent
    / "v2sam_offline_assets"
    / "fusion_ego2exo_full.pth"
)


def write_checkpoint(path: Path, payload: bytes) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", payload)


def run_inspector(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-S", str(SCRIPT), "--json", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )


class EvilReduce:
    def __init__(self, command: str):
        self.command = command

    def __reduce__(self):
        return os.system, (self.command,)


class SafeCheckpointInspectorTests(unittest.TestCase):
    @unittest.skipUnless(
        REAL_CHECKPOINT.is_file(),
        f"external checkpoint fixture not found: {REAL_CHECKPOINT}",
    )
    def test_real_checkpoint_reports_exact_tensor_structure_without_torch(self):
        result = run_inspector(REAL_CHECKPOINT)

        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["static_only"])
        self.assertEqual(report["tensor_count"], 1335)
        self.assertEqual(report["non_tensor_state_entries"], 0)
        self.assertEqual(
            report["prefix_counts"],
            {
                "constr_prompt_fcs": 4,
                "grounding_encoder": 900,
                "matcher": 60,
                "sparse_correspondence": 371,
            },
        )
        self.assertEqual(report["state_dict_key"], "state_dict")
        self.assertEqual(report["pickle_protocol"], 2)
        self.assertEqual(report["unknown_globals"], [])

    def test_unknown_reduce_global_is_rejected_without_execution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            sentinel = temp / "must-not-exist.txt"
            checkpoint = temp / "evil.pth"
            command = f'echo unsafe > "{sentinel}"'
            write_checkpoint(
                checkpoint,
                pickle.dumps(EvilReduce(command), protocol=2),
            )

            result = run_inspector(checkpoint)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsupported GLOBAL", result.stderr)
            self.assertFalse(sentinel.exists())

    def test_getattr_outside_history_statistics_is_rejected(self):
        # Protocol-2 pickle equivalent to
        # getattr(HistoryBuffer, "__subclasses__"). The inspector must never
        # execute the getattr; it only recognizes four literal method names.
        payload = (
            b"\x80\x02"
            b"c__builtin__\ngetattr\n"
            b"cmmengine.logging.history_buffer\nHistoryBuffer\n"
            b"X\x0e\x00\x00\x00__subclasses__"
            b"\x86R."
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "bad-getattr.pth"
            write_checkpoint(checkpoint, payload)

            result = run_inspector(checkpoint)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("getattr is not allowlisted", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
