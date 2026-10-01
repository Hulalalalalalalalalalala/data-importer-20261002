import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from data_importer import normalize_csv

ROOT = Path(__file__).resolve().parent


class ImporterTests(unittest.TestCase):
    def setUp(self):
        self.schema = json.loads((ROOT / "samples/schema.json").read_text())

    def test_types_whitespace_and_optional_null(self):
        result = normalize_csv(ROOT / "samples/customers.csv", self.schema)
        self.assertEqual(result["accepted"], 2)
        self.assertEqual(result["records"][0], {"name": "Maya", "orders": 3, "active": True})
        self.assertIsNone(result["records"][1]["orders"])

    def test_row_errors_preserve_valid_rows(self):
        result = normalize_csv(ROOT / "samples/mixed.csv", self.schema)
        self.assertEqual((result["accepted"], result["rejected"]), (1, 2))
        self.assertEqual([row["row"] for row in result["errors"]], [3, 4])
        self.assertIn("orders", result["errors"][0]["errors"][0])

    def test_invalid_schema_and_header(self):
        with self.assertRaises(ValueError):
            normalize_csv(ROOT / "samples/customers.csv", {"columns": [{"name": "name", "type": "date"}]})
        with self.assertRaises(ValueError):
            normalize_csv(ROOT / "samples/customers.csv", {"columns": [{"name": "other", "type": "string"}]})

    def test_cli_jsonl_and_exit_status(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            command = [sys.executable, str(ROOT / "data_importer.py"), str(ROOT / "samples/mixed.csv"), "--schema", str(ROOT / "samples/schema.json"), "--output", str(output), "--errors", str(errors)]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(json.loads(run.stdout), {"accepted": 1, "rejected": 2})
            self.assertEqual(len(output.read_text().splitlines()), 1)
            self.assertEqual(len(errors.read_text().splitlines()), 2)
            command[-1] = str(output)
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)


if __name__ == "__main__":
    unittest.main()
