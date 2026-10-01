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

    def test_source_mapping_renames_fields(self):
        schema = json.loads((ROOT / "samples/mapped_schema.json").read_text())
        result = normalize_csv(ROOT / "samples/customers_mapped.csv", schema)
        self.assertEqual(result["accepted"], 2)
        self.assertEqual(result["rejected"], 0)
        self.assertEqual(result["records"][0], {"name": "Maya", "orders": 3, "active": True})
        self.assertIsNone(result["records"][1]["orders"])

    def test_mapping_is_deterministic(self):
        schema = json.loads((ROOT / "samples/mapped_schema.json").read_text())
        first = normalize_csv(ROOT / "samples/customers_mapped.csv", schema)
        second = normalize_csv(ROOT / "samples/customers_mapped.csv", schema)
        self.assertEqual(first, second)

    def test_invalid_source_values_mention_target_name(self):
        base_column = {"name": "name", "type": "string"}
        for bad_source in (None, 3, True, False, [], {}, "", "   "):
            with self.subTest(bad_source=bad_source):
                schema = {"columns": [{**base_column, "source": bad_source}]}
                with self.assertRaises(ValueError) as context:
                    normalize_csv(ROOT / "samples/customers_mapped.csv", schema)
                self.assertIn("name", str(context.exception))

    def test_duplicate_sources_mention_targets_and_source(self):
        schema = {"columns": [
            {"name": "name", "type": "string", "source": "label"},
            {"name": "orders", "type": "integer", "source": "label"},
            {"name": "active", "type": "boolean"},
        ]}
        with self.assertRaises(ValueError) as context:
            normalize_csv(ROOT / "samples/customers_mapped.csv", schema)
        message = str(context.exception)
        self.assertIn("label", message)
        self.assertIn("name", message)
        self.assertIn("orders", message)

    def test_explicit_source_clashes_with_other_default(self):
        schema = {"columns": [
            {"name": "name", "type": "string", "source": "orders"},
            {"name": "orders", "type": "integer"},
        ]}
        with self.assertRaises(ValueError) as context:
            normalize_csv(ROOT / "samples/customers_mapped.csv", schema)
        message = str(context.exception)
        self.assertIn("orders", message)
        self.assertIn("name", message)

    def test_swapped_sources_and_target_name_overlap_allowed(self):
        schema = {"columns": [
            {"name": "name", "type": "string", "required": True, "source": "active"},
            {"name": "active", "type": "boolean", "required": True, "source": "name"},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = Path(directory) / "swapped.csv"
            csv_path.write_text("active,name\nMaya,true\n", encoding="utf-8")
            result = normalize_csv(csv_path, schema)
        self.assertEqual(result["records"], [{"name": "Maya", "active": True}])

    def test_header_mismatches_raise(self):
        schema = json.loads((ROOT / "samples/mapped_schema.json").read_text())
        cases = {
            "missing": "display_name,active\nMaya,true\n",
            "extra": "display_name,purchase_count,active,extra\nMaya,3,true,x\n",
            "wrong_order": "active,purchase_count,display_name\ntrue,3,Maya\n",
            "duplicate": "display_name,purchase_count,display_name\nMaya,3,X\n",
            "case_or_whitespace": " display_name ,purchase_count,active\nMaya,3,true\n",
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            for label, content in cases.items():
                csv_path = Path(directory) / f"{label}.csv"
                csv_path.write_text(content, encoding="utf-8")
                with self.subTest(label=label):
                    with self.assertRaises(ValueError):
                        normalize_csv(csv_path, schema)

    def test_row_errors_use_target_names_with_mapping(self):
        schema = json.loads((ROOT / "samples/mapped_schema.json").read_text())
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = Path(directory) / "bad.csv"
            csv_path.write_text(
                "display_name,purchase_count,active\nMaya,many,true\n,3,yes\n",
                encoding="utf-8",
            )
            result = normalize_csv(csv_path, schema)
        self.assertEqual((result["accepted"], result["rejected"]), (0, 2))
        self.assertEqual([row["row"] for row in result["errors"]], [2, 3])
        self.assertIn("orders", result["errors"][0]["errors"][0])
        self.assertIn("name", result["errors"][1]["errors"][0])

    def test_mapping_error_cli_exit_2_keeps_existing_outputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            output.write_text("KEEP-ME\n", encoding="utf-8")
            errors.write_text("KEEP-ME\n", encoding="utf-8")
            command = [
                sys.executable, str(ROOT / "data_importer.py"),
                str(ROOT / "samples/customers.csv"),
                "--schema", str(ROOT / "samples/mapped_schema.json"),
                "--output", str(output), "--errors", str(errors),
            ]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertIn("error", json.loads(run.stdout))
            self.assertEqual(output.read_text(), "KEEP-ME\n")
            self.assertEqual(errors.read_text(), "KEEP-ME\n")

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
