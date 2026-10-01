import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from data_importer import normalize_csv, normalize_jsonl

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

    def mapped_schema(self):
        return {"columns": [
            {"name": "name", "type": "string", "required": True, "source": "display_name"},
            {"name": "orders", "type": "integer", "source": "purchase_count"},
            {"name": "active", "type": "boolean", "required": True},
        ]}

    def write_csv(self, directory, text):
        path = Path(directory) / "data.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_source_mapping(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "display_name,purchase_count,active\nMaya,3,TRUE\n")
            result = normalize_csv(csv_path, self.mapped_schema())
            self.assertEqual(result["records"], [{"name": "Maya", "orders": 3, "active": True}])
            self.assertEqual((result["accepted"], result["rejected"]), (1, 0))

    def test_source_swap_and_target_name_overlap(self):
        schema = {"columns": [
            {"name": "first", "type": "string", "source": "second"},
            {"name": "second", "type": "string", "source": "first"},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "second,first\nalpha,beta\n")
            result = normalize_csv(csv_path, schema)
            self.assertEqual(result["records"], [{"first": "alpha", "second": "beta"}])

    def test_source_mapping_row_errors_and_counts(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "display_name,purchase_count,active\nMaya,3,TRUE\n,nope,TRUE\n")
            result = normalize_csv(csv_path, self.mapped_schema())
            self.assertEqual((result["accepted"], result["rejected"]), (1, 1))
            self.assertEqual(result["errors"][0]["row"], 3)
            self.assertTrue(any("name" in error for error in result["errors"][0]["errors"]))
            self.assertTrue(any("orders" in error for error in result["errors"][0]["errors"]))

    def test_source_mapping_header_mismatch(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            for header in ("name,purchase_count,active\n", "display_name,purchase_count\n",
                           "display_name,purchase_count,active,extra\n", "purchase_count,display_name,active\n"):
                csv_path = self.write_csv(directory, header)
                with self.assertRaises(ValueError):
                    normalize_csv(csv_path, self.mapped_schema())

    def test_invalid_source_values(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "display_name\nMaya\n")
            for bad in (None, 3, True, "", "   ", ["display_name"]):
                schema = {"columns": [{"name": "name", "type": "string", "source": bad}]}
                with self.assertRaises(ValueError) as caught:
                    normalize_csv(csv_path, schema)
                self.assertIn("name", str(caught.exception))

    def test_conflicting_sources(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "display_name,active\nMaya,TRUE\n")
            schema = {"columns": [
                {"name": "name", "type": "string", "source": "display_name"},
                {"name": "active", "type": "boolean", "source": "display_name"},
            ]}
            with self.assertRaises(ValueError) as caught:
                normalize_csv(csv_path, schema)
            message = str(caught.exception)
            self.assertIn("name", message)
            self.assertIn("active", message)
            self.assertIn("display_name", message)
            schema["columns"] = [
                {"name": "name", "type": "string"},
                {"name": "active", "type": "boolean", "source": "name"},
            ]
            with self.assertRaises(ValueError) as caught:
                normalize_csv(csv_path, schema)
            message = str(caught.exception)
            self.assertIn("name", message)
            self.assertIn("active", message)

    def test_cli_mapping_error_keeps_existing_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "name,orders,active\nMaya,3,TRUE\n")
            schema_path = Path(directory) / "schema.json"
            schema_path.write_text(json.dumps(self.mapped_schema()), encoding="utf-8")
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            output.write_text("keep me\n", encoding="utf-8")
            command = [sys.executable, str(ROOT / "data_importer.py"), str(csv_path), "--schema", str(schema_path), "--output", str(output), "--errors", str(errors)]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertIn("error", json.loads(run.stdout))
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")
            self.assertFalse(errors.exists())

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


class JsonlImporterTests(unittest.TestCase):
    def setUp(self):
        self.schema = json.loads((ROOT / "samples/schema.json").read_text())

    def mapped_schema(self):
        return {"columns": [
            {"name": "name", "type": "string", "required": True, "source": "display_name"},
            {"name": "orders", "type": "integer", "source": "purchase_count"},
            {"name": "active", "type": "boolean", "required": True},
        ]}

    def write_jsonl(self, directory, text, raw=False):
        path = Path(directory) / "data.jsonl"
        if raw:
            path.write_bytes(text)
        else:
            path.write_text(text, encoding="utf-8")
        return path

    def test_samples_types_and_counts(self):
        result = normalize_jsonl(ROOT / "samples/customers.jsonl", self.schema)
        self.assertEqual(result["accepted"], 2)
        self.assertEqual(result["records"][0], {"name": "Maya", "orders": 3, "active": True})
        self.assertIsNone(result["records"][1]["orders"])
        result = normalize_jsonl(ROOT / "samples/mixed.jsonl", self.schema)
        self.assertEqual((result["accepted"], result["rejected"]), (1, 2))
        self.assertEqual([row["row"] for row in result["errors"]], [2, 3])
        self.assertTrue(any("orders" in message for message in result["errors"][0]["errors"]))
        self.assertTrue(any("name" in message for message in result["errors"][1]["errors"]))

    def test_blank_lines_preserve_physical_rows(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory, '\n  \n{"display_name":"a","purchase_count":2,"active":false}\n\n')
            result = normalize_jsonl(path, self.mapped_schema())
            self.assertEqual(result["accepted"], 1)
            self.assertEqual(result["records"][0], {"name": "a", "orders": 2, "active": False})
            self.assertEqual(result["errors"], [])

    def test_empty_and_blank_only_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            for text in ("", "   \n\t\n  \n", "\n\n"):
                path = self.write_jsonl(directory, text)
                result = normalize_jsonl(path, self.mapped_schema())
                self.assertEqual((result["records"], result["errors"], result["accepted"], result["rejected"]), ([], [], 0, 0))

    def test_bom_is_accepted(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory, b'\xef\xbb\xbf{"display_name":"a","purchase_count":2,"active":false}\n', raw=True)
            result = normalize_jsonl(path, self.mapped_schema())
            self.assertEqual(result["records"], [{"name": "a", "orders": 2, "active": False}])
            self.assertEqual(result["errors"], [])

    def test_records_keep_input_order_and_mapping(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory, '{"active":true,"purchase_count":1,"display_name":"z"}\n'
                                               '{"display_name":"y","purchase_count":2,"active":false}\n')
            result = normalize_jsonl(path, self.mapped_schema())
            self.assertEqual(result["records"], [
                {"name": "z", "orders": 1, "active": True},
                {"name": "y", "orders": 2, "active": False},
            ])

    def test_parse_structure_and_constant_rejections(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            lines = ['{bad', '42', '[1,2]', '"str"', 'null', '{"a":1,"a":2}',
                     '{"display_name":"a","purchase_count":1,"active":NaN}',
                     '{"display_name":"a","purchase_count":Infinity,"active":true}',
                     '{"display_name":"a","purchase_count":-Infinity,"active":true}']
            path = self.write_jsonl(directory, "\n".join(lines) + "\n")
            result = normalize_jsonl(path, self.mapped_schema())
            self.assertEqual(result["accepted"], 0)
            self.assertEqual([row["row"] for row in result["errors"]], [1, 2, 3, 4, 5, 6, 7, 8, 9])
            self.assertTrue(result["errors"][0]["errors"][0].startswith("parse error"))
            for index in (1, 2, 3, 4):  # non-object rows -> structure errors
                self.assertIn("structure error", result["errors"][index]["errors"][0])
            self.assertIn("duplicate", result["errors"][5]["errors"][0])
            for index in (6, 7, 8):
                self.assertIn("parse error", result["errors"][index]["errors"][0])

    def test_missing_and_extra_source_keys(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory,
                                    '{"display_name":"a","active":true}\n'
                                    '{"display_name":"a","purchase_count":1,"active":true,"x":9}\n')
            result = normalize_jsonl(path, self.mapped_schema())
            self.assertEqual(result["accepted"], 0)
            self.assertIn("missing source keys", result["errors"][0]["errors"][0])
            self.assertIn("purchase_count", result["errors"][0]["errors"][0])
            self.assertIn("unexpected source keys", result["errors"][1]["errors"][0])
            self.assertIn("x", result["errors"][1]["errors"][0])

    def test_field_types(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            lines = [
                '{"display_name":"a","purchase_count":1.5,"active":true}',       # float
                '{"display_name":"a","purchase_count":1e3,"active":true}',       # exponent
                '{"display_name":"a","purchase_count":true,"active":true}',      # bool not int
                '{"display_name":["a"],"purchase_count":1,"active":true}',       # array
                '{"display_name":{"k":"v"},"purchase_count":1,"active":true}',   # object
                '{"display_name":7,"purchase_count":1,"active":true}',           # number not string
                '{"display_name":"a","purchase_count":"nope","active":true}',    # bad int string
                '{"display_name":"a","purchase_count":1,"active":"maybe"}',      # bad bool string
            ]
            path = self.write_jsonl(directory, "\n".join(lines) + "\n")
            result = normalize_jsonl(path, self.mapped_schema())
            self.assertEqual(result["accepted"], 0)
            self.assertEqual([row["row"] for row in result["errors"]], [1, 2, 3, 4, 5, 6, 7, 8])
            self.assertEqual(result["errors"][2]["errors"], ["orders: must be an integer"])
            self.assertTrue(result["errors"][3]["errors"][0].startswith("name:"))
            self.assertTrue(result["errors"][7]["errors"][0].startswith("active:"))

    def test_string_forms_of_integer_and_boolean(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory,
                                    '{"display_name":"a","purchase_count":" 7 ","active":"FaLsE"}\n'
                                    '{"display_name":" b ","purchase_count":-4,"active":"TRUE"}\n')
            result = normalize_jsonl(path, self.mapped_schema())
            self.assertEqual(result["records"], [
                {"name": "a", "orders": 7, "active": False},
                {"name": "b", "orders": -4, "active": True},
            ])

    def test_required_empty_and_null(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory,
                                    '{"display_name":"a","purchase_count":null,"active":null}\n'
                                    '{"display_name":"  ","purchase_count":1,"active":true}\n'
                                    '{"display_name":"a","purchase_count":"","active":true}\n'
                                    '{"display_name":"b","purchase_count":null,"active":false}\n')
            result = normalize_jsonl(path, self.mapped_schema())
            self.assertEqual(result["accepted"], 2)
            self.assertEqual(result["records"][0], {"name": "a", "orders": None, "active": True})
            self.assertEqual(result["records"][1], {"name": "b", "orders": None, "active": False})
            self.assertEqual([row["row"] for row in result["errors"]], [1, 2])
            self.assertIn("active: required value is empty", result["errors"][0]["errors"])

    def test_field_errors_follow_schema_order(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory, '{"display_name":9,"purchase_count":"x","active":"yes"}\n')
            result = normalize_jsonl(path, self.mapped_schema())
            fields = [message.split(":", 1)[0] for message in result["errors"][0]["errors"]]
            self.assertEqual(fields, ["name", "orders", "active"])

    def test_schema_validation_matches_csv(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory, '{"x":"y"}\n')
            with self.assertRaises(ValueError):
                normalize_jsonl(path, {"columns": [{"name": "x", "type": "date"}]})
            with self.assertRaises(ValueError):
                normalize_jsonl(path, {"columns": [{"name": "x", "type": "string", "source": " "}]})
            with self.assertRaises(ValueError):
                normalize_jsonl(path, {"columns": [
                    {"name": "a", "type": "string", "source": "k"},
                    {"name": "b", "type": "string", "source": "k"},
                ]})

    def test_missing_file_and_bad_encoding(self):
        with self.assertRaises(OSError):
            normalize_jsonl(ROOT / "does-not-exist.jsonl", self.schema)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory, b"\xff\xfe bad", raw=True)
            with self.assertRaises(UnicodeDecodeError):
                normalize_jsonl(path, self.schema)

    def test_deterministic_reprocessing(self):
        result = normalize_jsonl(ROOT / "samples/mixed.jsonl", self.schema)
        again = normalize_jsonl(ROOT / "samples/mixed.jsonl", self.schema)
        self.assertEqual(result, again)

    def test_cli_format_jsonl_and_exit_status(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            command = [sys.executable, str(ROOT / "data_importer.py"), str(ROOT / "samples/mixed.jsonl"),
                       "--schema", str(ROOT / "samples/schema.json"), "--output", str(output),
                       "--errors", str(errors), "--format", "jsonl"]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(json.loads(run.stdout), {"accepted": 1, "rejected": 2})
            self.assertEqual(len(output.read_text().splitlines()), 1)
            self.assertEqual(len(errors.read_text().splitlines()), 2)

    def test_cli_invalid_format_preserves_outputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            output.write_text("keep me\n", encoding="utf-8")
            command = [sys.executable, str(ROOT / "data_importer.py"), str(ROOT / "samples/customers.jsonl"),
                       "--schema", str(ROOT / "samples/schema.json"), "--output", str(output),
                       "--errors", str(errors), "--format", "xml"]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertIn("error", json.loads(run.stdout))
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")
            self.assertFalse(errors.exists())

    def test_cli_default_format_is_csv(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            command = [sys.executable, str(ROOT / "data_importer.py"), str(ROOT / "samples/customers.jsonl"),
                       "--schema", str(ROOT / "samples/schema.json"), "--output", str(output),
                       "--errors", str(errors)]
            run = subprocess.run(command, capture_output=True, text=True)
            # A .jsonl file parsed as CSV fails the header check; format is not inferred from the extension.
            self.assertEqual(run.returncode, 2)
            self.assertIn("error", json.loads(run.stdout))


if __name__ == "__main__":
    unittest.main()
