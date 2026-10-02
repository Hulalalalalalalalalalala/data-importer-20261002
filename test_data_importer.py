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

    def test_duplicate_groups_use_record_indices_and_converted_values(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            # orders 3, 3, null, null -> two groups [1,2] and [3,4]
            csv_path = self.write_csv(
                directory,
                "name,orders,active\na,3,TRUE\nb, 3 ,true\nc,,true\nd,,true\n")
            result = normalize_csv(csv_path, self.schema, duplicate_by=["orders"])
            self.assertEqual(len(result["records"]), 4)
            self.assertEqual((result["accepted"], result["rejected"]), (4, 0))
            self.assertEqual(result["duplicates"], [
                {"key": {"orders": 3}, "record_numbers": [1, 2]},
                {"key": {"orders": None}, "record_numbers": [3, 4]},
            ])
            # no option or None keeps the old result shape
            self.assertNotIn("duplicates", normalize_csv(csv_path, self.schema))
            self.assertNotIn("duplicates", normalize_csv(csv_path, self.schema, duplicate_by=None))
            # enabled but unique -> empty array
            unique = self.write_csv(directory, "name,orders,active\na,1,true\nb,2,true\n")
            self.assertEqual(normalize_csv(unique, self.schema, duplicate_by=["orders"])["duplicates"], [])

    def test_duplicate_report_ignores_rejected_rows(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(
                directory,
                "name,orders,active\n"
                "a,3,true\n"
                ",nope,true\n"
                "b,3,true\n"
                "x,9,maybe\n"
                "c,,true\n"
                "d,,true\n")
            result = normalize_csv(csv_path, self.schema, duplicate_by=["orders"])
            self.assertEqual((result["accepted"], result["rejected"]), (4, 2))
            self.assertEqual(len(result["records"]), 4)
            self.assertEqual(result["duplicates"], [
                {"key": {"orders": 3}, "record_numbers": [1, 2]},
                {"key": {"orders": None}, "record_numbers": [3, 4]},
            ])

    def test_duplicate_strings_case_sensitive_and_composite_keys(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(
                directory,
                "name,orders,active\n"
                "a,1,true\nA,1,true\n"
                "b,2,true\nb,5,false\n"
                "c,3,true\nc,3,true\n")
            by_name = normalize_csv(csv_path, self.schema, duplicate_by=["name"])
            self.assertEqual(by_name["duplicates"], [
                {"key": {"name": "b"}, "record_numbers": [3, 4]},
                {"key": {"name": "c"}, "record_numbers": [5, 6]},
            ])
            composite = normalize_csv(csv_path, self.schema, duplicate_by=["name", "orders"])
            self.assertEqual(composite["duplicates"], [
                {"key": {"name": "c", "orders": 3}, "record_numbers": [5, 6]},
            ])

    def test_duplicate_by_validation_raises_before_reading_input(self):
        missing = ROOT / "samples" / "does-not-exist.csv"
        cases = ["orders", (), [], ["orders", 3], [""], ["  "], ["orders", "orders"], ["ghost"]]
        for bad in cases:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    normalize_csv(missing, self.schema, duplicate_by=bad)
        with self.assertRaises(ValueError) as caught:
            normalize_csv(missing, self.schema, duplicate_by=["ghost"])
        self.assertIn("ghost", str(caught.exception))
        # output names match literally; source names are not accepted
        mapped = self.mapped_schema()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "display_name,purchase_count,active\na,1,true\n")
            for source_name in ("display_name", "purchase_count"):
                with self.assertRaises(ValueError) as caught:
                    normalize_csv(csv_path, mapped, duplicate_by=[source_name])
                self.assertIn(source_name, str(caught.exception))

    def test_cli_duplicate_by_flag(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(
                directory, "name,orders,active\na,3,true\nb,3,true\nc,,true\nd,,true\n")
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            command = [sys.executable, str(ROOT / "data_importer.py"), str(csv_path),
                       "--schema", str(ROOT / "samples/schema.json"),
                       "--output", str(output), "--errors", str(errors),
                       "--duplicate-by", "orders"]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            summary = json.loads(run.stdout)
            self.assertEqual(summary["accepted"], 4)
            self.assertEqual(summary["duplicates"], [
                {"key": {"orders": 3}, "record_numbers": [1, 2]},
                {"key": {"orders": None}, "record_numbers": [3, 4]},
            ])
            self.assertEqual(len(output.read_text().splitlines()), 4)
            # invalid field: exit 2, error JSON, no outputs created
            output2, errors2 = Path(directory) / "o.jsonl", Path(directory) / "e.jsonl"
            bad = command[:-2] + ["--output", str(output2), "--errors", str(errors2),
                                  "--duplicate-by", "ghost"]
            failed = subprocess.run(bad, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 2)
            self.assertIn("error", json.loads(failed.stdout))
            self.assertFalse(output2.exists())
            self.assertFalse(errors2.exists())



class JsonlImporterTests(unittest.TestCase):
    def setUp(self):
        self.schema = json.loads((ROOT / "samples/schema.json").read_text())

    def write_jsonl(self, directory, text, name="data.jsonl"):
        path = Path(directory) / name
        path.write_text(text, encoding="utf-8")
        return path

    def mapped_schema(self):
        return {"columns": [
            {"name": "name", "type": "string", "required": True, "source": "display_name"},
            {"name": "orders", "type": "integer", "source": "purchase_count"},
            {"name": "active", "type": "boolean", "required": True},
        ]}

    def test_samples_types_bom_blank_lines_and_null(self):
        result = normalize_jsonl(ROOT / "samples/customers.jsonl", self.schema)
        self.assertEqual((result["accepted"], result["rejected"]), (2, 0))
        self.assertEqual(result["records"][0], {"name": "Maya", "orders": 3, "active": True})
        self.assertIsNone(result["records"][1]["orders"])

    def test_mixed_sample_row_numbers_and_counts(self):
        result = normalize_jsonl(ROOT / "samples/mixed.jsonl", self.schema)
        self.assertEqual((result["accepted"], result["rejected"]), (1, 2))
        self.assertEqual([row["row"] for row in result["errors"]], [3, 4])
        self.assertTrue(any("orders" in message for message in result["errors"][0]["errors"]))
        self.assertTrue(any("name" in message for message in result["errors"][1]["errors"]))

    def test_empty_and_blank_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            empty = self.write_jsonl(directory, "", "empty.jsonl")
            blanks = self.write_jsonl(directory, "\n   \n\t\n", "blanks.jsonl")
            for path in (empty, blanks):
                result = normalize_jsonl(path, self.schema)
                self.assertEqual(result, {"records": [], "errors": [], "accepted": 0, "rejected": 0})

    def test_physical_row_numbers_skip_blank_lines(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory, '{"name": "A", "orders": 1, "active": true}\n\nnot json\n  \n[]\n')
            result = normalize_jsonl(path, self.schema)
            self.assertEqual([row["row"] for row in result["errors"]], [3, 5])
            self.assertTrue(result["errors"][0]["errors"][0].startswith("parse error"))
            self.assertTrue(result["errors"][1]["errors"][0].startswith("structure error"))

    def test_string_trimming_empty_and_required_rules(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory,
                                   '{"name": "  ", "orders": null, "active": true}\n'
                                   '{"name": "ok", "orders": "", "active": "FALSE"}\n'
                                   '{"name": "Z", "orders": "  ", "active": " True "}\n')
            result = normalize_jsonl(path, self.schema)
            self.assertEqual((result["accepted"], result["rejected"]), (2, 1))
            self.assertEqual(result["records"][0], {"name": "ok", "orders": None, "active": False})
            self.assertEqual(result["records"][1], {"name": "Z", "orders": None, "active": True})
            self.assertTrue(any("name" in message for message in result["errors"][0]["errors"]))

    def test_integer_rules(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            accepted = '{"name": "ok", "orders": %s, "active": true}'
            rejected_values = ("3.0", "1e2", "-0.5", "true", '"3.5"', '"1e2"', "[3]")
            lines = [accepted % "42", accepted % '"-8"']
            lines.extend(accepted % value for value in rejected_values)
            path = self.write_jsonl(directory, "\n".join(lines) + "\n")
            result = normalize_jsonl(path, self.schema)
            self.assertEqual((result["accepted"], result["rejected"]), (2, len(rejected_values)))
            self.assertEqual([record["orders"] for record in result["records"]], [42, -8])
            for row in result["errors"]:
                self.assertTrue(any("orders" in message for message in row["errors"]))

    def test_boolean_rules(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            lines = ['{"name": "a", "orders": null, "active": true}',
                     '{"name": "b", "orders": null, "active": false}',
                     '{"name": "c", "orders": null, "active": "TrUe"}',
                     '{"name": "d", "orders": null, "active": 1}',
                     '{"name": "e", "orders": null, "active": "yes"}',
                     '{"name": "f", "orders": null, "active": ["true"]}']
            path = self.write_jsonl(directory, "\n".join(lines) + "\n")
            result = normalize_jsonl(path, self.schema)
            self.assertEqual([record["active"] for record in result["records"]], [True, False, True])
            self.assertEqual(result["rejected"], 3)

    def test_arrays_objects_and_other_types_are_field_errors(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            lines = ['{"name": ["Maya"], "orders": {}, "active": true}',
                     '{"name": 4, "orders": null, "active": false}']
            path = self.write_jsonl(directory, "\n".join(lines) + "\n")
            result = normalize_jsonl(path, self.schema)
            self.assertEqual(result["rejected"], 2)
            first = result["errors"][0]["errors"]
            self.assertTrue(any(message.startswith("name:") for message in first))
            self.assertTrue(any(message.startswith("orders:") for message in first))
            self.assertTrue(any(message.startswith("name:") for message in result["errors"][1]["errors"]))

    def test_duplicate_missing_and_extra_keys(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            lines = ['{"name": "A", "name": "B", "orders": 1, "active": true}',
                     '{"name": "A", "active": true}',
                     '{"name": "A", "orders": 1, "active": true, "extra": 9}',
                     '{"orders": 1, "active": true, "extra": 9}']
            path = self.write_jsonl(directory, "\n".join(lines) + "\n")
            result = normalize_jsonl(path, self.schema)
            self.assertEqual(result["rejected"], 4)
            self.assertTrue(result["errors"][0]["errors"][0].startswith("structure error: duplicate key"))
            self.assertIn("orders", result["errors"][1]["errors"][0])
            self.assertIn("extra", result["errors"][2]["errors"][0])
            third = result["errors"][3]["errors"]
            self.assertTrue(any("missing" in message and "name" in message for message in third))
            self.assertTrue(any("unexpected" in message and "extra" in message for message in third))

    def test_constants_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            lines = ['{"name": "A", "orders": NaN, "active": true}',
                     '{"name": "A", "orders": Infinity, "active": true}',
                     '{"name": "A", "orders": -Infinity, "active": true}']
            path = self.write_jsonl(directory, "\n".join(lines) + "\n")
            result = normalize_jsonl(path, self.schema)
            self.assertEqual(result["rejected"], 3)
            self.assertTrue(all(row["errors"][0].startswith("parse error") for row in result["errors"]))

    def test_source_mapping_and_literal_key_match(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            text = '{"display_name": " Maya ", "purchase_count": "3", "active": "TRUE"}\n'
            path = self.write_jsonl(directory, text)
            result = normalize_jsonl(path, self.mapped_schema())
            self.assertEqual(result["records"], [{"name": "Maya", "orders": 3, "active": True}])
            wrong = self.write_jsonl(directory, '{"display_name": "x", "purchase_count": 1, "Active": true}\n',
                                     "wrong.jsonl")
            result = normalize_jsonl(wrong, self.mapped_schema())
            self.assertEqual(result["rejected"], 1)
            messages = result["errors"][0]["errors"]
            self.assertTrue(any("missing" in message and "active" in message for message in messages))
            self.assertTrue(any("unexpected" in message and "Active" in message for message in messages))

    def test_field_errors_use_output_names_in_schema_order(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory, '{"active": 1, "orders": "x", "name": 9}\n')
            result = normalize_jsonl(path, self.schema)
            prefixes = [message.split(":", 1)[0] for message in result["errors"][0]["errors"]]
            self.assertEqual(prefixes, ["name", "orders", "active"])

    def test_order_preserved_and_deterministic(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            lines = ['{"name": "c%d", "orders": %d, "active": true}' % (i, i) for i in range(5)]
            lines.insert(2, 'not json')
            path = self.write_jsonl(directory, "\n".join(lines) + "\n")
            first = normalize_jsonl(path, self.schema)
            second = normalize_jsonl(path, self.schema)
            self.assertEqual(first, second)
            self.assertEqual((first["accepted"], first["rejected"]), (5, 1))
            self.assertEqual(first["errors"][0]["row"], 3)
            self.assertEqual([record["name"] for record in first["records"]], ["c0", "c1", "c2", "c3", "c4"])

    def test_duplicate_int_equals_numeric_string_and_blank_lines_skip_indices(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            # integer 3 and the string " 3 " convert to the same key;
            # leading/inner blank lines must not consume record numbers.
            path = self.write_jsonl(directory,
                                    '\n  \n'
                                    '{"name": "a", "orders": 3, "active": true}\n'
                                    '\n'
                                    '{"name": "b", "orders": " 3 ", "active": true}\n'
                                    '{"name": "c", "orders": null, "active": true}\n'
                                    '{"name": "d", "orders": "", "active": true}\n')
            result = normalize_jsonl(path, self.schema, duplicate_by=["orders"])
            self.assertEqual((result["accepted"], result["rejected"]), (4, 0))
            self.assertEqual(result["duplicates"], [
                {"key": {"orders": 3}, "record_numbers": [1, 2]},
                {"key": {"orders": None}, "record_numbers": [3, 4]},
            ])
            self.assertNotIn("duplicates", normalize_jsonl(path, self.schema))
            self.assertEqual(normalize_jsonl(path, self.schema, duplicate_by=None).get("duplicates", None), None)

    def test_duplicate_report_skips_rejected_jsonl_lines(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory,
                                    '{"name": "a", "orders": 3, "active": true}\n'
                                    'not json\n'
                                    '{"name": "b", "orders": "3", "active": true}\n'
                                    '{"name": "c", "orders": 4, "active": true}\n')
            result = normalize_jsonl(path, self.schema, duplicate_by=["orders"])
            self.assertEqual((result["accepted"], result["rejected"]), (3, 1))
            self.assertEqual(result["duplicates"], [
                {"key": {"orders": 3}, "record_numbers": [1, 2]},
            ])

    def test_duplicate_by_validation_jsonl(self):
        missing = ROOT / "samples" / "does-not-exist.jsonl"
        for bad in ("orders", [], [3], ["  "], ["active", "active"], ["unknown"]):
            with self.assertRaises(ValueError):
                normalize_jsonl(missing, self.schema, duplicate_by=bad)

    def test_cli_duplicate_by_jsonl(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = self.write_jsonl(
                directory,
                '{"name": "a", "orders": 3, "active": true}\n'
                '{"name": "b", "orders": 3, "active": true}\n',
                name="source.jsonl")
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            command = [sys.executable, str(ROOT / "data_importer.py"), str(source),
                       "--schema", str(ROOT / "samples/schema.json"),
                       "--output", str(output), "--errors", str(errors),
                       "--format", "jsonl", "--duplicate-by", "orders",
                       "--duplicate-by", "active"]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout), {
                "accepted": 2, "rejected": 0,
                "duplicates": [{"key": {"orders": 3, "active": True}, "record_numbers": [1, 2]}],
            })

    def test_file_and_decode_errors(self):
        missing = ROOT / "samples" / "does-not-exist.jsonl"
        with self.assertRaises(OSError):
            normalize_jsonl(missing, self.schema)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "bad.bin"
            path.write_bytes(b"\xff\xfe{}\n")
            with self.assertRaises(UnicodeDecodeError):
                normalize_jsonl(path, self.schema)

    def test_schema_validation_shared(self):
        with self.assertRaises(ValueError):
            normalize_jsonl(ROOT / "samples/customers.jsonl",
                            {"columns": [{"name": "name", "type": "date"}]})
        with self.assertRaises(ValueError):
            normalize_jsonl(ROOT / "samples/customers.jsonl",
                            {"columns": [{"name": "a", "type": "string"},
                                         {"name": "b", "type": "string", "source": "a"}]})

    def test_cli_format_flag(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            base = [sys.executable, str(ROOT / "data_importer.py"),
                    str(ROOT / "samples/mixed.jsonl"), "--schema", str(ROOT / "samples/schema.json"),
                    "--output", str(output), "--errors", str(errors)]
            run = subprocess.run(base + ["--format", "jsonl"], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(json.loads(run.stdout), {"accepted": 1, "rejected": 2})
            self.assertEqual(len(output.read_text().splitlines()), 1)
            self.assertEqual(len(errors.read_text().splitlines()), 2)
            run_default = subprocess.run(base, capture_output=True, text=True)
            self.assertEqual(run_default.returncode, 2)
            run_bad = subprocess.run(base + ["--format", "xml"], capture_output=True, text=True)
            self.assertEqual(run_bad.returncode, 2)
            self.assertIn("error", json.loads(run_bad.stdout))

    def test_cli_jsonl_no_rejections_exit_zero(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            command = [sys.executable, str(ROOT / "data_importer.py"),
                       str(ROOT / "samples/customers.jsonl"), "--schema",
                       str(ROOT / "samples/schema.json"), "--output", str(output),
                       "--errors", str(errors), "--format", "jsonl"]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout), {"accepted": 2, "rejected": 0})
            self.assertEqual(errors.read_text(encoding="utf-8"), "")

    def test_cli_jsonl_error_keeps_existing_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = self.write_jsonl(directory, '{"name": "A", "orders": 1, "active": true}\n')
            schema_path = Path(directory) / "schema.json"
            schema_path.write_text(json.dumps(self.mapped_schema()), encoding="utf-8")
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            output.write_text("keep me\n", encoding="utf-8")
            command = [sys.executable, str(ROOT / "data_importer.py"), str(source),
                       "--schema", str(schema_path), "--output", str(output),
                       "--errors", str(errors), "--format", "jsonl"]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertIn("error", json.loads(run.stdout))
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")
            self.assertFalse(errors.exists())


if __name__ == "__main__":
    unittest.main()
