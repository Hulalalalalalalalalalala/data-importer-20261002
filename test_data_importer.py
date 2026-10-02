import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from data_importer import normalize_csv, normalize_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parent


def run_cli(source, output, errors, fmt=None, extra=()):
    command = [sys.executable, str(ROOT / "data_importer.py"), str(source),
               "--schema", str(ROOT / "samples/schema.json"),
               "--output", str(output), "--errors", str(errors)]
    if fmt is not None:
        command += ["--format", fmt]
    command += list(extra)
    return subprocess.run(command, capture_output=True, text=True)


def hidden_entries(directory):
    return [path.name for path in Path(directory).iterdir() if path.name.startswith(".")]



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

    def write_csv_bytes(self, directory, data):
        path = Path(directory) / "data.csv"
        path.write_bytes(data)
        return path

    def test_physical_rows_blank_and_multiline_fixture(self):
        text = ("name,orders,active\n\n\"Ma\nya\",3,true\nBad,x,true\n"
                "\"No\nra\",2,\nZ,3,false\n")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, text)
            result = normalize_csv(csv_path, self.schema, duplicate_by=["orders"])
        self.assertEqual((result["accepted"], result["rejected"]), (2, 2))
        self.assertEqual([row["row"] for row in result["errors"]], [5, 6])
        self.assertIn("orders", result["errors"][0]["errors"][0])
        self.assertEqual(result["errors"][1]["errors"], ["active: required value is empty"])
        self.assertEqual([record["name"] for record in result["records"]], ["Ma\nya", "Z"])
        self.assertEqual(result["duplicates"],
                         [{"key": {"orders": 3}, "record_numbers": [1, 2]}])

    def test_physical_rows_line_endings_bom_and_no_final_newline(self):
        base = (b"name,orders,active\r\n\r\n\"Ma\r\nya\",3,true\r\nBad,x,true\r\n"
                b"\"No\r\nra\",2,\r\nZ,3,false\r\n")
        lone_cr = base.replace(b"\r\n", b"\r")
        bom = b"\xef\xbb\xbf" + base
        no_final = base[:-2]  # drop trailing CRLF
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            for data in (base, lone_cr, bom, no_final):
                csv_path = self.write_csv_bytes(directory, data)
                result = normalize_csv(csv_path, self.schema)
                self.assertEqual((result["accepted"], result["rejected"]), (2, 2), data)
                self.assertEqual([row["row"] for row in result["errors"]], [5, 6], data)

    def test_physical_rows_multiline_reject_counts_once(self):
        # A quoted field spanning lines is one record; its error reports the
        # starting physical line. Consecutive/trailing blank lines add nothing.
        text = "name,orders,active\n\n\n\"a\nb\",nope,\n\"x\n\ny\",1,true\n\n"
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, text)
            result = normalize_csv(csv_path, self.schema)
        self.assertEqual((result["accepted"], result["rejected"]), (1, 1))
        self.assertEqual(result["errors"][0]["row"], 4)
        self.assertEqual([m.split(":", 1)[0] for m in result["errors"][0]["errors"]],
                         ["orders", "active"])
        self.assertEqual(result["records"][0]["name"], "x\n\ny")

    def test_physical_rows_whitespace_only_keeps_csv_rule(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "name,orders,active\nMaya,3,true\n   \n")
            result = normalize_csv(csv_path, self.schema)
        self.assertEqual(result["rejected"], 1)
        self.assertEqual(result["errors"][0]["row"], 3)
        self.assertEqual(result["errors"][0]["errors"], ["wrong number of cells"])

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


    def test_duplicate_report_groups_and_record_numbers(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(
                directory,
                "name,orders,active\n"
                "a,3,TRUE\n"
                "b,3,true\n"
                "bad,x,true\n"
                "c,,true\n"
                "d,,false\n")
            result = normalize_csv(csv_path, self.schema, duplicate_by=["orders"])
            self.assertEqual(len(result["records"]), 4)
            self.assertEqual((result["accepted"], result["rejected"]), (4, 1))
            self.assertEqual(result["duplicates"], [
                {"key": {"orders": 3}, "record_numbers": [1, 2]},
                {"key": {"orders": None}, "record_numbers": [3, 4]},
            ])

    def test_duplicate_report_disabled_or_absent(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "name,orders,active\na,3,TRUE\nb,3,true\n")
            self.assertNotIn("duplicates", normalize_csv(csv_path, self.schema))
            self.assertNotIn("duplicates", normalize_csv(csv_path, self.schema, None))
            unique = self.write_csv(directory, "name,orders,active\na,1,TRUE\nb,2,true\n")
            self.assertEqual(normalize_csv(unique, self.schema, duplicate_by=["orders"])["duplicates"], [])

    def test_duplicate_report_composite_and_first_seen_order(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(
                directory,
                "name,orders,active\n"
                "a,1,true\n"
                "b,2,false\n"
                "c,1,true\n"
                "d,2,true\n"
                "e,2,false\n")
            result = normalize_csv(csv_path, self.schema, duplicate_by=["orders", "active"])
            self.assertEqual(result["duplicates"], [
                {"key": {"orders": 1, "active": True}, "record_numbers": [1, 3]},
                {"key": {"orders": 2, "active": False}, "record_numbers": [2, 5]},
            ])

    def test_duplicate_by_invalid_config_raises_before_reading(self):
        missing = ROOT / "samples" / "does-not-exist.csv"
        cases = ["orders", [], ["orders", "orders"], [""], ["  "], [3], ("orders",),
                 ["nope"], ["active", "nope"]]
        for value in cases:
            with self.assertRaises(ValueError):
                normalize_csv(missing, self.schema, duplicate_by=value)
        with self.assertRaises(ValueError) as caught:
            normalize_csv(missing, self.schema, duplicate_by=["orders", "nope"])
        self.assertIn("nope", str(caught.exception))
        # a source name is not accepted as a field name
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "display_name,purchase_count,active\nMaya,3,TRUE\n")
            with self.assertRaises(ValueError):
                normalize_csv(csv_path, self.mapped_schema(), duplicate_by=["purchase_count"])

    def test_cli_duplicate_by_flag(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(
                directory,
                "name,orders,active\nA,3,TRUE\nB,3,true\nC,,true\nBAD,x,true\nD,,true\n")
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            command = [sys.executable, str(ROOT / "data_importer.py"), str(csv_path),
                       "--schema", str(ROOT / "samples/schema.json"),
                       "--output", str(output), "--errors", str(errors),
                       "--duplicate-by", "orders"]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            summary = json.loads(run.stdout)
            self.assertEqual(summary["accepted"], 4)
            self.assertEqual(summary["rejected"], 1)
            self.assertEqual(summary["duplicates"], [
                {"key": {"orders": 3}, "record_numbers": [1, 2]},
                {"key": {"orders": None}, "record_numbers": [3, 4]},
            ])
            self.assertEqual(len(output.read_text().splitlines()), 4)
            self.assertEqual(len(errors.read_text().splitlines()), 1)

    def test_cli_duplicate_by_bad_field_exit_two_keeps_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "name,orders,active\nA,3,TRUE\n")
            schema_path = Path(directory) / "schema.json"
            schema_path.write_text(json.dumps(self.mapped_schema()), encoding="utf-8")
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            output.write_text("keep me\n", encoding="utf-8")
            command = [sys.executable, str(ROOT / "data_importer.py"), str(csv_path),
                       "--schema", str(schema_path), "--output", str(output),
                       "--errors", str(errors), "--duplicate-by", "purchase_count"]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            payload = json.loads(run.stdout)
            self.assertIn("error", payload)
            self.assertIn("purchase_count", payload["error"])
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")
            self.assertFalse(errors.exists())


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


    def test_duplicate_report_jsonl_values_and_blank_lines(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(
                directory,
                '{"name": "a", "orders": 3, "active": true}\n'
                '{"name": "b", "orders": " 3 ", "active": true}\n'
                '\n'
                'not json\n'
                '{"name": "c", "orders": null, "active": true}\n'
                '{"name": "d", "orders": null, "active": false}\n')
            result = normalize_jsonl(path, self.schema, duplicate_by=["orders"])
            self.assertEqual((result["accepted"], result["rejected"]), (4, 1))
            self.assertEqual(len(result["records"]), 4)
            self.assertEqual(result["duplicates"], [
                {"key": {"orders": 3}, "record_numbers": [1, 2]},
                {"key": {"orders": None}, "record_numbers": [3, 4]},
            ])

    def test_duplicate_report_string_case_sensitive_and_validation(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(
                directory,
                '{"name": "X", "orders": 1, "active": true}\n'
                '{"name": "x", "orders": 2, "active": true}\n')
            self.assertEqual(normalize_jsonl(path, self.schema, duplicate_by=["name"])["duplicates"], [])
            missing = ROOT / "samples" / "does-not-exist.jsonl"
            for value in ("name", [], ["name", "name"], [""], ["  "], [3], ["nope"]):
                with self.assertRaises(ValueError):
                    normalize_jsonl(missing, self.schema, duplicate_by=value)
            self.assertNotIn("duplicates", normalize_jsonl(path, self.schema))

    def test_cli_jsonl_duplicate_by_flag(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(
                directory,
                '{"name": "a", "orders": 3, "active": true}\n'
                '{"name": "b", "orders": "3", "active": true}\n'
                '{"name": "c", "orders": null, "active": true}\n'
                '{"name": "d", "orders": null, "active": true}\n',
                name="source.jsonl")
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            command = [sys.executable, str(ROOT / "data_importer.py"), str(path),
                       "--schema", str(ROOT / "samples/schema.json"),
                       "--output", str(output), "--errors", str(errors), "--format", "jsonl",
                       "--duplicate-by", "orders"]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout), {
                "accepted": 4, "rejected": 0,
                "duplicates": [
                    {"key": {"orders": 3}, "record_numbers": [1, 2]},
                    {"key": {"orders": None}, "record_numbers": [3, 4]},
                ]})
            self.assertEqual(len(output.read_text().splitlines()), 4)
            self.assertEqual(errors.read_text(encoding="utf-8"), "")


class FilterEqTests(unittest.TestCase):
    """Single-field equality filtering for both input formats."""

    CSV_TEXT = ("name,orders,active\n"
                "Maya,3,true\n"      # 1 valid, orders 3, active true
                "Omar,,false\n"      # 2 valid, orders null
                "Nora,3,true\n"      # 3 valid, orders 3, active true
                "Liam,bad,true\n"    # 4 invalid orders -> error
                "Zoe,3,false\n")     # 5 valid, active false

    JSONL_TEXT = ('{"name": "Maya", "orders": 3, "active": true}\n'
                  '{"name": "Omar", "orders": null, "active": false}\n'
                  '{"name": "Nora", "orders": "3", "active": true}\n'
                  '\n'
                  '{"name": "Liam", "orders": "bad", "active": true}\n'
                  '{"name": "Zoe", "orders": 3, "active": false}\n'
                  '{"name": "Amy", "orders": "", "active": true}\n')

    def setUp(self):
        self.schema = json.loads((ROOT / "samples/schema.json").read_text())
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        directory = Path(self.temp.name)
        self.csv_path = directory / "data.csv"
        self.csv_path.write_text(self.CSV_TEXT, encoding="utf-8")
        self.jsonl_path = directory / "data.jsonl"
        self.jsonl_path.write_text(self.JSONL_TEXT, encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def mapped_schema(self):
        return {"columns": [
            {"name": "name", "type": "string", "required": True, "source": "display_name"},
            {"name": "orders", "type": "integer", "source": "purchase_count"},
            {"name": "active", "type": "boolean", "required": True},
        ]}

    def test_boolean_filter_counts_and_example(self):
        # Three valid rows match active=true; one valid row is filtered out;
        # the bad row is still an error: accepted 3... see per-format counts.
        for fmt, path in (("csv", self.csv_path), ("jsonl", self.jsonl_path)):
            with self.subTest(fmt=fmt):
                normalize = normalize_csv if fmt == "csv" else normalize_jsonl
                result = normalize(path, self.schema, filter_eq={"field": "active", "value": True})
                if fmt == "csv":
                    self.assertEqual((result["accepted"], result["filtered"], result["rejected"]), (2, 2, 1))
                    self.assertEqual([r["name"] for r in result["records"]], ["Maya", "Nora"])
                else:
                    # Amy (orders "" -> null) is a fourth active=true row
                    self.assertEqual((result["accepted"], result["filtered"], result["rejected"]), (3, 2, 1))
                    self.assertEqual([r["name"] for r in result["records"]], ["Maya", "Nora", "Amy"])
                self.assertEqual(result["errors"][0]["row"], 5)
                # rejected rows are never hidden behind the filter
                self.assertTrue(any("orders" in m for m in result["errors"][0]["errors"]))

    def test_spec_example_two_matched_one_filtered_one_rejected(self):
        # Three valid records, two of which match; one further error row.
        text = ("name,orders,active\n"
                "Maya,3,true\n"
                "Omar,2,false\n"
                "Nora,4,true\n"
                "Liam,x,true\n")
        path = Path(self.temp.name) / "spec.csv"
        path.write_text(text, encoding="utf-8")
        result = normalize_csv(path, self.schema, filter_eq={"field": "active", "value": True})
        self.assertEqual((result["accepted"], result["filtered"], result["rejected"]), (2, 1, 1))
        self.assertEqual([r["name"] for r in result["records"]], ["Maya", "Nora"])
        self.assertEqual(result["errors"][0]["row"], 5)

    def test_filter_does_not_count_mismatches_as_rejected(self):
        result = normalize_csv(self.csv_path, self.schema,
                               filter_eq={"field": "active", "value": True})
        self.assertEqual(len(result["records"]), result["accepted"])
        self.assertEqual(len(result["errors"]), result["rejected"])

    def test_null_filter_matches_normalized_null_only(self):
        result = normalize_csv(self.csv_path, self.schema,
                               filter_eq={"field": "orders", "value": None})
        self.assertEqual((result["accepted"], result["filtered"], result["rejected"]), (1, 3, 1))
        self.assertEqual(result["records"], [{"name": "Omar", "orders": None, "active": False}])
        # JSONL: null and an empty/whitespace string both normalize to null
        result = normalize_jsonl(self.jsonl_path, self.schema,
                                 filter_eq={"field": "orders", "value": None})
        self.assertEqual((result["accepted"], result["filtered"], result["rejected"]), (2, 3, 1))
        self.assertEqual([r["name"] for r in result["records"]], ["Omar", "Amy"])

    def test_string_filter_is_case_sensitive_and_literal(self):
        lower = normalize_csv(self.csv_path, self.schema,
                              filter_eq={"field": "name", "value": "maya"})
        self.assertEqual((lower["accepted"], lower["filtered"]), (0, 4))
        padded = normalize_csv(self.csv_path, self.schema,
                               filter_eq={"field": "name", "value": " Maya "})
        self.assertEqual((padded["accepted"], padded["filtered"]), (0, 4))
        exact = normalize_csv(self.csv_path, self.schema,
                              filter_eq={"field": "name", "value": "Maya"})
        self.assertEqual([r["name"] for r in exact["records"]], ["Maya"])
        # empty string never equals a normalized value (those became null)
        empty = normalize_csv(self.csv_path, self.schema,
                              filter_eq={"field": "name", "value": ""})
        self.assertEqual(empty["accepted"], 0)

    def test_integer_type_boundaries(self):
        match = normalize_csv(self.csv_path, self.schema,
                              filter_eq={"field": "orders", "value": 3})
        self.assertEqual([r["name"] for r in match["records"]], ["Maya", "Nora", "Zoe"])
        # JSONL string "3" was converted to integer 3, so it matches too
        match_j = normalize_jsonl(self.jsonl_path, self.schema,
                                  filter_eq={"field": "orders", "value": 3})
        self.assertEqual([r["name"] for r in match_j["records"]], ["Maya", "Nora", "Zoe"])
        for bad in (True, False, 3.0, "3", [3]):
            with self.assertRaises(ValueError) as caught:
                normalize_csv(self.csv_path, self.schema,
                              filter_eq={"field": "orders", "value": bad})
            self.assertIn("filter_eq", str(caught.exception))

    def test_boolean_and_string_value_type_boundaries(self):
        with self.assertRaises(ValueError):
            normalize_csv(self.csv_path, self.schema,
                          filter_eq={"field": "active", "value": "true"})
        with self.assertRaises(ValueError):
            normalize_csv(self.csv_path, self.schema,
                          filter_eq={"field": "active", "value": 1})
        with self.assertRaises(ValueError):
            normalize_csv(self.csv_path, self.schema,
                          filter_eq={"field": "name", "value": 3})
        with self.assertRaises(ValueError):
            normalize_csv(self.csv_path, self.schema,
                          filter_eq={"field": "name", "value": True})
        # null is valid for every field type
        for field in ("name", "orders", "active"):
            normalize_csv(self.csv_path, self.schema,
                          filter_eq={"field": field, "value": None})

    def test_no_match_keeps_empty_records_with_errors_intact(self):
        result = normalize_csv(self.csv_path, self.schema,
                               filter_eq={"field": "name", "value": "nobody"})
        self.assertEqual((result["accepted"], result["filtered"], result["rejected"]), (0, 4, 1))
        self.assertEqual(result["records"], [])
        self.assertEqual(len(result["errors"]), 1)

    def test_duplicates_use_retained_records_renumbered_from_one(self):
        # Zoe also has orders 3 but is filtered out; Liam is rejected; only
        # Maya and Nora remain as duplicate positions 1 and 2.
        result = normalize_csv(self.csv_path, self.schema, duplicate_by=["orders"],
                               filter_eq={"field": "active", "value": True})
        self.assertEqual(result["duplicates"],
                         [{"key": {"orders": 3}, "record_numbers": [1, 2]}])
        result_j = normalize_jsonl(self.jsonl_path, self.schema, duplicate_by=["orders"],
                                   filter_eq={"field": "active", "value": True})
        self.assertEqual(result_j["duplicates"],
                         [{"key": {"orders": 3}, "record_numbers": [1, 2]}])
        # null group forms over retained JSONL rows (Omar is filtered by
        # active=true, so test it with the null filter instead)
        null_groups = normalize_jsonl(self.jsonl_path, self.schema, duplicate_by=["orders"],
                                      filter_eq={"field": "orders", "value": None})
        self.assertEqual(null_groups["duplicates"],
                         [{"key": {"orders": None}, "record_numbers": [1, 2]}])

    def test_filtered_key_only_when_enabled(self):
        self.assertNotIn("filtered", normalize_csv(self.csv_path, self.schema))
        self.assertNotIn("filtered", normalize_csv(self.csv_path, self.schema, filter_eq=None))
        enabled_zero = normalize_csv(self.csv_path, self.schema,
                                     filter_eq={"field": "active", "value": True})
        self.assertIn("filtered", enabled_zero)

    def test_invalid_config_raises_before_reading_input(self):
        missing = ROOT / "samples" / "does-not-exist.csv"
        valid_object_cases = [
            {"field": "active"},                       # missing value
            {"value": True},                           # missing field
            {},                                       # both missing
            {"field": "active", "value": True, "x": 1},  # extra key
            {"field": 3, "value": 1},                 # non-string field
            {"field": "  ", "value": None},           # blank field
            {"field": "nope", "value": None},         # unknown field
            {"field": "orders", "value": True},       # bool for integer
            {"field": "orders", "value": 3.5},        # float for integer
            {"field": "active", "value": "true"},     # string for boolean
            {"field": "name", "value": 1},            # integer for string
        ]
        for condition in valid_object_cases:
            with self.subTest(condition=condition):
                with self.assertRaises(ValueError) as caught:
                    normalize_csv(missing, self.schema, filter_eq=condition)
                message = str(caught.exception)
                self.assertTrue("filter_eq" in message or str(condition.get("field", "")) in message,
                                message)
        # non-dict conditions
        for condition in ([], "active", 3, True, [{"field": "active", "value": True}]):
            with self.subTest(condition=condition):
                with self.assertRaises(ValueError):
                    normalize_csv(missing, self.schema, filter_eq=condition)
        # field matches schema name literally; source aliases are unknown
        with self.assertRaises(ValueError) as caught:
            normalize_csv(missing, self.mapped_schema(),
                          filter_eq={"field": "purchase_count", "value": 3})
        self.assertIn("purchase_count", str(caught.exception))

    def test_repeated_runs_are_identical(self):
        kwargs = {"duplicate_by": ["orders"], "filter_eq": {"field": "active", "value": True}}
        first = normalize_jsonl(self.jsonl_path, self.schema, **kwargs)
        second = normalize_jsonl(self.jsonl_path, self.schema, **kwargs)
        self.assertEqual(first, second)

    def test_cli_filter_summary_outputs_and_exit_codes(self):
        for fmt, path in (("csv", self.csv_path), ("jsonl", self.jsonl_path)):
            with self.subTest(fmt=fmt):
                output = Path(self.temp.name) / f"out-{fmt}.jsonl"
                errors = Path(self.temp.name) / f"err-{fmt}.jsonl"
                run = run_cli(path, output, errors, fmt=fmt,
                              extra=["--filter-eq", json.dumps({"field": "active", "value": True})])
                self.assertEqual(run.returncode, 1, run.stderr)
                summary = json.loads(run.stdout)
                self.assertEqual(summary["rejected"], 1)
                self.assertEqual(summary["filtered"], 2)
                self.assertEqual(summary["accepted"], 2 if fmt == "csv" else 3)
                self.assertEqual(len(output.read_text().splitlines()), summary["accepted"])
                self.assertEqual(len(errors.read_text().splitlines()), 1)
                self.assertNotIn("duplicates", summary)

    def test_cli_all_filtered_no_errors_exit_zero_empty_files(self):
        clean = Path(self.temp.name) / "clean.csv"
        clean.write_text("name,orders,active\nA,1,true\nB,2,true\n", encoding="utf-8")
        output = Path(self.temp.name) / "all-out.jsonl"
        errors = Path(self.temp.name) / "all-err.jsonl"
        run = run_cli(clean, output, errors,
                      extra=["--filter-eq", json.dumps({"field": "active", "value": False})])
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout),
                         {"accepted": 0, "rejected": 0, "filtered": 2})
        self.assertEqual(output.read_text(encoding="utf-8"), "")
        self.assertEqual(errors.read_text(encoding="utf-8"), "")

    def test_cli_all_filtered_with_row_errors_exit_one(self):
        output = Path(self.temp.name) / "none-out.jsonl"
        errors = Path(self.temp.name) / "none-err.jsonl"
        run = run_cli(self.csv_path, output, errors,
                      extra=["--filter-eq", json.dumps({"field": "name", "value": "nobody"})])
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual(json.loads(run.stdout),
                         {"accepted": 0, "rejected": 1, "filtered": 4})
        self.assertEqual(output.read_text(encoding="utf-8"), "")
        self.assertEqual(len(errors.read_text().splitlines()), 1)

    def test_cli_filter_with_duplicate_by_renumbers_retained(self):
        output = Path(self.temp.name) / "dup-out.jsonl"
        errors = Path(self.temp.name) / "dup-err.jsonl"
        run = run_cli(self.csv_path, output, errors,
                      extra=["--duplicate-by", "orders",
                             "--filter-eq", json.dumps({"field": "active", "value": True})])
        self.assertEqual(run.returncode, 1, run.stderr)
        summary = json.loads(run.stdout)
        self.assertEqual(summary["accepted"], 2)
        self.assertEqual(summary["filtered"], 2)
        self.assertEqual(summary["duplicates"],
                         [{"key": {"orders": 3}, "record_numbers": [1, 2]}])

    def test_cli_bad_filter_eq_exit_two_creates_nothing(self):
        conditions = ["{not json", "true", "false", "null", "42", '"active"', "[1, 2]",
                      json.dumps({"field": "active"}),
                      json.dumps({"field": "active", "value": True, "x": 1}),
                      json.dumps({"field": "nope", "value": None}),
                      json.dumps({"field": "orders", "value": True}),
                      json.dumps({"field": "  ", "value": None})]
        for index, raw in enumerate(conditions):
            with self.subTest(raw=raw):
                output = Path(self.temp.name) / f"bad-out-{index}.jsonl"
                errors = Path(self.temp.name) / f"bad-err-{index}.jsonl"
                run = run_cli(self.csv_path, output, errors, extra=["--filter-eq", raw])
                self.assertEqual(run.returncode, 2, run.stderr)
                payload = json.loads(run.stdout)
                self.assertEqual(set(payload), {"error"})
                self.assertTrue(payload["error"].strip())
                self.assertFalse(output.exists())
                self.assertFalse(errors.exists())

    def test_cli_bad_filter_eq_preserves_existing_outputs(self):
        output = Path(self.temp.name) / "keep-out.jsonl"
        errors = Path(self.temp.name) / "keep-err.jsonl"
        output.write_text("keep records\n", encoding="utf-8")
        errors.write_text("keep errors\n", encoding="utf-8")
        run = run_cli(self.csv_path, output, errors,
                      extra=["--filter-eq", json.dumps({"field": "nope", "value": None})])
        self.assertEqual(run.returncode, 2)
        self.assertEqual(output.read_text(encoding="utf-8"), "keep records\n")
        self.assertEqual(errors.read_text(encoding="utf-8"), "keep errors\n")

    def test_cli_filter_reruns_are_identical(self):
        output = Path(self.temp.name) / "rerun-out.jsonl"
        errors = Path(self.temp.name) / "rerun-err.jsonl"
        extra = ["--duplicate-by", "orders",
                 "--filter-eq", json.dumps({"field": "active", "value": True})]
        first = run_cli(self.csv_path, output, errors, extra=extra)
        out_after_first, err_after_first = output.read_bytes(), errors.read_bytes()
        second = run_cli(self.csv_path, output, errors, extra=extra)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(output.read_bytes(), out_after_first)
        self.assertEqual(errors.read_bytes(), err_after_first)


class AtomicExportTests(unittest.TestCase):
    """The two CLI outputs commit together or not at all."""

    def trio_run(self, directory, fmt, *, output=None, errors=None, preseed_output=False):
        source = ROOT / ("samples/trio.csv" if fmt == "csv" else "samples/trio.jsonl")
        output = output or Path(directory) / "data.jsonl"
        errors = errors or Path(directory) / "errors.jsonl"
        old_bytes = b"previous output bytes\n"
        if preseed_output:
            output.write_bytes(old_bytes)
        run = run_cli(source, output, errors, fmt=fmt)
        return run, output, errors, old_bytes if preseed_output else None

    def assert_error_payload(self, run, target):
        self.assertEqual(run.returncode, 2, run.stderr)
        payload = json.loads(run.stdout)
        self.assertEqual(set(payload), {"error"})
        self.assertIsInstance(payload["error"], str)
        self.assertTrue(payload["error"].strip())
        self.assertIn(str(target), payload["error"])

    def assert_unchanged(self, path, old_bytes):
        self.assertEqual(path.read_bytes(), old_bytes)

    def test_normal_exports_for_both_formats(self):
        for fmt, bad_row in (("csv", 4), ("jsonl", 3)):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                run, output, errors, _ = self.trio_run(directory, fmt, preseed_output=True)
                self.assertEqual(run.returncode, 1, run.stderr)
                self.assertEqual(json.loads(run.stdout), {"accepted": 2, "rejected": 1})
                records = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual(records, [
                    {"active": True, "name": "Maya", "orders": 3},
                    {"active": False, "name": "Omar", "orders": None},
                ])
                error_rows = [json.loads(line) for line in errors.read_text().splitlines()]
                self.assertEqual(len(error_rows), 1)
                self.assertEqual(error_rows[0]["row"], bad_row)
                self.assertTrue(any("orders" in message for message in error_rows[0]["errors"]))
                self.assertEqual(hidden_entries(directory), [])

    def test_success_replaces_existing_targets(self):
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                run, output, errors, _ = self.trio_run(directory, fmt, preseed_output=True)
                errors.write_text("stale errors\n", encoding="utf-8")
                # Re-run: both pre-existing targets are replaced wholesale.
                run = run_cli(ROOT / ("samples/trio.csv" if fmt == "csv" else "samples/trio.jsonl"),
                              output, errors, fmt=fmt)
                self.assertEqual(run.returncode, 1, run.stderr)
                self.assertEqual(len(output.read_text().splitlines()), 2)
                fresh_errors = [json.loads(line) for line in errors.read_text().splitlines()]
                self.assertTrue(fresh_errors[0]["errors"][0].startswith("orders:"))
                self.assertEqual(hidden_entries(directory), [])

    def test_missing_errors_parent_preserves_existing_output(self):
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                run, output, errors, old_bytes = self.trio_run(
                    directory, fmt, errors=Path(directory) / "nope" / "errors.jsonl",
                    preseed_output=True)
                self.assert_error_payload(run, errors)
                self.assert_unchanged(output, old_bytes)
                self.assertFalse(errors.exists())
                self.assertFalse(Path(directory, "nope").exists())
                self.assertEqual(hidden_entries(directory), [])

    def test_missing_errors_parent_leaves_absent_output_absent(self):
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                run, output, errors, _ = self.trio_run(
                    directory, fmt, errors=Path(directory) / "nope" / "errors.jsonl")
                self.assert_error_payload(run, errors)
                self.assertFalse(output.exists())
                self.assertFalse(errors.exists())
                self.assertFalse(Path(directory, "nope").exists())
                self.assertEqual(hidden_entries(directory), [])

    def test_missing_output_parent_preserves_both_targets(self):
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                output = Path(directory) / "nope" / "data.jsonl"
                errors = Path(directory) / "errors.jsonl"
                errors.write_bytes(b"old errors\n")
                run = run_cli(ROOT / ("samples/trio.csv" if fmt == "csv" else "samples/trio.jsonl"),
                              output, errors, fmt=fmt)
                self.assert_error_payload(run, output)
                self.assertFalse(output.exists())
                self.assertEqual(errors.read_bytes(), b"old errors\n")
                self.assertFalse(Path(directory, "nope").exists())
                self.assertEqual(hidden_entries(directory), [])

    def test_directory_targets_are_failures_without_side_effects(self):
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                source = ROOT / ("samples/trio.csv" if fmt == "csv" else "samples/trio.jsonl")
                # records target is a directory; errors target absent
                output_dir = Path(directory) / "data.jsonl"
                output_dir.mkdir()
                errors = Path(directory) / "errors.jsonl"
                run = run_cli(source, output_dir, errors, fmt=fmt)
                self.assert_error_payload(run, output_dir)
                self.assertTrue(output_dir.is_dir())
                self.assertFalse(errors.exists())
                # errors target is a directory while records target exists
                output = Path(directory) / "records.jsonl"
                output.write_bytes(b"keep records\n")
                errors_dir = Path(directory) / "bad-errors"
                errors_dir.mkdir()
                run = run_cli(source, output, errors_dir, fmt=fmt)
                self.assert_error_payload(run, errors_dir)
                self.assertEqual(output.read_bytes(), b"keep records\n")
                self.assertTrue(errors_dir.is_dir())
                self.assertEqual(hidden_entries(directory), [])

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root bypasses permission bits")
    def test_unwritable_existing_target_is_preserved(self):
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                source = ROOT / ("samples/trio.csv" if fmt == "csv" else "samples/trio.jsonl")
                output = Path(directory) / "data.jsonl"
                output.write_bytes(b"locked\n")
                output.chmod(0o444)
                errors = Path(directory) / "errors.jsonl"
                try:
                    run = run_cli(source, output, errors, fmt=fmt)
                    self.assert_error_payload(run, output)
                    self.assertEqual(output.read_bytes(), b"locked\n")
                    self.assertFalse(errors.exists())
                finally:
                    output.chmod(0o644)

    def test_source_and_schema_unchanged_by_failed_export(self):
        source = ROOT / "samples/trio.csv"
        schema = ROOT / "samples/schema.json"
        source_before, schema_before = source.read_bytes(), schema.read_bytes()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            run = run_cli(source, Path(directory) / "data.jsonl",
                          Path(directory) / "nope" / "errors.jsonl")
            self.assertEqual(run.returncode, 2)
        self.assertEqual(source.read_bytes(), source_before)
        self.assertEqual(schema.read_bytes(), schema_before)

    def test_write_jsonl_independent_semantics_unchanged(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "plain.jsonl"
            write_jsonl(path, [{"b": 1, "a": None}, {"a": "x", "b": 2}])
            self.assertEqual(path.read_text(encoding="utf-8"),
                             '{"a": null, "b": 1}\n{"a": "x", "b": 2}\n')


if __name__ == "__main__":
    unittest.main()
