import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from data_importer import normalize_csv, normalize_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parent


def run_cli(source, output, errors, fmt=None, extra=(), output_format=None):
    command = [sys.executable, str(ROOT / "data_importer.py"), str(source),
               "--schema", str(ROOT / "samples/schema.json"),
               "--output", str(output), "--errors", str(errors)]
    if fmt is not None:
        command += ["--format", fmt]
    if output_format is not None:
        command += ["--output-format", output_format]
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

    def test_filter_eq_keeps_matches_and_counts_filtered(self):
        # Three valid rows, two match active=true; a fourth row is invalid
        # and must still reach errors with its physical line number.
        text = ("name,orders,active\n"
                "a,1,true\n"
                "b,,true\n"
                "c,3,false\n"
                "d,x,true\n")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, text)
            result = normalize_csv(csv_path, self.schema,
                                   filter_eq={"field": "active", "value": True})
        self.assertEqual((result["accepted"], result["filtered"], result["rejected"]), (2, 1, 1))
        self.assertEqual([record["name"] for record in result["records"]], ["a", "b"])
        self.assertEqual(result["errors"][0]["row"], 5)
        self.assertTrue(all(record["active"] is True for record in result["records"]))

    def test_filter_eq_null_matches_only_normalized_empty(self):
        text = ("name,orders,active\n"
                "a,,true\n"
                "b,0,true\n"
                "c,,false\n")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, text)
            null_result = normalize_csv(csv_path, self.schema,
                                        filter_eq={"field": "orders", "value": None})
            zero_result = normalize_csv(csv_path, self.schema,
                                        filter_eq={"field": "orders", "value": 0})
        self.assertEqual([r["name"] for r in null_result["records"]], ["a", "c"])
        self.assertEqual(null_result["filtered"], 1)
        self.assertEqual([r["name"] for r in zero_result["records"]], ["b"])
        self.assertEqual(zero_result["filtered"], 2)

    def test_filter_eq_string_exact_case_and_padding(self):
        text = ("name,orders,active\n"
                "Maya,1,true\n"
                "maya,2,true\n")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, text)
            exact = normalize_csv(csv_path, self.schema,
                                  filter_eq={"field": "name", "value": "Maya"})
            folded = normalize_csv(csv_path, self.schema,
                                   filter_eq={"field": "name", "value": "maya"})
            padded = normalize_csv(csv_path, self.schema,
                                   filter_eq={"field": "name", "value": " Maya "})
        self.assertEqual([r["name"] for r in exact["records"]], ["Maya"])
        self.assertEqual([r["name"] for r in folded["records"]], ["maya"])
        self.assertEqual(padded["accepted"], 0)
        self.assertEqual(padded["filtered"], 2)

    def test_filter_eq_no_match_keeps_blank_line_order(self):
        text = "name,orders,active\na,1,true\n\nb,2,true\n"
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, text)
            result = normalize_csv(csv_path, self.schema,
                                   filter_eq={"field": "orders", "value": 9})
        self.assertEqual((result["accepted"], result["filtered"], result["rejected"]), (0, 2, 0))
        self.assertEqual(result["records"], [])
        self.assertEqual(result["errors"], [])

    def test_filter_eq_with_duplicate_by_renumbers_kept_records(self):
        text = ("name,orders,active\n"
                "a,3,true\n"
                "b,3,false\n"
                "c,3,true\n"
                "bad,x,true\n"
                "d,3,true\n")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, text)
            result = normalize_csv(csv_path, self.schema, duplicate_by=["orders"],
                                   filter_eq={"field": "active", "value": True})
        self.assertEqual((result["accepted"], result["filtered"], result["rejected"]), (3, 1, 1))
        self.assertEqual([r["name"] for r in result["records"]], ["a", "c", "d"])
        self.assertEqual(result["duplicates"],
                         [{"key": {"orders": 3}, "record_numbers": [1, 2, 3]}])

    def test_filter_eq_disabled_or_none_unchanged(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "name,orders,active\na,1,true\nb,2,false\n")
            omitted = normalize_csv(csv_path, self.schema)
            explicit_none = normalize_csv(csv_path, self.schema, filter_eq=None)
        self.assertEqual(omitted, explicit_none)
        self.assertEqual(set(omitted), {"records", "errors", "accepted", "rejected"})
        self.assertNotIn("filtered", omitted)

    def test_filter_eq_invalid_config_raises_before_reading(self):
        missing = ROOT / "samples" / "does-not-exist.csv"
        cases = [
            [], "x", 3, True,
            {"field": "active"},
            {"value": True},
            {},
            {"field": "active", "value": True, "extra": 1},
            {"field": 1, "value": True},
            {"field": "  ", "value": True},
            {"field": "nope", "value": True},
            {"field": "active", "value": "true"},
            {"field": "active", "value": 1},
            {"field": "orders", "value": True},
            {"field": "orders", "value": 1.0},
            {"field": "orders", "value": "1"},
            {"field": "name", "value": 1},
            {"field": "name", "value": None, "extra": 0},
        ]
        for condition in cases:
            with self.assertRaises(ValueError) as caught:
                normalize_csv(missing, self.schema, filter_eq=condition)
            self.assertIn("filter_eq", str(caught.exception))
        # source aliases are not recognized as field names
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "display_name,purchase_count,active\nMaya,3,TRUE\n")
            with self.assertRaises(ValueError) as caught:
                normalize_csv(csv_path, self.mapped_schema(),
                              filter_eq={"field": "purchase_count", "value": 3})
            self.assertIn("filter_eq", str(caught.exception))

    def test_cli_filter_eq_flag(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(
                directory,
                "name,orders,active\n"
                "a,1,true\n"
                "b,,true\n"
                "c,3,false\n"
                "d,x,true\n")
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            run = run_cli(csv_path, output, errors,
                          extra=("--filter-eq", json.dumps({"field": "active", "value": True})))
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(json.loads(run.stdout),
                             {"accepted": 2, "rejected": 1, "filtered": 1})
            self.assertEqual(len(output.read_text().splitlines()), 2)
            self.assertEqual(len(errors.read_text().splitlines()), 1)

    def test_cli_filter_eq_all_filtered_exit_zero_empty_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            run = run_cli(ROOT / "samples/customers.csv", output, errors,
                          extra=("--filter-eq", json.dumps({"field": "name", "value": "nobody"})))
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout),
                             {"accepted": 0, "rejected": 0, "filtered": 2})
            self.assertEqual(output.read_text(encoding="utf-8"), "")
            self.assertEqual(errors.read_text(encoding="utf-8"), "")

    def test_cli_filter_eq_bad_condition_exit_two_keeps_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "name,orders,active\nA,3,TRUE\n")
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            for condition in ("{bad json", "[1, 2]", json.dumps({"field": "nope", "value": 1}),
                              json.dumps({"field": "active"})):
                if output.exists():
                    output.unlink()
                output.write_text("keep me\n", encoding="utf-8")
                self.assertFalse(errors.exists())
                run = run_cli(csv_path, output, errors, extra=("--filter-eq", condition))
                self.assertEqual(run.returncode, 2, condition)
                payload = json.loads(run.stdout)
                self.assertEqual(set(payload), {"error"})
                self.assertTrue(payload["error"].strip())
                self.assertIn("filter_eq", payload["error"])
                self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")
                self.assertFalse(errors.exists())

    def test_cli_filter_eq_is_deterministic_across_runs(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(
                directory,
                "name,orders,active\n"
                "a,3,true\nb,3,false\nc,,true\n")
            outputs = []
            condition = json.dumps({"field": "orders", "value": None})
            for suffix in (1, 2):
                output, errors = Path(directory) / f"data{suffix}.jsonl", Path(directory) / f"errors{suffix}.jsonl"
                run = run_cli(csv_path, output, errors, extra=("--filter-eq", condition))
                outputs.append((run.stdout, output.read_bytes(), errors.read_bytes()))
            self.assertEqual(outputs[0], outputs[1])


    def test_default_fills_empty_for_optional_and_required_csv(self):
        schema = {"columns": [
            {"name": "name", "type": "string", "required": True, "default": "  Nobody "},
            {"name": "orders", "type": "integer", "default": 0},
            {"name": "active", "type": "boolean", "required": True, "default": False},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(
                directory,
                "name,orders,active\n"
                " , ,  \n"
                "Maya,3,true\n")
            result = normalize_csv(csv_path, schema)
        self.assertEqual((result["accepted"], result["rejected"]), (2, 0))
        self.assertEqual(result["records"][0], {"name": "Nobody", "orders": 0, "active": False})
        self.assertEqual(result["records"][1], {"name": "Maya", "orders": 3, "active": True})

    def test_default_nonempty_invalid_value_is_not_masked_csv(self):
        schema = {"columns": [
            {"name": "name", "type": "string", "default": "Nobody"},
            {"name": "orders", "type": "integer", "default": 3},
            {"name": "active", "type": "boolean", "default": True},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(
                directory,
                "name,orders,active\n"
                "a,bad,true\n"
                ",,\n"
                "b,3,true\n")
            result = normalize_csv(csv_path, schema)
        self.assertEqual((result["accepted"], result["rejected"]), (2, 1))
        self.assertEqual(result["errors"][0]["row"], 2)
        self.assertTrue(any("orders" in message for message in result["errors"][0]["errors"]))
        self.assertEqual([record["name"] for record in result["records"]], ["Nobody", "b"])

    def test_default_does_not_relax_cell_count_csv(self):
        schema = {"columns": [
            {"name": "name", "type": "string", "default": "Nobody"},
            {"name": "orders", "type": "integer", "default": 3},
            {"name": "active", "type": "boolean", "default": True},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "name,orders,active\nMaya,3\n")
            result = normalize_csv(csv_path, schema)
        self.assertEqual(result["rejected"], 1)
        self.assertEqual(result["errors"][0]["errors"], ["wrong number of cells"])

    def test_default_feeds_filter_and_duplicates_csv(self):
        schema = {"columns": [
            {"name": "name", "type": "string", "required": True},
            {"name": "orders", "type": "integer", "default": 3},
            {"name": "active", "type": "boolean", "required": True},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(
                directory,
                "name,orders,active\n"
                "a,3,true\n"
                "b,,true\n"
                "bad,x,true\n"
                "c,3,true\n")
            filtered = normalize_csv(csv_path, schema,
                                     filter_eq={"field": "orders", "value": 3})
            duplicates = normalize_csv(csv_path, schema, duplicate_by=["orders"])
        self.assertEqual([r["name"] for r in filtered["records"]], ["a", "b", "c"])
        self.assertEqual((filtered["accepted"], filtered["filtered"], filtered["rejected"]), (3, 0, 1))
        self.assertEqual(duplicates["duplicates"],
                         [{"key": {"orders": 3}, "record_numbers": [1, 2, 3]}])

    def test_invalid_default_raises_before_reading_csv(self):
        missing = ROOT / "samples" / "does-not-exist.csv"
        bad_defaults = [
            ("string", 3), ("string", True), ("string", None), ("string", ""),
            ("string", "   "), ("string", ["x"]), ("string", {"x": 1}),
            ("integer", "3"), ("integer", 3.0), ("integer", True), ("integer", None),
            ("integer", [3]),
            ("boolean", "true"), ("boolean", 1), ("boolean", 0), ("boolean", None),
            ("boolean", {"x": 1}),
        ]
        for kind, default in bad_defaults:
            schema = {"columns": [{"name": "f", "type": kind, "default": default}]}
            with self.assertRaises(ValueError) as caught:
                normalize_csv(missing, schema)
            message = str(caught.exception)
            self.assertIn("f", message, (kind, default))
            self.assertIn("default", message, (kind, default))
            self.assertIn(repr(default), message, (kind, default))
        # valid zero/false defaults are accepted
        for kind, default in (("integer", 0), ("boolean", False)):
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                csv_path = self.write_csv(directory, "f\n \n")
                result = normalize_csv(
                    csv_path, {"columns": [{"name": "f", "type": kind, "default": default}]})
                self.assertEqual(result["records"], [{"f": default}])

    def test_schema_dict_unchanged_with_defaults_csv(self):
        import copy
        schema = {"columns": [
            {"name": "name", "type": "string", "required": True, "default": "  Maya "},
            {"name": "orders", "type": "integer", "default": 3},
            {"name": "active", "type": "boolean", "required": True},
        ]}
        snapshot = copy.deepcopy(schema)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "name,orders,active\n,,true\n")
            normalize_csv(csv_path, schema)
        self.assertEqual(schema, snapshot)

    def test_cli_bad_default_exit_two_keeps_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "name,orders,active\nA,3,TRUE\n")
            schema_path = Path(directory) / "schema.json"
            schema_path.write_text(json.dumps({"columns": [
                {"name": "name", "type": "string", "required": True},
                {"name": "orders", "type": "integer", "default": "3"},
                {"name": "active", "type": "boolean", "required": True},
            ]}), encoding="utf-8")
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            output.write_text("keep me\n", encoding="utf-8")
            command = [sys.executable, str(ROOT / "data_importer.py"), str(csv_path),
                       "--schema", str(schema_path), "--output", str(output),
                       "--errors", str(errors)]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            payload = json.loads(run.stdout)
            self.assertEqual(set(payload), {"error"})
            self.assertTrue(payload["error"].strip())
            self.assertIn("orders", payload["error"])
            self.assertIn("default", payload["error"])
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")
            self.assertFalse(errors.exists())

    def enum_schema(self):
        return {"columns": [
            {"name": "name", "type": "string", "required": True},
            {"name": "orders", "type": "integer", "allowed_values": [0, 3], "default": 3},
            {"name": "active", "type": "boolean", "required": True},
        ]}

    def test_allowed_values_enum_default_and_type_error_csv(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(
                directory,
                "name,orders,active\n"
                "a,,true\n"
                'b,"3",true\n'
                "c,0,false\n"
                "d,4,true\n"
                "e,bad,true\n")
            result = normalize_csv(csv_path, self.enum_schema())
        self.assertEqual((result["accepted"], result["rejected"]), (3, 2))
        self.assertEqual([record["orders"] for record in result["records"]], [3, 3, 0])
        self.assertEqual([row["row"] for row in result["errors"]], [5, 6])
        self.assertEqual(result["errors"][0]["errors"],
                         ["orders: value 4 is not one of allowed_values [0, 3]"])
        type_error = result["errors"][1]["errors"]
        self.assertEqual(len(type_error), 1)
        self.assertNotIn("allowed_values", type_error[0])

    def test_allowed_values_null_required_and_schema_order_csv(self):
        schema = {"columns": [
            {"name": "o", "type": "integer", "allowed_values": [1, 2]},
            {"name": "r", "type": "string", "required": True, "allowed_values": ["x"]},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "o,r\n,x\n9,9\n")
            result = normalize_csv(csv_path, schema)
        self.assertEqual(result["records"], [{"o": None, "r": "x"}])
        self.assertEqual(result["errors"][0]["errors"], [
            "o: value 9 is not one of allowed_values [1, 2]",
            "r: value '9' is not one of allowed_values ['x']",
        ])

    def test_allowed_values_does_not_fix_cell_count_csv(self):
        schema = {"columns": [
            {"name": "a", "type": "integer", "allowed_values": [1]},
            {"name": "b", "type": "integer", "allowed_values": [2]},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "a,b\n1\n")
            result = normalize_csv(csv_path, schema)
        self.assertEqual(result["errors"][0]["errors"], ["wrong number of cells"])

    def test_allowed_values_string_and_boolean_verbatim_csv(self):
        strings = {"columns": [{"name": "f", "type": "string",
                                "allowed_values": ["Maya", "x y"]}]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "f\nMaya\nmaya\nx y\n")
            result = normalize_csv(csv_path, strings)
        self.assertEqual([record["f"] for record in result["records"]], ["Maya", "x y"])
        self.assertEqual(result["rejected"], 1)
        booleans = {"columns": [{"name": "f", "type": "boolean",
                                 "allowed_values": [True]}]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "f\ntrue\nfalse\n")
            result = normalize_csv(csv_path, booleans)
        self.assertEqual(result["records"], [{"f": True}])
        self.assertEqual(result["rejected"], 1)

    def test_allowed_values_precedes_filter_and_duplicates_csv(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(
                directory,
                "name,orders,active\n"
                "a,3,true\n"
                "b,,true\n"
                "d,4,true\n")
            filtered = normalize_csv(csv_path, self.enum_schema(),
                                     filter_eq={"field": "orders", "value": 3})
            duplicates = normalize_csv(csv_path, self.enum_schema(),
                                       duplicate_by=["orders"])
        self.assertEqual([r["name"] for r in filtered["records"]], ["a", "b"])
        self.assertEqual((filtered["accepted"], filtered["filtered"], filtered["rejected"]),
                         (2, 0, 1))
        self.assertEqual(duplicates["duplicates"],
                         [{"key": {"orders": 3}, "record_numbers": [1, 2]}])

    def test_invalid_allowed_values_raises_before_reading_csv(self):
        import copy
        missing = ROOT / "samples" / "does-not-exist.csv"
        bad_attributes = [("integer", []), ("integer", "x"), ("integer", True),
                          ("integer", {}), ("integer", None)]
        bad_entries = [
            ("integer", [1, True]), ("integer", [1, None]), ("integer", [1, [2]]),
            ("integer", [1, {"a": 1}]), ("integer", [1, 1.0]), ("integer", [1, 1]),
            ("string", ["a", "a"]), ("string", [1]), ("string", [None]),
            ("string", [["a"]]),
            ("boolean", [True, 1]), ("boolean", [False, 0]), ("boolean", [None]),
        ]
        for kind, values in bad_attributes:
            schema = {"columns": [{"name": "f", "type": kind, "allowed_values": values}]}
            with self.assertRaises(ValueError) as caught:
                normalize_csv(missing, schema)
            message = str(caught.exception)
            self.assertIn("f", message, (kind, values))
            self.assertIn("allowed_values", message, (kind, values))
            self.assertIn(repr(values), message, (kind, values))
        for kind, values in bad_entries:
            schema = {"columns": [{"name": "f", "type": kind, "allowed_values": values}]}
            with self.assertRaises(ValueError) as caught:
                normalize_csv(missing, schema)
            message = str(caught.exception)
            self.assertIn("f", message, (kind, values))
            self.assertIn("allowed_values", message, (kind, values))
            # the offending entry is named verbatim
            self.assertTrue(any(repr(entry) in message for entry in values),
                            (kind, values, message))
        # a processed default outside the list is rejected with no input read
        for schema in (
                {"columns": [{"name": "f", "type": "integer",
                              "default": 4, "allowed_values": [0, 3]}]},
                {"columns": [{"name": "f", "type": "string",
                              "default": " x ", "allowed_values": [" x "]}]}):
            with self.assertRaises(ValueError) as caught:
                normalize_csv(missing, schema)
            self.assertIn("allowed_values", str(caught.exception))
        # the processed default inside the list is fine
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "f\n \n")
            schema = {"columns": [{"name": "f", "type": "string",
                                   "default": " x ", "allowed_values": ["x"]}]}
            self.assertEqual(normalize_csv(csv_path, schema)["records"], [{"f": "x"}])
        # caller's schema is never mutated
        schema = self.enum_schema()
        snapshot = copy.deepcopy(schema)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "name,orders,active\nd,4,true\n")
            normalize_csv(csv_path, schema)
        self.assertEqual(schema, snapshot)

    def test_cli_bad_allowed_values_exit_two_keeps_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "name,orders,active\nA,3,TRUE\n")
            schema_path = Path(directory) / "schema.json"
            schema_path.write_text(json.dumps({"columns": [
                {"name": "name", "type": "string", "required": True},
                {"name": "orders", "type": "integer", "allowed_values": [0, 0]},
                {"name": "active", "type": "boolean", "required": True},
            ]}), encoding="utf-8")
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            output.write_text("keep me\n", encoding="utf-8")
            command = [sys.executable, str(ROOT / "data_importer.py"), str(csv_path),
                       "--schema", str(schema_path), "--output", str(output),
                       "--errors", str(errors)]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            payload = json.loads(run.stdout)
            self.assertEqual(set(payload), {"error"})
            self.assertTrue(payload["error"].strip())
            self.assertIn("orders", payload["error"])
            self.assertIn("allowed_values", payload["error"])
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")
            self.assertFalse(errors.exists())

    def test_cli_defaults_applied_and_exported(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(
                directory, "name,orders,active\n,,true\nMaya,,false\n")
            schema_path = Path(directory) / "schema.json"
            schema_path.write_text(json.dumps({"columns": [
                {"name": "name", "type": "string", "required": True, "default": "Nobody"},
                {"name": "orders", "type": "integer", "default": 3},
                {"name": "active", "type": "boolean", "required": True},
            ]}), encoding="utf-8")
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            command = [sys.executable, str(ROOT / "data_importer.py"), str(csv_path),
                       "--schema", str(schema_path), "--output", str(output),
                       "--errors", str(errors)]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout), {"accepted": 2, "rejected": 0})
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(records, [
                {"active": True, "name": "Nobody", "orders": 3},
                {"active": False, "name": "Maya", "orders": 3},
            ])


class CsvQuoteSyntaxTests(unittest.TestCase):
    """A quoted field left open at EOF or junk after a closing quote is a
    fatal parse error for the whole CSV, not silently folded into data."""

    UNTERMINATED = "unterminated"
    AFTER_CLOSE = "after closing quote"

    def setUp(self):
        self.schema = json.loads((ROOT / "samples/schema.json").read_text())

    def write_csv(self, directory, text):
        path = Path(directory) / "data.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def write_csv_bytes(self, directory, data):
        path = Path(directory) / "data.csv"
        path.write_bytes(data)
        return path

    def assert_parse_failure(self, path, start_line, reason_part):
        with self.assertRaises(ValueError) as caught:
            normalize_csv(path, self.schema)
        message = str(caught.exception)
        self.assertIn("CSV parse error", message)
        self.assertIn(str(path), message)
        self.assertIn(f"record starting line {start_line}", message)
        self.assertIn(reason_part, message)
        return message

    def test_unterminated_quoted_field_at_eof(self):
        # The exact motivating input: the final quoted field never closes.
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, 'name,orders,active\nMaya,3,"true')
            message = self.assert_parse_failure(csv_path, 2, self.UNTERMINATED)
            self.assertNotIn(self.AFTER_CLOSE, message)

    def test_text_after_closing_quote_is_fatal(self):
        # "Ma"ya must not collapse into the name; letters, spaces and tabs
        # right after a closing quote are all illegal.
        cases = (
            ('name,orders,active\n"Ma"ya,3,true\n', 2),
            ('name,orders,active\n"Ma" ya,3,true\n', 2),
            ('name,orders,active\n"Ma"\tya,3,true\n', 2),
            ('name,orders,active\nMaya,3,"true"x', 2),
            ('name,orders,active\nMaya,3,"true" ,\n', 2),
            ('name,orders,active\nMaya,3,true\n"x"y,2,false\n', 3),
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            for text, line in cases:
                csv_path = self.write_csv(directory, text)
                message = self.assert_parse_failure(csv_path, line, self.AFTER_CLOSE)
                self.assertNotIn(self.UNTERMINATED, message, text)

    def test_error_line_points_to_multiline_record_start(self):
        # Even though the problem is discovered on a later physical line
        # (or at EOF), the message names where the record began. Blank
        # lines and newlines inside quoted fields all occupy line numbers.
        unterminated = ("name,orders,active\n"
                        "Maya,3,true\n"
                        "\n"
                        '"a\nb",3,true\n'
                        '"x\ny,3,true')
        after_close = ("name,orders,active\n"
                       "\n\n"
                       '"a\nb",3,true\n'
                       '"x\ny"z,3,true\n')
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, unterminated)
            self.assert_parse_failure(csv_path, 6, self.UNTERMINATED)
            csv_path = self.write_csv(directory, after_close)
            self.assert_parse_failure(csv_path, 6, self.AFTER_CLOSE)

    def test_quote_errors_after_valid_and_ordinary_error_rows(self):
        # Earlier accepted records and ordinary row errors cannot make the
        # later quote failure a row error: the call raises, returning
        # nothing at all.
        for text in ("name,orders,active\nMaya,3,true\nBad,x,true\n\"oops,3,true\n",
                     "name,orders,active\nMaya,3,true\nBad,x,true\n\"Ma\"ya,3,true\n"):
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                csv_path = self.write_csv(directory, text)
                self.assert_parse_failure(csv_path, 4, "quote")

    def test_quote_errors_in_header_report_line_one(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, 'name,orders,"active')
            self.assert_parse_failure(csv_path, 1, self.UNTERMINATED)
            csv_path = self.write_csv(directory, 'name,orders,"active"x\nMaya,3,true\n')
            self.assert_parse_failure(csv_path, 1, self.AFTER_CLOSE)

    def test_valid_commas_newlines_and_doubled_quotes(self):
        text = ('name,orders,active\n'
                '"Ma,ya",3,true\n'
                '"Ma""ya",2,true\n'
                '"No\nra",4,false\n')
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, text)
            result = normalize_csv(csv_path, self.schema)
        self.assertEqual((result["accepted"], result["rejected"]), (3, 0))
        self.assertEqual([record["name"] for record in result["records"]],
                         ["Ma,ya", 'Ma"ya', "No\nra"])

    def test_valid_doubled_quote_and_eof_without_newline(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, 'name,orders,active\n"Ma""ya",3,true')
            result = normalize_csv(csv_path, self.schema)
        self.assertEqual(result["records"], [{"name": 'Ma"ya', "orders": 3, "active": True}])

    def test_line_endings_bom_and_no_final_newline(self):
        variants = (
            b'name,orders,active\r\nMaya,3,"true',
            b'name,orders,active\rMaya,3,"true',
            b'\xef\xbb\xbfname,orders,active\nMaya,3,"true',
            b'name,orders,active\r\n"Ma"ya,3,true\r\n',
            b'name,orders,active\r"Ma"ya,3,true\r',
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            for data in variants:
                csv_path = self.write_csv_bytes(directory, data)
                with self.assertRaises(ValueError) as caught:
                    normalize_csv(csv_path, self.schema)
                self.assertIn("CSV parse error", str(caught.exception), data)
            # valid quoted multiline fields under every line convention,
            # BOM and a missing final newline keep their old semantics:
            # line bytes inside a quoted field are retained verbatim.
            for data, expected_name in (
                    (b'name,orders,active\r\n"Ma\r\nya",3,true\r\n', "Ma\r\nya"),
                    (b'name,orders,active\r"Ma\rya",3,true\r', "Ma\rya"),
                    (b'\xef\xbb\xbfname,orders,active\n"Maya",3,true\n', "Maya"),
                    (b'name,orders,active\n"Maya",3,true', "Maya")):
                csv_path = self.write_csv_bytes(directory, data)
                result = normalize_csv(csv_path, self.schema)
                self.assertEqual(result["accepted"], 1, data)
                self.assertEqual(result["records"][0]["name"], expected_name, data)

    def test_unquoted_double_quote_stays_literal(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, 'name,orders,active\nMa"ya,3,true\n')
            result = normalize_csv(csv_path, self.schema)
        self.assertEqual(result["records"], [{"name": 'Ma"ya', "orders": 3, "active": True}])

    def test_cli_quote_errors_exit_two_and_keep_outputs(self):
        bad_texts = {
            "unterminated": "name,orders,active\nMaya,3,true\nBad,x,true\n\"oops,3,true\n",
            "after": "name,orders,active\nMaya,3,true\nBad,x,true\n\"Ma\"ya,3,true\n",
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            for kind, text in bad_texts.items():
                csv_path = self.write_csv(directory, text)
                for fmt in (None, "csv"):
                    output.write_bytes(b"previous output bytes\n")
                    errors.write_bytes(b"previous error bytes\n")
                    run = run_cli(csv_path, output, errors, fmt=fmt)
                    self.assertEqual(run.returncode, 2, (kind, run.stderr))
                    payload = json.loads(run.stdout)
                    self.assertEqual(set(payload), {"error"})
                    self.assertTrue(payload["error"].strip())
                    self.assertIn("CSV parse error", payload["error"])
                    self.assertIn(str(csv_path), payload["error"])
                    self.assertIn("record starting line 4", payload["error"])
                    self.assertIn(kind if kind == "unterminated" else self.AFTER_CLOSE,
                                  payload["error"])
                    self.assertEqual(output.read_bytes(), b"previous output bytes\n")
                    self.assertEqual(errors.read_bytes(), b"previous error bytes\n")
            # absent outputs stay absent and no success summary is printed
            output.unlink()
            errors.unlink()
            run = run_cli(csv_path, output, errors)
            self.assertEqual(run.returncode, 2)
            self.assertEqual(set(json.loads(run.stdout)), {"error"})
            self.assertFalse(output.exists() or errors.exists())

    def test_cli_quote_rules_apply_to_csv_format_only(self):
        # The same characters as JSONL remain an ordinary per-line parse
        # error: exit 1, other lines processed, both outputs written.
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            jsonl_path = Path(directory) / "data.jsonl"
            jsonl_path.write_text('{"name": "Maya", "orders": 3, "active": true}\n'
                                  'Maya,3,"true\n', encoding="utf-8")
            output, errors = Path(directory) / "out.jsonl", Path(directory) / "err.jsonl"
            run = run_cli(jsonl_path, output, errors, fmt="jsonl")
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(json.loads(run.stdout), {"accepted": 1, "rejected": 1})
            self.assertEqual(len(output.read_text().splitlines()), 1)
            self.assertEqual(len(errors.read_text().splitlines()), 1)


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

    def test_filter_eq_jsonl_matches_counts_and_error_rows(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(
                directory,
                '{"name": "a", "orders": 1, "active": true}\n'
                '{"name": "b", "orders": null, "active": true}\n'
                '{"name": "c", "orders": 3, "active": false}\n'
                'not json\n')
            result = normalize_jsonl(path, self.schema,
                                    filter_eq={"field": "active", "value": True})
        self.assertEqual((result["accepted"], result["filtered"], result["rejected"]), (2, 1, 1))
        self.assertEqual([r["name"] for r in result["records"]], ["a", "b"])
        self.assertEqual(result["errors"][0]["row"], 4)
        self.assertTrue(result["errors"][0]["errors"][0].startswith("parse error"))

    def test_filter_eq_jsonl_null_and_integer_type_boundaries(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(
                directory,
                '{"name": "a", "orders": null, "active": true}\n'
                '{"name": "b", "orders": 0, "active": true}\n'
                '{"name": "c", "orders": -8, "active": true}\n'
                '{"name": "d", "orders": "", "active": true}\n')
            null_rows = normalize_jsonl(path, self.schema,
                                        filter_eq={"field": "orders", "value": None})
            zero_rows = normalize_jsonl(path, self.schema,
                                        filter_eq={"field": "orders", "value": 0})
            minus_rows = normalize_jsonl(path, self.schema,
                                         filter_eq={"field": "orders", "value": -8})
        self.assertEqual([r["name"] for r in null_rows["records"]], ["a", "d"])
        self.assertEqual(null_rows["filtered"], 2)
        self.assertEqual([r["name"] for r in zero_rows["records"]], ["b"])
        self.assertEqual([r["name"] for r in minus_rows["records"]], ["c"])

    def test_filter_eq_jsonl_string_and_boolean_boundaries(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(
                directory,
                '{"name": "Maya", "orders": null, "active": true}\n'
                '{"name": "maya", "orders": null, "active": false}\n'
                '{"name": " Maya ", "orders": null, "active": true}\n')
            upper = normalize_jsonl(path, self.schema,
                                    filter_eq={"field": "name", "value": "Maya"})
            padded = normalize_jsonl(path, self.schema,
                                     filter_eq={"field": "name", "value": " Maya "})
            falsy = normalize_jsonl(path, self.schema,
                                    filter_eq={"field": "active", "value": False})
        # trimming happens during conversion, but the condition itself is never trimmed
        self.assertEqual([r["name"] for r in upper["records"]], ["Maya", "Maya"])
        self.assertEqual(padded["accepted"], 0)
        self.assertEqual([r["name"] for r in falsy["records"]], ["maya"])
        self.assertEqual(falsy["filtered"], 2)

    def test_filter_eq_jsonl_no_match_with_errors(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(
                directory,
                '{"name": "a", "orders": 1, "active": true}\n[]\n')
            result = normalize_jsonl(path, self.schema,
                                    filter_eq={"field": "name", "value": "zzz"})
        self.assertEqual((result["accepted"], result["filtered"], result["rejected"]), (0, 1, 1))
        self.assertEqual(result["records"], [])
        self.assertEqual(result["errors"][0]["row"], 2)

    def test_filter_eq_jsonl_with_duplicate_by_renumbers_kept(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(
                directory,
                '{"name": "a", "orders": 3, "active": true}\n'
                '{"name": "b", "orders": 3, "active": false}\n'
                '{"name": "c", "orders": 3, "active": true}\n'
                '{"name": "d", "orders": 3, "active": true}\n')
            result = normalize_jsonl(path, self.schema, duplicate_by=["orders"],
                                     filter_eq={"field": "active", "value": True})
        self.assertEqual((result["accepted"], result["filtered"], result["rejected"]), (3, 1, 0))
        self.assertEqual([r["name"] for r in result["records"]], ["a", "c", "d"])
        self.assertEqual(result["duplicates"],
                         [{"key": {"orders": 3}, "record_numbers": [1, 2, 3]}])

    def test_filter_eq_jsonl_disabled_structure_unchanged(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory, '{"name": "a", "orders": 1, "active": true}\n')
            omitted = normalize_jsonl(path, self.schema)
            explicit_none = normalize_jsonl(path, self.schema, filter_eq=None)
        self.assertEqual(omitted, explicit_none)
        self.assertEqual(set(omitted), {"records", "errors", "accepted", "rejected"})

    def test_filter_eq_jsonl_invalid_config_raises_before_reading(self):
        missing = ROOT / "samples" / "does-not-exist.jsonl"
        cases = [
            [], "x", 3, False,
            {"field": "active"},
            {"value": True},
            {},
            {"field": "active", "value": True, "extra": 1},
            {"field": None, "value": True},
            {"field": "   ", "value": True},
            {"field": "nope", "value": True},
            {"field": "active", "value": "true"},
            {"field": "active", "value": 0},
            {"field": "orders", "value": False},
            {"field": "orders", "value": 1.5},
            {"field": "orders", "value": "1"},
            {"field": "name", "value": 0},
        ]
        for condition in cases:
            with self.assertRaises(ValueError) as caught:
                normalize_jsonl(missing, self.schema, filter_eq=condition)
            self.assertIn("filter_eq", str(caught.exception))

    def test_default_fills_null_and_empty_string_jsonl(self):
        schema = {"columns": [
            {"name": "name", "type": "string", "required": True, "default": "Nobody"},
            {"name": "orders", "type": "integer", "default": 0},
            {"name": "active", "type": "boolean", "required": True, "default": False},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(
                directory,
                '{"name": null, "orders": "", "active": "  "}\n'
                '{"name": "  ", "orders": null, "active": null}\n'
                '{"name": "Maya", "orders": "3", "active": "TRUE"}\n')
            result = normalize_jsonl(path, schema)
        self.assertEqual((result["accepted"], result["rejected"]), (3, 0))
        self.assertEqual(result["records"][0], {"name": "Nobody", "orders": 0, "active": False})
        self.assertEqual(result["records"][1], {"name": "Nobody", "orders": 0, "active": False})
        self.assertEqual(result["records"][2], {"name": "Maya", "orders": 3, "active": True})

    def test_default_nonempty_invalid_value_is_not_masked_jsonl(self):
        schema = {"columns": [
            {"name": "name", "type": "string", "default": "Nobody"},
            {"name": "orders", "type": "integer", "default": 3},
            {"name": "active", "type": "boolean", "default": True},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(
                directory,
                '{"name": "a", "orders": "bad", "active": true}\n'
                '{"name": null, "orders": null, "active": null}\n'
                '{"name": "b", "orders": 4, "active": false}\n')
            result = normalize_jsonl(path, schema)
        self.assertEqual((result["accepted"], result["rejected"]), (2, 1))
        self.assertEqual(result["errors"][0]["row"], 1)
        self.assertTrue(any("orders" in message for message in result["errors"][0]["errors"]))
        self.assertEqual([record["name"] for record in result["records"]], ["Nobody", "b"])

    def test_default_does_not_relax_key_set_or_duplicate_keys_jsonl(self):
        schema = {"columns": [
            {"name": "name", "type": "string", "required": True, "default": "Nobody"},
            {"name": "orders", "type": "integer", "default": 3},
            {"name": "active", "type": "boolean", "required": True, "default": True},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(
                directory,
                '{"name": "A", "name": "B", "orders": 1, "active": true}\n'
                '{"name": "A", "active": true}\n'
                '{"name": "A", "orders": 1, "active": true, "extra": 9}\n')
            result = normalize_jsonl(path, schema)
        self.assertEqual(result["rejected"], 3)
        self.assertTrue(result["errors"][0]["errors"][0].startswith("structure error: duplicate key"))
        self.assertIn("orders", result["errors"][1]["errors"][0])
        self.assertIn("extra", result["errors"][2]["errors"][0])

    def test_default_feeds_filter_and_duplicates_jsonl(self):
        schema = {"columns": [
            {"name": "name", "type": "string", "required": True},
            {"name": "orders", "type": "integer", "default": 3},
            {"name": "active", "type": "boolean", "required": True},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(
                directory,
                '{"name": "a", "orders": "3", "active": true}\n'
                '{"name": "b", "orders": null, "active": true}\n'
                '{"name": "bad", "orders": "x", "active": true}\n'
                '{"name": "c", "orders": 3, "active": true}\n')
            filtered = normalize_jsonl(path, schema,
                                      filter_eq={"field": "orders", "value": 3})
            duplicates = normalize_jsonl(path, schema, duplicate_by=["orders"])
        self.assertEqual([r["name"] for r in filtered["records"]], ["a", "b", "c"])
        self.assertEqual((filtered["accepted"], filtered["filtered"], filtered["rejected"]), (3, 0, 1))
        self.assertEqual(duplicates["duplicates"],
                         [{"key": {"orders": 3}, "record_numbers": [1, 2, 3]}])

    def enum_schema(self):
        return {"columns": [
            {"name": "name", "type": "string", "required": True},
            {"name": "orders", "type": "integer", "allowed_values": [0, 3], "default": 3},
            {"name": "active", "type": "boolean", "required": True},
        ]}

    def test_allowed_values_enum_default_and_type_error_jsonl(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(
                directory,
                '{"name": "a", "orders": null, "active": true}\n'
                '{"name": "b", "orders": "3", "active": true}\n'
                '{"name": "c", "orders": 0, "active": false}\n'
                '{"name": "d", "orders": 4, "active": true}\n'
                '{"name": "e", "orders": "bad", "active": true}\n')
            result = normalize_jsonl(path, self.enum_schema())
        self.assertEqual((result["accepted"], result["rejected"]), (3, 2))
        self.assertEqual([record["orders"] for record in result["records"]], [3, 3, 0])
        self.assertEqual([row["row"] for row in result["errors"]], [4, 5])
        self.assertEqual(result["errors"][0]["errors"],
                         ["orders: value 4 is not one of allowed_values [0, 3]"])
        self.assertEqual(result["errors"][1]["errors"], ["orders: expected integer"])

    def test_allowed_values_null_required_and_arrays_jsonl(self):
        schema = {"columns": [
            {"name": "o", "type": "integer", "allowed_values": [1, 2]},
            {"name": "r", "type": "string", "required": True, "allowed_values": ["x"]},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            null_row = self.write_jsonl(directory, '{"o": null, "r": "x"}\n', "n.jsonl")
            result = normalize_jsonl(null_row, schema)
            self.assertEqual(result["records"], [{"o": None, "r": "x"}])
            multi = self.write_jsonl(directory, '{"o": 9, "r": "z"}\n', "m.jsonl")
            result = normalize_jsonl(multi, schema)
            self.assertEqual(result["errors"][0]["errors"], [
                "o: value 9 is not one of allowed_values [1, 2]",
                "r: value 'z' is not one of allowed_values ['x']",
            ])
            # arrays/objects keep the original type error only
            arrays = self.write_jsonl(directory, '{"o": [1], "r": "x"}\n', "a.jsonl")
            result = normalize_jsonl(arrays, schema)
            self.assertEqual(result["errors"][0]["errors"], ["o: expected integer"])

    def test_allowed_values_does_not_relax_key_set_jsonl(self):
        schema = {"columns": [
            {"name": "a", "type": "integer", "allowed_values": [1]},
            {"name": "b", "type": "integer", "allowed_values": [2]},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory, '{"a": 9}\n')
            result = normalize_jsonl(path, schema)
        self.assertTrue(result["errors"][0]["errors"][0].startswith("structure error"))

    def test_allowed_values_precedes_filter_and_duplicates_jsonl(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(
                directory,
                '{"name": "a", "orders": 3, "active": true}\n'
                '{"name": "b", "orders": null, "active": true}\n'
                '{"name": "d", "orders": 4, "active": true}\n')
            filtered = normalize_jsonl(path, self.enum_schema(),
                                      filter_eq={"field": "orders", "value": 3})
            duplicates = normalize_jsonl(path, self.enum_schema(),
                                         duplicate_by=["orders"])
        self.assertEqual([r["name"] for r in filtered["records"]], ["a", "b"])
        self.assertEqual((filtered["accepted"], filtered["filtered"], filtered["rejected"]),
                         (2, 0, 1))
        self.assertEqual(duplicates["duplicates"],
                         [{"key": {"orders": 3}, "record_numbers": [1, 2]}])

    def test_invalid_allowed_values_raises_before_reading_jsonl(self):
        missing = ROOT / "samples" / "does-not-exist.jsonl"
        for kind, values in [("integer", []), ("integer", "x"), ("integer", None),
                             ("integer", [True]), ("integer", [1, None]),
                             ("integer", [1, 1]), ("string", ["a", "a"]),
                             ("string", [1]), ("boolean", [0])]:
            schema = {"columns": [{"name": "f", "type": kind, "allowed_values": values}]}
            with self.assertRaises(ValueError) as caught:
                normalize_jsonl(missing, schema)
            message = str(caught.exception)
            self.assertIn("allowed_values", message)
            self.assertIn("f", message)

    def test_cli_jsonl_bad_allowed_values_exit_two_keeps_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory, '{"name": "A", "orders": 1, "active": true}\n',
                                    name="source.jsonl")
            schema_path = Path(directory) / "schema.json"
            schema_path.write_text(json.dumps({"columns": [
                {"name": "name", "type": "string", "required": True},
                {"name": "orders", "type": "integer", "allowed_values": [3, 3]},
                {"name": "active", "type": "boolean", "required": True},
            ]}), encoding="utf-8")
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            output.write_text("keep me\n", encoding="utf-8")
            command = [sys.executable, str(ROOT / "data_importer.py"), str(path),
                       "--schema", str(schema_path), "--output", str(output),
                       "--errors", str(errors), "--format", "jsonl"]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            payload = json.loads(run.stdout)
            self.assertEqual(set(payload), {"error"})
            self.assertTrue(payload["error"].strip())
            self.assertIn("orders", payload["error"])
            self.assertIn("allowed_values", payload["error"])
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")
            self.assertFalse(errors.exists())

    def test_invalid_default_raises_before_reading_jsonl(self):
        import copy
        missing = ROOT / "samples" / "does-not-exist.jsonl"
        bad_defaults = [
            ("string", None), ("string", 3), ("string", ""), ("string", "  "),
            ("integer", None), ("integer", "3"), ("integer", False),
            ("boolean", None), ("boolean", 0), ("boolean", "true"),
        ]
        for kind, default in bad_defaults:
            schema = {"columns": [{"name": "f", "type": kind, "default": default}]}
            with self.assertRaises(ValueError) as caught:
                normalize_jsonl(missing, schema)
            message = str(caught.exception)
            self.assertIn("f", message)
            self.assertIn("default", message)
            self.assertIn(repr(default), message)
        # caller's schema is not mutated by validation or normalization
        schema = {"columns": [
            {"name": "name", "type": "string", "required": True, "default": " Maya "},
            {"name": "orders", "type": "integer", "default": 3},
            {"name": "active", "type": "boolean", "default": True},
        ]}
        snapshot = copy.deepcopy(schema)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory, '{"name": null, "orders": null, "active": null}\n')
            normalize_jsonl(path, schema)
        self.assertEqual(schema, snapshot)

    def test_cli_jsonl_bad_default_exit_two_keeps_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory, '{"name": "A", "orders": 1, "active": true}\n',
                                    name="source.jsonl")
            schema_path = Path(directory) / "schema.json"
            schema_path.write_text(json.dumps({"columns": [
                {"name": "name", "type": "string", "required": True},
                {"name": "orders", "type": "integer", "default": True},
                {"name": "active", "type": "boolean", "required": True},
            ]}), encoding="utf-8")
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            output.write_text("keep me\n", encoding="utf-8")
            command = [sys.executable, str(ROOT / "data_importer.py"), str(path),
                       "--schema", str(schema_path), "--output", str(output),
                       "--errors", str(errors), "--format", "jsonl"]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            payload = json.loads(run.stdout)
            self.assertEqual(set(payload), {"error"})
            self.assertTrue(payload["error"].strip())
            self.assertIn("orders", payload["error"])
            self.assertIn("default", payload["error"])
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")
            self.assertFalse(errors.exists())

    def test_cli_jsonl_filter_eq_flag_and_exit_codes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(
                directory,
                '{"name": "a", "orders": 1, "active": true}\n'
                '{"name": "b", "orders": null, "active": false}\n'
                '{"name": "c", "orders": 2, "active": true}\n'
                'bad line\n',
                name="source.jsonl")
            clean = self.write_jsonl(
                directory,
                '{"name": "a", "orders": 1, "active": true}\n'
                '{"name": "b", "orders": null, "active": false}\n'
                '{"name": "c", "orders": 2, "active": true}\n',
                name="clean.jsonl")
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            command = [sys.executable, str(ROOT / "data_importer.py"), str(path),
                       "--schema", str(ROOT / "samples/schema.json"),
                       "--output", str(output), "--errors", str(errors), "--format", "jsonl",
                       "--filter-eq", json.dumps({"field": "active", "value": True})]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(json.loads(run.stdout),
                             {"accepted": 2, "rejected": 1, "filtered": 1})
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([r["name"] for r in records], ["a", "c"])
            self.assertEqual(len(errors.read_text().splitlines()), 1)
            # no match still surfaces the parse error: rejected row never hidden
            run = subprocess.run(command[:-1] + [json.dumps({"field": "name", "value": "zzz"})],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(json.loads(run.stdout),
                             {"accepted": 0, "rejected": 1, "filtered": 3})
            self.assertEqual(output.read_text(encoding="utf-8"), "")
            self.assertEqual(len(errors.read_text().splitlines()), 1)
            # clean input, all filtered: empty files and exit 0
            run = subprocess.run(
                [sys.executable, str(ROOT / "data_importer.py"), str(clean),
                 "--schema", str(ROOT / "samples/schema.json"),
                 "--output", str(output), "--errors", str(errors), "--format", "jsonl",
                 "--filter-eq", json.dumps({"field": "name", "value": "zzz"})],
                capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout),
                             {"accepted": 0, "rejected": 0, "filtered": 3})
            self.assertEqual(output.read_text(encoding="utf-8"), "")
            self.assertEqual(errors.read_text(encoding="utf-8"), "")

    def test_cli_jsonl_filter_eq_bad_condition_exit_two_keeps_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory, '{"name": "A", "orders": 1, "active": true}\n',
                                    name="source.jsonl")
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            for condition in ("", "{bad", "3", '"x"',
                              json.dumps([{"field": "active", "value": True}]),
                              json.dumps({"field": "active", "value": "true"})):
                output.write_text("keep me\n", encoding="utf-8")
                if errors.exists():
                    errors.unlink()
                command = [sys.executable, str(ROOT / "data_importer.py"), str(path),
                           "--schema", str(ROOT / "samples/schema.json"),
                           "--output", str(output), "--errors", str(errors),
                           "--format", "jsonl", "--filter-eq", condition]
                run = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(run.returncode, 2, condition)
                payload = json.loads(run.stdout)
                self.assertEqual(set(payload), {"error"})
                self.assertIn("filter_eq", payload["error"])
                self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")
                self.assertFalse(errors.exists())


class MissingValuesTests(unittest.TestCase):
    """A column may declare text markers that count as missing values."""

    def marker_schema(self, **orders_overrides):
        column = {"name": "orders", "type": "integer", "missing_values": ["N/A"]}
        column.update(orders_overrides)
        return {"columns": [
            {"name": "name", "type": "string", "required": True},
            column,
            {"name": "active", "type": "boolean", "required": True},
        ]}

    def write_source(self, directory, text, fmt):
        path = Path(directory) / ("data.csv" if fmt == "csv" else "data.jsonl")
        path.write_text(text, encoding="utf-8")
        return path

    def normalize(self, path, schema, fmt, **kwargs):
        return (normalize_csv if fmt == "csv" else normalize_jsonl)(path, schema, **kwargs)

    ENUM_ROWS = {
        "csv": ("name,orders,active\n"
                "a, N/A ,true\n"
                "b,3,true\n"
                "c,0,false\n"
                "d,4,true\n"
                "e,bad,true\n"),
        "jsonl": ('{"name": "a", "orders": " N/A ", "active": true}\n'
                  '{"name": "b", "orders": "3", "active": true}\n'
                  '{"name": "c", "orders": 0, "active": false}\n'
                  '{"name": "d", "orders": 4, "active": true}\n'
                  '{"name": "e", "orders": "bad", "active": true}\n'),
    }

    def test_marker_default_enum_and_type_error(self):
        # " N/A " and "3" both become 3 via the default, 0 stays 0,
        # 4 hits the enum and "bad" hits the integer type error.
        schema = self.marker_schema(default=3, allowed_values=[0, 3])
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, self.ENUM_ROWS[fmt], fmt)
                result = self.normalize(path, schema, fmt)
            self.assertEqual((result["accepted"], result["rejected"]), (3, 2))
            self.assertEqual([record["orders"] for record in result["records"]], [3, 3, 0])
            self.assertEqual([row["row"] for row in result["errors"]],
                             [5, 6] if fmt == "csv" else [4, 5])
            self.assertEqual(result["errors"][0]["errors"],
                             ["orders: value 4 is not one of allowed_values [0, 3]"])
            type_error = result["errors"][1]["errors"]
            self.assertEqual(len(type_error), 1)
            self.assertTrue(type_error[0].startswith("orders:"))
            self.assertNotIn("allowed_values", type_error[0])

    def test_marker_match_is_case_sensitive(self):
        # Declaring "n/a" does not match the input "N/A".
        schema = self.marker_schema(missing_values=["n/a"], default=3)
        for fmt, text in (
                ("csv", "name,orders,active\na,N/A,true\nb,n/a,true\n"),
                ("jsonl", '{"name": "a", "orders": "N/A", "active": true}\n'
                         '{"name": "b", "orders": "n/a", "active": true}\n')):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, text, fmt)
                result = self.normalize(path, schema, fmt)
            self.assertEqual((result["accepted"], result["rejected"]), (1, 1))
            self.assertEqual(result["records"], [{"name": "b", "orders": 3, "active": True}])
            self.assertEqual(result["errors"][0]["row"], 2 if fmt == "csv" else 1)
            self.assertTrue(result["errors"][0]["errors"][0].startswith("orders:"))

    def test_marker_without_default_null_or_required(self):
        optional = self.marker_schema()
        required = self.marker_schema(required=True)
        for fmt, text in (
                ("csv", "name,orders,active\na,N/A,true\nb,2,true\n"),
                ("jsonl", '{"name": "a", "orders": "N/A", "active": true}\n'
                         '{"name": "b", "orders": 2, "active": true}\n')):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, text, fmt)
                null_result = self.normalize(path, optional, fmt)
                required_result = self.normalize(path, required, fmt)
            self.assertEqual((null_result["accepted"], null_result["rejected"]), (2, 0))
            self.assertEqual([record["orders"] for record in null_result["records"]], [None, 2])
            self.assertEqual((required_result["accepted"], required_result["rejected"]), (1, 1))
            self.assertEqual(required_result["errors"][0]["row"], 2 if fmt == "csv" else 1)
            self.assertEqual(required_result["errors"][0]["errors"],
                             ["orders: required value is empty"])

    def test_marker_feeds_filter_and_duplicates(self):
        schema = self.marker_schema(default=3, allowed_values=[0, 3])
        texts = {
            "csv": ("name,orders,active\n"
                    "a,N/A,true\n"
                    "b,3,true\n"
                    "bad,x,true\n"
                    "c,0,true\n"),
            "jsonl": ('{"name": "a", "orders": "N/A", "active": true}\n'
                      '{"name": "b", "orders": "3", "active": true}\n'
                      '{"name": "bad", "orders": "x", "active": true}\n'
                      '{"name": "c", "orders": 0, "active": true}\n'),
        }
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, texts[fmt], fmt)
                filtered = self.normalize(path, schema, fmt,
                                          filter_eq={"field": "orders", "value": 3})
                duplicates = self.normalize(path, schema, fmt, duplicate_by=["orders"])
            self.assertEqual([r["name"] for r in filtered["records"]], ["a", "b"])
            self.assertEqual((filtered["accepted"], filtered["filtered"], filtered["rejected"]),
                             (2, 1, 1))
            self.assertEqual(duplicates["duplicates"],
                             [{"key": {"orders": 3}, "record_numbers": [1, 2]}])

    def test_marker_does_not_relax_structure(self):
        schema = self.marker_schema(default=3)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_source(directory, "name,orders,active\na,N/A\n", "csv")
            result = normalize_csv(csv_path, schema)
            self.assertEqual(result["errors"][0]["errors"], ["wrong number of cells"])
            jsonl_path = self.write_source(
                directory, '{"name": "a", "orders": "N/A"}\n'
                           '{"name": "a", "orders": "N/A", "orders": "N/A", "active": true}\n'
                           '{"name": "a", "orders": "N/A", "active": true, "extra": 1}\n',
                "jsonl")
            result = normalize_jsonl(jsonl_path, schema)
        self.assertEqual(result["rejected"], 3)
        self.assertTrue(result["errors"][0]["errors"][0].startswith("structure error: missing"))
        self.assertTrue(result["errors"][1]["errors"][0].startswith(
            "structure error: duplicate key"))
        self.assertTrue(result["errors"][2]["errors"][0].startswith(
            "structure error: unexpected"))

    def test_jsonl_non_string_values_never_match(self):
        schema = {"columns": [
            {"name": "i", "type": "integer", "missing_values": ["1"]},
            {"name": "b", "type": "boolean", "missing_values": ["true"]},
            {"name": "s", "type": "string", "missing_values": ["x"]},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_source(
                directory,
                # numbers and booleans are not stringified for matching
                '{"i": 1, "b": true, "s": "1"}\n'
                # the string forms do match the markers and become null
                '{"i": "1", "b": "true", "s": "x"}\n'
                # arrays never match and keep the ordinary type error
                '{"i": [1], "b": true, "s": "y"}\n'
                # null and "" keep their existing semantics
                '{"i": null, "b": "", "s": ""}\n',
                "jsonl")
            result = normalize_jsonl(path, schema)
        self.assertEqual((result["accepted"], result["rejected"]), (3, 1))
        self.assertEqual(result["records"][0], {"i": 1, "b": True, "s": "1"})
        self.assertEqual(result["records"][1], {"i": None, "b": None, "s": None})
        self.assertEqual(result["records"][2], {"i": None, "b": None, "s": None})
        self.assertEqual(result["errors"][0]["row"], 3)
        self.assertEqual(result["errors"][0]["errors"], ["i: expected integer"])

    def test_default_is_not_matched_against_markers(self):
        # A default equal to a marker is emitted verbatim, not re-marked.
        schema = {"columns": [
            {"name": "name", "type": "string",
             "default": "N/A", "missing_values": ["N/A"]},
        ]}
        for fmt, text in (("csv", "name\n \nN/A\nx\n"),
                          ("jsonl", '{"name": null}\n{"name": "N/A"}\n{"name": "x"}\n')):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, text, fmt)
                result = self.normalize(path, schema, fmt)
            self.assertEqual(result["records"],
                             [{"name": "N/A"}, {"name": "N/A"}, {"name": "x"}])

    def test_invalid_missing_values_raises_before_reading(self):
        import copy
        missing = {"csv": ROOT / "samples" / "does-not-exist.csv",
                   "jsonl": ROOT / "samples" / "does-not-exist.jsonl"}
        cases = [
            ("x", "x"), (3, 3), (True, True), (None, None), ({}, {}), ([], []),
            ([1], 1), ([True], True), ([None], None), ([["x"]], ["x"]),
            ([{"x": 1}], {"x": 1}), ([""], ""), (["  "], "  "),
            (["N/A", "N/A"], "N/A"), ([" N/A ", "N/A"], "N/A"),
            (["n/a", "N/A", "n/a"], "n/a"),
        ]
        for fmt in ("csv", "jsonl"):
            normalize = normalize_csv if fmt == "csv" else normalize_jsonl
            for values, offending in cases:
                with self.subTest(fmt=fmt, values=values):
                    schema = {"columns": [
                        {"name": "orders", "type": "integer", "missing_values": values}]}
                    snapshot = copy.deepcopy(schema)
                    with self.assertRaises(ValueError) as caught:
                        normalize(missing[fmt], schema)
                    message = str(caught.exception)
                    self.assertIn("missing_values", message)
                    self.assertIn("orders", message)
                    self.assertIn(repr(offending), message)
                    self.assertEqual(schema, snapshot)
        # distinct case-folded markers are fine; the missing file fails next
        schema = {"columns": [
            {"name": "orders", "type": "integer", "missing_values": ["N/A", "n/a"]}]}
        with self.assertRaises(OSError):
            normalize_csv(missing["csv"], schema)

    def test_cli_bad_missing_values_exit_two_keeps_files(self):
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                source = Path(directory) / f"source.{fmt}"
                source.write_text(
                    "name,orders,active\nA,3,true\n" if fmt == "csv"
                    else '{"name": "A", "orders": 3, "active": true}\n',
                    encoding="utf-8")
                schema_path = Path(directory) / "schema.json"
                schema_path.write_text(json.dumps({"columns": [
                    {"name": "name", "type": "string", "required": True},
                    {"name": "orders", "type": "integer", "missing_values": ["N/A", " N/A "]},
                    {"name": "active", "type": "boolean", "required": True},
                ]}), encoding="utf-8")
                output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
                output.write_text("keep me\n", encoding="utf-8")
                command = [sys.executable, str(ROOT / "data_importer.py"), str(source),
                           "--schema", str(schema_path), "--output", str(output),
                           "--errors", str(errors), "--format", fmt]
                run = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(run.returncode, 2, run.stderr)
                payload = json.loads(run.stdout)
                self.assertEqual(set(payload), {"error"})
                self.assertTrue(payload["error"].strip())
                self.assertIn("orders", payload["error"])
                self.assertIn("missing_values", payload["error"])
                self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")
                self.assertFalse(errors.exists())


class IntegerRangeTests(unittest.TestCase):
    """An integer column may declare inclusive minimum/maximum bounds."""

    def ranged_schema(self, **orders_overrides):
        column = {"name": "orders", "type": "integer",
                  "minimum": 0, "maximum": 5, "default": 3,
                  "missing_values": ["N/A"]}
        column.update(orders_overrides)
        return {"columns": [
            {"name": "name", "type": "string", "required": True},
            column,
            {"name": "active", "type": "boolean", "required": True},
        ]}

    def write_source(self, directory, text, fmt):
        path = Path(directory) / ("data.csv" if fmt == "csv" else "data.jsonl")
        path.write_text(text, encoding="utf-8")
        return path

    def normalize(self, path, schema, fmt, **kwargs):
        return (normalize_csv if fmt == "csv" else normalize_jsonl)(path, schema, **kwargs)

    RANGE_ROWS = {
        "csv": ("name,orders,active\n"
                "a,0,true\n"
                "b,3,true\n"
                "c,5,true\n"
                "d,-1,true\n"
                "e,6,true\n"
                "f,bad,true\n"
                "g, N/A ,true\n"),
        "jsonl": ('{"name": "a", "orders": "0", "active": true}\n'
                  '{"name": "b", "orders": 3, "active": true}\n'
                  '{"name": "c", "orders": "5", "active": true}\n'
                  '{"name": "d", "orders": -1, "active": true}\n'
                  '{"name": "e", "orders": 6, "active": true}\n'
                  '{"name": "f", "orders": "bad", "active": true}\n'
                  '{"name": "g", "orders": " N/A ", "active": true}\n'),
    }

    def test_range_bounds_default_and_marker(self):
        # 0, 3, 5 and the marker (defaulted to 3) are accepted; -1 and 6
        # are out of range and "bad" keeps only its integer type error.
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, self.RANGE_ROWS[fmt], fmt)
                result = self.normalize(path, self.ranged_schema(), fmt)
            self.assertEqual((result["accepted"], result["rejected"]), (4, 3))
            self.assertEqual([record["orders"] for record in result["records"]],
                             [0, 3, 5, 3])
            self.assertEqual([row["row"] for row in result["errors"]],
                             [5, 6, 7] if fmt == "csv" else [4, 5, 6])
            self.assertEqual(result["errors"][0]["errors"],
                             ["orders: value -1 is less than minimum 0"])
            self.assertEqual(result["errors"][1]["errors"],
                             ["orders: value 6 is greater than maximum 5"])
            type_error = result["errors"][2]["errors"]
            self.assertEqual(len(type_error), 1)
            self.assertTrue(type_error[0].startswith("orders:"))
            self.assertNotIn("minimum", type_error[0])
            self.assertNotIn("maximum", type_error[0])

    def test_range_feeds_filter_and_duplicates(self):
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, self.RANGE_ROWS[fmt], fmt)
                combined = self.normalize(path, self.ranged_schema(), fmt,
                                          duplicate_by=["orders"],
                                          filter_eq={"field": "orders", "value": 3})
            # out-of-range rows are rejected, never filtered
            self.assertEqual((combined["accepted"], combined["filtered"],
                              combined["rejected"]), (2, 2, 3))
            self.assertEqual([record["orders"] for record in combined["records"]], [3, 3])
            self.assertEqual(combined["duplicates"],
                             [{"key": {"orders": 3}, "record_numbers": [1, 2]}])

    def test_single_ended_and_equal_bounds(self):
        cases = (
            ({"minimum": -2}, [-2, -1, 0, 99], [-3]),
            ({"maximum": 2}, [-99, 0, 2], [3]),
            ({"minimum": 4, "maximum": 4}, [4], [3, 5]),
            ({"minimum": 0, "maximum": 0}, [0], [-1, 1]),
        )
        for bounds, accepted_values, rejected_values in cases:
            with self.subTest(bounds=bounds):
                schema = {"columns": [{"name": "f", "type": "integer", **bounds}]}
                lines = [str(value) for value in accepted_values + rejected_values]
                with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                    csv_path = self.write_source(directory, "f\n" + "\n".join(lines) + "\n", "csv")
                    result = normalize_csv(csv_path, schema)
                self.assertEqual([record["f"] for record in result["records"]],
                                 accepted_values)
                self.assertEqual(result["rejected"], len(rejected_values))

    def test_null_and_required_empty_ignore_bounds(self):
        optional = {"columns": [{"name": "f", "type": "integer", "minimum": 2}]}
        required = {"columns": [{"name": "f", "type": "integer",
                                 "required": True, "maximum": 2}]}
        for fmt, empty_cell, low, high in (
                ("csv", " ", "1", "3"), ("jsonl", "null", "1", "3")):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                optional_path = self.write_source(
                    directory, (f"f\n{empty_cell}\n{low}\n" if fmt == "csv"
                                else f'{{"f": {empty_cell}}}\n{{"f": {low}}}\n'), fmt)
                null_result = self.normalize(optional_path, optional, fmt)
                required_path = self.write_source(
                    directory, (f"f\n{empty_cell}\n{high}\n" if fmt == "csv"
                                else f'{{"f": {empty_cell}}}\n{{"f": {high}}}\n'), fmt)
                required_result = self.normalize(required_path, required, fmt)
            # an empty optional stays null and is never range-checked
            self.assertEqual(null_result["records"], [{"f": None}])
            self.assertEqual(null_result["errors"][0]["errors"],
                             [f"f: value {low} is less than minimum 2"])
            # an empty required keeps its ordinary error
            self.assertEqual(required_result["errors"][0]["errors"],
                             ["f: required value is empty"])
            self.assertEqual(required_result["errors"][1]["errors"],
                             [f"f: value {high} is greater than maximum 2"])

    def test_enum_and_type_failures_keep_only_their_error(self):
        schema = {"columns": [
            {"name": "o", "type": "integer",
             "allowed_values": [1, 9], "minimum": 0, "maximum": 5},
            {"name": "r", "type": "integer", "minimum": 0},
        ]}
        # 9 passes the enum but violates the range; 8 violates the enum
        # (its range violation is not reported); "x" keeps the type error.
        for fmt, text in (
                ("csv", "o,r\n9,1\n8,1\nx,1\n"),
                ("jsonl", '{"o": 9, "r": 1}\n{"o": 8, "r": 1}\n{"o": "x", "r": 1}\n')):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, text, fmt)
                result = self.normalize(path, schema, fmt)
            self.assertEqual((result["accepted"], result["rejected"]), (0, 3))
            self.assertEqual(result["errors"][0]["errors"],
                             ["o: value 9 is greater than maximum 5"])
            self.assertEqual(result["errors"][1]["errors"],
                             ["o: value 8 is not one of allowed_values [1, 9]"])
            self.assertEqual(len(result["errors"][2]["errors"]), 1)
            self.assertTrue(result["errors"][2]["errors"][0].startswith("o:"))

    def test_enum_entries_out_of_range_are_not_a_config_error(self):
        # An allowed_values entry outside the range is legal; records
        # converted to it still fail the range check.
        schema = {"columns": [{"name": "f", "type": "integer",
                               "allowed_values": [1, 9], "maximum": 5}]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_source(directory, "f\n1\n9\n", "csv")
            result = normalize_csv(csv_path, schema)
        self.assertEqual(result["records"], [{"f": 1}])
        self.assertEqual(result["errors"][0]["errors"],
                         ["f: value 9 is greater than maximum 5"])

    def test_multi_field_errors_in_schema_order(self):
        schema = {"columns": [
            {"name": "a", "type": "integer", "minimum": 0},
            {"name": "b", "type": "integer", "maximum": 0},
        ]}
        for fmt, text in (("csv", "a,b\n-1,2\n"),
                          ("jsonl", '{"a": -1, "b": 2}\n')):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, text, fmt)
                result = self.normalize(path, schema, fmt)
            self.assertEqual(result["errors"][0]["errors"], [
                "a: value -1 is less than minimum 0",
                "b: value 2 is greater than maximum 0",
            ])

    def test_invalid_range_config_raises_before_reading(self):
        import copy
        missing = {"csv": ROOT / "samples" / "does-not-exist.csv",
                   "jsonl": ROOT / "samples" / "does-not-exist.jsonl"}
        bad_columns = [
            # bounds must be non-boolean integers
            {"name": "f", "type": "integer", "minimum": True},
            {"name": "f", "type": "integer", "maximum": False},
            {"name": "f", "type": "integer", "minimum": None},
            {"name": "f", "type": "integer", "maximum": None},
            {"name": "f", "type": "integer", "minimum": "0"},
            {"name": "f", "type": "integer", "maximum": "5"},
            {"name": "f", "type": "integer", "minimum": 1.0},
            {"name": "f", "type": "integer", "maximum": 2.5},
            {"name": "f", "type": "integer", "minimum": [0]},
            # bounds on non-integer columns
            {"name": "f", "type": "string", "minimum": 0},
            {"name": "f", "type": "string", "maximum": 5},
            {"name": "f", "type": "boolean", "minimum": 0},
            {"name": "f", "type": "boolean", "maximum": 5},
            # minimum above maximum
            {"name": "f", "type": "integer", "minimum": 6, "maximum": 5},
            # default out of range, even though no record would use it
            {"name": "f", "type": "integer", "default": -1, "minimum": 0},
            {"name": "f", "type": "integer", "default": 6, "maximum": 5},
            {"name": "f", "type": "integer", "default": 4,
             "minimum": 0, "maximum": 3},
        ]
        for fmt in ("csv", "jsonl"):
            normalize = normalize_csv if fmt == "csv" else normalize_jsonl
            for column in bad_columns:
                with self.subTest(fmt=fmt, column=column):
                    schema = {"columns": [column]}
                    snapshot = copy.deepcopy(schema)
                    with self.assertRaises(ValueError) as caught:
                        normalize(missing[fmt], schema)
                    message = str(caught.exception)
                    self.assertIn("f", message)
                    attribute = "maximum" if "maximum" in column else "minimum"
                    if "default" in column and not (
                            "minimum" in column and "maximum" in column
                            and column["minimum"] > column["maximum"]):
                        attribute = "default"
                    self.assertIn(attribute, message)
                    self.assertEqual(schema, snapshot)
        # negative and zero bounds are legal; the missing file fails next
        schema = {"columns": [{"name": "f", "type": "integer",
                               "minimum": -10, "maximum": 0, "default": -3}]}
        with self.assertRaises(OSError):
            normalize_csv(missing["csv"], schema)

    def test_cli_bad_range_exit_two_keeps_files(self):
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                source = Path(directory) / f"source.{fmt}"
                source.write_text(
                    "name,orders,active\nA,3,true\n" if fmt == "csv"
                    else '{"name": "A", "orders": 3, "active": true}\n',
                    encoding="utf-8")
                schema_path = Path(directory) / "schema.json"
                schema_path.write_text(json.dumps({"columns": [
                    {"name": "name", "type": "string", "required": True},
                    {"name": "orders", "type": "integer", "minimum": 5, "maximum": 0},
                    {"name": "active", "type": "boolean", "required": True},
                ]}), encoding="utf-8")
                output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
                output.write_text("keep me\n", encoding="utf-8")
                command = [sys.executable, str(ROOT / "data_importer.py"), str(source),
                           "--schema", str(schema_path), "--output", str(output),
                           "--errors", str(errors), "--format", fmt]
                run = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(run.returncode, 2, run.stderr)
                payload = json.loads(run.stdout)
                self.assertEqual(set(payload), {"error"})
                self.assertTrue(payload["error"].strip())
                self.assertIn("orders", payload["error"])
                self.assertIn("minimum", payload["error"])
                self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")
                self.assertFalse(errors.exists())

    def test_cli_range_row_errors_exit_one_and_export(self):
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                source = self.write_source(directory, self.RANGE_ROWS[fmt], fmt)
                schema_path = Path(directory) / "schema.json"
                schema_path.write_text(json.dumps(self.ranged_schema()), encoding="utf-8")
                output, errors = Path(directory) / "records.jsonl", Path(directory) / "errors.jsonl"
                command = [sys.executable, str(ROOT / "data_importer.py"), str(source),
                           "--schema", str(schema_path), "--output", str(output),
                           "--errors", str(errors), "--format", fmt]
                run = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(run.returncode, 1, run.stderr)
                self.assertEqual(json.loads(run.stdout), {"accepted": 4, "rejected": 3})
                records = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual([record["orders"] for record in records], [0, 3, 5, 3])
                error_rows = [json.loads(line) for line in errors.read_text().splitlines()]
                self.assertEqual(len(error_rows), 3)
                self.assertEqual(hidden_entries(directory), [])


class LteFieldRelationTests(unittest.TestCase):
    """An integer column may declare lte_field naming another integer
    output column; its final value must not exceed the target's."""

    def lohi_schema(self, **lo_overrides):
        lo = {"name": "lo", "type": "integer", "lte_field": "hi"}
        lo.update(lo_overrides)
        return {"columns": [lo, {"name": "hi", "type": "integer"}]}

    def write_source(self, directory, text, fmt):
        path = Path(directory) / ("data.csv" if fmt == "csv" else "data.jsonl")
        path.write_text(text, encoding="utf-8")
        return path

    def normalize(self, path, schema, fmt, **kwargs):
        return (normalize_csv if fmt == "csv" else normalize_jsonl)(path, schema, **kwargs)

    ROWS = {
        "csv": "lo,hi\n1,3\n4,3\n,3\n2,2\nbad,3\n",
        "jsonl": ('{"lo": 1, "hi": 3}\n'
                  '{"lo": 4, "hi": 3}\n'
                  '{"lo": null, "hi": 3}\n'
                  '{"lo": 2, "hi": 2}\n'
                  '{"lo": "bad", "hi": 3}\n'),
    }

    def test_five_record_scenario_both_formats(self):
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, self.ROWS[fmt], fmt)
                result = self.normalize(path, self.lohi_schema(), fmt)
            self.assertEqual((result["accepted"], result["rejected"]), (3, 2))
            self.assertEqual(result["records"], [
                {"lo": 1, "hi": 3},
                {"lo": None, "hi": 3},
                {"lo": 2, "hi": 2},
            ])
            self.assertEqual([row["row"] for row in result["errors"]],
                             [3, 6] if fmt == "csv" else [2, 5])
            self.assertEqual(result["errors"][0]["errors"],
                             ["lo: value 4 is greater than lte_field 'hi' value 3"])
            type_error = result["errors"][1]["errors"]
            self.assertEqual(len(type_error), 1)
            self.assertTrue(type_error[0].startswith("lo:"))
            self.assertNotIn("lte_field", type_error[0])

    def test_equality_is_legal(self):
        for fmt, text in (("csv", "lo,hi\n5,5\n0,0\n-2,-2\n"),
                          ("jsonl", '{"lo": 5, "hi": 5}\n{"lo": 0, "hi": 0}\n'
                                    '{"lo": -2, "hi": -2}\n')):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, text, fmt)
                result = self.normalize(path, self.lohi_schema(), fmt)
            self.assertEqual((result["accepted"], result["rejected"]), (3, 0), fmt)
            self.assertEqual(result["errors"], [])

    def test_null_on_either_side_skips_the_relation(self):
        schema = {"columns": [
            {"name": "lo", "type": "integer", "lte_field": "hi"},
            {"name": "hi", "type": "integer"},
        ]}
        for fmt, text in (("csv", "lo,hi\n9,\n,9\n,\n"),
                          ("jsonl", '{"lo": 9, "hi": null}\n{"lo": null, "hi": 9}\n'
                                    '{"lo": null, "hi": null}\n')):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, text, fmt)
                result = self.normalize(path, schema, fmt)
            self.assertEqual((result["accepted"], result["rejected"]), (3, 0), fmt)

    def test_defaults_participate_in_the_relation(self):
        schema = {"columns": [
            {"name": "lo", "type": "integer", "default": 9, "lte_field": "hi"},
            {"name": "hi", "type": "integer"},
        ]}
        for fmt, text in (("csv", "lo,hi\n,1\n,\n3,3\n"),
                          ("jsonl", '{"lo": null, "hi": 1}\n{"lo": null, "hi": null}\n'
                                    '{"lo": 3, "hi": 3}\n')):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, text, fmt)
                result = self.normalize(path, schema, fmt)
            self.assertEqual((result["accepted"], result["rejected"]), (2, 1), fmt)
            self.assertEqual(result["errors"][0]["errors"],
                             ["lo: value 9 is greater than lte_field 'hi' value 1"])
            self.assertEqual(result["records"], [{"lo": 9, "hi": None},
                                                 {"lo": 3, "hi": 3}])

    def test_defaults_on_both_sides_make_a_row_error_not_a_config_error(self):
        # Two defaults standing in a violating relation are legal config;
        # the violation surfaces per row, with no extra default checks.
        schema = {"columns": [
            {"name": "lo", "type": "integer", "default": 9, "lte_field": "hi"},
            {"name": "hi", "type": "integer", "default": 0},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_source(directory, "lo,hi\n,\n1,1\n", "csv")
            result = normalize_csv(csv_path, schema)
        self.assertEqual((result["accepted"], result["rejected"]), (1, 1))
        self.assertEqual(result["errors"][0]["errors"],
                         ["lo: value 9 is greater than lte_field 'hi' value 0"])

    def test_relation_skips_participant_required_type_enum_range_errors(self):
        schema = {"columns": [
            {"name": "lo", "type": "integer", "maximum": 5,
             "allowed_values": [1, 4, 7], "lte_field": "hi"},
            {"name": "hi", "type": "integer", "required": True,
             "allowed_values": [3, 4]},
        ]}
        # 7 passes conversion+enum but fails lo's own range, and 4 > 3:
        # the lo range error is kept and the relation skipped.
        # bad hi keeps its type error and the relation is skipped.
        # empty hi keeps its required error and the relation is skipped.
        # 4/4 equal is fine; 1/3 is fine.
        for fmt, text in (("csv", "lo,hi\n7,3\n4,x\n4,\n4,4\n1,3\n"),
                          ("jsonl", '{"lo": 7, "hi": 3}\n{"lo": 4, "hi": "x"}\n'
                                    '{"lo": 4, "hi": null}\n{"lo": 4, "hi": 4}\n'
                                    '{"lo": 1, "hi": 3}\n')):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, text, fmt)
                result = self.normalize(path, schema, fmt)
            self.assertEqual((result["accepted"], result["rejected"]), (2, 3), fmt)
            for row in result["errors"]:
                self.assertEqual(len(row["errors"]), 1)
                self.assertNotIn("lte_field", row["errors"][0])

    def test_other_relations_still_checked_when_one_is_skipped(self):
        schema = {"columns": [
            {"name": "a", "type": "integer", "lte_field": "b"},
            {"name": "b", "type": "integer"},
            {"name": "c", "type": "integer", "lte_field": "b"},
        ]}
        for fmt, text in (("csv", "a,b,c\nx,1,8\n"),
                          ("jsonl", '{"a": "x", "b": 1, "c": 8}\n')):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, text, fmt)
                result = self.normalize(path, schema, fmt)
            self.assertEqual(result["errors"][0]["errors"], [
                "a: invalid literal for int() with base 10: 'x'"
                if fmt == "csv" else "a: expected integer",
                "c: value 8 is greater than lte_field 'b' value 1",
            ])

    def test_errors_filed_in_schema_column_order(self):
        # a (column 1) points forward at b (column 2); a's relation error
        # must still precede a type error on a later column.
        schema = {"columns": [
            {"name": "a", "type": "integer", "lte_field": "b"},
            {"name": "b", "type": "integer"},
            {"name": "c", "type": "integer", "lte_field": "b"},
        ]}
        for fmt, text in (("csv", "a,b,c\n9,1,8\n"),
                          ("jsonl", '{"a": 9, "b": 1, "c": 8}\n')):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, text, fmt)
                result = self.normalize(path, schema, fmt)
            self.assertEqual(result["errors"][0]["errors"], [
                "a: value 9 is greater than lte_field 'b' value 1",
                "c: value 8 is greater than lte_field 'b' value 1",
            ])

    def test_reference_direction_independent_of_declaration_order(self):
        schema = {"columns": [
            {"name": "hi", "type": "integer"},
            {"name": "mid", "type": "integer", "lte_field": "hi"},
            {"name": "lo", "type": "integer", "lte_field": "mid"},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_source(
                directory, "hi,mid,lo\n3,2,1\n0,2,1\n", "csv")
            result = normalize_csv(path, schema)
        self.assertEqual((result["accepted"], result["rejected"]), (1, 1))
        self.assertEqual(result["errors"][0]["errors"],
                         ["mid: value 2 is greater than lte_field 'hi' value 0"])
        self.assertEqual(result["records"], [{"hi": 3, "mid": 2, "lo": 1}])

    def test_relation_precedes_filter_and_duplicates(self):
        # A violating row is rejected even when the filter matches it, and
        # duplicate_by counts only the records surviving filtering.
        rows = ("lo,hi\n1,3\n4,3\n1,9\n0,9\n"
                , '{"lo": 1, "hi": 3}\n{"lo": 4, "hi": 3}\n'
                  '{"lo": 1, "hi": 9}\n{"lo": 0, "hi": 9}\n')
        for fmt, text in zip(("csv", "jsonl"), rows):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, text, fmt)
                result = self.normalize(
                    path, self.lohi_schema(), fmt,
                    filter_eq={"field": "hi", "value": 3})
            self.assertEqual((result["accepted"], result["filtered"],
                              result["rejected"]), (1, 2, 1), fmt)
            self.assertEqual(result["records"], [{"lo": 1, "hi": 3}])
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_source(directory, rows[0], "csv")
            result = normalize_csv(path, self.lohi_schema(), duplicate_by=["lo"])
        # (4,3) is rejected; the two valid lo=1 records form the only group.
        self.assertEqual(result["duplicates"],
                         [{"key": {"lo": 1}, "record_numbers": [1, 2]}])

    def test_later_rows_keep_being_processed(self):
        for fmt, text in (("csv", "lo,hi\n4,3\n1,1\n4,3\n"),
                          ("jsonl", '{"lo": 4, "hi": 3}\n{"lo": 1, "hi": 1}\n'
                                    '{"lo": 4, "hi": 3}\n')):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, text, fmt)
                result = self.normalize(path, self.lohi_schema(), fmt)
            self.assertEqual((result["accepted"], result["rejected"]), (1, 2), fmt)

    def test_invalid_lte_field_config_raises_before_reading(self):
        import copy
        missing = {"csv": ROOT / "samples" / "does-not-exist.csv",
                   "jsonl": ROOT / "samples" / "does-not-exist.jsonl"}
        bad_columns = [
            {"name": "lo", "type": "integer", "lte_field": 3},
            {"name": "lo", "type": "integer", "lte_field": True},
            {"name": "lo", "type": "integer", "lte_field": None},
            {"name": "lo", "type": "integer", "lte_field": ["hi"]},
            {"name": "lo", "type": "integer", "lte_field": ""},
            {"name": "lo", "type": "integer", "lte_field": "   "},
            {"name": "lo", "type": "integer", "lte_field": "nope"},
            {"name": "lo", "type": "integer", "lte_field": "lo"},
            {"name": "lo", "type": "integer", "lte_field": "s"},
            {"name": "s", "type": "string", "lte_field": "lo"},
            {"name": "lo", "type": "boolean", "lte_field": "hi"},
        ]
        for fmt in ("csv", "jsonl"):
            normalize = normalize_csv if fmt == "csv" else normalize_jsonl
            for column in bad_columns:
                others = [c for c in (
                    {"name": "lo", "type": "integer"},
                    {"name": "hi", "type": "integer"},
                    {"name": "s", "type": "string"},
                ) if c["name"] != column["name"]]
                schema = {"columns": [column] + others}
                with self.subTest(fmt=fmt, column=column):
                    snapshot = copy.deepcopy(schema)
                    with self.assertRaises(ValueError) as caught:
                        normalize(missing[fmt], schema)
                    message = str(caught.exception)
                    self.assertIn("lte_field", message)
                    self.assertIn(column["name"], message)
                    self.assertIn(repr(column["lte_field"]), message)
                    self.assertEqual(schema, snapshot)

    def test_source_alias_is_not_a_lte_field_target(self):
        missing = ROOT / "samples" / "does-not-exist.csv"
        schema = {"columns": [
            {"name": "lo", "type": "integer", "source": "low", "lte_field": "high"},
            {"name": "hi", "type": "integer", "source": "high"},
        ]}
        with self.assertRaises(ValueError) as caught:
            normalize_csv(missing, schema)
        self.assertIn("lte_field", str(caught.exception))
        # the output name is accepted and processing reaches file open
        schema["columns"][0]["lte_field"] = "hi"
        with self.assertRaises(OSError):
            normalize_csv(missing, schema)

    def test_structure_check_still_runs_first(self):
        missing = ROOT / "samples" / "does-not-exist.csv"
        schema = {"columns": [
            {"name": "lo", "type": "date", "lte_field": 3},
        ]}
        with self.assertRaises(ValueError) as caught:
            normalize_csv(missing, schema)
        self.assertNotIn("lte_field", str(caught.exception))

    def test_cli_relation_row_errors_exit_one_and_export(self):
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                source = self.write_source(directory, self.ROWS[fmt], fmt)
                schema_path = Path(directory) / "schema.json"
                schema_path.write_text(json.dumps(self.lohi_schema()), encoding="utf-8")
                output, errors = Path(directory) / "records.jsonl", Path(directory) / "errors.jsonl"
                command = [sys.executable, str(ROOT / "data_importer.py"), str(source),
                           "--schema", str(schema_path), "--output", str(output),
                           "--errors", str(errors), "--format", fmt]
                run = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(run.returncode, 1, run.stderr)
                self.assertEqual(json.loads(run.stdout), {"accepted": 3, "rejected": 2})
                records = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual(records, [
                    {"hi": 3, "lo": 1}, {"hi": 3, "lo": None}, {"hi": 2, "lo": 2}])
                error_rows = [json.loads(line) for line in errors.read_text().splitlines()]
                self.assertEqual([row["row"] for row in error_rows],
                                 [3, 6] if fmt == "csv" else [2, 5])
                self.assertEqual(error_rows[0]["errors"],
                                 ["lo: value 4 is greater than lte_field 'hi' value 3"])
                self.assertEqual(hidden_entries(directory), [])

    def test_cli_relation_errors_with_csv_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = self.write_source(directory, self.ROWS["csv"], "csv")
            schema_path = Path(directory) / "schema.json"
            schema_path.write_text(json.dumps(self.lohi_schema()), encoding="utf-8")
            output, errors = Path(directory) / "records.csv", Path(directory) / "errors.jsonl"
            command = [sys.executable, str(ROOT / "data_importer.py"), str(source),
                       "--schema", str(schema_path), "--output", str(output),
                       "--errors", str(errors), "--output-format", "csv"]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"),
                             "lo,hi\n1,3\n,3\n2,2\n")
            self.assertEqual(len(errors.read_text().splitlines()), 2)

    def test_cli_bad_lte_field_exit_two_keeps_files(self):
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                source = Path(directory) / f"source.{fmt}"
                source.write_text(
                    "lo,hi\n1,3\n" if fmt == "csv"
                    else '{"lo": 1, "hi": 3}\n', encoding="utf-8")
                schema_path = Path(directory) / "schema.json"
                schema_path.write_text(json.dumps({"columns": [
                    {"name": "lo", "type": "integer", "lte_field": "nope"},
                    {"name": "hi", "type": "integer"},
                ]}), encoding="utf-8")
                output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
                output.write_text("keep me\n", encoding="utf-8")
                command = [sys.executable, str(ROOT / "data_importer.py"), str(source),
                           "--schema", str(schema_path), "--output", str(output),
                           "--errors", str(errors), "--format", fmt]
                run = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(run.returncode, 2, run.stderr)
                payload = json.loads(run.stdout)
                self.assertEqual(set(payload), {"error"})
                self.assertTrue(payload["error"].strip())
                self.assertIn("lte_field", payload["error"])
                self.assertIn("lo", payload["error"])
                self.assertIn("nope", payload["error"])
                self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")
                self.assertFalse(errors.exists())


class DeduplicateByTests(unittest.TestCase):
    """Optional keep-first deduplication over final converted values."""

    MISSING = {"csv": ROOT / "samples" / "does-not-exist.csv",
               "jsonl": ROOT / "samples" / "does-not-exist.jsonl"}

    def schema(self, **overrides):
        columns = [
            {"name": "name", "type": "string", "required": True},
            {"name": "orders", "type": "integer", "required": True},
            {"name": "active", "type": "boolean", "required": True},
        ]
        return {"columns": columns}

    def write_source(self, directory, text, fmt):
        path = Path(directory) / ("data.csv" if fmt == "csv" else "data.jsonl")
        path.write_text(text, encoding="utf-8")
        return path

    def normalize(self, path, schema, fmt, **kwargs):
        return (normalize_csv if fmt == "csv" else normalize_jsonl)(path, schema, **kwargs)

    # name/orders after conversion, plus one illegal orders row at the end
    ROWS = {
        "csv": ("name,orders,active\n"
                "A,3,true\n"
                "A,3,true\n"
                "B,3,true\n"
                "A,4,true\n"
                "B,3,true\n"
                "C,bad,true\n"),
        "jsonl": ('{"name": "A", "orders": 3, "active": true}\n'
                  '{"name": "A", "orders": "3", "active": true}\n'
                  '{"name": "B", "orders": 3, "active": true}\n'
                  '{"name": "A", "orders": 4, "active": true}\n'
                  '{"name": "B", "orders": 3, "active": true}\n'
                  '{"name": "C", "orders": "bad", "active": true}\n'),
    }

    def test_equivalent_samples_keep_first_with_same_records_and_counts(self):
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, self.ROWS[fmt], fmt)
                result = self.normalize(path, self.schema(), fmt,
                                        deduplicate_by=["name", "orders"])
            self.assertEqual((result["accepted"], result["deduplicated"],
                              result["rejected"]), (3, 2, 1), fmt)
            self.assertEqual(result["records"], [
                {"name": "A", "orders": 3, "active": True},
                {"name": "B", "orders": 3, "active": True},
                {"name": "A", "orders": 4, "active": True},
            ], fmt)
            self.assertNotIn("filtered", result)
            self.assertNotIn("duplicates", result)
            # the illegal row reaches errors once, at its physical line,
            # never masked by filtering or deduplication
            self.assertEqual(len(result["errors"]), 1)
            self.assertEqual(result["errors"][0]["row"], 7 if fmt == "csv" else 6)
            self.assertTrue(any("orders" in m for m in result["errors"][0]["errors"]))

    def test_first_record_kept_whole_and_order_preserved(self):
        # equal key, different non-key field: the first full record wins
        # and nothing is merged or overwritten; survivors stay in order
        texts = {
            "csv": ("name,orders,active\n"
                    "A,3,true\n"
                    "A,4,false\n"
                    "B,1,true\n"),
            "jsonl": ('{"name": "A", "orders": 3, "active": true}\n'
                      '{"name": "A", "orders": 4, "active": false}\n'
                      '{"name": "B", "orders": 1, "active": true}\n'),
        }
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, texts[fmt], fmt)
                result = self.normalize(path, self.schema(), fmt,
                                        deduplicate_by=["name"])
            self.assertEqual((result["accepted"], result["deduplicated"],
                              result["rejected"]), (2, 1, 0), fmt)
            self.assertEqual(result["records"], [
                {"name": "A", "orders": 3, "active": True},
                {"name": "B", "orders": 1, "active": True},
            ], fmt)

    def test_composite_key_needs_every_field_equal(self):
        texts = {
            "csv": ("name,orders,active\nA,3,true\nA,4,true\nB,3,true\nA,3,false\n"),
            "jsonl": ('{"name": "A", "orders": 3, "active": true}\n'
                      '{"name": "A", "orders": 4, "active": true}\n'
                      '{"name": "B", "orders": 3, "active": true}\n'
                      '{"name": "A", "orders": 3, "active": false}\n'),
        }
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, texts[fmt], fmt)
                result = self.normalize(path, self.schema(), fmt,
                                        deduplicate_by=["name", "orders"])
            # only the first row has a full-key duplicate
            self.assertEqual((result["accepted"], result["deduplicated"]), (3, 1), fmt)
            self.assertEqual([r["name"] for r in result["records"]], ["A", "A", "B"], fmt)

    def test_null_matches_null_and_strings_are_case_sensitive(self):
        schema = {"columns": [
            {"name": "name", "type": "string"},
            {"name": "orders", "type": "integer"},
        ]}
        texts = {
            "csv": ("name,orders\nX,1\nx,2\n,3\n,4\n"),
            "jsonl": ('{"name": "X", "orders": 1}\n'
                      '{"name": "x", "orders": 2}\n'
                      '{"name": null, "orders": 3}\n'
                      '{"name": null, "orders": 4}\n'),
        }
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, texts[fmt], fmt)
                result = self.normalize(path, schema, fmt,
                                        deduplicate_by=["name"])
            self.assertEqual(result["records"], [
                {"name": "X", "orders": 1},
                {"name": "x", "orders": 2},
                {"name": None, "orders": 3},
            ], fmt)
            self.assertEqual(result["deduplicated"], 1, fmt)

    def test_no_duplicates_reports_zero_and_keeps_everything(self):
        texts = {
            "csv": "name,orders,active\nA,1,true\nB,2,false\n",
            "jsonl": ('{"name": "A", "orders": 1, "active": true}\n'
                      '{"name": "B", "orders": 2, "active": false}\n'),
        }
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, texts[fmt], fmt)
                result = self.normalize(path, self.schema(), fmt,
                                        deduplicate_by=["name"])
            self.assertEqual(result["accepted"], 2)
            self.assertEqual(result["deduplicated"], 0)

    def test_omitted_or_none_leaves_result_unchanged(self):
        texts = {
            "csv": "name,orders,active\nA,3,true\nA,3,false\n",
            "jsonl": ('{"name": "A", "orders": 3, "active": true}\n'
                      '{"name": "A", "orders": 3, "active": false}\n'),
        }
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, texts[fmt], fmt)
                omitted = self.normalize(path, self.schema(), fmt)
                explicit_none = self.normalize(path, self.schema(), fmt,
                                               deduplicate_by=None)
            self.assertEqual(omitted, explicit_none)
            self.assertEqual(set(omitted), {"records", "errors", "accepted", "rejected"})
            self.assertNotIn("deduplicated", omitted)

    def test_filter_runs_before_dedup_and_counts_stay_separate(self):
        # valid rows: A3t, A3f, A3t, B1f -- filtering active=true keeps
        # the first and third (identical), which dedup then collapses;
        # filtered counts only the predicate removals (2), dedup only 1.
        texts = {
            "csv": ("name,orders,active\n"
                    "A,3,true\nA,3,false\nA,3,true\nB,1,false\n"),
            "jsonl": ('{"name": "A", "orders": 3, "active": true}\n'
                      '{"name": "A", "orders": 3, "active": false}\n'
                      '{"name": "A", "orders": 3, "active": true}\n'
                      '{"name": "B", "orders": 1, "active": false}\n'),
        }
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, texts[fmt], fmt)
                result = self.normalize(
                    path, self.schema(), fmt,
                    deduplicate_by=["name", "orders"],
                    filter_eq={"field": "active", "value": True})
            self.assertEqual((result["accepted"], result["filtered"],
                              result["deduplicated"], result["rejected"]),
                             (1, 2, 1, 0), fmt)
            self.assertEqual(result["records"],
                             [{"name": "A", "orders": 3, "active": True}], fmt)

    def test_empty_result_after_filter_has_zero_deduplicated(self):
        texts = {
            "csv": "name,orders,active\nA,3,true\nA,3,true\n",
            "jsonl": ('{"name": "A", "orders": 3, "active": true}\n'
                      '{"name": "A", "orders": 3, "active": true}\n'),
        }
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, texts[fmt], fmt)
                result = self.normalize(
                    path, self.schema(), fmt, deduplicate_by=["name", "orders"],
                    filter_eq={"field": "name", "value": "zzz"})
            self.assertEqual((result["accepted"], result["filtered"],
                              result["deduplicated"]), (0, 2, 0), fmt)
            self.assertEqual(result["records"], [])

    def test_duplicate_report_runs_over_final_records(self):
        texts = {
            "csv": ("name,orders,active\n"
                    "A,3,true\nB,3,true\nA,3,false\nA,4,true\n"),
            "jsonl": ('{"name": "A", "orders": 3, "active": true}\n'
                      '{"name": "B", "orders": 3, "active": true}\n'
                      '{"name": "A", "orders": 3, "active": false}\n'
                      '{"name": "A", "orders": 4, "active": true}\n'),
        }
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, texts[fmt], fmt)
                result = self.normalize(path, self.schema(), fmt,
                                        deduplicate_by=["name", "orders"],
                                        duplicate_by=["orders"])
            self.assertEqual((result["accepted"], result["deduplicated"]), (3, 1), fmt)
            # positions restart at 1 in the final, deduplicated records
            self.assertEqual(result["duplicates"],
                             [{"key": {"orders": 3}, "record_numbers": [1, 2]}], fmt)

    def test_invalid_rows_are_not_deduplicated_or_masked(self):
        texts = {
            "csv": ("name,orders,active\n"
                    "A,3,true\nA,3,true\nBad,x,true\nA,3,true\n"),
            "jsonl": ('{"name": "A", "orders": 3, "active": true}\n'
                      '{"name": "A", "orders": 3, "active": true}\n'
                      '{"name": "Bad", "orders": "x", "active": true}\n'
                      '{"name": "A", "orders": 3, "active": true}\n'),
        }
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, texts[fmt], fmt)
                result = self.normalize(path, self.schema(), fmt,
                                        deduplicate_by=["name", "orders"])
            self.assertEqual((result["accepted"], result["deduplicated"],
                              result["rejected"]), (1, 2, 1), fmt)
            self.assertEqual(result["records"],
                             [{"name": "A", "orders": 3, "active": True}], fmt)
            self.assertEqual(result["errors"][0]["row"], 4 if fmt == "csv" else 3)

    def test_invalid_config_raises_before_reading(self):
        import copy
        cases = ["orders", [], ["orders", "orders"], [""], ["  "], [3],
                 ("orders",), ["nope"], ["active", "nope"]]
        for fmt in ("csv", "jsonl"):
            normalize = normalize_csv if fmt == "csv" else normalize_jsonl
            for value in cases:
                with self.subTest(fmt=fmt, value=value):
                    with self.assertRaises(ValueError) as caught:
                        normalize(self.MISSING[fmt], self.schema(),
                                  deduplicate_by=value)
                    message = str(caught.exception)
                    self.assertIn("deduplicate_by", message)
        with self.assertRaises(ValueError) as caught:
            normalize_csv(self.MISSING["csv"], self.schema(),
                          deduplicate_by=["orders", "nope"])
        self.assertIn("nope", str(caught.exception))
        # a source alias is not accepted as a field name
        mapped = {"columns": [
            {"name": "name", "type": "string", "required": True, "source": "display_name"},
            {"name": "orders", "type": "integer", "source": "purchase_count"},
            {"name": "active", "type": "boolean", "required": True},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_source(
                directory, "display_name,purchase_count,active\nMaya,3,TRUE\n", "csv")
            with self.assertRaises(ValueError) as caught:
                normalize_csv(csv_path, mapped, deduplicate_by=["purchase_count"])
            self.assertIn("deduplicate_by", str(caught.exception))
        # the caller's list and schema are never mutated
        fields = ["orders", "name"]
        snapshot = copy.deepcopy(fields)
        schema = self.schema()
        schema_snapshot = copy.deepcopy(schema)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_source(
                directory, self.ROWS["csv"], "csv")
            normalize_csv(path, schema, deduplicate_by=fields)
        self.assertEqual(fields, snapshot)
        self.assertEqual(schema, schema_snapshot)

    def test_cli_flag_summary_outputs_and_exit_code(self):
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, self.ROWS[fmt], fmt)
                output, errors = Path(directory) / "out.jsonl", Path(directory) / "errors.jsonl"
                run = run_cli(path, output, errors, fmt=fmt,
                              extra=("--deduplicate-by", "name",
                                     "--deduplicate-by", "orders"))
                self.assertEqual(run.returncode, 1, run.stderr)
                self.assertEqual(json.loads(run.stdout),
                                 {"accepted": 3, "rejected": 1, "deduplicated": 2})
                records = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual(records, [
                    {"active": True, "name": "A", "orders": 3},
                    {"active": True, "name": "B", "orders": 3},
                    {"active": True, "name": "A", "orders": 4},
                ])
                self.assertEqual(len(errors.read_text().splitlines()), 1)

    def test_cli_csv_output_also_deduplicates(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_source(directory, self.ROWS["csv"], "csv")
            output, errors = Path(directory) / "out.csv", Path(directory) / "errors.jsonl"
            run = run_cli(path, output, errors, fmt="csv", output_format="csv",
                          extra=("--deduplicate-by", "name",
                                 "--deduplicate-by", "orders"))
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(json.loads(run.stdout),
                             {"accepted": 3, "rejected": 1, "deduplicated": 2})
            self.assertEqual(output.read_text(encoding="utf-8"),
                             "name,orders,active\n"
                             "A,3,true\nB,3,true\nA,4,true\n")

    def test_cli_bad_deduplicate_field_exit_two_keeps_files(self):
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                path = self.write_source(directory, self.ROWS[fmt], fmt)
                output, errors = Path(directory) / "out.jsonl", Path(directory) / "errors.jsonl"
                for flag in (("--deduplicate-by", "nope"),
                             ("--deduplicate-by", "name", "--deduplicate-by", "name")):
                    output.write_text("keep me\n", encoding="utf-8")
                    if errors.exists():
                        errors.unlink()
                    run = run_cli(path, output, errors, fmt=fmt, extra=flag)
                    self.assertEqual(run.returncode, 2, flag)
                    payload = json.loads(run.stdout)
                    self.assertEqual(set(payload), {"error"})
                    self.assertIn("deduplicate_by", payload["error"])
                    self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")
                    self.assertFalse(errors.exists())


class SchemaStructureTests(unittest.TestCase):
    """Malformed schema shapes raise a clear ValueError before the data
    file is opened, identically for normalize_csv and normalize_jsonl."""

    MISSING = {"csv": ROOT / "samples" / "does-not-exist.csv",
               "jsonl": ROOT / "samples" / "does-not-exist.jsonl"}

    def assert_structure_error(self, schema, *fragments):
        messages = []
        for normalize, missing in ((normalize_csv, self.MISSING["csv"]),
                                   (normalize_jsonl, self.MISSING["jsonl"])):
            with self.assertRaises(ValueError) as caught:
                normalize(missing, schema)
            message = str(caught.exception)
            self.assertIn("schema", message)
            for fragment in fragments:
                self.assertIn(fragment, message, (schema, message))
            messages.append(message)
        return messages[0]

    def test_schema_must_be_an_object_with_columns(self):
        self.assert_structure_error({}, "columns")
        self.assert_structure_error([], "object")
        self.assert_structure_error(None, "object")
        self.assert_structure_error("columns", "object")

    def test_columns_must_be_a_non_empty_list(self):
        self.assert_structure_error({"columns": None}, "columns", "None")
        self.assert_structure_error({"columns": []}, "non-empty")
        self.assert_structure_error({"columns": {}}, "non-empty")
        self.assert_structure_error({"columns": "name"}, "non-empty")

    def test_column_must_be_an_object(self):
        self.assert_structure_error({"columns": [None]}, "column 1", "object", "None")
        self.assert_structure_error(
            {"columns": [{"name": "a", "type": "string"}, "a"]}, "column 2", "object")

    def test_missing_name_and_type_are_named(self):
        self.assert_structure_error({"columns": [{"type": "string"}]}, "column 1", "name")
        self.assert_structure_error({"columns": [{"name": "a"}]}, "column 1", "type")
        message = self.assert_structure_error({"columns": [{}]}, "column 1")
        self.assertIn("name", message)

    def test_name_must_be_a_non_blank_string(self):
        self.assert_structure_error({"columns": [{"name": ["a"], "type": "string"}]},
                                    "column 1", "name", repr(["a"]))
        self.assert_structure_error({"columns": [{"name": 3, "type": "string"}]},
                                    "column 1", "name", "3")
        self.assert_structure_error({"columns": [{"name": "", "type": "string"}]},
                                    "column 1", "non-blank")
        self.assert_structure_error({"columns": [{"name": "   ", "type": "string"}]},
                                    "column 1", "non-blank")

    def test_duplicate_name_points_at_later_column(self):
        schema = {"columns": [{"name": "a", "type": "string"},
                              {"name": "b", "type": "integer"},
                              {"name": "a", "type": "boolean"}]}
        message = self.assert_structure_error(schema, "column 3", "'a'", "column 1")
        self.assertNotIn("column 2", message)

    def test_names_are_case_sensitive_and_never_trimmed(self):
        # Distinct raw names pass structural validation; with the data
        # file absent the call then fails opening it, not on the schema.
        schema = {"columns": [{"name": "A", "type": "string"},
                              {"name": "a", "type": "string"},
                              {"name": " a ", "type": "string"}]}
        for normalize, missing in ((normalize_csv, self.MISSING["csv"]),
                                   (normalize_jsonl, self.MISSING["jsonl"])):
            with self.assertRaises(OSError):
                normalize(missing, schema)

    def test_unknown_type_names_value_and_column(self):
        self.assert_structure_error({"columns": [{"name": "a", "type": "date"}]},
                                    "column 1", "type", "'date'")
        self.assert_structure_error({"columns": [{"name": "a", "type": ["string"]}]},
                                    "column 1", "type", repr(["string"]))

    def test_required_must_be_a_boolean_when_present(self):
        self.assert_structure_error(
            {"columns": [{"name": "a", "type": "string", "required": "yes"}]},
            "column 1", "required", "'yes'")
        self.assert_structure_error(
            {"columns": [{"name": "a", "type": "string", "required": 1}]},
            "column 1", "required", "boolean")
        # booleans (and omission) are fine: the missing data file fails next
        for required in (True, False):
            schema = {"columns": [{"name": "a", "type": "string", "required": required}]}
            with self.assertRaises(OSError):
                normalize_csv(self.MISSING["csv"], schema)

    def test_only_the_first_error_is_reported(self):
        # top level beats column problems
        message = self.assert_structure_error({"columns": None}, "columns")
        self.assertNotIn("column 1", message)
        # an earlier column beats a later one
        message = self.assert_structure_error(
            {"columns": [{"name": ["x"], "type": "string"}, None]}, "column 1")
        self.assertNotIn("column 2", message)
        # within a column: name before duplicate/type/required
        message = self.assert_structure_error(
            {"columns": [{"name": "a", "type": "string"},
                         {"name": "a", "type": "date", "required": "yes"}]},
            "column 2", "duplicates")
        self.assertNotIn("date", message)
        self.assertNotIn("required", message)

    def test_unknown_attributes_are_ignored(self):
        schema = {"extra": 1, "columns": [
            {"name": "name", "type": "string", "bogus": True},
            {"name": "orders", "type": "integer"},
            {"name": "active", "type": "boolean"},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = Path(directory) / "data.csv"
            csv_path.write_text("name,orders,active\nMaya,3,true\n", encoding="utf-8")
            result = normalize_csv(csv_path, schema)
        self.assertEqual(result["records"], [{"name": "Maya", "orders": 3, "active": True}])

    def test_schema_is_not_mutated_by_failed_validation(self):
        import copy
        schema = {"columns": [{"name": "a", "type": "string"},
                              {"name": "a", "type": "date", "required": "yes"}]}
        snapshot = copy.deepcopy(schema)
        self.assert_structure_error(schema, "column 2")
        self.assertEqual(schema, snapshot)

    def run_cli_with_schema(self, directory, schema_obj, fmt, source_name):
        source = Path(directory) / source_name
        source.write_text("name,orders,active\nMaya,3,true\n"
                          if fmt == "csv" else
                          '{"name": "Maya", "orders": 3, "active": true}\n',
                          encoding="utf-8")
        schema_path = Path(directory) / "schema.json"
        schema_path.write_text(json.dumps(schema_obj), encoding="utf-8")
        output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
        output.write_text("keep me\n", encoding="utf-8")
        command = [sys.executable, str(ROOT / "data_importer.py"), str(source),
                   "--schema", str(schema_path), "--output", str(output),
                   "--errors", str(errors), "--format", fmt]
        run = subprocess.run(command, capture_output=True, text=True)
        return run, source, schema_path, output, errors

    def test_cli_structure_error_exit_two_keeps_everything(self):
        bad_schemas = ({}, {"columns": None}, {"columns": [None]},
                       {"columns": [{"name": "a", "type": "string"},
                                    {"name": "a", "type": "string"}]},
                       {"columns": [{"name": "a", "type": "string", "required": "yes"}]})
        for fmt in ("csv", "jsonl"):
            for schema_obj in bad_schemas:
                with self.subTest(fmt=fmt, schema=schema_obj):
                    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                        run, source, schema_path, output, errors = self.run_cli_with_schema(
                            directory, schema_obj, fmt,
                            "source.csv" if fmt == "csv" else "source.jsonl")
                        self.assertEqual(run.returncode, 2, run.stderr)
                        payload = json.loads(run.stdout)
                        self.assertEqual(set(payload), {"error"})
                        self.assertTrue(payload["error"].strip())
                        self.assertIn("schema", payload["error"])
                        self.assertNotIn("accepted", payload)
                        self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")
                        self.assertFalse(errors.exists())
                        self.assertEqual(source.read_bytes(),
                                         ("name,orders,active\nMaya,3,true\n" if fmt == "csv"
                                          else '{"name": "Maya", "orders": 3, "active": true}\n'
                                          ).encode())
                        self.assertEqual(json.loads(schema_path.read_text(encoding="utf-8")),
                                         schema_obj)

    def test_cli_structure_error_beats_missing_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            schema_path = Path(directory) / "schema.json"
            schema_path.write_text("{}", encoding="utf-8")
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            command = [sys.executable, str(ROOT / "data_importer.py"),
                       str(Path(directory) / "absent.csv"),
                       "--schema", str(schema_path), "--output", str(output),
                       "--errors", str(errors)]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            payload = json.loads(run.stdout)
            self.assertEqual(set(payload), {"error"})
            self.assertIn("schema", payload["error"])
            self.assertFalse(output.exists() or errors.exists())

    def test_cli_path_alias_precheck_still_runs_first(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = Path(directory) / "data.csv"
            source.write_text("name,orders,active\nMaya,3,true\n", encoding="utf-8")
            schema_path = Path(directory) / "schema.json"
            schema_path.write_text("{}", encoding="utf-8")
            errors = Path(directory) / "errors.jsonl"
            command = [sys.executable, str(ROOT / "data_importer.py"), str(source),
                       "--schema", str(schema_path), "--output", str(source),
                       "--errors", str(errors)]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            payload = json.loads(run.stdout)
            self.assertEqual(set(payload), {"error"})
            self.assertIn("same file", payload["error"])
            self.assertEqual(source.read_text(encoding="utf-8"),
                             "name,orders,active\nMaya,3,true\n")
            self.assertFalse(errors.exists())


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


class PathIsolationTests(unittest.TestCase):
    """Outputs must not alias the source/schema or each other.

    Identity is the resolved pathname (same path or symlink alias) or, for
    two existing files, the same device/inode pair -- so distinct hard
    links to one file are refused, while independent files with identical
    contents are allowed.
    """

    CSV_TEXT = ("name,orders,active\n"
                "Maya,3,true\n"
                "Omar,,false\n")
    JSONL_TEXT = ('{"name": "Maya", "orders": 3, "active": true}\n'
                  '{"name": "Omar", "orders": null, "active": false}\n')

    def write_inputs(self, directory, fmt):
        source = Path(directory, "data.csv" if fmt == "csv" else "data.jsonl")
        source.write_text(self.CSV_TEXT if fmt == "csv" else self.JSONL_TEXT,
                          encoding="utf-8")
        schema = Path(directory, "schema.json")
        schema.write_bytes((ROOT / "samples" / "schema.json").read_bytes())
        return source, schema

    def run_importer(self, directory, source, schema, output, errors, fmt=None):
        command = [sys.executable, str(ROOT / "data_importer.py"), str(source),
                   "--schema", str(schema), "--output", str(output),
                   "--errors", str(errors)]
        if fmt is not None:
            command += ["--format", fmt]
        return subprocess.run(command, capture_output=True, text=True, cwd=directory)

    def assert_isolation_error(self, run, *named):
        self.assertEqual(run.returncode, 2, run.stderr)
        payload = json.loads(run.stdout)
        self.assertEqual(set(payload), {"error"})
        self.assertTrue(payload["error"].strip())
        self.assertNotIn("accepted", payload)
        self.assertNotIn("rejected", payload)
        for name in named:
            self.assertIn(name, payload["error"])

    def assert_same_file(self, left, right, links=2):
        left_stat, right_stat = left.stat(), right.stat()
        self.assertEqual((left_stat.st_dev, left_stat.st_nlink),
                         (right_stat.st_dev, links))
        self.assertEqual(left_stat.st_ino, right_stat.st_ino)

    def test_each_output_hardlinked_to_each_input_is_rejected(self):
        # Each output slot (records/errors) hard-linked to each input
        # (source/schema): four pairings. The alias is passed as a relative
        # path (from the CLI working directory), mixing spellings.
        for fmt in ("csv", "jsonl"):
            for alias_slot, input_role, alias_name, other_name in (
                    ("records", "source", "out.csv", "errors.jsonl"),
                    ("records", "schema", "out.json", "errors.jsonl"),
                    ("errors", "source", "errors.csv", "out.jsonl"),
                    ("errors", "schema", "errors.json", "out.jsonl")):
                with self.subTest(fmt=fmt, slot=alias_slot, alias=input_role):
                    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                        source, schema = self.write_inputs(directory, fmt)
                        input_path = source if input_role == "source" else schema
                        input_name = ("data." + ("csv" if fmt == "csv" else "jsonl")
                                      if input_role == "source" else "schema.json")
                        alias, other = Path(directory, alias_name), Path(directory, other_name)
                        os.link(input_path, alias)
                        if alias_slot == "records":
                            output, errors = alias, other
                        else:
                            output, errors = other, alias
                        original = input_path.read_bytes()
                        run = self.run_importer(
                            directory, "data.csv" if fmt == "csv" else "data.jsonl",
                            Path(directory, "schema.json"), output, errors, fmt)
                        self.assert_isolation_error(run, alias_name, input_name)
                        # Bytes, the hard-link relationship and link count
                        # are all untouched; the other output was not made.
                        self.assertEqual(input_path.read_bytes(), original)
                        self.assertEqual(alias.read_bytes(), original)
                        self.assert_same_file(input_path, alias)
                        self.assertFalse(other.exists())

    def test_two_existing_outputs_hardlinked_together_are_rejected(self):
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                source, schema = self.write_inputs(directory, fmt)
                output = Path(directory, "a.jsonl")
                errors = Path(directory, "b.jsonl")
                output.write_bytes(b"previous output bytes\n")
                os.link(output, errors)
                run = self.run_importer(directory, source, schema, output, errors, fmt)
                self.assert_isolation_error(run, "a.jsonl", "b.jsonl")
                self.assertEqual(output.read_bytes(), b"previous output bytes\n")
                self.assertEqual(errors.read_bytes(), b"previous output bytes\n")
                self.assert_same_file(output, errors)

    def test_identical_content_independent_files_are_allowed(self):
        for fmt in ("csv", "jsonl"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                source, schema = self.write_inputs(directory, fmt)
                twin = Path(directory, "twin")
                twin.write_bytes(source.read_bytes())
                self.assertNotEqual(source.stat().st_ino, twin.stat().st_ino)
                errors = Path(directory, "errors.jsonl")
                run = self.run_importer(directory, source, schema, twin, errors, fmt)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(json.loads(run.stdout), {"accepted": 2, "rejected": 0})
                # The independent output was overwritten with JSONL; the
                # source was not touched even though the bytes matched.
                self.assertEqual(source.read_bytes(),
                                 (self.CSV_TEXT if fmt == "csv" else self.JSONL_TEXT).encode())
                records = [json.loads(line) for line in twin.read_text().splitlines()]
                self.assertEqual([record["name"] for record in records], ["Maya", "Omar"])

    def test_symlink_aliases_still_rejected_and_link_kept(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source, schema = self.write_inputs(directory, "csv")
            link = Path(directory, "link.csv")
            link.symlink_to(source.name)
            errors = Path(directory, "errors.jsonl")
            run = self.run_importer(directory, source, schema, link, errors)
            self.assert_isolation_error(run, "link.csv", "data.csv")
            self.assertTrue(link.is_symlink())
            self.assertEqual(os.readlink(link), source.name)
            self.assertFalse(errors.exists())
            # two symlinks spelling one not-yet-existing output name
            first, second = Path(directory, "l1"), Path(directory, "l2")
            first.symlink_to("shared.jsonl")
            second.symlink_to("shared.jsonl")
            run = self.run_importer(directory, source, schema, first, second)
            self.assert_isolation_error(run, "l1", "l2")
            self.assertTrue(first.is_symlink() and second.is_symlink())
            self.assertFalse((Path(directory, "shared.jsonl")).exists())

    def test_same_output_pathname_still_rejected_without_creating_it(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source, schema = self.write_inputs(directory, "csv")
            run = self.run_importer(directory, source, schema, "same.jsonl", "same.jsonl")
            self.assert_isolation_error(run, "same.jsonl")
            self.assertFalse((Path(directory, "same.jsonl")).exists())

    def test_nonexistent_distinct_outputs_keep_pathname_rule_only(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source, schema = self.write_inputs(directory, "csv")
            output, errors = Path(directory, "out.jsonl"), Path(directory, "err.jsonl")
            run = self.run_importer(directory, source, schema, output, errors)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue(output.exists() and errors.exists())

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0,
                     "root bypasses permission bits")
    def test_unstatable_path_is_io_error_exit_two_without_outputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source, _ = self.write_inputs(directory, "csv")
            locked = Path(directory, "locked")
            locked.mkdir()
            hidden_schema = locked / "schema.json"
            hidden_schema.write_bytes((ROOT / "samples" / "schema.json").read_bytes())
            locked.chmod(0o000)
            try:
                output, errors = Path(directory, "out.jsonl"), Path(directory, "err.jsonl")
                run = self.run_importer(directory, source, hidden_schema, output, errors)
                self.assertEqual(run.returncode, 2, run.stderr)
                payload = json.loads(run.stdout)
                self.assertEqual(set(payload), {"error"})
                self.assertIn("schema.json", payload["error"])
                self.assertFalse(output.exists() or errors.exists())
            finally:
                locked.chmod(0o755)


class CsvOutputTests(unittest.TestCase):
    """--output-format csv writes exact LF-terminated CSV text while the
    errors output stays JSONL; input format is selected independently."""

    def write_file(self, directory, name, text):
        path = Path(directory) / name
        path.write_text(text, encoding="utf-8")
        return path

    def run_with_schema(self, source, schema_path, output, errors,
                        input_format, output_format):
        command = [sys.executable, str(ROOT / "data_importer.py"), str(source),
                   "--schema", str(schema_path), "--output", str(output),
                   "--errors", str(errors), "--format", input_format,
                   "--output-format", output_format]
        return subprocess.run(command, capture_output=True, text=True)

    def test_csv_output_exact_bytes_and_jsonl_errors(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output, errors = Path(directory) / "data.csv", Path(directory) / "errors.jsonl"
            run = run_cli(ROOT / "samples/trio.csv", output, errors,
                          fmt="csv", output_format="csv")
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(json.loads(run.stdout), {"accepted": 2, "rejected": 1})
            self.assertEqual(output.read_bytes(),
                             b"name,orders,active\n"
                             b"Maya,3,true\n"
                             b"Omar,,false\n")
            error_rows = [json.loads(line) for line in errors.read_text().splitlines()]
            self.assertEqual(len(error_rows), 1)
            self.assertEqual(error_rows[0]["row"], 4)
            self.assertTrue(any("orders" in m for m in error_rows[0]["errors"]))
            self.assertEqual(hidden_entries(directory), [])

    def test_same_samples_csv_and_jsonl_inputs_give_identical_csv_bytes(self):
        schema = {"columns": [
            {"name": "label", "type": "string"},
            {"name": "n", "type": "integer"},
            {"name": "on", "type": "boolean", "required": True},
        ]}
        expected = (
            "label,n,on\n"
            '"中文,带逗号",3,true\n'
            '"引号""测试",-7,false\n'
            '"第一行\n第二行\r第三行",,true\n'
            ",0,false\n"
        ).encode("utf-8")
        csv_input = expected.decode("utf-8")  # canonical text is valid input too
        jsonl_input = "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in (
                {"label": "中文,带逗号", "n": 3, "on": True},
                {"label": '引号"测试', "n": -7, "on": False},
                {"label": "第一行\n第二行\r第三行", "n": None, "on": True},
                {"label": None, "n": 0, "on": False},
            ))
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            schema_path = self.write_file(directory, "schema.json",
                                          json.dumps(schema, ensure_ascii=False))
            csv_path = self.write_file(directory, "data.csv", csv_input)
            jsonl_path = self.write_file(directory, "data.jsonl", jsonl_input)
            results = []
            for source, input_format in ((csv_path, "csv"), (jsonl_path, "jsonl")):
                output = Path(directory) / f"out-{input_format}.csv"
                errors = Path(directory) / f"err-{input_format}.jsonl"
                run = self.run_with_schema(source, schema_path, output, errors,
                                           input_format, "csv")
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(errors.read_text(encoding="utf-8"), "")
                results.append(output.read_bytes())
        self.assertEqual(results[0], expected)
        self.assertEqual(results[1], expected)

    def test_csv_header_uses_output_names_not_sources(self):
        schema = {"columns": [
            {"name": "name", "type": "string", "required": True, "source": "display_name"},
            {"name": "orders", "type": "integer", "source": "purchase_count"},
            {"name": "active", "type": "boolean", "required": True},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            schema_path = self.write_file(directory, "schema.json", json.dumps(schema))
            csv_path = self.write_file(
                directory, "data.csv",
                "display_name,purchase_count,active\nMaya,3,TRUE\n")
            output, errors = Path(directory) / "out.csv", Path(directory) / "err.jsonl"
            run = self.run_with_schema(csv_path, schema_path, output, errors, "csv", "csv")
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(output.read_bytes(),
                             b"name,orders,active\nMaya,3,true\n")

    def test_csv_quoting_rules_single_column_empty_is_quoted(self):
        schema = {"columns": [{"name": "f", "type": "string"}]}
        csv_input = ('f\n'
                     '""\n'
                     'plain\n'
                     '"a,b"\n'
                     '"a""b"\n'
                     '"a\nb"\n'
                     '"a\rb"\n')
        jsonl_input = "".join(
            json.dumps({"f": value}, ensure_ascii=False) + "\n"
            for value in (None, "plain", "a,b", 'a"b', "a\nb", "a\rb"))
        expected = ('f\n'
                    '""\n'
                    'plain\n'
                    '"a,b"\n'
                    '"a""b"\n'
                    '"a\nb"\n'
                    '"a\rb"\n').encode("utf-8")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            schema_path = self.write_file(directory, "schema.json", json.dumps(schema))
            produced = []
            for source, input_format, text in (
                    ("data.csv", "csv", csv_input),
                    ("data.jsonl", "jsonl", jsonl_input)):
                path = self.write_file(directory, source, text)
                output, errors = Path(directory) / f"out-{input_format}.csv", \
                    Path(directory) / f"err-{input_format}.jsonl"
                run = self.run_with_schema(path, schema_path, output, errors,
                                           input_format, "csv")
                self.assertEqual(run.returncode, 0, run.stderr)
                produced.append(output.read_bytes())
        self.assertEqual(produced[0], expected)
        self.assertEqual(produced[1], expected)

    def test_csv_header_itself_follows_quoting_rule(self):
        schema = {"columns": [{"name": "a,b", "type": "string"}]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            schema_path = self.write_file(directory, "schema.json", json.dumps(schema))
            csv_path = self.write_file(directory, "data.csv", '"a,b"\nx\n')
            jsonl_path = self.write_file(
                directory, "data.jsonl", json.dumps({"a,b": "x"}) + "\n")
            for source, input_format in ((csv_path, "csv"), (jsonl_path, "jsonl")):
                output = Path(directory) / f"out-{input_format}.csv"
                errors = Path(directory) / f"err-{input_format}.jsonl"
                run = self.run_with_schema(source, schema_path, output, errors,
                                           input_format, "csv")
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(output.read_bytes(), b'"a,b"\nx\n')

    def test_csv_header_only_when_no_records_kept(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            # every valid record filtered out: header alone, exit 0
            output, errors = Path(directory) / "data.csv", Path(directory) / "errors.jsonl"
            run = run_cli(
                ROOT / "samples/customers.csv", output, errors,
                fmt="csv", output_format="csv",
                extra=("--filter-eq", json.dumps({"field": "name", "value": "nobody"})))
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout),
                             {"accepted": 0, "rejected": 0, "filtered": 2})
            self.assertEqual(output.read_bytes(), b"name,orders,active\n")
            self.assertEqual(errors.read_bytes(), b"")
            # rejected rows but none accepted: still header alone, exit 1
            source = self.write_file(
                directory, "bad.csv", "name,orders,active\nBad,x,true\n")
            run = run_cli(source, output, errors, fmt="csv", output_format="csv")
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(json.loads(run.stdout), {"accepted": 0, "rejected": 1})
            self.assertEqual(output.read_bytes(), b"name,orders,active\n")
            self.assertEqual(len(errors.read_text().splitlines()), 1)

    def test_output_format_defaults_jsonl_and_is_independent_of_input(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            out, err = Path(directory) / "records.txt", Path(directory) / "errors.txt"
            # JSONL input, flag omitted: JSONL output despite any names
            run = run_cli(ROOT / "samples/trio.jsonl", out, err, fmt="jsonl")
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertTrue(out.read_text(encoding="utf-8").startswith("{"))
            # CSV input with explicit jsonl output: JSONL
            run = run_cli(ROOT / "samples/trio.csv", out, err,
                          fmt="csv", output_format="jsonl")
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertTrue(out.read_text(encoding="utf-8").startswith("{"))
            # JSONL input with csv output: CSV header
            run = run_cli(ROOT / "samples/trio.jsonl", out, err,
                          fmt="jsonl", output_format="csv")
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(out.read_text(encoding="utf-8").splitlines()[0],
                             "name,orders,active")

    def test_csv_output_is_utf8_without_bom_and_uses_lf(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output, errors = Path(directory) / "data.csv", Path(directory) / "errors.jsonl"
            run = run_cli(ROOT / "samples/customers.csv", output, errors,
                          fmt="csv", output_format="csv")
            self.assertEqual(run.returncode, 0, run.stderr)
            data = output.read_bytes()
            self.assertFalse(data.startswith(b"\xef\xbb\xbf"))
            # every record terminator is LF; the samples contain no CR cells
            self.assertNotIn(b"\r", data)
            self.assertTrue(data.endswith(b"\n"))

    def test_invalid_output_format_ordering_message_and_untouched_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = self.write_file(
                directory, "data.csv", "name,orders,active\nMaya,3,true\n")
            schema_path = self.write_file(
                directory, "schema.json",
                (ROOT / "samples/schema.json").read_text(encoding="utf-8"))
            output, errors = Path(directory) / "out.csv", Path(directory) / "errors.jsonl"
            # unreadable inputs are never opened: even missing paths report
            # only the output-format error
            missing = Path(directory) / "nope.csv"
            missing_schema = Path(directory) / "nope.json"
            command = [sys.executable, str(ROOT / "data_importer.py"), str(missing),
                       "--schema", str(missing_schema), "--output", str(output),
                       "--errors", str(errors), "--output-format", "xml"]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            payload = json.loads(run.stdout)
            self.assertEqual(set(payload), {"error"})
            self.assertIn("output-format", payload["error"])
            self.assertIn("xml", payload["error"])
            self.assertFalse(output.exists() or errors.exists())
            # ... but path isolation is checked first
            command = [sys.executable, str(ROOT / "data_importer.py"), str(source),
                       "--schema", str(schema_path), "--output", str(source),
                       "--errors", str(errors), "--output-format", "xml"]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertIn("same file", json.loads(run.stdout)["error"])
            # pre-existing outputs keep their bytes; absent ones stay absent
            output.write_bytes(b"previous output bytes\n")
            command = [sys.executable, str(ROOT / "data_importer.py"), str(source),
                       "--schema", str(schema_path), "--output", str(output),
                       "--errors", str(errors), "--output-format", "XML"]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            payload = json.loads(run.stdout)
            self.assertIn("output-format", payload["error"])
            self.assertIn("XML", payload["error"])
            self.assertEqual(output.read_bytes(), b"previous output bytes\n")
            self.assertFalse(errors.exists())
            # with both formats invalid, output-format is reported first
            command = [sys.executable, str(ROOT / "data_importer.py"), str(missing),
                       "--schema", str(missing_schema), "--output", str(output),
                       "--errors", str(errors), "--format", "xml",
                       "--output-format", "yaml"]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertIn("output-format", json.loads(run.stdout)["error"])
            self.assertEqual(output.read_bytes(), b"previous output bytes\n")
            self.assertFalse(errors.exists())
            self.assertEqual(hidden_entries(directory), [])

    def test_csv_write_failures_keep_both_outputs_and_create_no_dirs(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            # records target is a directory; errors target absent
            output_dir = Path(directory) / "out.csv"
            output_dir.mkdir()
            errors = Path(directory) / "errors.jsonl"
            run = run_cli(ROOT / "samples/trio.csv", output_dir, errors,
                          fmt="csv", output_format="csv")
            self.assertEqual(run.returncode, 2, run.stderr)
            payload = json.loads(run.stdout)
            self.assertEqual(set(payload), {"error"})
            self.assertIn(str(output_dir), payload["error"])
            self.assertTrue(output_dir.is_dir())
            self.assertFalse(errors.exists())
            # missing output parent: existing errors file keeps its bytes
            missing_parent_output = Path(directory) / "nope" / "out.csv"
            errors.write_bytes(b"old errors\n")
            run = run_cli(ROOT / "samples/trio.csv", missing_parent_output, errors,
                          fmt="csv", output_format="csv")
            self.assertEqual(run.returncode, 2)
            self.assertIn(str(missing_parent_output), json.loads(run.stdout)["error"])
            self.assertFalse(missing_parent_output.exists())
            self.assertFalse(Path(directory, "nope").exists())
            self.assertEqual(errors.read_bytes(), b"old errors\n")
            # missing errors parent: existing CSV output keeps its bytes
            output = Path(directory) / "records.csv"
            output.write_bytes(b"old csv\n")
            missing_errors = Path(directory) / "dir" / "errors.jsonl"
            run = run_cli(ROOT / "samples/trio.csv", output, missing_errors,
                          fmt="csv", output_format="csv")
            self.assertEqual(run.returncode, 2)
            self.assertIn(str(missing_errors), json.loads(run.stdout)["error"])
            self.assertEqual(output.read_bytes(), b"old csv\n")
            self.assertFalse(Path(directory, "dir").exists())
            self.assertEqual(hidden_entries(directory), [])

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root bypasses permission bits")
    def test_unwritable_csv_output_target_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output, errors = Path(directory) / "data.csv", Path(directory) / "errors.jsonl"
            output.write_bytes(b"locked\n")
            output.chmod(0o444)
            try:
                run = run_cli(ROOT / "samples/trio.csv", output, errors,
                              fmt="csv", output_format="csv")
                self.assertEqual(run.returncode, 2)
                self.assertIn(str(output), json.loads(run.stdout)["error"])
                self.assertEqual(output.read_bytes(), b"locked\n")
                self.assertFalse(errors.exists())
            finally:
                output.chmod(0o644)


if __name__ == "__main__":
    unittest.main()
