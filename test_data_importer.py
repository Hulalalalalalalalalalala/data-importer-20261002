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

    def enum_schema(self):
        return {"columns": [
            {"name": "name", "type": "string", "required": True},
            {"name": "orders", "type": "integer", "allowed_values": [0, 3], "default": 3},
            {"name": "active", "type": "boolean", "required": True},
        ]}

    def test_allowed_values_enum_csv(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(
                directory,
                "name,orders,active\n"
                "a,,true\n"
                "b,3,true\n"
                "c,0,false\n"
                "d,4,true\n"
                "e,bad,true\n")
            result = normalize_csv(csv_path, self.enum_schema())
        self.assertEqual((result["accepted"], result["rejected"]), (3, 2))
        self.assertEqual([record["orders"] for record in result["records"]], [3, 3, 0])
        self.assertEqual([row["row"] for row in result["errors"]], [5, 6])
        enum_error = result["errors"][0]["errors"][0]
        self.assertIn("orders", enum_error)
        self.assertIn("allowed_values", enum_error)
        # a non-empty value that fails conversion reports only the type error
        type_error = result["errors"][1]["errors"][0]
        self.assertIn("orders", type_error)
        self.assertNotIn("allowed_values", type_error)

    def test_allowed_values_string_entries_kept_verbatim_csv(self):
        schema = {"columns": [
            {"name": "name", "type": "string", "allowed_values": ["Maya", " Omar "]},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "name\nMaya\nmaya\n Omar \nOmar\n")
            result = normalize_csv(csv_path, schema)
        # input is trimmed during conversion; the entries are not, and case stays
        self.assertEqual([record["name"] for record in result["records"]], ["Maya"])
        self.assertEqual(result["rejected"], 3)

    def test_allowed_values_boolean_and_null_exemption_csv(self):
        schema = {"columns": [
            {"name": "orders", "type": "integer", "allowed_values": [0, 3]},
            {"name": "active", "type": "boolean", "required": True, "allowed_values": [True]},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(
                directory,
                "orders,active\n"
                ",true\n"
                "3,true\n"
                "1,false\n"
                "0,\n")
            result = normalize_csv(csv_path, schema)
        # optional empty stays null and is exempt; required empty keeps its error
        self.assertEqual(result["records"][0]["orders"], None)
        self.assertEqual((result["accepted"], result["rejected"]), (2, 2))
        self.assertEqual(result["errors"][0]["errors"],
                         ["orders: value 1 is not in allowed_values",
                          "active: value False is not in allowed_values"])
        self.assertEqual(result["errors"][1]["errors"], ["active: required value is empty"])

    def test_allowed_values_invalid_config_raises_before_reading_csv(self):
        missing = ROOT / "samples" / "does-not-exist.csv"
        bad_lists = [None, 3, "x", {}, (), [], [None], [[1]], [{"a": 1}], [True],
                     ["3"], [3.0], [0, 0], [3, 3]]
        for values in bad_lists:
            schema = {"columns": [{"name": "orders", "type": "integer",
                                   "allowed_values": values}]}
            with self.assertRaises(ValueError) as caught:
                normalize_csv(missing, schema)
            message = str(caught.exception)
            self.assertIn("allowed_values", message, values)
            self.assertIn("orders", message, values)
        for kind, values in (("string", ["a", "a"]), ("boolean", [True, True]),
                             ("boolean", [1]), ("string", [3])):
            schema = {"columns": [{"name": "f", "type": kind, "allowed_values": values}]}
            with self.assertRaises(ValueError) as caught:
                normalize_csv(missing, schema)
            self.assertIn("allowed_values", str(caught.exception), (kind, values))
            self.assertIn("f", str(caught.exception), (kind, values))
        # a processed default outside the list is a configuration error even
        # when no input row would have used it
        schema = {"columns": [{"name": "orders", "type": "integer",
                               "allowed_values": [0, 3], "default": 4}]}
        with self.assertRaises(ValueError) as caught:
            normalize_csv(missing, schema)
        self.assertIn("allowed_values", str(caught.exception))
        self.assertIn("orders", str(caught.exception))

    def test_allowed_values_schema_not_mutated_csv(self):
        import copy
        schema = self.enum_schema()
        snapshot = copy.deepcopy(schema)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "name,orders,active\na,,true\nb,4,true\n")
            normalize_csv(csv_path, schema)
        self.assertEqual(schema, snapshot)

    def test_allowed_values_before_filter_and_duplicates_csv(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(
                directory,
                "name,orders,active\n"
                "a,3,true\n"
                "b,4,true\n"
                "c,,true\n"
                "d,0,false\n")
            result = normalize_csv(csv_path, self.enum_schema(), duplicate_by=["orders"],
                                   filter_eq={"field": "active", "value": True})
        # the enum-rejected row reaches errors and is never hidden by the filter
        self.assertEqual((result["accepted"], result["filtered"], result["rejected"]), (2, 1, 1))
        self.assertEqual([r["name"] for r in result["records"]], ["a", "c"])
        self.assertEqual(result["errors"][0]["row"], 3)
        self.assertEqual(result["duplicates"],
                         [{"key": {"orders": 3}, "record_numbers": [1, 2]}])

    def test_allowed_values_csv_jsonl_results_consistent(self):
        csv_text = ("name,orders,active\n"
                    "a,,true\n"
                    "b,3,true\n"
                    "c,0,false\n"
                    "d,4,true\n"
                    "e,bad,true\n")
        jsonl_text = ('{"name": "a", "orders": null, "active": true}\n'
                      '{"name": "b", "orders": "3", "active": true}\n'
                      '{"name": "c", "orders": 0, "active": false}\n'
                      '{"name": "d", "orders": 4, "active": true}\n'
                      '{"name": "e", "orders": "bad", "active": true}\n')
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, csv_text)
            jsonl_path = Path(directory) / "data.jsonl"
            jsonl_path.write_text(jsonl_text, encoding="utf-8")
            csv_result = normalize_csv(csv_path, self.enum_schema())
            jsonl_result = normalize_jsonl(jsonl_path, self.enum_schema())
        self.assertEqual(csv_result["records"], jsonl_result["records"])
        self.assertEqual((csv_result["accepted"], csv_result["rejected"]),
                         (jsonl_result["accepted"], jsonl_result["rejected"]))
        # the CSV header occupies physical line 1; enum messages themselves match
        shifted = [{"row": row["row"] - 1, "errors": row["errors"]}
                   for row in csv_result["errors"]]
        for csv_row, jsonl_row in zip(shifted, jsonl_result["errors"]):
            self.assertEqual(csv_row["row"], jsonl_row["row"])
            for csv_message, jsonl_message in zip(csv_row["errors"], jsonl_row["errors"]):
                if "allowed_values" in csv_message:
                    self.assertEqual(csv_message, jsonl_message)

    def test_cli_allowed_values_bad_config_exit_two_keeps_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(directory, "name,orders,active\nA,3,TRUE\n")
            schema_path = Path(directory) / "schema.json"
            bad_schemas = [
                {"columns": [
                    {"name": "name", "type": "string", "required": True},
                    {"name": "orders", "type": "integer", "allowed_values": []},
                    {"name": "active", "type": "boolean", "required": True}]},
                {"columns": [
                    {"name": "name", "type": "string", "required": True},
                    {"name": "orders", "type": "integer", "allowed_values": [0, 3], "default": 4},
                    {"name": "active", "type": "boolean", "required": True}]},
            ]
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            for bad_schema in bad_schemas:
                schema_path.write_text(json.dumps(bad_schema), encoding="utf-8")
                output.write_text("keep me\n", encoding="utf-8")
                if errors.exists():
                    errors.unlink()
                command = [sys.executable, str(ROOT / "data_importer.py"), str(csv_path),
                           "--schema", str(schema_path), "--output", str(output),
                           "--errors", str(errors)]
                run = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(run.returncode, 2, run.stderr)
                payload = json.loads(run.stdout)
                self.assertEqual(set(payload), {"error"})
                self.assertTrue(payload["error"].strip())
                self.assertIn("allowed_values", payload["error"])
                self.assertIn("orders", payload["error"])
                self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")
                self.assertFalse(errors.exists())

    def test_cli_allowed_values_applied_and_exported(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = self.write_csv(
                directory, "name,orders,active\na,,true\nb,0,false\nc,4,true\n")
            schema_path = Path(directory) / "schema.json"
            schema_path.write_text(json.dumps(self.enum_schema()), encoding="utf-8")
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            command = [sys.executable, str(ROOT / "data_importer.py"), str(csv_path),
                       "--schema", str(schema_path), "--output", str(output),
                       "--errors", str(errors)]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(json.loads(run.stdout), {"accepted": 2, "rejected": 1})
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([record["orders"] for record in records], [3, 0])
            error_rows = [json.loads(line) for line in errors.read_text().splitlines()]
            self.assertEqual(error_rows[0]["row"], 4)
            self.assertIn("allowed_values", error_rows[0]["errors"][0])


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


    def enum_schema(self):
        return {"columns": [
            {"name": "name", "type": "string", "required": True},
            {"name": "orders", "type": "integer", "allowed_values": [0, 3], "default": 3},
            {"name": "active", "type": "boolean", "required": True},
        ]}

    def test_allowed_values_enum_jsonl(self):
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
        enum_error = result["errors"][0]["errors"][0]
        self.assertIn("orders", enum_error)
        self.assertIn("allowed_values", enum_error)
        type_error = result["errors"][1]["errors"][0]
        self.assertIn("orders", type_error)
        self.assertNotIn("allowed_values", type_error)

    def test_allowed_values_jsonl_schema_order_and_null_exemption(self):
        schema = {"columns": [
            {"name": "name", "type": "string", "allowed_values": ["x"]},
            {"name": "orders", "type": "integer", "allowed_values": [0, 3]},
            {"name": "active", "type": "boolean", "allowed_values": [True]},
        ]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(
                directory,
                '{"name": "y", "orders": 7, "active": false}\n'
                '{"name": "x", "orders": null, "active": true}\n')
            result = normalize_jsonl(path, schema)
        self.assertEqual((result["accepted"], result["rejected"]), (1, 1))
        self.assertIsNone(result["records"][0]["orders"])
        prefixes = [message.split(":", 1)[0] for message in result["errors"][0]["errors"]]
        self.assertEqual(prefixes, ["name", "orders", "active"])
        self.assertTrue(all("allowed_values" in message
                            for message in result["errors"][0]["errors"]))

    def test_allowed_values_jsonl_structure_not_repaired(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(
                directory,
                '{"name": "a", "active": true}\n'
                '{"name": "a", "orders": 4, "active": true, "extra": 1}\n')
            result = normalize_jsonl(path, self.enum_schema())
        self.assertEqual(result["rejected"], 2)
        self.assertTrue(result["errors"][0]["errors"][0].startswith("structure error"))
        self.assertTrue(result["errors"][1]["errors"][0].startswith("structure error"))

    def test_allowed_values_invalid_config_raises_before_reading_jsonl(self):
        import copy
        missing = ROOT / "samples" / "does-not-exist.jsonl"
        bad_lists = [None, 3, "x", {}, (), [], [None], [[1]], [{"a": 1}], [False],
                     ["0"], [0.0], [0, 0], [3, 3]]
        for values in bad_lists:
            schema = {"columns": [{"name": "orders", "type": "integer",
                                   "allowed_values": values}]}
            with self.assertRaises(ValueError) as caught:
                normalize_jsonl(missing, schema)
            message = str(caught.exception)
            self.assertIn("allowed_values", message, values)
            self.assertIn("orders", message, values)
        schema = {"columns": [{"name": "orders", "type": "integer",
                               "allowed_values": [0, 3], "default": 9}]}
        with self.assertRaises(ValueError) as caught:
            normalize_jsonl(missing, schema)
        self.assertIn("allowed_values", str(caught.exception))
        self.assertIn("orders", str(caught.exception))
        # caller's schema is not mutated by validation or normalization
        schema = self.enum_schema()
        snapshot = copy.deepcopy(schema)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory, '{"name": "a", "orders": 4, "active": true}\n')
            normalize_jsonl(path, schema)
        self.assertEqual(schema, snapshot)

    def test_allowed_values_jsonl_before_filter_and_duplicates(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(
                directory,
                '{"name": "a", "orders": 3, "active": true}\n'
                '{"name": "b", "orders": 4, "active": true}\n'
                '{"name": "c", "orders": null, "active": true}\n'
                '{"name": "d", "orders": 0, "active": false}\n')
            result = normalize_jsonl(path, self.enum_schema(), duplicate_by=["orders"],
                                     filter_eq={"field": "active", "value": True})
        self.assertEqual((result["accepted"], result["filtered"], result["rejected"]), (2, 1, 1))
        self.assertEqual([r["name"] for r in result["records"]], ["a", "c"])
        self.assertEqual(result["errors"][0]["row"], 2)
        self.assertEqual(result["duplicates"],
                         [{"key": {"orders": 3}, "record_numbers": [1, 2]}])

    def test_cli_jsonl_allowed_values_bad_config_exit_two_keeps_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(directory, '{"name": "A", "orders": 1, "active": true}\n',
                                    name="source.jsonl")
            schema_path = Path(directory) / "schema.json"
            schema_path.write_text(json.dumps({"columns": [
                {"name": "name", "type": "string", "required": True},
                {"name": "orders", "type": "integer", "allowed_values": [1, 1]},
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
            self.assertIn("allowed_values", payload["error"])
            self.assertIn("orders", payload["error"])
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")
            self.assertFalse(errors.exists())

    def test_cli_jsonl_allowed_values_applied_and_exported(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = self.write_jsonl(
                directory,
                '{"name": "a", "orders": null, "active": true}\n'
                '{"name": "b", "orders": 0, "active": false}\n'
                '{"name": "c", "orders": 4, "active": true}\n',
                name="source.jsonl")
            schema_path = Path(directory) / "schema.json"
            schema_path.write_text(json.dumps(self.enum_schema()), encoding="utf-8")
            output, errors = Path(directory) / "data.jsonl", Path(directory) / "errors.jsonl"
            command = [sys.executable, str(ROOT / "data_importer.py"), str(path),
                       "--schema", str(schema_path), "--output", str(output),
                       "--errors", str(errors), "--format", "jsonl"]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(json.loads(run.stdout), {"accepted": 2, "rejected": 1})
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([record["orders"] for record in records], [3, 0])
            error_rows = [json.loads(line) for line in errors.read_text().splitlines()]
            self.assertEqual(error_rows[0]["row"], 3)
            self.assertIn("allowed_values", error_rows[0]["errors"][0])


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


if __name__ == "__main__":
    unittest.main()
