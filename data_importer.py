"""Normalize CSV rows using a small explicit column schema."""
import argparse
import csv
import json
from pathlib import Path


def normalize_csv(source, schema):
    columns = schema["columns"]
    names = [column["name"] for column in columns]
    if not names or len(set(names)) != len(names):
        raise ValueError("schema column names must be nonempty and unique")
    if any(column["type"] not in ("string", "integer", "boolean") for column in columns):
        raise ValueError("unsupported column type")
    sources = []
    for column in columns:
        name = column["name"]
        if "source" not in column:
            sources.append(name)
            continue
        origin = column["source"]
        if not isinstance(origin, str) or not origin.strip():
            raise ValueError(f"column {name!r}: source must be a non-empty, non-blank string")
        sources.append(origin)
    owners = {}
    for column, origin in zip(columns, sources):
        name = column["name"]
        if origin in owners:
            raise ValueError(f"columns {owners[origin]!r} and {name!r} share source {origin!r}")
        owners[origin] = name
    records, errors = [], []
    with Path(source).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != sources:
            raise ValueError("CSV header must match schema column sources and order exactly")
        for row_number, row in enumerate(reader, start=2):
            record = {}
            row_errors = []
            if None in row or any(value is None for value in row.values()):
                row_errors.append("wrong number of cells")
            else:
                for column, origin in zip(columns, sources):
                    name, kind = column["name"], column["type"]
                    value = row[origin].strip()
                    try:
                        if not value:
                            if column.get("required", False):
                                raise ValueError("required value is empty")
                            record[name] = None
                        elif kind == "integer":
                            record[name] = int(value)
                        elif kind == "boolean":
                            if value.casefold() not in ("true", "false"):
                                raise ValueError("boolean must be true or false")
                            record[name] = value.casefold() == "true"
                        else:
                            record[name] = value
                    except ValueError as exc:
                        row_errors.append(f"{name}: {exc}")
            if row_errors:
                errors.append({"row": row_number, "errors": row_errors})
            else:
                records.append(record)
    return {"records": records, "errors": errors, "accepted": len(records), "rejected": len(errors)}


def write_jsonl(path, records):
    Path(path).write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--errors", required=True)
    args = parser.parse_args()
    try:
        destinations = [Path(args.output).resolve(), Path(args.errors).resolve()]
        inputs = [Path(args.source).resolve(), Path(args.schema).resolve()]
        if destinations[0] == destinations[1] or any(path in inputs for path in destinations):
            raise ValueError("output paths must be distinct from each other and inputs")
        result = normalize_csv(args.source, json.loads(Path(args.schema).read_text(encoding="utf-8")))
        write_jsonl(args.output, result["records"])
        write_jsonl(args.errors, result["errors"])
        print(json.dumps({"accepted": result["accepted"], "rejected": result["rejected"]}))
        return 1 if result["rejected"] else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
