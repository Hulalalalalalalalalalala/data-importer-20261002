"""Normalize CSV rows or JSONL objects using a small explicit schema."""
import argparse
import csv
import json
from pathlib import Path


def _prepare_schema(schema):
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
    return columns, sources


def normalize_csv(source, schema):
    columns, sources = _prepare_schema(schema)
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


def _convert_jsonl_value(column, value):
    """Convert one decoded JSON value. Returns (converted, error_message)."""
    name, kind = column["name"], column["type"]
    if isinstance(value, str):
        value = value.strip()
    if value is None or value == "":
        if column.get("required", False):
            return None, f"{name}: required value is empty"
        return None, None
    if kind == "string":
        if not isinstance(value, str):
            return None, f"{name}: expected string"
        return value, None
    if kind == "integer":
        if isinstance(value, bool):
            return None, f"{name}: expected integer"
        if isinstance(value, int):
            return value, None
        if isinstance(value, str):
            try:
                return int(value), None
            except ValueError:
                pass
        return None, f"{name}: expected integer"
    if isinstance(value, bool):
        return value, None
    if isinstance(value, str):
        folded = value.casefold()
        if folded in ("true", "false"):
            return folded == "true", None
    return None, f"{name}: boolean must be true or false"


def normalize_jsonl(source, schema):
    columns, sources = _prepare_schema(schema)
    expected = set(sources)
    records, errors = [], []
    decoded_pairs = []

    def pairs_hook(pairs):
        decoded_pairs.append(pairs)
        return dict(pairs)

    def reject_constant(token):
        raise ValueError(f"constant {token} is not allowed")

    with Path(source).open(encoding="utf-8-sig", newline="") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                decoded_pairs.clear()
                obj = json.loads(line, parse_constant=reject_constant, object_pairs_hook=pairs_hook)
            except (ValueError, RecursionError) as exc:
                errors.append({"row": row_number, "errors": [f"parse error: {exc}"]})
                continue
            if not isinstance(obj, dict):
                errors.append({"row": row_number, "errors": ["structure error: top-level value must be an object"]})
                continue
            top_keys = [key for key, _ in decoded_pairs[-1]]
            seen = set()
            duplicate = None
            for key in top_keys:
                if key in seen:
                    duplicate = key
                    break
                seen.add(key)
            if duplicate is not None:
                errors.append({"row": row_number, "errors": [f"structure error: duplicate key {duplicate!r}"]})
                continue
            row_errors = []
            missing = [origin for origin in sources if origin not in seen]
            extra = sorted(seen - expected)
            if missing:
                row_errors.append("structure error: missing source keys: " + ", ".join(missing))
            if extra:
                row_errors.append("structure error: unexpected source keys: " + ", ".join(extra))
            if row_errors:
                errors.append({"row": row_number, "errors": row_errors})
                continue
            record = {}
            for column, origin in zip(columns, sources):
                converted, message = _convert_jsonl_value(column, obj[origin])
                if message is not None:
                    row_errors.append(message)
                else:
                    record[column["name"]] = converted
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
    parser.add_argument("--format", default="csv", help="input format: csv (default) or jsonl")
    args = parser.parse_args()
    try:
        destinations = [Path(args.output).resolve(), Path(args.errors).resolve()]
        inputs = [Path(args.source).resolve(), Path(args.schema).resolve()]
        if destinations[0] == destinations[1] or any(path in inputs for path in destinations):
            raise ValueError("output paths must be distinct from each other and inputs")
        if args.format not in ("csv", "jsonl"):
            raise ValueError("format must be csv or jsonl")
        normalize = normalize_csv if args.format == "csv" else normalize_jsonl
        result = normalize(args.source, json.loads(Path(args.schema).read_text(encoding="utf-8")))
        write_jsonl(args.output, result["records"])
        write_jsonl(args.errors, result["errors"])
        print(json.dumps({"accepted": result["accepted"], "rejected": result["rejected"]}))
        return 1 if result["rejected"] else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
