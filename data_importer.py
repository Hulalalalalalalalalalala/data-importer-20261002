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
    return columns, names, sources


def _convert_value(kind, value):
    """Convert a trimmed CSV cell into the column type; raise ValueError on failure."""
    if not value:
        raise ValueError("required value is empty")
    if kind == "integer":
        return int(value)
    if kind == "boolean":
        if value.casefold() not in ("true", "false"):
            raise ValueError("boolean must be true or false")
        return value.casefold() == "true"
    return value


def _convert_json_value(kind, value):
    """Convert a trimmed JSONL field value into the column type; raise ValueError on failure."""
    if kind == "string":
        if not isinstance(value, str):
            raise ValueError("must be a string")
        return value
    if kind == "integer":
        if isinstance(value, bool):
            raise ValueError("must be an integer")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value)
        raise ValueError("must be an integer")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.casefold() not in ("true", "false"):
            raise ValueError("boolean must be true or false")
        return value.casefold() == "true"
    raise ValueError("must be a boolean")


def normalize_csv(source, schema):
    columns, names, sources = _prepare_schema(schema)
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
                        else:
                            record[name] = _convert_value(kind, value)
                    except ValueError as exc:
                        row_errors.append(f"{name}: {exc}")
            if row_errors:
                errors.append({"row": row_number, "errors": row_errors})
            else:
                records.append(record)
    return {"records": records, "errors": errors, "accepted": len(records), "rejected": len(errors)}


def _reject_constant(value):
    raise ValueError(f"{value} is not valid JSON")


def normalize_jsonl(source, schema):
    columns, names, sources = _prepare_schema(schema)
    expected = set(sources)
    records, errors = [], []
    # utf-8-sig accepts a leading BOM and decodes plain UTF-8 identically otherwise.
    with Path(source).open(encoding="utf-8-sig", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row_errors = []
            record = None
            object_pairs = []

            def pairs_hook(pairs):
                object_pairs.append(pairs)
                return dict(pairs)

            try:
                obj = json.loads(line, parse_constant=_reject_constant, object_pairs_hook=pairs_hook)
            except ValueError as exc:
                row_errors.append(f"parse error: {exc}")
                obj = None
            if not row_errors:
                if not isinstance(obj, dict):
                    row_errors.append("structure error: top-level value must be a JSON object")
                else:
                    # For a top-level object the outermost pairs hook runs last.
                    root_pairs = object_pairs[-1]
                    seen = set()
                    duplicates = []
                    for key, _ in root_pairs:
                        if key in seen and key not in duplicates:
                            duplicates.append(key)
                        seen.add(key)
                    if duplicates:
                        row_errors.append("structure error: duplicate source key: " + ", ".join(repr(key) for key in duplicates))
                    else:
                        missing = [origin for origin in sources if origin not in seen]
                        extra = [origin for origin in obj if origin not in expected]
                        if missing or extra:
                            details = []
                            if missing:
                                details.append("missing source keys: " + ", ".join(missing))
                            if extra:
                                details.append("unexpected source keys: " + ", ".join(extra))
                            row_errors.append("structure error: " + "; ".join(details))
                        else:
                            record = {}
                            for column, origin in zip(columns, sources):
                                name, kind = column["name"], column["type"]
                                value = obj[origin]
                                try:
                                    if isinstance(value, str):
                                        value = value.strip()
                                    if value is None or (isinstance(value, str) and not value):
                                        if column.get("required", False):
                                            raise ValueError("required value is empty")
                                        record[name] = None
                                    else:
                                        record[name] = _convert_json_value(kind, value)
                                except ValueError as exc:
                                    row_errors.append(f"{name}: {exc}")
            if row_errors:
                errors.append({"row": line_number, "errors": row_errors})
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
    parser.add_argument("--format", default="csv")
    args = parser.parse_args()
    try:
        if args.format not in ("csv", "jsonl"):
            raise ValueError("--format must be csv or jsonl")
        destinations = [Path(args.output).resolve(), Path(args.errors).resolve()]
        inputs = [Path(args.source).resolve(), Path(args.schema).resolve()]
        if destinations[0] == destinations[1] or any(path in inputs for path in destinations):
            raise ValueError("output paths must be distinct from each other and inputs")
        schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
        normalize = normalize_csv if args.format == "csv" else normalize_jsonl
        result = normalize(args.source, schema)
        write_jsonl(args.output, result["records"])
        write_jsonl(args.errors, result["errors"])
        print(json.dumps({"accepted": result["accepted"], "rejected": result["rejected"]}))
        return 1 if result["rejected"] else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
