"""Normalize CSV rows or JSONL objects using a small explicit schema."""
import argparse
import csv
import json
import os
import stat
import tempfile
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


def _prepare_duplicate_by(duplicate_by, names):
    """Validate the optional duplicate-report fields against output names.

    Returns a new list or None when reporting is disabled. Raises ValueError
    before the input is opened.
    """
    if duplicate_by is None:
        return None
    if not isinstance(duplicate_by, list):
        raise ValueError("duplicate_by must be a list of schema field names")
    if not duplicate_by:
        raise ValueError("duplicate_by must not be empty")
    known = set(names)
    seen = set()
    for member in duplicate_by:
        if not isinstance(member, str) or not member.strip():
            raise ValueError(f"duplicate_by entry {member!r} must be a non-blank string")
        if member in seen:
            raise ValueError(f"duplicate_by field {member!r} is repeated")
        if member not in known:
            raise ValueError(f"duplicate_by field {member!r} is not a schema column name")
        seen.add(member)
    return list(duplicate_by)


def _find_duplicates(records, duplicate_by):
    """Group accepted records by the converted values of duplicate_by fields."""
    groups = {}
    for record_number, record in enumerate(records, start=1):
        value_key = tuple(record[field] for field in duplicate_by)
        groups.setdefault(value_key, []).append(record_number)
    return [
        {"key": dict(zip(duplicate_by, values)), "record_numbers": numbers}
        for values, numbers in groups.items()
        if len(numbers) > 1
    ]


def normalize_csv(source, schema, duplicate_by=None):
    columns, sources = _prepare_schema(schema)
    duplicate_fields = _prepare_duplicate_by(duplicate_by, [column["name"] for column in columns])
    records, errors = [], []
    with Path(source).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            header = None
        if header != sources:
            raise ValueError("CSV header must match schema column sources and order exactly")
        # reader.line_num is the physical line where the row just read ends;
        # LF, CRLF (counted once) and a lone CR all terminate a line. A record's
        # starting physical line is one past the end of the preceding row, so
        # skipped empty lines and newlines inside quoted fields shift it down.
        previous_end = reader.line_num
        for cells in reader:
            start_line = previous_end + 1
            previous_end = reader.line_num
            if not cells:
                # An ordinary empty physical line: skipped as a record but it
                # still occupies a physical line number (tracked above).
                continue
            record = {}
            row_errors = []
            if len(cells) != len(sources):
                row_errors.append("wrong number of cells")
            else:
                for column, value in zip(columns, cells):
                    name, kind = column["name"], column["type"]
                    value = value.strip()
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
                errors.append({"row": start_line, "errors": row_errors})
            else:
                records.append(record)
    result = {"records": records, "errors": errors, "accepted": len(records), "rejected": len(errors)}
    if duplicate_fields is not None:
        result["duplicates"] = _find_duplicates(records, duplicate_fields)
    return result


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


def normalize_jsonl(source, schema, duplicate_by=None):
    columns, sources = _prepare_schema(schema)
    duplicate_fields = _prepare_duplicate_by(duplicate_by, [column["name"] for column in columns])
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
    result = {"records": records, "errors": errors, "accepted": len(records), "rejected": len(errors)}
    if duplicate_fields is not None:
        result["duplicates"] = _find_duplicates(records, duplicate_fields)
    return result


def _encode_jsonl(records):
    return "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)


def write_jsonl(path, records):
    Path(path).write_text(_encode_jsonl(records), encoding="utf-8")


def _check_destination(path):
    """Validate one output target before any file is created or moved.

    Raises OSError naming the target when it is an existing directory, when
    its parent directory is missing, or when the location is not writable.
    Parent directories are never created.
    """
    if path.is_dir():
        raise IsADirectoryError(f"cannot write output to {path}: target is a directory")
    parent = path.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"cannot write output to {path}: parent directory does not exist")
    if not os.access(parent, os.W_OK):
        raise PermissionError(f"cannot write output to {path}: directory is not writable")
    if path.exists() and not os.access(path, os.W_OK):
        raise PermissionError(f"cannot write output to {path}: file is not writable")


def _reserve_path(path, suffix):
    """Reserve and return a unique temporary name next to path."""
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=suffix)
    os.close(descriptor)
    return Path(name)


def _export_outputs(output_path, errors_path, records, row_errors):
    """Replace both output targets or leave both exactly as they were.

    Payloads are staged in the target directories first; existing targets are
    then moved aside and the staged files renamed into place. A failure at any
    point restores the originals (and removes newly created targets), so a
    run ends with both files replaced or with both targets unchanged.
    """
    targets = [output_path, errors_path]
    payloads = [_encode_jsonl(records).encode("utf-8"),
                _encode_jsonl(row_errors).encode("utf-8")]
    umask = os.umask(0)
    os.umask(umask)
    default_mode = 0o666 & ~umask
    # Preflight both targets before touching anything: a directory target, a
    # missing parent or an unwritable location fails with no side effects.
    for target in targets:
        _check_destination(target)
    staged = []
    reserved = []
    try:
        for target, payload in zip(targets, payloads):
            try:
                descriptor, name = tempfile.mkstemp(
                    dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
                staged.append(Path(name))
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                raise OSError(f"failed to write output {target}: {exc}") from exc
        backups = [None, None]
        modes = [None, None]
        moved = []
        committed = []
        try:
            for index, target in enumerate(targets):
                if target.exists():
                    # Preserve the permissions write_text() would have kept
                    # when truncating an existing file.
                    modes[index] = stat.S_IMODE(target.stat().st_mode)
                    backup = _reserve_path(target, ".bak")
                    reserved.append((index, backup))
                    os.replace(target, backup)
                    backups[index] = backup
                    moved.append(index)
            for index, (target, tmp) in enumerate(zip(targets, staged)):
                os.chmod(tmp, modes[index] if modes[index] is not None else default_mode)
                os.replace(tmp, target)
                committed.append(index)
        except OSError as exc:
            failed = target
            # Put every moved-aside original back; targets it overwrote or
            # created are removed first. A restored backup drops out of
            # reserved so a failed restore leaves the .bak recoverable.
            for index in moved:
                try:
                    targets[index].unlink(missing_ok=True)
                    os.replace(backups[index], targets[index])
                    reserved = [(i, name) for i, name in reserved if i != index]
                except OSError:
                    pass
            # A target that did not exist beforehand must stay absent.
            for index in committed:
                if index not in moved:
                    try:
                        targets[index].unlink(missing_ok=True)
                    except OSError:
                        pass
            raise OSError(f"failed to write output {failed}: {exc}") from exc
        # Both staged files are in place; the moved-aside originals go away.
        for _, backup in reserved:
            backup.unlink(missing_ok=True)
        reserved.clear()
    finally:
        for tmp in staged:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        # Only empty reservations whose move never happened land here.
        for _, backup in reserved:
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--errors", required=True)
    parser.add_argument("--format", default="csv", help="input format: csv (default) or jsonl")
    parser.add_argument("--duplicate-by", action="append", metavar="FIELD",
                        help="output field to report duplicate accepted records by; repeatable")
    args = parser.parse_args()
    try:
        destinations = [Path(args.output).resolve(), Path(args.errors).resolve()]
        inputs = [Path(args.source).resolve(), Path(args.schema).resolve()]
        if destinations[0] == destinations[1] or any(path in inputs for path in destinations):
            raise ValueError("output paths must be distinct from each other and inputs")
        if args.format not in ("csv", "jsonl"):
            raise ValueError("format must be csv or jsonl")
        normalize = normalize_csv if args.format == "csv" else normalize_jsonl
        result = normalize(args.source, json.loads(Path(args.schema).read_text(encoding="utf-8")),
                           duplicate_by=args.duplicate_by)
        _export_outputs(Path(args.output), Path(args.errors), result["records"], result["errors"])
        summary = {"accepted": result["accepted"], "rejected": result["rejected"]}
        if args.duplicate_by is not None:
            summary["duplicates"] = result["duplicates"]
        print(json.dumps(summary))
        return 1 if result["rejected"] else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
