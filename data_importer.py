"""Normalize CSV rows or JSONL objects using a small explicit schema."""
import argparse
import contextlib
import csv
import json
import os
import tempfile
from pathlib import Path


def _prepare_defaults(columns):
    """Validate the optional target-typed ``default`` of each column.

    Returns a mapping of output field name to the processed default
    (string defaults are trimmed). A malformed default is a configuration
    error naming the default value and output field, raised before the
    input is read; the caller's schema dicts are never mutated.
    """
    defaults = {}
    for column in columns:
        name, kind = column["name"], column["type"]
        if "default" not in column:
            continue
        default = column["default"]
        if kind == "string":
            if not isinstance(default, str):
                raise ValueError(f"column {name!r} default {default!r} must be a string")
            processed = default.strip()
            if not processed:
                raise ValueError(f"column {name!r} default {default!r} must not be blank")
            defaults[name] = processed
        elif kind == "integer":
            if isinstance(default, bool) or not isinstance(default, int):
                raise ValueError(
                    f"column {name!r} default {default!r} must be a non-boolean integer")
            defaults[name] = default
        else:
            if not isinstance(default, bool):
                raise ValueError(f"column {name!r} default {default!r} must be a boolean")
            defaults[name] = default
    return defaults


def _enum_field_error(name, value, allowed):
    """Return the enum error for a converted value, or None when it is allowed.

    Null is never checked: an empty optional without a default stays null
    and membership is decided after type conversion and empty handling.
    """
    values = allowed.get(name)
    if values is not None and value is not None and value not in values:
        return f"{name}: value {value!r} is not one of allowed_values {values!r}"
    return None


def _prepare_allowed_values(columns, defaults):
    """Validate the optional ``allowed_values`` list of each column.

    Returns a mapping of output field name to a fresh list holding the
    declared entries verbatim (strings are neither trimmed nor case
    folded). The attribute must be a non-empty list of target-typed
    values: strings for ``string`` columns, non-boolean integers for
    ``integer`` columns, booleans for ``boolean`` columns; null, arrays
    and objects are never valid entries, and repeated entries (value
    equality, verbatim for strings) are refused. A processed default
    outside the list is a configuration error even when no row needs it.
    Raises ValueError naming ``allowed_values`` and the output field
    before the input is read; the caller's schema is never mutated.
    """
    allowed = {}
    for column in columns:
        name, kind = column["name"], column["type"]
        if "allowed_values" not in column:
            continue
        values = column["allowed_values"]
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"column {name!r} allowed_values {values!r} must be a non-empty list")
        checked = []
        for entry in values:
            if kind == "string":
                if not isinstance(entry, str):
                    raise ValueError(
                        f"column {name!r} allowed_values entry {entry!r} must be a string")
            elif kind == "integer":
                if isinstance(entry, bool) or not isinstance(entry, int):
                    raise ValueError(
                        f"column {name!r} allowed_values entry {entry!r} "
                        "must be a non-boolean integer")
            elif not isinstance(entry, bool):
                raise ValueError(
                    f"column {name!r} allowed_values entry {entry!r} must be a boolean")
            if entry in checked:
                raise ValueError(
                    f"column {name!r} allowed_values entry {entry!r} is repeated")
            checked.append(entry)
        allowed[name] = checked
        if name in defaults and defaults[name] not in checked:
            raise ValueError(
                f"column {name!r} default {defaults[name]!r} is not one of "
                f"allowed_values {checked!r}")
    return allowed


def _validate_schema_structure(schema):
    """Validate the structural shape of the schema before any file is read.

    The schema must be an object holding a non-empty ``columns`` list;
    each column must be an object carrying ``name`` (a non-blank string,
    kept verbatim -- never trimmed -- and unique case-sensitively) and
    ``type`` (one of ``string``, ``integer``, ``boolean``), plus an
    optional ``required`` that, when present, must be a boolean. Checks
    run top level first, then column by column in order (column
    structure, name, duplicate name, type, required) and only the first
    problem is reported: the ValueError message names ``schema`` and the
    one-based column position, names a missing key, carries the
    offending value verbatim, and points a duplicate name at the later
    column. Unknown attributes are ignored and the caller's schema is
    never mutated.
    """
    if not isinstance(schema, dict):
        raise ValueError(f"schema must be an object, got {schema!r}")
    if "columns" not in schema:
        raise ValueError("schema is missing 'columns'")
    columns = schema["columns"]
    if not isinstance(columns, list) or not columns:
        raise ValueError(f"schema 'columns' must be a non-empty list, got {columns!r}")
    seen = {}
    for index, column in enumerate(columns, start=1):
        if not isinstance(column, dict):
            raise ValueError(f"schema column {index} must be an object, got {column!r}")
        for key in ("name", "type"):
            if key not in column:
                raise ValueError(f"schema column {index} is missing {key!r}")
        name = column["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"schema column {index} name {name!r} must be a non-blank string")
        if name in seen:
            raise ValueError(
                f"schema column {index} name {name!r} duplicates column {seen[name]}")
        seen[name] = index
        kind = column["type"]
        if kind not in ("string", "integer", "boolean"):
            raise ValueError(
                f"schema column {index} type {kind!r} must be one of "
                "'string', 'integer', 'boolean'")
        if "required" in column and not isinstance(column["required"], bool):
            raise ValueError(
                f"schema column {index} required {column['required']!r} must be a boolean")


def _prepare_schema(schema):
    _validate_schema_structure(schema)
    columns = schema["columns"]
    defaults = _prepare_defaults(columns)
    allowed = _prepare_allowed_values(columns, defaults)
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
    return columns, sources, defaults, allowed


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


def _prepare_filter_eq(filter_eq, columns):
    """Validate the optional single-field equality filter.

    The condition is an object with exactly ``field`` and ``value`` keys;
    field matches an output name literally (source aliases are not
    recognized) and value must fit the target type, with null allowed for
    every type. Returns (field, value) or None when filtering is disabled.
    Raises ValueError before the input is read.
    """
    if filter_eq is None:
        return None
    if not isinstance(filter_eq, dict):
        raise ValueError("filter_eq must be an object with 'field' and 'value' keys")
    keys = set(filter_eq)
    if keys != {"field", "value"}:
        details = []
        missing = [key for key in ("field", "value") if key not in keys]
        extra = sorted(keys - {"field", "value"})
        if missing:
            details.append("missing key(s): " + ", ".join(missing))
        if extra:
            details.append("unexpected key(s): " + ", ".join(extra))
        raise ValueError("filter_eq must contain exactly 'field' and 'value' keys (" + "; ".join(details) + ")")
    field = filter_eq["field"]
    if not isinstance(field, str):
        raise ValueError("filter_eq field must be a string")
    if not field.strip():
        raise ValueError("filter_eq field must be a non-blank string")
    by_name = {column["name"]: column for column in columns}
    if field not in by_name:
        raise ValueError(f"filter_eq field {field!r} is not a schema column name")
    value = filter_eq["value"]
    kind = by_name[field]["type"]
    if value is not None:
        if kind == "string" and not isinstance(value, str):
            raise ValueError(f"filter_eq value for field {field!r} must be a string or null")
        if kind == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"filter_eq value for field {field!r} must be a non-boolean integer or null")
        if kind == "boolean" and not isinstance(value, bool):
            raise ValueError(f"filter_eq value for field {field!r} must be a boolean or null")
    return field, value


def _build_result(records, errors, duplicate_fields, filter_condition):
    """Apply the equality filter to fully validated records, then counts."""
    if filter_condition is None:
        kept, filtered_count = records, 0
    else:
        field, expected = filter_condition
        kept = [record for record in records if record[field] == expected]
        filtered_count = len(records) - len(kept)
    result = {"records": kept, "errors": errors, "accepted": len(kept),
              "rejected": len(errors)}
    if filter_condition is not None:
        result["filtered"] = filtered_count
    if duplicate_fields is not None:
        result["duplicates"] = _find_duplicates(kept, duplicate_fields)
    return result


def _csv_parse_error(source, start_line, reason):
    """Build the ValueError for a fatal CSV quote-syntax failure.

    The message carries the literal "CSV parse error", the input path, the
    offending record's starting physical line (header is line 1; a data
    record starts on the physical line after the preceding row ends), and
    a reason distinguishing a quote left open at end of file from text
    appearing right after a closing quote.
    """
    return ValueError(
        f"CSV parse error in {source} at record starting line {start_line}: {reason}")


def _csv_error_reason(exc):
    """Map a strict-reader ``csv.Error`` onto the two fatal quote problems.

    ``strict=True`` raises "unexpected end of data" for a quoted field
    still open at EOF and "',' expected after '\"'" for any character
    other than comma/newline/EOF right after a closing quote (spaces and
    tabs included). Anything else is reported verbatim.
    """
    message = str(exc)
    if message == "unexpected end of data":
        return ("unterminated quoted field: a double-quoted field was still "
                "open when the file ended")
    if "expected after" in message:
        return ("invalid text after closing quote: only a comma, a line "
                "ending or the end of the file may follow a closing "
                "double quote")
    return message


def normalize_csv(source, schema, duplicate_by=None, filter_eq=None):
    columns, sources, defaults, allowed = _prepare_schema(schema)
    names = [column["name"] for column in columns]
    duplicate_fields = _prepare_duplicate_by(duplicate_by, names)
    filter_condition = _prepare_filter_eq(filter_eq, columns)
    records, errors = [], []
    # strict=True turns the two quote-syntax problems that a lenient reader
    # silently folds into the data into csv.Error: a quoted field still open
    # at EOF and any character other than comma/newline/EOF right after a
    # closing quote (a space or tab included). A doubled "" inside a quoted
    # field still decodes to one literal double quote, and a double quote
    # inside an unquoted field stays an ordinary character.
    with Path(source).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        # reader.line_num is the physical line where the row just read ends;
        # LF, CRLF (counted once) and a lone CR all terminate a line. A
        # record's starting physical line is one past the end of the
        # preceding row, so skipped empty lines and newlines inside quoted
        # fields shift it down. When the strict reader reports a syntax
        # error the failing record starts on previous_end + 1, even though
        # the problem is only detected at a later physical line or at EOF.
        previous_end = 0
        header = None
        try:
            for cells in reader:
                start_line = previous_end + 1
                previous_end = reader.line_num
                if header is None:
                    header = cells
                    if header != sources:
                        raise ValueError(
                            "CSV header must match schema column sources and order exactly")
                    continue
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
                                if name in defaults:
                                    converted = defaults[name]
                                elif column.get("required", False):
                                    raise ValueError("required value is empty")
                                else:
                                    converted = None
                            elif kind == "integer":
                                converted = int(value)
                            elif kind == "boolean":
                                if value.casefold() not in ("true", "false"):
                                    raise ValueError("boolean must be true or false")
                                converted = value.casefold() == "true"
                            else:
                                converted = value
                        except ValueError as exc:
                            row_errors.append(f"{name}: {exc}")
                            continue
                        enum_error = _enum_field_error(name, converted, allowed)
                        if enum_error is not None:
                            row_errors.append(enum_error)
                        else:
                            record[name] = converted
                if row_errors:
                    errors.append({"row": start_line, "errors": row_errors})
                else:
                    records.append(record)
        except csv.Error as exc:
            raise _csv_parse_error(
                source, previous_end + 1, _csv_error_reason(exc)) from None
        if header is None:
            raise ValueError("CSV header must match schema column sources and order exactly")
    return _build_result(records, errors, duplicate_fields, filter_condition)


def _convert_jsonl_value(column, value, defaults):
    """Convert one decoded JSON value. Returns (converted, error_message)."""
    name, kind = column["name"], column["type"]
    if isinstance(value, str):
        value = value.strip()
    if value is None or value == "":
        if name in defaults:
            return defaults[name], None
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


def normalize_jsonl(source, schema, duplicate_by=None, filter_eq=None):
    columns, sources, defaults, allowed = _prepare_schema(schema)
    names = [column["name"] for column in columns]
    duplicate_fields = _prepare_duplicate_by(duplicate_by, names)
    filter_condition = _prepare_filter_eq(filter_eq, columns)
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
                converted, message = _convert_jsonl_value(column, obj[origin], defaults)
                if message is not None:
                    row_errors.append(message)
                    continue
                enum_error = _enum_field_error(column["name"], converted, allowed)
                if enum_error is not None:
                    row_errors.append(enum_error)
                else:
                    record[column["name"]] = converted
            if row_errors:
                errors.append({"row": row_number, "errors": row_errors})
            else:
                records.append(record)
    return _build_result(records, errors, duplicate_fields, filter_condition)


def write_jsonl(path, records):
    Path(path).write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def _stage_jsonl(destination, records):
    """Serialize records into a temp file in the destination's directory.

    Parent directories are never created and the destination is not
    touched. Returns (staged_path, existed); raised OSError names the
    target path.
    """
    target = Path(destination)
    parent = target.parent
    if not parent.exists() or not parent.is_dir():
        raise OSError(f"cannot write {target}: parent directory does not exist")
    if target.exists() and target.is_dir():
        raise OSError(f"cannot write {target}: target is a directory")
    if target.exists() and not os.access(target, os.W_OK):
        raise OSError(f"cannot write {target}: target is not writable")
    payload = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    try:
        handle, staged_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=parent)
    except OSError as exc:
        raise OSError(f"cannot write {target}: {exc}") from None
    staged = parent / staged_name
    try:
        out = os.fdopen(handle, "w", encoding="utf-8", newline="")
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(handle)
        with contextlib.suppress(OSError):
            staged.unlink()
        raise
    try:
        with out:
            out.write(payload)
    except OSError as exc:
        with contextlib.suppress(OSError):
            staged.unlink()
        raise OSError(f"cannot write {target}: {exc}") from None
    except BaseException:
        with contextlib.suppress(OSError):
            staged.unlink()
        raise
    return staged, target.exists() and not target.is_dir()


def _install_staged(target, staged, existed):
    """Replace target with staged, keeping its old bytes recoverable.

    Returns the backup path (or None). If the swap fails after the
    original was moved aside, the original is restored at target before
    the OSError (naming target) is raised.
    """
    mode = None
    backup = None
    try:
        if existed:
            mode = target.stat().st_mode & 0o7777
            # Reserve a unique backup name so an unrelated user file can
            # never be clobbered; os.replace below overwrites the empty
            # placeholder with the original file.
            backup_handle, backup_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".backup", dir=target.parent)
            os.close(backup_handle)
            backup = target.parent / backup_name
            os.replace(target, backup)
        os.replace(staged, target)
        if mode is not None:
            with contextlib.suppress(OSError):
                target.chmod(mode)
        else:
            # mkstemp creates 0600; a freshly created output historically
            # got the usual 0666 & ~umask mode from open().
            umask = os.umask(0)
            os.umask(umask)
            with contextlib.suppress(OSError):
                target.chmod(0o666 & ~umask)
    except OSError as exc:
        if backup is not None:
            if target.exists():
                # The original never left target; just drop the placeholder.
                with contextlib.suppress(OSError):
                    backup.unlink()
            else:
                with contextlib.suppress(OSError):
                    os.replace(backup, target)
        raise OSError(f"cannot write {target}: {exc}") from None
    return backup


def _write_jsonl_outputs_atomic(output_path, records, errors_path, errors):
    """Write both JSONL outputs as one all-or-nothing operation.

    Either both files are fully written (replacing existing files), or a
    filesystem failure leaves every target exactly as it was beforehand:
    pre-existing files keep their bytes and previously absent files stay
    absent. Parent directories are never created.
    """
    output_target, errors_target = Path(output_path), Path(errors_path)
    staged_output, output_existed = _stage_jsonl(output_target, records)
    try:
        staged_errors, errors_existed = _stage_jsonl(errors_target, errors)
    except BaseException:
        with contextlib.suppress(OSError):
            staged_output.unlink()
        raise
    installs = []
    try:
        installs.append((output_target,
                         _install_staged(output_target, staged_output, output_existed)))
        installs.append((errors_target,
                         _install_staged(errors_target, staged_errors, errors_existed)))
    except BaseException:
        # Undo only fully completed installs; a failed _install_staged
        # has already restored that target itself.
        for target, backup in installs:
            with contextlib.suppress(OSError):
                if backup is not None:
                    os.replace(backup, target)
                else:
                    target.unlink()
        for leftover in (staged_errors, staged_output):
            with contextlib.suppress(OSError):
                leftover.unlink()
        raise
    for _, backup in installs:
        if backup is not None:
            with contextlib.suppress(OSError):
                backup.unlink()


def _check_path_isolation(source, schema, output, errors):
    """Reject when an output aliases an input or the other output.

    Runs before either input is read or either output is written. Two
    paths alias when their resolved pathnames are equal (identical paths,
    including relative/absolute spellings and symbolic-link aliases) or,
    when both paths currently exist, when they share a device and inode
    (distinct hard links to the same file). Outputs that do not exist yet
    only take part in the pathname comparison, so two fresh names are not
    refused merely because they could later be linked together. File
    names, extensions and directories never decide identity, and two
    independent files with identical contents are always allowed. Any I/O
    error resolving or stating a path propagates as OSError naming it.
    """
    roles = (
        ("source", Path(source)),
        ("schema", Path(schema)),
        ("records output", Path(output)),
        ("errors output", Path(errors)),
    )

    def resolve(path):
        try:
            return path.resolve()
        except OSError as exc:
            raise OSError(f"cannot check path {path}: {exc}") from None

    def file_identity(labeled_path, resolved):
        try:
            result = resolved.stat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise OSError(f"cannot check path {labeled_path}: {exc}") from None
        return result.st_dev, result.st_ino

    entries = [(label, path, resolve(path)) for label, path in roles]
    # Pairs involving at least one output, in a fixed order: the two
    # outputs, then records output against the inputs, then errors output.
    pairs = ((2, 3), (2, 0), (2, 1), (3, 0), (3, 1))
    for left_index, right_index in pairs:
        left_label, left_path, left_resolved = entries[left_index]
        right_label, right_path, right_resolved = entries[right_index]
        if left_resolved == right_resolved:
            raise ValueError(
                f"paths must be distinct: {left_label} {left_path} and "
                f"{right_label} {right_path} point to the same file")
        left_identity = file_identity(left_path, left_resolved)
        right_identity = file_identity(right_path, right_resolved)
        # A not-yet-existing path (or a dangling link to one) cannot
        # currently be hard-linked to an existing file; the resolved
        # pathname comparison above still applies to both spellings.
        if left_identity is None or right_identity is None:
            continue
        if left_identity == right_identity:
            raise ValueError(
                f"paths must be distinct: {left_label} {left_path} and "
                f"{right_label} {right_path} point to the same existing file")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--errors", required=True)
    parser.add_argument("--format", default="csv", help="input format: csv (default) or jsonl")
    parser.add_argument("--duplicate-by", action="append", metavar="FIELD",
                        help="output field to report duplicate accepted records by; repeatable")
    parser.add_argument("--filter-eq", metavar="JSON",
                        help='equality condition as JSON, e.g. {"field": "active", "value": true}')
    args = parser.parse_args()
    try:
        _check_path_isolation(args.source, args.schema, args.output, args.errors)
        if args.format not in ("csv", "jsonl"):
            raise ValueError("format must be csv or jsonl")
        filter_eq = None
        if args.filter_eq is not None:
            try:
                filter_eq = json.loads(args.filter_eq)
            except ValueError as exc:
                raise ValueError(f"filter_eq is not valid JSON: {exc}") from None
            if not isinstance(filter_eq, dict):
                raise ValueError("filter_eq must be a JSON object with 'field' and 'value' keys")
        normalize = normalize_csv if args.format == "csv" else normalize_jsonl
        result = normalize(args.source, json.loads(Path(args.schema).read_text(encoding="utf-8")),
                           duplicate_by=args.duplicate_by, filter_eq=filter_eq)
        _write_jsonl_outputs_atomic(args.output, result["records"], args.errors, result["errors"])
        summary = {"accepted": result["accepted"], "rejected": result["rejected"]}
        if args.filter_eq is not None:
            summary["filtered"] = result["filtered"]
        if args.duplicate_by is not None:
            summary["duplicates"] = result["duplicates"]
        print(json.dumps(summary))
        return 1 if result["rejected"] else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
