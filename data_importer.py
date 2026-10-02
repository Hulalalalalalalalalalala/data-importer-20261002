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


def _prepare_ranges(columns, defaults):
    """Validate the optional ``minimum``/``maximum`` bounds of each column.

    Returns a mapping of output field name to a ``(minimum, maximum)``
    pair, with None for an unset end; either end may be declared alone
    and equal ends accept exactly that one integer. Bounds may appear
    only on ``integer`` columns and each must be a non-boolean integer
    (negative values and zero are legal): null, strings, floats and
    booleans are configuration errors, as is a minimum greater than the
    maximum. A processed default outside the declared range is a
    configuration error even when no row needs it. Raises ValueError
    naming the output field, the attribute and the offending value
    before the input is read; the caller's schema is never mutated.
    """
    ranges = {}
    for column in columns:
        name, kind = column["name"], column["type"]
        bounds = {}
        for attribute in ("minimum", "maximum"):
            if attribute not in column:
                continue
            bound = column[attribute]
            if kind != "integer":
                raise ValueError(
                    f"column {name!r} {attribute} {bound!r} requires an integer column")
            if isinstance(bound, bool) or not isinstance(bound, int):
                raise ValueError(
                    f"column {name!r} {attribute} {bound!r} must be a non-boolean integer")
            bounds[attribute] = bound
        if not bounds:
            continue
        minimum = bounds.get("minimum")
        maximum = bounds.get("maximum")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(
                f"column {name!r} minimum {minimum!r} is greater than "
                f"maximum {maximum!r}")
        if name in defaults:
            default = defaults[name]
            if minimum is not None and default < minimum:
                raise ValueError(
                    f"column {name!r} default {default!r} is less than "
                    f"minimum {minimum!r}")
            if maximum is not None and default > maximum:
                raise ValueError(
                    f"column {name!r} default {default!r} is greater than "
                    f"maximum {maximum!r}")
        ranges[name] = (minimum, maximum)
    return ranges


def _range_field_error(name, value, ranges):
    """Return the range error for a converted value, or None when in range.

    Null is never checked: an empty optional without a default stays null
    and the inclusive bounds only see non-null integers, after type
    conversion, empty handling and the enum check.
    """
    bounds = ranges.get(name)
    if bounds is None or value is None:
        return None
    minimum, maximum = bounds
    if minimum is not None and value < minimum:
        return f"{name}: value {value!r} is less than minimum {minimum!r}"
    if maximum is not None and value > maximum:
        return f"{name}: value {value!r} is greater than maximum {maximum!r}"
    return None


def _lte_field_errors(record, field_errors, lte_fields):
    """Check every declared ``lte_field`` relation over final integers.

    Runs after structure checks, marker matching, default filling, type
    conversion, enum and range checks. A relation is skipped when either
    final value is null or either participating field already has a
    required, type, enum or range error; the original errors are kept and
    every other relation is still checked. Relations are evaluated in
    schema column order and returned as ``(name, message)`` pairs so the
    caller can file each error under its declaring column's position; a
    violation names ``lte_field``, both output names and the two actual
    integers (the local value larger than the target value fails;
    equality is legal).
    """
    errors = []
    for name, target in lte_fields:
        if name in field_errors or target in field_errors:
            continue
        local = record.get(name)
        other = record.get(target)
        if local is None or other is None:
            continue
        if local > other:
            errors.append((
                name,
                f"{name}: value {local!r} is greater than lte_field "
                f"{target!r} value {other!r}"))
    return errors


def _prepare_lte_fields(columns):
    """Validate the optional ``lte_field`` reference of each integer column.

    Returns a list of ``(name, target)`` pairs in schema column order, one
    entry per declaring column. The attribute must be a non-blank string
    naming another ``integer`` column by its output name (source aliases
    are not recognized); a non-string, a blank string, an unknown field,
    a self reference, a non-integer target, or declaring ``lte_field`` on
    a non-integer column is a configuration error. Raises ValueError
    naming ``lte_field``, the output field and the attribute value before
    the input is read; the caller's schema is never mutated.
    """
    by_name = {column["name"]: column for column in columns}
    lte_fields = []
    for column in columns:
        name, kind = column["name"], column["type"]
        if "lte_field" not in column:
            continue
        target = column["lte_field"]
        if not isinstance(target, str):
            raise ValueError(
                f"column {name!r} lte_field {target!r} must be a string naming "
                "an integer column")
        if not target.strip():
            raise ValueError(
                f"column {name!r} lte_field {target!r} must be a non-blank string")
        if kind != "integer":
            raise ValueError(
                f"column {name!r} lte_field {target!r} requires an integer column")
        if target == name:
            raise ValueError(
                f"column {name!r} lte_field {target!r} must not reference the column itself")
        if target not in by_name:
            raise ValueError(
                f"column {name!r} lte_field {target!r} is not a schema column name")
        if by_name[target]["type"] != "integer":
            raise ValueError(
                f"column {name!r} lte_field {target!r} must reference an integer column")
        lte_fields.append((name, target))
    return lte_fields


def _prepare_missing_values(columns):
    """Validate the optional ``missing_values`` marker list of each column.

    Returns a mapping of output field name to a fresh list of the declared
    markers, each trimmed of surrounding whitespace. The attribute must be
    a non-empty list of strings; an entry that trims to empty or repeats
    an earlier trimmed entry (compared case-sensitively) is invalid.
    Raises ValueError naming ``missing_values``, the output field and the
    offending value before the input is read; the caller's schema is
    never mutated.
    """
    missing = {}
    for column in columns:
        name = column["name"]
        if "missing_values" not in column:
            continue
        values = column["missing_values"]
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"column {name!r} missing_values {values!r} must be a non-empty list")
        checked = []
        for entry in values:
            if not isinstance(entry, str):
                raise ValueError(
                    f"column {name!r} missing_values entry {entry!r} must be a string")
            marker = entry.strip()
            if not marker:
                raise ValueError(
                    f"column {name!r} missing_values entry {entry!r} must not be blank")
            if marker in checked:
                raise ValueError(
                    f"column {name!r} missing_values entry {entry!r} is repeated")
            checked.append(marker)
        missing[name] = checked
    return missing


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
    markers = _prepare_missing_values(columns)
    ranges = _prepare_ranges(columns, defaults)
    lte_fields = _prepare_lte_fields(columns)
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
    return columns, sources, defaults, allowed, markers, ranges, lte_fields


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


def _prepare_deduplicate_by(deduplicate_by, names):
    """Validate the optional first-of-group dedup fields against output names.

    Returns a new list or None when deduplication is disabled. The value
    must be a non-empty list of non-blank strings, each naming a schema
    output column literally (never trimmed; source aliases are not recognized), with
    no repeat and no unknown field. Raises ValueError naming
    ``deduplicate_by`` and the offending value before the input is opened; the
    caller's list and schema are never mutated.
    """
    if deduplicate_by is None:
        return None
    if not isinstance(deduplicate_by, list):
        raise ValueError(
            f"deduplicate_by {deduplicate_by!r} must be a non-empty list of schema field names")
    if not deduplicate_by:
        raise ValueError(f"deduplicate_by {deduplicate_by!r} must not be empty")
    known = set(names)
    seen = set()
    fields = []
    for member in deduplicate_by:
        if not isinstance(member, str) or not member.strip():
            raise ValueError(f"deduplicate_by entry {member!r} must be a non-blank string")
        if member in seen:
            raise ValueError(f"deduplicate_by field {member!r} is repeated")
        if member not in known:
            raise ValueError(f"deduplicate_by field {member!r} is not a schema column name")
        seen.add(member)
        fields.append(member)
    return fields


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


def _deduplicate_records(records, deduplicate_fields):
    """Keep the first record of each key tuple, in first-seen order.

    Keys are the converted values of the given output fields compared by
    equality: strings case-sensitively, null equal to null, integers and
    booleans by value. A later record whose other fields differ is dropped
    whole -- it never merges with or overwrites the kept record.
    Returns (kept, removed_count).
    """
    seen_keys = set()
    kept = []
    for record in records:
        key = tuple(record[field] for field in deduplicate_fields)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        kept.append(record)
    return kept, len(records) - len(kept)


def _build_result(records, errors, duplicate_fields, filter_condition, deduplicate_fields=None):
    """Filter fully validated records, deduplicate the survivors, then counts."""
    if filter_condition is None:
        kept, filtered_count = records, 0
    else:
        field, expected = filter_condition
        kept = [record for record in records if record[field] == expected]
        filtered_count = len(records) - len(kept)
    if deduplicate_fields is not None:
        kept, deduplicated_count = _deduplicate_records(kept, deduplicate_fields)
    else:
        deduplicated_count = 0
    result = {"records": kept, "errors": errors, "accepted": len(kept),
              "rejected": len(errors)}
    if filter_condition is not None:
        result["filtered"] = filtered_count
    if deduplicate_fields is not None:
        result["deduplicated"] = deduplicated_count
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


def normalize_csv(source, schema, duplicate_by=None, filter_eq=None, deduplicate_by=None):
    columns, sources, defaults, allowed, markers, ranges, lte_fields = _prepare_schema(schema)
    names = [column["name"] for column in columns]
    column_index = {name: index for index, name in enumerate(names)}
    duplicate_fields = _prepare_duplicate_by(duplicate_by, names)
    filter_condition = _prepare_filter_eq(filter_eq, columns)
    deduplicate_fields = _prepare_deduplicate_by(deduplicate_by, names)
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
                if len(cells) != len(sources):
                    row_errors = ["wrong number of cells"]
                else:
                    # One error slot per column keeps per-field errors and
                    # the lte_field relation error attributed to a column in
                    # schema column order, even when the relation points
                    # backward at an earlier column.
                    slots = [[] for _ in columns]
                    field_error_names = set()
                    for index, (column, value) in enumerate(zip(columns, cells)):
                        name, kind = column["name"], column["type"]
                        value = value.strip()
                        if value in markers.get(name, ()):
                            # A declared missing marker matches the trimmed
                            # cell exactly (case-sensitive) before any type
                            # conversion and then follows the empty-value
                            # flow: default, required error or null.
                            value = ""
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
                            slots[index].append(f"{name}: {exc}")
                            field_error_names.add(name)
                            continue
                        enum_error = _enum_field_error(name, converted, allowed)
                        if enum_error is not None:
                            slots[index].append(enum_error)
                            field_error_names.add(name)
                            continue
                        range_error = _range_field_error(name, converted, ranges)
                        if range_error is not None:
                            slots[index].append(range_error)
                            field_error_names.add(name)
                        else:
                            record[name] = converted
                    for name, message in _lte_field_errors(
                            record, field_error_names, lte_fields):
                        slots[column_index[name]].append(message)
                    row_errors = [message for slot in slots for message in slot]
                if row_errors:
                    errors.append({"row": start_line, "errors": row_errors})
                else:
                    records.append(record)
        except csv.Error as exc:
            raise _csv_parse_error(
                source, previous_end + 1, _csv_error_reason(exc)) from None
        if header is None:
            raise ValueError("CSV header must match schema column sources and order exactly")
    return _build_result(records, errors, duplicate_fields, filter_condition,
                       deduplicate_fields)


def _convert_jsonl_value(column, value, defaults, markers):
    """Convert one decoded JSON value. Returns (converted, error_message)."""
    name, kind = column["name"], column["type"]
    if isinstance(value, str):
        value = value.strip()
        if value in markers.get(name, ()):
            # Only JSON strings take part in marker matching (numbers,
            # booleans, arrays and objects never do); a hit follows the
            # empty-value flow before any type conversion.
            value = ""
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


def normalize_jsonl(source, schema, duplicate_by=None, filter_eq=None, deduplicate_by=None):
    columns, sources, defaults, allowed, markers, ranges, lte_fields = _prepare_schema(schema)
    names = [column["name"] for column in columns]
    column_index = {name: index for index, name in enumerate(names)}
    duplicate_fields = _prepare_duplicate_by(duplicate_by, names)
    filter_condition = _prepare_filter_eq(filter_eq, columns)
    deduplicate_fields = _prepare_deduplicate_by(deduplicate_by, names)
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
            # One slot per column keeps field errors and the lte_field
            # relation error attributed to a column in schema order.
            slots = [[] for _ in columns]
            field_error_names = set()
            for index, (column, origin) in enumerate(zip(columns, sources)):
                name = column["name"]
                converted, message = _convert_jsonl_value(column, obj[origin], defaults, markers)
                if message is not None:
                    slots[index].append(message)
                    field_error_names.add(name)
                    continue
                enum_error = _enum_field_error(name, converted, allowed)
                if enum_error is not None:
                    slots[index].append(enum_error)
                    field_error_names.add(name)
                    continue
                range_error = _range_field_error(name, converted, ranges)
                if range_error is not None:
                    slots[index].append(range_error)
                    field_error_names.add(name)
                else:
                    record[name] = converted
            for name, message in _lte_field_errors(
                    record, field_error_names, lte_fields):
                slots[column_index[name]].append(message)
            row_errors = [message for slot in slots for message in slot]
            if row_errors:
                errors.append({"row": row_number, "errors": row_errors})
            else:
                records.append(record)
    return _build_result(records, errors, duplicate_fields, filter_condition,
                       deduplicate_fields)


def write_jsonl(path, records):
    Path(path).write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def _csv_cell(value, quote_empty=False):
    """Serialize one normalized value to a single CSV cell.

    Integers are decimal text (booleans, checked first, are lowercase
    ``true``/``false``), null is an empty cell, and strings keep their
    normalized content including any embedded CR/LF characters. The same
    quoting rule applies to every cell (headers included): a cell that
    contains a comma, double quote, CR or LF is wrapped in double quotes
    with each interior double quote doubled. An empty cell stays bare,
    except in a single-column table (``quote_empty``), where it must be a
    quoted empty string so the row cannot collapse into a skipped blank
    line; every other cell stays unquoted.
    """
    if value is None:
        text = ""
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, int):
        text = str(value)
    else:
        text = value
    if text == "" and not quote_empty:
        return text
    if text == "" or any(char in text for char in (",", '"', "\r", "\n")):
        return '"' + text.replace('"', '""') + '"'
    return text


def _csv_payload(names, records):
    """Serialize the header and kept records to LF-terminated CSV text.

    The header uses the schema output names in declared order (never
    ``source`` aliases); data rows follow kept-record order. The output is
    UTF-8 text with no BOM and every line ends in LF, including the header
    and the final record (an empty result still yields the header alone).
    """
    single_column = len(names) == 1
    lines = [",".join(_csv_cell(name, single_column) for name in names)]
    lines.extend(
        ",".join(_csv_cell(record[name], single_column) for name in names)
        for record in records)
    return "\n".join(lines) + "\n"


def _stage_output(destination, payload):
    """Serialize payload text into a temp file in the destination's directory.

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


def _write_outputs_atomic(output_path, output_payload, errors_path, errors):
    """Write both output files as one all-or-nothing operation.

    Either both files are fully written (replacing existing files), or a
    filesystem failure leaves every target exactly as it was beforehand:
    pre-existing files keep their bytes and previously absent files stay
    absent. Parent directories are never created. The records output
    carries output_payload (JSONL or CSV text); the errors output is
    always JSONL.
    """
    errors_payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in errors)
    output_target, errors_target = Path(output_path), Path(errors_path)
    staged_output, output_existed = _stage_output(output_target, output_payload)
    try:
        staged_errors, errors_existed = _stage_output(errors_target, errors_payload)
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
    parser.add_argument("--output-format", default="jsonl",
                        help="records output format: jsonl (default) or csv")
    parser.add_argument("--duplicate-by", action="append", metavar="FIELD",
                        help="output field to report duplicate accepted records by; repeatable")
    parser.add_argument("--deduplicate-by", action="append", metavar="FIELD",
                        help="output field to deduplicate accepted records by, keeping the first; repeatable")
    parser.add_argument("--filter-eq", metavar="JSON",
                        help='equality condition as JSON, e.g. {"field": "active", "value": true}')
    args = parser.parse_args()
    try:
        _check_path_isolation(args.source, args.schema, args.output, args.errors)
        # The output format is validated after the path-isolation check and
        # before either input is read, independently of --format: names and
        # extensions never decide either format.
        if args.output_format not in ("csv", "jsonl"):
            raise ValueError(
                f"output-format must be csv or jsonl, got {args.output_format!r}")
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
        schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
        result = normalize(args.source, schema,
                           duplicate_by=args.duplicate_by, filter_eq=filter_eq,
                           deduplicate_by=args.deduplicate_by)
        if args.output_format == "csv":
            names = [column["name"] for column in schema["columns"]]
            output_payload = _csv_payload(names, result["records"])
        else:
            output_payload = "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in result["records"])
        _write_outputs_atomic(args.output, output_payload, args.errors, result["errors"])
        summary = {"accepted": result["accepted"], "rejected": result["rejected"]}
        if args.filter_eq is not None:
            summary["filtered"] = result["filtered"]
        if args.deduplicate_by is not None:
            summary["deduplicated"] = result["deduplicated"]
        if args.duplicate_by is not None:
            summary["duplicates"] = result["duplicates"]
        print(json.dumps(summary))
        return 1 if result["rejected"] else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
