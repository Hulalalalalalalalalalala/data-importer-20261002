"""Normalize CSV rows or JSONL objects using a small explicit schema."""
import argparse
import contextlib
import csv
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path


def _prepare_unicode_normalization(columns):
    """Validate the optional per-column ``unicode_normalization`` attribute.

    Returns the set of output field names that opt in. The attribute may
    appear only on ``string`` columns and its value must be the exact,
    case-sensitive string ``"NFKC"``: any other value (other casing or
    spelling, a non-string) or a declaration on an ``integer`` or
    ``boolean`` column is a configuration error. Raises ValueError naming
    ``unicode_normalization``, the output field and the offending value
    before the input is read; the caller's schema is never mutated.
    """
    normalizing = set()
    for column in columns:
        name, kind = column["name"], column["type"]
        if "unicode_normalization" not in column:
            continue
        mode = column["unicode_normalization"]
        if mode != "NFKC" or not isinstance(mode, str):
            raise ValueError(
                f"column {name!r} unicode_normalization {mode!r} must be the "
                "exact string 'NFKC'")
        if kind != "string":
            raise ValueError(
                f"column {name!r} unicode_normalization {mode!r} requires a string column")
        normalizing.add(name)
    return normalizing


def _prepare_casefold(columns):
    """Validate the optional per-column ``casefold`` attribute.

    Returns the set of output field names that opt in. The attribute may
    appear only on ``string`` columns and its value must be a boolean:
    ``true`` enables Unicode case folding (Python ``str.casefold()``) of
    the final string, ``false`` keeps the previous behavior, and a
    non-boolean value or a declaration on an ``integer`` or ``boolean``
    column is a configuration error. Raises ValueError naming
    ``casefold``, the output field and the offending value before the
    input is read; the caller's schema is never mutated.
    """
    folding = set()
    for column in columns:
        name, kind = column["name"], column["type"]
        if "casefold" not in column:
            continue
        enabled = column["casefold"]
        if not isinstance(enabled, bool):
            raise ValueError(
                f"column {name!r} casefold {enabled!r} must be a boolean")
        if kind != "string":
            raise ValueError(
                f"column {name!r} casefold {enabled!r} requires a string column")
        if enabled:
            folding.add(name)
    return folding


def _prepare_collapse_whitespace(columns):
    """Validate the optional per-column ``collapse_whitespace`` attribute.

    Returns the set of output field names that opt in. The attribute may
    appear only on ``string`` columns and its value must be a boolean:
    ``true`` enables merging of every interior run of whitespace into a
    single ASCII space, ``false`` keeps the previous behavior, and a
    non-boolean value or a declaration on an ``integer`` or ``boolean``
    column is a configuration error. Raises ValueError naming
    ``collapse_whitespace``, the output field and the offending value
    before the input is read; the caller's schema is never mutated.
    """
    collapsing = set()
    for column in columns:
        name, kind = column["name"], column["type"]
        if "collapse_whitespace" not in column:
            continue
        enabled = column["collapse_whitespace"]
        if not isinstance(enabled, bool):
            raise ValueError(
                f"column {name!r} collapse_whitespace {enabled!r} must be a boolean")
        if kind != "string":
            raise ValueError(
                f"column {name!r} collapse_whitespace {enabled!r} requires a string column")
        if enabled:
            collapsing.add(name)
    return collapsing


def _collapse_whitespace(value):
    """Replace each interior run of whitespace with one ASCII space.

    Whitespace follows Python ``str.isspace()`` exactly (so the
    zero-width space U+200B, which is not whitespace, is kept verbatim)
    and every maximal run becomes a single ``" "``. Runs are merged but
    no character is deleted, so a non-empty string stays non-empty and
    the value -- already trimmed on arrival, and NFKC-normalized plus
    re-trimmed before this step when that normalization is enabled --
    gains no fresh leading or trailing space.
    """
    pieces = []
    inside_run = False
    for char in value:
        if char.isspace():
            if not inside_run:
                pieces.append(" ")
                inside_run = True
        else:
            pieces.append(char)
            inside_run = False
    return "".join(pieces)


def _process_string_text(value, name, normalizing, collapsing, casefolding):
    """Finish a non-empty string value through trim, NFKC, collapse and fold.

    Runs the column's declared string normalizations in the established
    order (the value arrives already trimmed): NFKC plus a second trim
    when the column opts in, then whitespace collapsing when the column
    opts in, then case folding. Returns the processed text; the
    empty-value flows (default, required, null) are handled by the
    caller when NFKC collapses the text to blank.
    """
    if name in normalizing:
        value = _nfkc(value)
    if name in collapsing:
        value = _collapse_whitespace(value)
    if name in casefolding:
        value = value.casefold()
    return value


def _prepare_value_maps(columns, normalizing, collapsing, casefolding):
    """Validate the optional per-column ``value_map`` attribute.

    Returns a mapping of output field name to the processed entries as a
    list of ``(processed_key, target)`` pairs kept in declared order. The
    attribute is accepted only on ``string`` columns and must be a
    non-empty object whose keys and mapped values are strings; declared
    keys and targets are first trimmed, then run through the column's
    existing string normalization in the same order an incoming string
    gets (NFKC plus a second trim when enabled, then whitespace
    collapsing when enabled, then case folding). A key that processes to
    blank, a non-string key or value, a declaration on an ``integer`` or
    ``boolean`` column, or two keys that process to the same text are
    configuration errors -- the latter even when both keys map to the
    same target, and keys merged equal by whitespace collapsing count as
    a clash. Raises ValueError naming ``value_map``, the output field and
    the offending value before the input is read; the caller's schema
    dicts are never mutated.
    """
    maps = {}
    for column in columns:
        name, kind = column["name"], column["type"]
        if "value_map" not in column:
            continue
        configured = column["value_map"]
        if kind != "string":
            raise ValueError(
                f"column {name!r} value_map {configured!r} requires a string column")
        if not isinstance(configured, dict) or not configured:
            raise ValueError(
                f"column {name!r} value_map {configured!r} must be a non-empty object")
        entries = []
        seen = set()
        for key, target in configured.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"column {name!r} value_map key {key!r} must be a string")
            if not isinstance(target, str):
                raise ValueError(
                    f"column {name!r} value_map value {target!r} for key {key!r} "
                    "must be a string")
            processed_key = key.strip()
            if not processed_key:
                raise ValueError(
                    f"column {name!r} value_map key {key!r} must not be blank")
            processed_target = target.strip()
            if not processed_target:
                raise ValueError(
                    f"column {name!r} value_map value {target!r} for key {key!r} "
                    "must not be blank")
            processed_key = _process_string_text(
                processed_key, name, normalizing, collapsing, casefolding)
            if not processed_key:
                raise ValueError(
                    f"column {name!r} value_map key {key!r} must not normalize to blank")
            processed_target = _process_string_text(
                processed_target, name, normalizing, collapsing, casefolding)
            if not processed_target:
                raise ValueError(
                    f"column {name!r} value_map value {target!r} for key {key!r} "
                    "must not normalize to blank")
            if processed_key in seen:
                raise ValueError(
                    f"column {name!r} value_map key {key!r} normalizes to a repeated "
                    f"key {processed_key!r}")
            seen.add(processed_key)
            entries.append((processed_key, processed_target))
        maps[name] = entries
    return maps


def _mapped_string(value, entries):
    """Look a normalized string up in a processed ``value_map`` exactly once.

    The match is exact against the processed keys, in declared order
    (the first duplicate key cannot survive preparation); a hit returns
    the processed target and a miss returns the value unchanged. The
    result is never matched against the map again, so a target spelling
    that also appears as a key is not chained.
    """
    for processed_key, processed_target in entries:
        if value == processed_key:
            return processed_target
    return value


def _prepare_defaults(columns, normalizing=frozenset(), collapsing=frozenset(),
                      casefolding=frozenset()):
    """Validate the optional target-typed ``default`` of each column.

    Returns a mapping of output field name to the processed default
    (string defaults are trimmed; a string column opting into NFKC also
    has its default NFKC-normalized and re-trimmed; a string column
    opting into whitespace collapsing then has its interior runs
    merged; and a string column opting into case folding then has it
    casefolded, in that order). A malformed default is a configuration
    error naming the default value and output field, raised before the
    input is read; the caller's schema dicts are never mutated. A
    normalized string default that becomes blank is refused here so the
    later ``allowed_values`` membership check keeps using the final
    emitted text.
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
            if name in normalizing:
                processed = unicodedata.normalize("NFKC", processed).strip()
                if not processed:
                    raise ValueError(
                        f"column {name!r} default {default!r} must not normalize to blank")
            if name in collapsing:
                # Collapsing merges runs into spaces but deletes nothing,
                # so a non-empty, already-trimmed default stays non-empty.
                processed = _collapse_whitespace(processed)
            if name in casefolding:
                # Case folding never blanks a non-blank string, so the
                # folded default needs no further emptiness check.
                processed = processed.casefold()
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


def _prepare_allowed_values(columns, defaults, value_maps=frozenset()):
    """Validate the optional ``allowed_values`` list of each column.

    Returns a mapping of output field name to a fresh list holding the
    declared entries verbatim (strings are neither trimmed nor case
    folded). The attribute must be a non-empty list of target-typed
    values: strings for ``string`` columns, non-boolean integers for
    ``integer`` columns, booleans for ``boolean`` columns; null, arrays
    and objects are never valid entries, and repeated entries (value
    equality, verbatim for strings) are refused. A processed default
    outside the list is a configuration error even when no row needs it;
    when the column declares a ``value_map`` the default is judged after
    its one mapping lookup (a filled default is itself mapped), so the
    checked default is the mapped target. Raises ValueError naming
    ``allowed_values`` and the output field before the input is read; the
    caller's schema is never mutated.
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
        if name in defaults:
            effective_default = defaults[name]
            column_map = value_maps.get(name)
            if column_map is not None:
                effective_default = _mapped_string(effective_default, column_map)
            if effective_default not in checked:
                raise ValueError(
                    f"column {name!r} default {effective_default!r} is not one of "
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


def _pattern_field_error(name, value, patterns):
    """Return the pattern error for a converted value, or None when it matches.

    Null is never checked: an empty optional without a default stays null
    and the full-string match only sees non-null strings, after type
    conversion, empty handling and the enum check. A matching value keeps
    its final text verbatim; the expression is the declared one, compiled
    with Python's standard ``re`` semantics (case-sensitive by default,
    inline flags honored).
    """
    compiled = patterns.get(name)
    if compiled is None or value is None:
        return None
    if compiled.fullmatch(value) is None:
        return (f"{name}: value {value!r} does not match pattern "
                f"{compiled.pattern!r}")
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


def _required_when_field_errors(record, field_errors, required_when):
    """Check every declared ``required_when`` condition over final values.

    Runs after structure checks, marker matching, default filling, type
    conversion and the enum, range, pattern and lte_field checks. A
    condition is skipped when either participating field already carries
    an error; the original errors are kept and every other condition on
    the row is still checked. Conditions are evaluated in schema column
    order and returned as ``(name, message)`` pairs so the caller can
    file each error under its declaring column's position. The rule
    fires only when the condition column's final boolean is true and the
    declaring column's final value is null -- a false or null condition
    never fires, and a final 0 or false on the declaring column counts
    as a value. A violation names ``required_when``, both output names
    and the true condition value.
    """
    errors = []
    for name, target in required_when:
        if name in field_errors or target in field_errors:
            continue
        if record.get(target) is not True:
            continue
        if record.get(name) is not None:
            continue
        errors.append((
            name,
            f"{name}: value is null but required_when {target!r} is true"))
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


def _prepare_required_when(columns):
    """Validate the optional ``required_when`` reference of each column.

    Returns a list of ``(name, target)`` pairs in schema column order, one
    entry per declaring column. The attribute may appear on a column of
    any type and must be a non-blank string naming another ``boolean``
    column by its output name, matched verbatim and case-sensitively (the
    text is never trimmed and source aliases are not recognized); the
    condition column may be declared before or after the declaring
    column. A non-string, a blank string, an unknown field, a self
    reference or a non-boolean target is a configuration error. Raises
    ValueError naming ``required_when``, the declaring output field and
    the attribute value before the input is read; the caller's schema is
    never mutated.
    """
    by_name = {column["name"]: column for column in columns}
    required_when = []
    for column in columns:
        name = column["name"]
        if "required_when" not in column:
            continue
        target = column["required_when"]
        if not isinstance(target, str):
            raise ValueError(
                f"column {name!r} required_when {target!r} must be a string naming "
                "a boolean column")
        if not target.strip():
            raise ValueError(
                f"column {name!r} required_when {target!r} must be a non-blank string")
        if target == name:
            raise ValueError(
                f"column {name!r} required_when {target!r} must not reference "
                "the column itself")
        if target not in by_name:
            raise ValueError(
                f"column {name!r} required_when {target!r} is not a schema column name")
        if by_name[target]["type"] != "boolean":
            raise ValueError(
                f"column {name!r} required_when {target!r} must reference "
                "a boolean column")
        required_when.append((name, target))
    return required_when


def _prepare_patterns(columns, defaults, value_maps=frozenset()):
    """Validate the optional ``pattern`` regular expression of each column.

    Returns a mapping of output field name to the compiled expression; the
    declared expression text is kept verbatim (``compiled.pattern`` hands
    it back) and the match later covers the whole string under Python's
    standard ``re`` semantics -- case-sensitive by default, with inline
    flags such as ``(?i)`` honored. The attribute must be a non-blank
    string and may appear only on a ``string`` column; a non-string or
    whitespace-only value, a declaration on an ``integer`` or ``boolean``
    column, or an expression the ``re`` compiler rejects is a
    configuration error. A processed ``default`` (already trimmed and, for
    an NFKC column, normalized) that does not match is a configuration
    error even when no row needs it, while an ``allowed_values`` entry
    that fails the expression is not. When the column declares a
    ``value_map`` the filled default is itself mapped once, so the
    constraint is checked against the mapped target. Raises ValueError
    naming ``pattern``, the output field and the offending value (the
    default error also names ``default``) before the input is read; the
    caller's schema is never mutated.
    """
    patterns = {}
    for column in columns:
        name, kind = column["name"], column["type"]
        if "pattern" not in column:
            continue
        pattern = column["pattern"]
        if not isinstance(pattern, str):
            raise ValueError(f"column {name!r} pattern {pattern!r} must be a string")
        if not pattern.strip():
            raise ValueError(
                f"column {name!r} pattern {pattern!r} must be a non-blank string")
        if kind != "string":
            raise ValueError(
                f"column {name!r} pattern {pattern!r} requires a string column")
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise ValueError(
                f"column {name!r} pattern {pattern!r} is not a valid regular "
                f"expression: {exc}") from None
        patterns[name] = compiled
        if name in defaults:
            effective_default = defaults[name]
            column_map = value_maps.get(name)
            if column_map is not None:
                effective_default = _mapped_string(effective_default, column_map)
            if compiled.fullmatch(effective_default) is None:
                raise ValueError(
                    f"column {name!r} default {effective_default!r} does not match "
                    f"pattern {pattern!r}")
    return patterns


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


def _prepare_boolean_aliases(columns):
    """Validate the optional ``boolean_aliases`` mapping of each boolean column.

    Returns a mapping of output field name to a fresh dict keyed by the
    normalized alias text (the declared key trimmed and Unicode
    casefolded) holding the target boolean verbatim. The attribute must be
    a non-empty object and may appear only on a ``boolean`` column; every
    key must be a non-blank string and every mapped value a boolean. Two
    keys that normalize to the same text are invalid even when they map to
    the same boolean, and a key normalizing to ``true`` or ``false`` is
    refused so the built-in spellings keep their fixed meaning. Raises
    ValueError naming ``boolean_aliases``, the output field and the
    offending value before the input is read; the caller's schema is never
    mutated.
    """
    aliases = {}
    for column in columns:
        name, kind = column["name"], column["type"]
        if "boolean_aliases" not in column:
            continue
        configured = column["boolean_aliases"]
        if kind != "boolean":
            raise ValueError(
                f"column {name!r} boolean_aliases {configured!r} requires a boolean column")
        if not isinstance(configured, dict) or not configured:
            raise ValueError(
                f"column {name!r} boolean_aliases {configured!r} must be a non-empty object")
        normalized = {}
        for key, mapped in configured.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    f"column {name!r} boolean_aliases key {key!r} must be a non-blank string")
            if not isinstance(mapped, bool):
                raise ValueError(
                    f"column {name!r} boolean_aliases value {mapped!r} for key {key!r} "
                    "must be a boolean")
            folded = key.strip().casefold()
            if folded in ("true", "false"):
                raise ValueError(
                    f"column {name!r} boolean_aliases key {key!r} must not normalize to "
                    f"the built-in boolean spelling {folded!r}")
            if folded in normalized:
                raise ValueError(
                    f"column {name!r} boolean_aliases key {key!r} normalizes to a repeated "
                    f"key {folded!r}")
            normalized[folded] = mapped
        aliases[name] = normalized
    return aliases


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
    normalizing = _prepare_unicode_normalization(columns)
    casefolding = _prepare_casefold(columns)
    collapsing = _prepare_collapse_whitespace(columns)
    value_maps = _prepare_value_maps(columns, normalizing, collapsing, casefolding)
    defaults = _prepare_defaults(columns, normalizing, collapsing, casefolding)
    allowed = _prepare_allowed_values(columns, defaults, value_maps)
    markers = _prepare_missing_values(columns)
    aliases = _prepare_boolean_aliases(columns)
    ranges = _prepare_ranges(columns, defaults)
    lte_fields = _prepare_lte_fields(columns)
    patterns = _prepare_patterns(columns, defaults, value_maps)
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
    # The required_when references are validated only after every
    # pre-existing configuration check (structure, per-column attributes
    # and sources) has passed.
    required_when = _prepare_required_when(columns)
    return columns, sources, defaults, allowed, markers, aliases, ranges, lte_fields, normalizing, collapsing, patterns, casefolding, required_when, value_maps


def _prepare_field_list(fields, names, option):
    """Validate a list option of output field names shared by the two
    report/dedup features.

    Returns a new list (the caller's list is never mutated or retained)
    or None when the option is disabled. The value must be a non-empty
    list of non-blank strings naming schema output columns literally
    (names are neither trimmed nor case folded, and source aliases are
    not recognized); a non-list, an empty list, a non-string or blank
    entry, a repeat, or an unknown field raises ValueError naming the
    option and the offending value before the input is read.
    """
    if fields is None:
        return None
    if not isinstance(fields, list):
        raise ValueError(f"{option} must be a list of schema field names")
    if not fields:
        raise ValueError(f"{option} must not be empty")
    known = set(names)
    seen = set()
    for member in fields:
        if not isinstance(member, str) or not member.strip():
            raise ValueError(f"{option} entry {member!r} must be a non-blank string")
        if member in seen:
            raise ValueError(f"{option} field {member!r} is repeated")
        if member not in known:
            raise ValueError(f"{option} field {member!r} is not a schema column name")
        seen.add(member)
    return list(fields)


def _prepare_duplicate_by(duplicate_by, names):
    """Validate the optional duplicate-report fields against output names.

    Returns a new list or None when reporting is disabled. Raises ValueError
    before the input is opened.
    """
    return _prepare_field_list(duplicate_by, names, "duplicate_by")


def _prepare_deduplicate_by(deduplicate_by, names):
    """Validate the optional keep-first dedup fields against output names.

    Returns a new list or None when deduplication is disabled. Raises
    ValueError naming ``deduplicate_by`` and the offending value before
    the input is opened.
    """
    if deduplicate_by is None:
        return None
    if not isinstance(deduplicate_by, list):
        raise ValueError(
            f"deduplicate_by {deduplicate_by!r} must be a list of schema field names")
    if not deduplicate_by:
        raise ValueError(f"deduplicate_by {deduplicate_by!r} must not be empty")
    return _prepare_field_list(deduplicate_by, names, "deduplicate_by")


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


def _prepare_filter_in(filter_in, columns):
    """Validate the optional single-field set-membership filter.

    The condition is an object with exactly ``field`` and ``values`` keys;
    field matches an output name literally (source aliases are not
    recognized) and values must be a non-empty list of values fitting the
    target type, with null allowed for every type. Entries are kept
    verbatim (strings are never trimmed, normalized or case folded) and
    must be unique by value equality (identical strings, equal integers or
    booleans, repeated nulls); booleans are never valid entries of an
    integer column. Returns (field, values) or None when filtering is
    disabled. Raises ValueError before the input is read; every message
    names ``filter_in`` and the offending value.
    """
    if filter_in is None:
        return None
    if not isinstance(filter_in, dict):
        raise ValueError(
            f"filter_in {filter_in!r} must be an object with 'field' and 'values' keys")
    keys = set(filter_in)
    if keys != {"field", "values"}:
        details = []
        missing = [key for key in ("field", "values") if key not in keys]
        extra = sorted(keys - {"field", "values"})
        if missing:
            details.append("missing key(s): " + ", ".join(missing))
        if extra:
            details.append("unexpected key(s): " + ", ".join(extra))
        raise ValueError(
            f"filter_in {filter_in!r} must contain exactly 'field' and 'values' keys ("
            + "; ".join(details) + ")")
    field = filter_in["field"]
    if not isinstance(field, str):
        raise ValueError(f"filter_in field {field!r} must be a string")
    if not field.strip():
        raise ValueError(f"filter_in field {field!r} must be a non-blank string")
    by_name = {column["name"]: column for column in columns}
    if field not in by_name:
        raise ValueError(f"filter_in field {field!r} is not a schema column name")
    values = filter_in["values"]
    if not isinstance(values, list) or not values:
        raise ValueError(f"filter_in values {values!r} must be a non-empty list")
    kind = by_name[field]["type"]
    checked = []
    for entry in values:
        if entry is not None:
            if kind == "string" and not isinstance(entry, str):
                raise ValueError(
                    f"filter_in values entry {entry!r} for field {field!r} "
                    "must be a string or null")
            if kind == "integer" and (isinstance(entry, bool) or not isinstance(entry, int)):
                raise ValueError(
                    f"filter_in values entry {entry!r} for field {field!r} "
                    "must be a non-boolean integer or null")
            if kind == "boolean" and not isinstance(entry, bool):
                raise ValueError(
                    f"filter_in values entry {entry!r} for field {field!r} "
                    "must be a boolean or null")
        if entry in checked:
            raise ValueError(
                f"filter_in values entry {entry!r} for field {field!r} is repeated")
        checked.append(entry)
    return field, checked


def _prepare_filter_range(filter_range, columns):
    """Validate the optional single-field inclusive integer range filter.

    The condition is an object holding ``field`` plus at least one of
    ``minimum`` and ``maximum`` and no other keys; field matches an output
    name literally (source aliases are not recognized) and must name an
    ``integer`` column. Each declared bound must be a non-boolean integer
    (negative values and zero are legal) and is used verbatim -- never
    converted; a missing end imposes no limit on that side, equal ends
    accept exactly that one integer, and a minimum greater than the
    maximum is a configuration error. Returns (field, minimum, maximum)
    with None for an undeclared end, or None when filtering is disabled.
    Raises ValueError naming ``filter_range`` and the offending value
    before the input is read; the caller's condition is never mutated.
    """
    if filter_range is None:
        return None
    if not isinstance(filter_range, dict):
        raise ValueError(
            f"filter_range {filter_range!r} must be an object with 'field' and "
            "'minimum' and/or 'maximum' keys")
    keys = set(filter_range)
    allowed = ("field", "minimum", "maximum")
    if "field" not in keys or not keys & {"minimum", "maximum"} or keys - set(allowed):
        details = []
        missing = []
        if "field" not in keys:
            missing.append("field")
        if not keys & {"minimum", "maximum"}:
            missing.append("minimum or maximum")
        extra = sorted(keys - set(allowed))
        if missing:
            details.append("missing key(s): " + ", ".join(missing))
        if extra:
            details.append("unexpected key(s): " + ", ".join(extra))
        raise ValueError(
            f"filter_range {filter_range!r} must contain 'field' and at least one of "
            "'minimum' or 'maximum' (" + "; ".join(details) + ")")
    field = filter_range["field"]
    if not isinstance(field, str):
        raise ValueError(f"filter_range field {field!r} must be a string")
    if not field.strip():
        raise ValueError(f"filter_range field {field!r} must be a non-blank string")
    by_name = {column["name"]: column for column in columns}
    if field not in by_name:
        raise ValueError(f"filter_range field {field!r} is not a schema column name")
    if by_name[field]["type"] != "integer":
        raise ValueError(
            f"filter_range field {field!r} must name an integer column")
    minimum = filter_range.get("minimum")
    maximum = filter_range.get("maximum")
    for attribute, bound in (("minimum", minimum), ("maximum", maximum)):
        if attribute not in filter_range:
            continue
        if isinstance(bound, bool) or not isinstance(bound, int):
            raise ValueError(
                f"filter_range {attribute} {bound!r} must be a non-boolean integer")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(
            f"filter_range minimum {minimum!r} is greater than maximum {maximum!r}")
    return field, minimum, maximum


def _deduplicate_records(records, deduplicate_by):
    """Keep the first record of each equal-key group, in input order.

    Two records collide only when every named field's final value is
    equal (strings case-sensitively, null equal to null). The first
    record in input order is kept complete -- non-key fields are never
    merged or overwritten -- and the remaining records keep their
    relative order. Returns (kept, removed_count).
    """
    seen_keys = set()
    kept = []
    for record in records:
        key = tuple(record[field] for field in deduplicate_by)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        kept.append(record)
    return kept, len(records) - len(kept)


def _build_result(records, errors, duplicate_fields, filter_condition,
                  deduplicate_fields=None, filter_in_condition=None,
                  filter_range_condition=None):
    """Filter fully validated records, then keep-first dedup, then counts.

    Order is: validation (already done by the caller), ``filter_eq``,
    ``filter_in`` and ``filter_range`` (a record survives only when every
    enabled predicate matches, so the conditions combine as AND;
    ``filtered`` counts each removed record once), keep-first
    deduplication over the filter survivors, and finally the duplicate
    report over the records that remain.
    """
    filtering = (filter_condition is not None or filter_in_condition is not None
                 or filter_range_condition is not None)
    filtered_records = records
    if filtering:
        eq_field, eq_expected = (
            filter_condition if filter_condition is not None else (None, None))
        in_field, in_values = (
            filter_in_condition if filter_in_condition is not None else (None, None))
        range_field, range_minimum, range_maximum = (
            filter_range_condition if filter_range_condition is not None
            else (None, None, None))

        def matches(record):
            if filter_condition is not None and record[eq_field] != eq_expected:
                return False
            if filter_in_condition is not None and record[in_field] not in in_values:
                return False
            if filter_range_condition is not None:
                # The bounds are inclusive; a null final value never matches.
                value = record[range_field]
                if value is None:
                    return False
                if range_minimum is not None and value < range_minimum:
                    return False
                if range_maximum is not None and value > range_maximum:
                    return False
            return True

        filtered_records = [record for record in records if matches(record)]
        filtered_count = len(records) - len(filtered_records)
    else:
        filtered_count = 0
    if deduplicate_fields is None:
        kept, deduplicated_count = filtered_records, 0
    else:
        kept, deduplicated_count = _deduplicate_records(
            filtered_records, deduplicate_fields)
    result = {"records": kept, "errors": errors, "accepted": len(kept),
              "rejected": len(errors)}
    if filtering:
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


def _nfkc(value):
    """NFKC-normalize an already trimmed string and trim it again.

    Compatibility composition folds full-width letters onto ASCII and
    merges combining-mark sequences (``e`` + U+0301 -> ``é``); NFKC may
    leave fresh leading/trailing whitespace (full-width space U+3000
    becomes an ASCII space), so the result is trimmed once more.
    """
    return unicodedata.normalize("NFKC", value).strip()


def normalize_csv(source, schema, duplicate_by=None, filter_eq=None,
                  deduplicate_by=None, filter_in=None, filter_range=None):
    columns, sources, defaults, allowed, markers, aliases, ranges, lte_fields, normalizing, collapsing, patterns, casefolding, required_when, value_maps = _prepare_schema(schema)
    names = [column["name"] for column in columns]
    column_index = {name: index for index, name in enumerate(names)}
    duplicate_fields = _prepare_duplicate_by(duplicate_by, names)
    deduplicate_fields = _prepare_deduplicate_by(deduplicate_by, names)
    filter_condition = _prepare_filter_eq(filter_eq, columns)
    filter_in_condition = _prepare_filter_in(filter_in, columns)
    filter_range_condition = _prepare_filter_range(filter_range, columns)
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
                                folded = value.casefold()
                                if folded in ("true", "false"):
                                    converted = folded == "true"
                                elif folded in aliases.get(name, ()):
                                    converted = aliases[name][folded]
                                else:
                                    raise ValueError("boolean must be true or false")
                            else:
                                converted = value
                                if name in normalizing:
                                    converted = _nfkc(converted)
                                    if not converted:
                                        # A non-empty trimmed value whose
                                        # NFKC normalization collapses to
                                        # blank follows the ordinary
                                        # empty-value flow; markers are not
                                        # matched a second time and a filled
                                        # default was normalized at load.
                                        if name in defaults:
                                            converted = defaults[name]
                                        elif column.get("required", False):
                                            raise ValueError("required value is empty")
                                        else:
                                            converted = None
                                if isinstance(converted, str) and name in collapsing:
                                    # Whitespace collapsing runs after NFKC
                                    # (and its second trim) and before case
                                    # folding and the single value_map
                                    # lookup: every interior run of
                                    # str.isspace characters becomes one
                                    # ASCII space. A filled default was
                                    # collapsed at load, so collapsing it
                                    # again is idempotent; null stays null.
                                    converted = _collapse_whitespace(converted)
                                if isinstance(converted, str) and name in casefolding:
                                    # Case folding runs last, on the final
                                    # string (a filled default was already
                                    # folded at load; folding it again is
                                    # idempotent), before the enum, range
                                    # and pattern checks.
                                    converted = converted.casefold()
                        except ValueError as exc:
                            slots[index].append(f"{name}: {exc}")
                            field_error_names.add(name)
                            continue
                        if isinstance(converted, str) and name in value_maps:
                            # One exact lookup over the processed keys on
                            # the final string (a filled default is mapped
                            # just like any other value); null stays null
                            # and the target is never matched again.
                            converted = _mapped_string(converted, value_maps[name])
                        enum_error = _enum_field_error(name, converted, allowed)
                        if enum_error is not None:
                            slots[index].append(enum_error)
                            field_error_names.add(name)
                            continue
                        range_error = _range_field_error(name, converted, ranges)
                        if range_error is not None:
                            slots[index].append(range_error)
                            field_error_names.add(name)
                            continue
                        pattern_error = _pattern_field_error(name, converted, patterns)
                        if pattern_error is not None:
                            slots[index].append(pattern_error)
                            field_error_names.add(name)
                        else:
                            record[name] = converted
                    for name, message in _lte_field_errors(
                            record, field_error_names, lte_fields):
                        slots[column_index[name]].append(message)
                        # An lte_field violation is a pre-existing field
                        # error as far as the required_when conditions
                        # below are concerned.
                        field_error_names.add(name)
                    for name, message in _required_when_field_errors(
                            record, field_error_names, required_when):
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
                         deduplicate_fields, filter_in_condition,
                         filter_range_condition)


def _convert_jsonl_value(column, value, defaults, markers, aliases,
                         normalizing=frozenset(), collapsing=frozenset(),
                         casefolding=frozenset()):
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
        if name in normalizing:
            normalized = _nfkc(value)
            if not normalized:
                # NFKC collapsed a non-empty string to blank: the ordinary
                # empty-value flow applies, without matching markers again;
                # a filled default was normalized at load.
                if name in defaults:
                    return defaults[name], None
                if column.get("required", False):
                    return None, f"{name}: required value is empty"
                return None, None
            value = normalized
        if name in collapsing:
            # Whitespace collapsing runs after NFKC (and its second trim)
            # and before case folding: every interior run of
            # str.isspace characters becomes one ASCII space. A filled
            # default was collapsed at load, so collapsing it again is
            # idempotent.
            value = _collapse_whitespace(value)
        if name in casefolding:
            # Case folding runs last, on the final string, before the
            # enum, range and pattern checks.
            value = value.casefold()
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
        column_aliases = aliases.get(name)
        if column_aliases is not None and folded in column_aliases:
            return column_aliases[folded], None
    return None, f"{name}: boolean must be true or false"


def normalize_jsonl(source, schema, duplicate_by=None, filter_eq=None,
                    deduplicate_by=None, filter_in=None, filter_range=None):
    columns, sources, defaults, allowed, markers, aliases, ranges, lte_fields, normalizing, collapsing, patterns, casefolding, required_when, value_maps = _prepare_schema(schema)
    names = [column["name"] for column in columns]
    column_index = {name: index for index, name in enumerate(names)}
    duplicate_fields = _prepare_duplicate_by(duplicate_by, names)
    deduplicate_fields = _prepare_deduplicate_by(deduplicate_by, names)
    filter_condition = _prepare_filter_eq(filter_eq, columns)
    filter_in_condition = _prepare_filter_in(filter_in, columns)
    filter_range_condition = _prepare_filter_range(filter_range, columns)
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
                converted, message = _convert_jsonl_value(
                    column, obj[origin], defaults, markers, aliases, normalizing,
                    collapsing, casefolding)
                if message is not None:
                    slots[index].append(message)
                    field_error_names.add(name)
                    continue
                if isinstance(converted, str) and name in value_maps:
                    # One exact lookup over the processed keys on the
                    # final string (a filled default is mapped just like
                    # any other value); null stays null, a non-string
                    # value keeps its type error and never reaches here,
                    # and the target is never matched again.
                    converted = _mapped_string(converted, value_maps[name])
                enum_error = _enum_field_error(name, converted, allowed)
                if enum_error is not None:
                    slots[index].append(enum_error)
                    field_error_names.add(name)
                    continue
                range_error = _range_field_error(name, converted, ranges)
                if range_error is not None:
                    slots[index].append(range_error)
                    field_error_names.add(name)
                    continue
                pattern_error = _pattern_field_error(name, converted, patterns)
                if pattern_error is not None:
                    slots[index].append(pattern_error)
                    field_error_names.add(name)
                else:
                    record[name] = converted
            for name, message in _lte_field_errors(
                    record, field_error_names, lte_fields):
                slots[column_index[name]].append(message)
                # An lte_field violation is a pre-existing field error
                # as far as the required_when conditions below are
                # concerned.
                field_error_names.add(name)
            for name, message in _required_when_field_errors(
                    record, field_error_names, required_when):
                slots[column_index[name]].append(message)
            row_errors = [message for slot in slots for message in slot]
            if row_errors:
                errors.append({"row": row_number, "errors": row_errors})
            else:
                records.append(record)
    return _build_result(records, errors, duplicate_fields, filter_condition,
                         deduplicate_fields, filter_in_condition,
                         filter_range_condition)


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
                        help="output field to keep the first record by; repeatable")
    parser.add_argument("--filter-eq", metavar="JSON",
                        help='equality condition as JSON, e.g. {"field": "active", "value": true}')
    parser.add_argument("--filter-in", metavar="JSON",
                        help='set-membership condition as JSON, e.g. {"field": "orders", "values": [3, null]}')
    parser.add_argument("--filter-range", metavar="JSON",
                        help='inclusive integer range condition as JSON, e.g. {"field": "orders", "minimum": 3, "maximum": 5}')
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
        filter_in = None
        if args.filter_in is not None:
            try:
                filter_in = json.loads(args.filter_in)
            except ValueError as exc:
                raise ValueError(f"filter_in is not valid JSON: {exc}") from None
            if not isinstance(filter_in, dict):
                raise ValueError("filter_in must be a JSON object with 'field' and 'values' keys")
        filter_range = None
        if args.filter_range is not None:
            try:
                filter_range = json.loads(args.filter_range)
            except ValueError as exc:
                raise ValueError(f"filter_range is not valid JSON: {exc}") from None
            if not isinstance(filter_range, dict):
                raise ValueError(
                    "filter_range must be a JSON object with 'field' and "
                    "'minimum' and/or 'maximum' keys")
        normalize = normalize_csv if args.format == "csv" else normalize_jsonl
        schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
        result = normalize(args.source, schema,
                           duplicate_by=args.duplicate_by, filter_eq=filter_eq,
                           deduplicate_by=args.deduplicate_by, filter_in=filter_in,
                           filter_range=filter_range)
        if args.output_format == "csv":
            names = [column["name"] for column in schema["columns"]]
            output_payload = _csv_payload(names, result["records"])
        else:
            output_payload = "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in result["records"])
        _write_outputs_atomic(args.output, output_payload, args.errors, result["errors"])
        summary = {"accepted": result["accepted"], "rejected": result["rejected"]}
        if (args.filter_eq is not None or args.filter_in is not None
                or args.filter_range is not None):
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
