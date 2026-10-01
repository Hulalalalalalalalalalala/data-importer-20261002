# Data Importer

Normalize a CSV file to JSONL using a small explicit schema. Requires Python 3.10+; no installation dependencies.

`python3 data_importer.py samples/customers.csv --schema samples/schema.json --output customers.jsonl --errors errors.jsonl` converts the two valid sample rows. `samples/mixed.csv` contains one accepted row and two rejected rows. Each rejected row is written separately with its logical CSV record number (header is 1) and field errors; accepted rows remain available.

Schema columns specify name, type (`string`, `integer`, `boolean`) and optional `required`. A column may also declare an optional `source`: the CSV column it reads from, while `name` remains the output field name; omitting `source` reads the column named by `name`. An explicit `source` must be a non-empty, non-blank string, and the effective source names must be unique across columns (a source may coincide with another field's `name`, and two fields may swap sources). The header must match the effective source names and their order exactly, compared literally without trimming or case folding. Text is trimmed, absent optional values become JSON null, and booleans accept true/false without case sensitivity. Output objects use sorted keys. The input is processed in memory. Outputs replace existing files but cannot alias either input or each other; parent output directories must already exist. CLI exit codes are 0 for no rejected rows, 1 for row errors and 2 for invalid configuration (including mapping or header mismatches) or I/O errors.

The public API is `normalize_csv(source, schema)`, returning records, errors and counts. `write_jsonl(path, records)` persists either sequence. Run `python3 -B -m unittest -v` for API conversion and CLI checks.
