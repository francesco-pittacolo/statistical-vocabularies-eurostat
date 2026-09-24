import csv
import json
import re
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config import DATASET, TABLE_DIR, OUTPUT_DIR


D_FILE = OUTPUT_DIR / f"1_D_t_{DATASET}_results.json"
OUTPUT_FILE = OUTPUT_DIR / f"2_S_t_{DATASET}_results.json"


with open(D_FILE, "r", encoding="utf-8") as f:
    D = json.load(f)


def is_numeric_string(value):
    value = str(value).strip()
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", value))


def clean_value(value):
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    if value in {":", ": ", "..", "..."}:
        return None

    if is_numeric_string(value):
        return None

    return value


def pandas_style_headers(header):
    counts = {}
    result = []

    for col in header:
        col = str(col).strip()

        if col not in counts:
            counts[col] = 0
            result.append(col)
        else:
            counts[col] += 1
            result.append(f"{col}.{counts[col]}")

    return result


def process_table(csv_file):
    table_name = csv_file.stem

    try:
        with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)

            header = next(reader)
            header = [str(col).strip() for col in header]
            header = pandas_style_headers(header)

            time_columns = {
                str(value).strip()
                for value in D.get(table_name, [])
            }

            without_time = len(time_columns) == 0

            # First try to locate the temporal boundary using D(t).
            first_time_index = None

            for idx, col in enumerate(header):
                if col in time_columns:
                    first_time_index = idx
                    break

            # Structural fallback when no explicit temporal match is available.
            if first_time_index is None:
                for idx, col in enumerate(header):
                    if "\\" in col:
                        first_time_index = idx + 1
                        break

            if first_time_index is None:
                keep_until = len(header)
            else:
                keep_until = first_time_index

            column_names = header[:keep_until]
            column_values = [set() for _ in column_names]

            for row in reader:
                if not row:
                    continue

                max_idx = min(keep_until, len(row))

                for idx in range(max_idx):
                    value = clean_value(row[idx])

                    if value is not None:
                        column_values[idx].add(value)

            table_vocab = {}

            for idx, col in enumerate(column_names):
                col_name = col.strip().split("\\")[0]
                values = column_values[idx]

                if values:
                    table_vocab[col_name] = sorted(values)

            return table_name, table_vocab, without_time, None

    except Exception as e:
        return table_name, {}, False, str(e)


def main():
    if not TABLE_DIR.exists():
        raise FileNotFoundError(f"Table directory not found: {TABLE_DIR}")

    if not D_FILE.exists():
        raise FileNotFoundError(
            f"Step 1 output not found: {D_FILE}\n"
            "Run Step 1 before Step 2."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    S = {}

    csv_files = sorted(TABLE_DIR.glob("*.csv"))
    total_tables = len(csv_files)

    tables_with_S = 0
    tables_without_time = []
    errors = []

    workers = max(1, cpu_count() - 1)

    print(f"Dataset: {DATASET}")
    print(f"Tables: {total_tables}")
    print(f"Workers: {workers}")
    print()

    with Pool(processes=workers) as pool:
        results = pool.imap_unordered(
            process_table,
            csv_files,
            chunksize=10,
        )

        for i, result in enumerate(results, start=1):
            table_name, table_vocab, without_time, error = result

            S[table_name] = table_vocab

            if table_vocab:
                tables_with_S += 1

            if without_time:
                tables_without_time.append(table_name)

            if error is not None:
                errors.append({
                    "table": table_name,
                    "error": error,
                })

                print(f"\nError processing {table_name}.csv: {error}")

            print(
                f"Processing {i}/{total_tables} "
                f"({i / total_tables * 100:.2f}%)",
                end="\r",
                flush=True,
            )

    S = dict(sorted(S.items()))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(S, f, indent=4, ensure_ascii=False)

    print()
    print()
    print("========== SUMMARY ==========")
    print(f"Dataset:                      {DATASET}")
    print(f"Tables processed:             {total_tables}")
    print(f"Tables with S(t):             {tables_with_S}")
    print(f"Tables without detected D(t): {len(tables_without_time)}")
    print(f"Errors:                       {len(errors)}")
    print(f"Saved file:                   {OUTPUT_FILE}")

    if tables_without_time:
        print("\n========== TABLES WITHOUT D(t) ==========")

        for table in sorted(tables_without_time):
            print("-", table)

    if errors:
        print("\n========== ERRORS ==========")

        for error in errors:
            print("-", error["table"], ":", error["error"])


if __name__ == "__main__":
    main()