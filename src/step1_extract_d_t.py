import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config import DATASET, TABLE_DIR, OUTPUT_DIR


OUTPUT_FILE = OUTPUT_DIR / f"1_D_t_{DATASET}_results.json"


# -------------------------
# Time patterns
# -------------------------

YEAR_ONLY = re.compile(r"^\s*(19\d{2}|20\d{2}|2100)(\.0)?\s*$")
YEAR_MONTH = re.compile(r"^\s*(19\d{2}|20\d{2}|2100)-\d{2}\s*$")
YEAR_DAY = re.compile(r"^\s*(19\d{2}|20\d{2}|2100)-\d{2}-\d{2}\s*$")
YEAR_SEMESTER = re.compile(
    r"^\s*(19\d{2}|20\d{2}|2100)-S[1-2]\s*$",
    re.IGNORECASE,
)


def is_time_header(col):
    """
    Decide whether a column name represents a time interval.
    """
    col = str(col).strip()

    # Annual
    # Example: 2000, 2000.0
    if YEAR_ONLY.match(col):
        return True

    # Monthly
    # Example: 2008-01
    if YEAR_MONTH.match(col):
        return True

    # Daily
    # Example: 2020-01-31
    if YEAR_DAY.match(col):
        return True

    # Semester
    # Example: 1985-S1
    if YEAR_SEMESTER.match(col):
        return True

    # Quarter
    # Example: 2020-Q1 or Q1 2020
    if re.search(r"(19\d{2}|20\d{2}|2100)", col):
        if re.search(r"Q[1-4]", col, re.IGNORECASE):
            return True

    # Month names
    # Example: Janvier-Mars 1995
    month_words = [
        "janvier", "février", "fevrier",
        "mars", "avril", "mai", "juin",
        "juillet", "août", "aout",
        "septembre", "octobre",
        "novembre", "décembre", "decembre",
        "january", "february", "march",
        "april", "may", "june",
        "july", "august", "september",
        "october", "november", "december",
    ]

    lower = col.lower()

    if any(month in lower for month in month_words):
        if re.search(r"(19\d{2}|20\d{2}|2100)", col):
            return True

    return False


def main():
    if not TABLE_DIR.exists():
        raise FileNotFoundError(f"Table directory not found: {TABLE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    D = {}

    csv_files = sorted(TABLE_DIR.glob("*.csv"))
    total_tables = len(csv_files)

    tables_with_time = 0
    tables_without_time = []
    tables_with_errors = []

    for i, csv_file in enumerate(csv_files, start=1):
        print(
            f"Processing {i}/{total_tables} "
            f"({i / total_tables * 100:.2f}%) - {csv_file.name}",
            end="\r",
            flush=True,
        )

        try:
            # Try normal UTF-8 first
            try:
                df = pd.read_csv(csv_file, nrows=0, encoding="utf-8")
            except UnicodeDecodeError:
                # Fallback for unusual encodings
                df = pd.read_csv(csv_file, nrows=0, encoding="latin1")

            intervals = sorted({
                str(col).strip()
                for col in df.columns
                if is_time_header(col)
            })

            D[csv_file.stem] = intervals

            if intervals:
                tables_with_time += 1
            else:
                tables_without_time.append(csv_file.name)

        except Exception as e:
            tables_with_errors.append({
                "file": csv_file.name,
                "error": str(e),
            })

    # Move to next line after progress
    print()

    # -------------------------
    # Save JSON
    # -------------------------

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(D, f, indent=4, ensure_ascii=False)

    # -------------------------
    # Summary
    # -------------------------

    print("========== SUMMARY ==========")
    print(f"Dataset:                      {DATASET}")
    print(f"Total tables processed:       {total_tables}")
    print(f"Tables with D(t):             {tables_with_time}")
    print(f"Tables without time columns:  {len(tables_without_time)}")
    print(f"Tables with errors:           {len(tables_with_errors)}")
    print(f"Saved file:                   {OUTPUT_FILE}")

    if tables_without_time:
        print("\nTables without detected time columns:")
        for table in tables_without_time:
            print("-", table)

    if tables_with_errors:
        print("\nTables with reading errors:")
        for error in tables_with_errors:
            print("-", error["file"], ":", error["error"])


if __name__ == "__main__":
    main()