import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config import DATASET, OUTPUT_DIR


V_TABLES_FILE = OUTPUT_DIR / f"4_V_t_{DATASET}_results.json"
V_TOTAL_FILE = OUTPUT_DIR / f"5_V_total_{DATASET}_results.json"

OUTPUT_FILE = OUTPUT_DIR / f"6_V_classified_{DATASET}_results.json"

CSV_DIR = OUTPUT_DIR / "vocabularies"

MEASURES_CSV = CSV_DIR / "measures.csv"
DIMENSION_NAMES_CSV = CSV_DIR / "dimension_names.csv"
DIMENSION_VALUES_CSV = CSV_DIR / "dimension_values.csv"
UNITS_CSV = CSV_DIR / "units.csv"
OTHER_CSV = CSV_DIR / "other.csv"


# -------------------------
# Helper functions
# -------------------------

def normalize(text):
    """
    Basic normalization used only for comparisons.
    Original terms are preserved in the output.
    """
    if text is None:
        return ""

    return " ".join(str(text).strip().lower().split())


def clean(text):
    """
    Convert a value to a clean string.
    """
    if text is None:
        return ""

    return str(text).strip()


def save_csv(filename, terms):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term"])

        for term in sorted(terms, key=str.lower):
            writer.writerow([term])


def main():
    if not V_TABLES_FILE.exists():
        raise FileNotFoundError(f"Step 4 output not found: {V_TABLES_FILE}\nRun Step 4 before Step 6.")

    if not V_TOTAL_FILE.exists():
        raise FileNotFoundError(f"Step 5 output not found: {V_TOTAL_FILE}\nRun Step 5 before Step 6.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load data
    # -------------------------

    print("Loading data...")

    with open(V_TABLES_FILE, "r", encoding="utf-8") as f:
        V_tables = json.load(f)

    with open(V_TOTAL_FILE, "r", encoding="utf-8") as f:
        V_total = json.load(f)

    print(f"Dataset: {DATASET}")
    print(f"Number of tables: {len(V_tables)}")
    print(f"Number of vocabulary terms: {len(V_total)}")

    # -------------------------
    # Structural information
    # -------------------------

    dimension_names = set()
    dimension_values_by_name = defaultdict(set)
    title_terms = set()

    for dimensions in V_tables.values():
        if not isinstance(dimensions, dict):
            continue

        for dimension_name, values in dimensions.items():
            dimension_name = clean(dimension_name)

            if not isinstance(values, list):
                continue

            # TITLE values are measure candidates
            if normalize(dimension_name) == "title":
                for value in values:
                    value = clean(value)

                    if value:
                        title_terms.add(value)

                continue

            # Dimension name
            if dimension_name:
                dimension_names.add(dimension_name)

            # Dimension values
            for value in values:
                value = clean(value)

                if value:
                    dimension_values_by_name[dimension_name].add(value)

    # -------------------------
    # Normalized lookups
    # -------------------------

    dimension_names_normalized = {normalize(term) for term in dimension_names}
    title_terms_normalized = {normalize(term) for term in title_terms}

    # -------------------------
    # Unit-related dimensions
    # -------------------------

    UNIT_DIMENSION_NAMES = {
        "unit",
        "unit of measure",
        "measurement unit",
        "time frequency",
        "frequency",
    }

    unit_dimension_names_normalized = {normalize(name) for name in UNIT_DIMENSION_NAMES}

    # -------------------------
    # Build A and U candidates
    # -------------------------

    dimension_values = set()
    unit_values = set()

    for dimension_name, values in dimension_values_by_name.items():
        normalized_dimension_name = normalize(dimension_name)

        if normalized_dimension_name in unit_dimension_names_normalized:
            unit_values.update(values)
        else:
            dimension_values.update(values)

    dimension_values_normalized = {normalize(term) for term in dimension_values}
    unit_values_normalized = {normalize(term) for term in unit_values}

    # -------------------------
    # Classification
    # -------------------------

    classified = {
        "M": {},
        "N": {},
        "A": {},
        "U": {},
        "O": {},
    }

    for original_term, info in V_total.items():
        term = clean(original_term)

        if not term:
            continue

        normalized_term = normalize(term)

        is_dimension_name = normalized_term in dimension_names_normalized
        is_unit_value = normalized_term in unit_values_normalized
        is_dimension_value = normalized_term in dimension_values_normalized
        is_title_term = normalized_term in title_terms_normalized

        # N -> U -> A -> M -> O
        if is_dimension_name:
            category = "N"
        elif is_unit_value:
            category = "U"
        elif is_dimension_value:
            category = "A"
        elif is_title_term:
            category = "M"
        else:
            category = "O"

        classified[category][term] = info

    # -------------------------
    # Save classified JSON
    # -------------------------

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(classified, f, indent=4, ensure_ascii=False)

    # -------------------------
    # Save vocabulary CSVs
    # -------------------------

    save_csv(MEASURES_CSV, classified["M"].keys())
    save_csv(DIMENSION_NAMES_CSV, classified["N"].keys())
    save_csv(DIMENSION_VALUES_CSV, classified["A"].keys())
    save_csv(UNITS_CSV, classified["U"].keys())

    if classified["O"]:
        save_csv(OTHER_CSV, classified["O"].keys())

    # -------------------------
    # Summary
    # -------------------------

    print()
    print("========== STEP 6 SUMMARY ==========")

    for category in ["M", "N", "A", "U", "O"]:
        print(f"{category}: {len(classified[category])}")

    print()
    print("Unit-related dimensions:")

    for dimension_name in sorted(dimension_values_by_name):
        if normalize(dimension_name) in unit_dimension_names_normalized:
            print(f"- {dimension_name}: {len(dimension_values_by_name[dimension_name])} values")

    print()
    print(f"Classified vocabulary: {OUTPUT_FILE}")
    print(f"Measures:              {MEASURES_CSV}")
    print(f"Dimension names:       {DIMENSION_NAMES_CSV}")
    print(f"Dimension values:      {DIMENSION_VALUES_CSV}")
    print(f"Units:                 {UNITS_CSV}")

    if classified["O"]:
        print(f"Other:                 {OTHER_CSV}")


if __name__ == "__main__":
    main()