import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config import DATASET, OUTPUT_DIR, REFERENCE_DIR


S_FILE = OUTPUT_DIR / f"2_S_t_{DATASET}_results.json"
NUTS_FILE = REFERENCE_DIR / "NUTS2021-NUTS2024.xlsx"

GEO_OUTPUT = OUTPUT_DIR / f"3_Geo_t_{DATASET}_results.json"
V_OUTPUT = OUTPUT_DIR / f"3_V_t_{DATASET}_results.json"


# -------------------------
# Geographical reference
# -------------------------

COUNTRY_NAMES = {
    "AT": "Austria",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "CY": "Cyprus",
    "CZ": "Czechia",
    "DE": "Germany",
    "DK": "Denmark",
    "EE": "Estonia",
    "EL": "Greece",
    "ES": "Spain",
    "FI": "Finland",
    "FR": "France",
    "HR": "Croatia",
    "HU": "Hungary",
    "IE": "Ireland",
    "IT": "Italy",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "LV": "Latvia",
    "MT": "Malta",
    "NL": "Netherlands",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "SE": "Sweden",
    "SI": "Slovenia",
    "SK": "Slovakia",
}


def normalize_geo(value):
    """
    Basic normalization used for geographical matching.

    The original value is not modified in the output.
    """
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def get_geo_candidates(value):
    """
    Return the original value together with variants obtained by
    removing parenthesized annotations one at a time.
    """
    value = str(value).strip()

    candidates = {value}
    queue = [value]

    while queue:
        current = queue.pop()

        for match in re.finditer(r"\s*\([^()]*\)", current):
            new_value = (
                current[:match.start()] + current[match.end():]
            ).strip()

            if new_value and new_value not in candidates:
                candidates.add(new_value)
                queue.append(new_value)

    return candidates


def load_geo_terms():
    nuts_df = pd.read_excel(NUTS_FILE, sheet_name="NUTS2024")

    geo_terms = set()

    # Country codes
    geo_terms.update(
        nuts_df["Country code"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    # NUTS codes
    geo_terms.update(
        nuts_df["NUTS Code"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    # NUTS labels
    geo_terms.update(
        nuts_df["NUTS label"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    # Country names
    geo_terms.update(COUNTRY_NAMES.values())

    # Cyrillic and Greek variants
    try:
        cyr_df = pd.read_excel(
            NUTS_FILE,
            sheet_name="Cyrillic & Greek to Latin",
        )

        for col in cyr_df.columns:
            geo_terms.update(
                cyr_df[col]
                .dropna()
                .astype(str)
                .str.strip()
            )

    except Exception as e:
        print("Warning loading Cyrillic/Greek variants:", e)

    return geo_terms


def main():
    if not S_FILE.exists():
        raise FileNotFoundError(
            f"Step 2 output not found: {S_FILE}\n"
            "Run Step 2 before Step 3."
        )

    if not NUTS_FILE.exists():
        raise FileNotFoundError(
            f"NUTS reference file not found: {NUTS_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load S(t)
    # -------------------------

    with open(S_FILE, "r", encoding="utf-8") as f:
        S = json.load(f)

    # -------------------------
    # Build geographical lookup
    # -------------------------

    geo_terms = load_geo_terms()

    geo_lookup = {
        normalize_geo(term)
        for term in geo_terms
    }

    # -------------------------
    # Build Geo(t) and V(t)
    # -------------------------

    Geo = {}
    V = {}

    tables = list(S.items())
    total_tables = len(tables)

    for i, (table, dimensions) in enumerate(tables, start=1):
        print(
            f"Processing {i}/{total_tables} "
            f"({i / total_tables * 100:.2f}%) - {table}",
            end="\r",
            flush=True,
        )

        geo_values = set()
        v_dimensions = {}

        for dimension_name, values in dimensions.items():
            remaining_values = []

            for value in values:
                is_geo = False

                # Try the original value and variants with
                # parenthesized annotations removed.
                for candidate in get_geo_candidates(value):
                    normalized = normalize_geo(candidate)

                    if normalized in geo_lookup:
                        is_geo = True
                        break

                if is_geo:
                    # Keep the original value from the table.
                    geo_values.add(value)
                else:
                    remaining_values.append(value)

            if remaining_values:
                v_dimensions[dimension_name] = sorted(
                    set(remaining_values)
                )

        Geo[table] = sorted(geo_values)
        V[table] = v_dimensions

    # -------------------------
    # Save outputs
    # -------------------------

    with open(GEO_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(Geo, f, indent=4, ensure_ascii=False)

    with open(V_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(V, f, indent=4, ensure_ascii=False)

    # -------------------------
    # Statistics
    # -------------------------

    tables_with_geo = sum(
        1
        for values in Geo.values()
        if values
    )

    distinct_geo_values = {
        value
        for values in Geo.values()
        for value in values
    }

    total_geo_values = sum(
        len(values)
        for values in Geo.values()
    )

    tables_with_v = sum(
        1
        for dimensions in V.values()
        if dimensions
    )

    distinct_v_values = {
        value
        for dimensions in V.values()
        for values in dimensions.values()
        for value in values
    }

    total_v_values = sum(
        len(values)
        for dimensions in V.values()
        for values in dimensions.values()
    )

    total_v_dimensions = sum(
        len(dimensions)
        for dimensions in V.values()
    )

    # -------------------------
    # Summary
    # -------------------------

    print()
    print()
    print("========== SUMMARY ==========")
    print(f"Dataset:                            {DATASET}")
    print(f"Tables processed:                   {len(S)}")
    print(f"Geographic reference terms:         {len(geo_terms)}")

    print("\n--- Geo(t) ---")
    print(f"Tables with Geo(t):                 {tables_with_geo}")
    print(f"Geographic values across tables:    {total_geo_values}")
    print(f"Distinct geographic values found:   {len(distinct_geo_values)}")

    print("\n--- V(t) ---")
    print(f"Tables with V(t):                   {tables_with_v}")
    print(f"Dimensions remaining across tables: {total_v_dimensions}")
    print(f"Vocabulary values across tables:    {total_v_values}")
    print(f"Distinct vocabulary values:         {len(distinct_v_values)}")

    print(f"\nGeo saved: {GEO_OUTPUT}")
    print(f"V saved:   {V_OUTPUT}")


if __name__ == "__main__":
    main()