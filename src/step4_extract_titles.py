import json
import re
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config import DATASET, OUTPUT_DIR, REFERENCE_DIR


V_FILE = OUTPUT_DIR / f"3_V_t_{DATASET}_results.json"

NUTS_FILE = REFERENCE_DIR / "NUTS2021-NUTS2024.xlsx"
TITLE_FILE = REFERENCE_DIR / "file-names_to_titles_eurostat_unfiltered.csv"

OUTPUT_FILE = OUTPUT_DIR / f"4_V_t_{DATASET}_results.json"


# -------------------------
# Country names
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


def load_titles():
    titles_df = pd.read_csv(TITLE_FILE, encoding="utf-8")

    titles = {}

    for _, row in titles_df.iterrows():
        table = str(row["filename"]).replace(".csv", "")
        titles[table] = str(row["title"])

    return titles


def load_geo_references():
    nuts_df = pd.read_excel(
        NUTS_FILE,
        sheet_name="NUTS2024",
    )

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
    cyr_df = None

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
        print("Warning loading Cyrillic/Greek sheet:", e)

    # Geographic terms used for title matching.
    # Country codes and NUTS codes are deliberately excluded.
    title_geo_terms = set()

    title_geo_terms.update(
        nuts_df["NUTS label"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    title_geo_terms.update(COUNTRY_NAMES.values())

    if cyr_df is not None:
        for col in cyr_df.columns:
            title_geo_terms.update(
                cyr_df[col]
                .dropna()
                .astype(str)
                .str.strip()
            )

    title_geo_terms = {
        term
        for term in title_geo_terms
        if term
    }

    return geo_terms, title_geo_terms


def build_geo_patterns(title_geo_terms):
    """
    Compile geographical expressions for title matching.

    Longer terms are processed first to avoid partial matches.
    """
    return [
        (
            re.compile(
                r"(?<!\w)" + re.escape(term) + r"(?!\w)",
                re.IGNORECASE,
            ),
            term,
        )
        for term in sorted(
            title_geo_terms,
            key=len,
            reverse=True,
        )
    ]


def remove_geography_from_title(text, geo_patterns):
    result = text

    for pattern, _ in geo_patterns:
        result = pattern.sub("", result)

    return result


def remove_dates(text):
    protected = {}

    def protect(match):
        key = f"__KEEP_{len(protected)}__"
        protected[key] = match.group(0)
        return key

    # Preserve index-base expressions such as (2015 = 100).
    text = re.sub(
        r"\(\s*\d{4}\s*=\s*100\s*\)",
        protect,
        text,
    )

    # Date ranges
    text = re.sub(
        r"\b\d{4}\s*[-–]\s*\d{4}\b",
        "",
        text,
    )

    text = re.sub(
        r"\bbetween\s+\d{4}\s+and\s+\d{4}\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bfrom\s+\d{4}\s+onwards\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bfrom\s+\d{4}\s+to\s+\d{4}\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bdata\s+from\s+\d{4}\s+to\s+\d{4}\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\buntil\s+\d{4}\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bup\s+to\s+\d{4}\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bsince\s+\d{4}\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Year-month / date / semester / quarter
    text = re.sub(
        r"\b(?:19|20|21)\d{2}-\d{2}-\d{2}\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\b(?:19|20|21)\d{2}-\d{2}\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\b(?:19|20|21)\d{2}-S[12]\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\b(?:19|20|21)\d{2}-Q[1-4]\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bQ[1-4]\s+(?:19|20|21)\d{2}\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Month names + year
    months = (
        r"January|February|March|April|May|June|July|August|"
        r"September|October|November|December|"
        r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|"
        r"janvier|février|fevrier|mars|avril|mai|juin|juillet|"
        r"août|aout|septembre|octobre|novembre|décembre|decembre"
    )

    text = re.sub(
        rf"\b(?:{months})\s+(?:19|20|21)\d{{2}}\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        rf"\b(?:19|20|21)\d{{2}}\s+(?:{months})\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Other temporal expressions
    text = re.sub(
        r"\b(survey)\s+\d{4}\b",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bin\s+\d{4}(?=,|\s|$)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Standalone year
    text = re.sub(
        r"\b(?:19|20|21)\d{2}\b",
        "",
        text,
    )

    # Restore protected expressions
    for key, value in protected.items():
        text = text.replace(key, value)

    return text


def cleanup(text):
    # Empty parentheses
    text = re.sub(r"\(\s*\)", "", text)

    # Multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Spaces before punctuation
    text = re.sub(r"\s+([,.;:])", r"\1", text)

    # Repeated hyphens
    text = re.sub(r"\s*-\s*-\s*", " - ", text)

    # Trailing hyphen
    text = re.sub(r"\s*-\s*$", "", text)

    # Opening parenthesis followed by comma
    text = re.sub(r"\(\s*,", "(", text)

    return text.strip(" -,")


def main():
    if not V_FILE.exists():
        raise FileNotFoundError(
            f"Step 3 output not found: {V_FILE}\n"
            "Run Step 3 before Step 4."
        )

    if not NUTS_FILE.exists():
        raise FileNotFoundError(
            f"NUTS reference file not found: {NUTS_FILE}"
        )

    if not TITLE_FILE.exists():
        raise FileNotFoundError(
            f"Title file not found: {TITLE_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load V(t)
    with open(V_FILE, "r", encoding="utf-8") as f:
        V = json.load(f)

    # Load titles
    titles = load_titles()

    # Geographic reference
    geo_terms, title_geo_terms = load_geo_references()
    geo_patterns = build_geo_patterns(title_geo_terms)

    print(f"Dataset: {DATASET}")
    print(f"Geographic reference terms: {len(geo_terms)}")
    print(
        "Geographic terms used for title matching:",
        len(title_geo_terms),
    )
    print()

    # -------------------------
    # Process titles
    # -------------------------

    tables = list(V.keys())
    total_tables = len(tables)

    titles_found = 0
    titles_added = 0
    titles_empty = 0
    titles_missing = 0

    for i, table in enumerate(tables, start=1):
        print(
            f"Processing {i}/{total_tables} "
            f"({i / total_tables * 100:.2f}%) - {table}",
            end="\r",
            flush=True,
        )

        if table not in titles:
            titles_missing += 1
            continue

        titles_found += 1
        title = titles[table]

        # Remove geographical information
        title = remove_geography_from_title(
            title,
            geo_patterns,
        )

        # Remove temporal information
        title = remove_dates(title)

        # Clean remaining text
        title = cleanup(title)

        if title:
            V[table]["TITLE"] = [title]
            titles_added += 1
        else:
            titles_empty += 1

    # -------------------------
    # Save
    # -------------------------

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(V, f, indent=4, ensure_ascii=False)

    distinct_title_remainders = {
        dimensions["TITLE"][0]
        for dimensions in V.values()
        if "TITLE" in dimensions and dimensions["TITLE"]
    }

    # -------------------------
    # Summary
    # -------------------------

    print()
    print()
    print("========== SUMMARY ==========")
    print(f"Dataset:                             {DATASET}")
    print(f"Tables in V(t):                      {len(V)}")
    print(f"Titles found:                        {titles_found}")
    print(f"Titles added to V(t):                {titles_added}")
    print(
        f"Distinct title remainders:           "
        f"{len(distinct_title_remainders)}"
    )
    print(f"Empty title remainders:              {titles_empty}")
    print(f"Missing titles:                      {titles_missing}")
    print(f"Geographic reference terms:          {len(geo_terms)}")
    print(
        f"Geographic terms used for titles:    "
        f"{len(title_geo_terms)}"
    )
    print(f"Saved file:                          {OUTPUT_FILE}")


if __name__ == "__main__":
    main()