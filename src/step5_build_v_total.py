import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config import DATASET, OUTPUT_DIR


INPUT_FILE = OUTPUT_DIR / f"4_V_t_{DATASET}_results.json"
OUTPUT_FILE = OUTPUT_DIR / f"5_V_total_{DATASET}_results.json"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Step 4 output not found: {INPUT_FILE}\n"
            "Run Step 4 before Step 5."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load V(t)
    # -------------------------

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        V_tables = json.load(f)

    # -------------------------
    # Build global vocabulary
    # -------------------------

    V_total = {}

    for table, dimensions in V_tables.items():
        for context, values in dimensions.items():

            # Add the dimension name itself.
            # TITLE is excluded because it is not a real dimension name.
            if context != "TITLE":
                if context not in V_total:
                    V_total[context] = {
                        "tables": [],
                        "contexts": [],
                    }

                if table not in V_total[context]["tables"]:
                    V_total[context]["tables"].append(table)

                if "DIMENSION_NAME" not in V_total[context]["contexts"]:
                    V_total[context]["contexts"].append("DIMENSION_NAME")

            # Add the values associated with the context.
            for value in values:
                if value not in V_total:
                    V_total[value] = {
                        "tables": [],
                        "contexts": [],
                    }

                if table not in V_total[value]["tables"]:
                    V_total[value]["tables"].append(table)

                if context not in V_total[value]["contexts"]:
                    V_total[value]["contexts"].append(context)

    # -------------------------
    # Sort
    # -------------------------

    V_total = dict(sorted(V_total.items()))

    for term in V_total:
        V_total[term]["tables"] = sorted(V_total[term]["tables"])
        V_total[term]["contexts"] = sorted(V_total[term]["contexts"])

    # -------------------------
    # Statistics
    # -------------------------

    dimension_name_terms = sum(
        1
        for info in V_total.values()
        if "DIMENSION_NAME" in info["contexts"]
    )

    title_terms = sum(
        1
        for info in V_total.values()
        if "TITLE" in info["contexts"]
    )

    multi_table_terms = sum(
        1
        for info in V_total.values()
        if len(info["tables"]) > 1
    )

    # -------------------------
    # Save
    # -------------------------

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(V_total, f, indent=4, ensure_ascii=False)

    # -------------------------
    # Summary
    # -------------------------

    print("========== SUMMARY ==========")
    print(f"Dataset:                           {DATASET}")
    print(f"Vocabulary size:                   {len(V_total)}")
    print(f"Terms used as dimension names:     {dimension_name_terms}")
    print(f"Terms coming from titles:          {title_terms}")
    print(f"Terms appearing in multiple tables:{multi_table_terms}")
    print(f"Saved file:                        {OUTPUT_FILE}")


if __name__ == "__main__":
    main()