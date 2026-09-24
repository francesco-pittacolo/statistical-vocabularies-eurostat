"""
Quality evaluation for Step 6.

Usage:
    python evaluate/evaluate_step_6.py prepare
    python evaluate/evaluate_step_6.py score

Two manually annotated samples are used:

1. Random sample:
   100 terms sampled from the global vocabulary. This is the main sample
   for overall performance evaluation.

2. Stratified sample:
   25 terms per predicted category (M, N, A, U). This is useful for
   class-wise analysis.

Predicted categories are stored separately in metadata files and are hidden
from the annotation files.
"""

import argparse
import csv
import json
import random
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_OUTPUT_DIR = ROOT_DIR / "results" / "step_outputs" / "2000"
EVALUATION_DIR = ROOT_DIR / "results" / "evaluation" / "step6"

V_TOTAL_FILE = DATA_OUTPUT_DIR / "5_V_total_2000_results.json"
CLASSIFIED_FILE = DATA_OUTPUT_DIR / "6_V_classified_2000_results.json"

RANDOM_ANNOTATION_FILE = EVALUATION_DIR / "random_annotations.csv"
RANDOM_METADATA_FILE = EVALUATION_DIR / "random_metadata.json"

STRATIFIED_ANNOTATION_FILE = EVALUATION_DIR / "stratified_annotations.csv"
STRATIFIED_METADATA_FILE = EVALUATION_DIR / "stratified_metadata.json"

REPORT_FILE = EVALUATION_DIR / "evaluation_report.json"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SAMPLE_SIZE = 100
SAMPLE_PER_CATEGORY = 25
SEED = 42

CATEGORIES = ["M", "N", "A", "U"]


# ============================================================
# UTILITIES
# ============================================================

def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def annotation_file_has_ground_truth(path):
    if not path.exists():
        return False

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if "gold_category" not in (reader.fieldnames or []):
            return False

        return any(str(row.get("gold_category", "")).strip() for row in reader)


def confirm_overwrite(annotation_file, sample_name):
    if not annotation_file_has_ground_truth(annotation_file):
        return True

    print()
    print(f"WARNING: the {sample_name} annotation file contains ground-truth labels:")
    print(f"  {annotation_file}")
    print()

    answer = input(
        f"Regenerating the {sample_name} sample will overwrite these annotations. "
        "Continue? [y/N]: "
    ).strip().lower()

    return answer in {"y", "yes"}


# ============================================================
# RANDOM SAMPLE
# ============================================================

def prepare_random_sample():
    if not confirm_overwrite(RANDOM_ANNOTATION_FILE, "random"):
        print("Random sample preserved.")
        return

    v_total = load_json(V_TOTAL_FILE)
    classified = load_json(CLASSIFIED_FILE)

    predicted_by_term = {}

    for category in CATEGORIES:
        for term in classified.get(category, {}):
            predicted_by_term[term] = category

    population = []

    for term, info in v_total.items():
        if term not in predicted_by_term:
            continue

        population.append({
            "term": term,
            "source_tables": "; ".join(str(x) for x in info.get("tables", [])),
            "contexts": "; ".join(str(x) for x in info.get("contexts", [])),
            "predicted_category": predicted_by_term[term],
        })

    if len(population) < RANDOM_SAMPLE_SIZE:
        raise ValueError(
            f"Only {len(population)} eligible terms are available, "
            f"but {RANDOM_SAMPLE_SIZE} are required."
        )

    random.seed(SEED)
    sample = random.sample(population, RANDOM_SAMPLE_SIZE)
    random.shuffle(sample)

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {}

    with RANDOM_ANNOTATION_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "term", "source_tables", "contexts", "gold_category"])

        for i, item in enumerate(sample, start=1):
            metadata[str(i)] = {
                "term": item["term"],
                "predicted_category": item["predicted_category"],
            }

            writer.writerow([
                i,
                item["term"],
                item["source_tables"],
                item["contexts"],
                "",
            ])

    with RANDOM_METADATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print("Random blind sample created.")
    print(f"Annotation file: {RANDOM_ANNOTATION_FILE}")
    print(f"Metadata file:   {RANDOM_METADATA_FILE}")
    print(f"Rows: {len(sample)}")
    print("Fill only gold_category with M, N, A or U.")


# ============================================================
# STRATIFIED SAMPLE
# ============================================================

def prepare_stratified_sample():
    if not confirm_overwrite(STRATIFIED_ANNOTATION_FILE, "stratified"):
        print("Stratified sample preserved.")
        return

    classified = load_json(CLASSIFIED_FILE)
    sample = []

    random.seed(SEED)

    for category in CATEGORIES:
        terms = sorted(classified.get(category, {}).keys())

        if len(terms) < SAMPLE_PER_CATEGORY:
            raise ValueError(
                f"Category {category} contains only {len(terms)} terms, "
                f"but {SAMPLE_PER_CATEGORY} are required."
            )

        selected = random.sample(terms, SAMPLE_PER_CATEGORY)

        for term in selected:
            info = classified[category][term]

            sample.append({
                "term": term,
                "source_tables": "; ".join(str(x) for x in info.get("tables", [])),
                "contexts": "; ".join(str(x) for x in info.get("contexts", [])),
                "predicted_category": category,
            })

    random.shuffle(sample)

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {}

    with STRATIFIED_ANNOTATION_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "term", "source_tables", "contexts", "gold_category"])

        for i, item in enumerate(sample, start=1):
            metadata[str(i)] = {
                "term": item["term"],
                "predicted_category": item["predicted_category"],
            }

            writer.writerow([
                i,
                item["term"],
                item["source_tables"],
                item["contexts"],
                "",
            ])

    with STRATIFIED_METADATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print("Stratified blind sample created.")
    print(f"Annotation file: {STRATIFIED_ANNOTATION_FILE}")
    print(f"Metadata file:   {STRATIFIED_METADATA_FILE}")
    print(f"Rows: {len(sample)}")
    print("Fill only gold_category with M, N, A or U.")


# ============================================================
# PREPARE
# ============================================================

def prepare():
    prepare_random_sample()
    prepare_stratified_sample()


# ============================================================
# ANNOTATION READING
# ============================================================

def read_annotation(path):
    rows = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {"id", "term", "gold_category"}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for row in reader:
            gold = row["gold_category"].strip().upper()

            if not gold:
                raise ValueError(f"Row {row['id']} has no gold_category.")

            if gold not in CATEGORIES:
                raise ValueError(
                    f"Row {row['id']} has invalid gold_category '{gold}'."
                )

            rows.append({
                "id": row["id"],
                "term": row["term"],
                "gold": gold,
            })

    return rows


# ============================================================
# EVALUATION
# ============================================================

def evaluate(annotation_file, metadata_file):
    rows = read_annotation(annotation_file)
    metadata = load_json(metadata_file)

    confusion = {
        true_cat: {pred_cat: 0 for pred_cat in CATEGORIES}
        for true_cat in CATEGORIES
    }

    for row in rows:
        meta = metadata.get(str(row["id"]))

        if meta is None:
            raise ValueError(f"No metadata found for id {row['id']}.")

        if meta["term"] != row["term"]:
            raise ValueError(f"Term mismatch for id {row['id']}.")

        predicted = meta["predicted_category"]
        confusion[row["gold"]][predicted] += 1

    total = len(rows)
    correct = sum(confusion[category][category] for category in CATEGORIES)
    accuracy = correct / total if total else 0.0

    per_class = {}

    for category in CATEGORIES:
        tp = confusion[category][category]

        fp = sum(
            confusion[true_cat][category]
            for true_cat in CATEGORIES
            if true_cat != category
        )

        fn = sum(
            confusion[category][pred_cat]
            for pred_cat in CATEGORIES
            if pred_cat != category
        )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        per_class[category] = {
            "support": tp + fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    macro_f1 = (
        sum(item["f1"] for item in per_class.values()) / len(per_class)
        if per_class
        else 0.0
    )

    return {
        "n": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


# ============================================================
# SCORE
# ============================================================

def score():
    random_result = evaluate(RANDOM_ANNOTATION_FILE, RANDOM_METADATA_FILE)
    stratified_result = evaluate(
        STRATIFIED_ANNOTATION_FILE,
        STRATIFIED_METADATA_FILE,
    )

    report = {
        "random_sample": random_result,
        "stratified_sample": stratified_result,
    }

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("STEP 6 EVALUATION")
    print("=" * 70)

    for name, result in [
        ("RANDOM SAMPLE", random_result),
        ("STRATIFIED SAMPLE", stratified_result),
    ]:
        print()
        print(name)
        print("-" * 70)
        print(f"N:        {result['n']}")
        print(f"Accuracy: {result['accuracy']:.1%}")
        print(f"Macro F1: {result['macro_f1']:.1%}")

        for category, metrics in result["per_class"].items():
            print(
                f"{category}: "
                f"P={metrics['precision']:.1%} "
                f"R={metrics['recall']:.1%} "
                f"F1={metrics['f1']:.1%} "
                f"support={metrics['support']}"
            )

    print(f"\nSaved report to: {REPORT_FILE}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quality evaluation for Step 6.")
    parser.add_argument("mode", choices=["prepare", "score"])
    args = parser.parse_args()

    if args.mode == "prepare":
        prepare()
    else:
        score()