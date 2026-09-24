"""
Quality evaluation for Step 7.

Usage:
    python evaluate/evaluate_step_7.py prepare
    python evaluate/evaluate_step_7.py score

The evaluation sample is drawn from the measures obtained from the
2,000-table subset and is distributed across low-, middle-, and high-score
regions.

The annotation file is blind. Predicted domains, scores, methods, and
confidence values are stored separately in a metadata file.

Primary-domain assignment is the main quantitative evaluation task.

Secondary domains are treated as optional additional output and are
analysed descriptively rather than included in the main performance metrics.
"""

import argparse
import json
import random
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_OUTPUT_DIR = ROOT_DIR / "results" / "step_outputs" / "2000" / "step7"
EVALUATION_DIR = ROOT_DIR / "results" / "evaluation" / "step7"

FINAL_FILE = DATA_OUTPUT_DIR / "7_M_final_classification_2000_results.json"
DETAILS_FILE = DATA_OUTPUT_DIR / "7_M_classification_details_2000_results.json"


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_SIZE = 150
SEED = 42

ANNOTATION_FILE = EVALUATION_DIR / f"annotations_{SAMPLE_SIZE}.json"
METADATA_FILE = EVALUATION_DIR / f"metadata_{SAMPLE_SIZE}.json"
REPORT_FILE = EVALUATION_DIR / f"evaluation_report_{SAMPLE_SIZE}.json"


# ============================================================
# UTILITIES
# ============================================================

def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def annotation_file_has_ground_truth(path):
    if not path.exists():
        return False

    annotation = load_json(path)

    if not isinstance(annotation, list):
        return False

    for row in annotation:
        gold_domain = str(row.get("gold_domain", "") or "").strip()
        gold_secondary = str(row.get("gold_secondary_domain", "") or "").strip()

        if gold_domain or gold_secondary:
            return True

    return False


def confirm_overwrite():
    if not annotation_file_has_ground_truth(ANNOTATION_FILE):
        return True

    print()
    print("WARNING: the Step 7 annotation file contains ground-truth labels:")
    print(f"  {ANNOTATION_FILE}")
    print()

    answer = input(
        "Regenerating this evaluation sample will overwrite these annotations. "
        "Continue? [y/N]: "
    ).strip().lower()

    return answer in {"y", "yes"}


# ============================================================
# PREPARE EVALUATION SAMPLE
# ============================================================

def prepare():
    if not confirm_overwrite():
        print("Step 7 evaluation sample preserved.")
        return

    final = load_json(FINAL_FILE)
    details = load_json(DETAILS_FILE)

    records = []

    for measure, result in final.items():
        predicted_domain = result.get("domain")

        if not predicted_domain:
            continue

        detail = details.get(measure, {})
        score = float(detail.get("final_blended_score", 0.0))

        records.append({
            "measure": measure,
            "predicted_domain": predicted_domain,
            "secondary_domain": result.get("secondary_domain"),
            "score": score,
            "method": result.get("method", "unknown"),
            "confidence": result.get("confidence"),
        })

    if len(records) < SAMPLE_SIZE:
        raise ValueError(
            f"Only {len(records)} classified measures are available, "
            f"but {SAMPLE_SIZE} are required."
        )

    records.sort(key=lambda x: x["score"])

    third = len(records) // 3

    low = records[:third]
    middle = records[third:2 * third]
    high = records[2 * third:]

    base_size = SAMPLE_SIZE // 3
    remainder = SAMPLE_SIZE % 3

    group_sizes = [
        base_size + (1 if remainder > 0 else 0),
        base_size + (1 if remainder > 1 else 0),
        base_size,
    ]

    if group_sizes[0] > len(low):
        raise ValueError(
            f"Low-score region contains only {len(low)} measures, "
            f"but {group_sizes[0]} are required."
        )

    if group_sizes[1] > len(middle):
        raise ValueError(
            f"Middle-score region contains only {len(middle)} measures, "
            f"but {group_sizes[1]} are required."
        )

    if group_sizes[2] > len(high):
        raise ValueError(
            f"High-score region contains only {len(high)} measures, "
            f"but {group_sizes[2]} are required."
        )

    random.seed(SEED)

    sample = []
    sample.extend(random.sample(low, group_sizes[0]))
    sample.extend(random.sample(middle, group_sizes[1]))
    sample.extend(random.sample(high, group_sizes[2]))

    random.shuffle(sample)

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    annotation = []
    metadata = {}

    for i, row in enumerate(sample, start=1):
        annotation.append({
            "id": i,
            "measure": row["measure"],
            "gold_domain": "",
            "gold_secondary_domain": "",
        })

        metadata[str(i)] = {
            "measure": row["measure"],
            "predicted_domain": row["predicted_domain"],
            "secondary_domain": row["secondary_domain"],
            "score": row["score"],
            "method": row["method"],
            "confidence": row["confidence"],
        }

    with ANNOTATION_FILE.open("w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)

    with METADATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    predicted_secondary_count = sum(
        bool(row["secondary_domain"])
        for row in sample
    )

    print("=" * 70)
    print("STEP 7 BLIND SAMPLE")
    print("=" * 70)

    print(f"Sample size:      {len(annotation)}")
    print(f"Low-score:        {group_sizes[0]}")
    print(f"Middle-score:     {group_sizes[1]}")
    print(f"High-score:       {group_sizes[2]}")
    print(f"Annotation file:  {ANNOTATION_FILE}")
    print(f"Metadata file:    {METADATA_FILE}")
    print(f"Predicted secondary domains in sample: {predicted_secondary_count}")

    print()
    print("Fill gold_domain for every record.")
    print(
        "gold_secondary_domain is optional and should only be filled "
        "when a second domain is substantively relevant."
    )


# ============================================================
# READ ANNOTATED SAMPLE
# ============================================================

def read_rows():
    annotation = load_json(ANNOTATION_FILE)
    metadata = load_json(METADATA_FILE)

    rows = []

    for row in annotation:
        gold = row.get("gold_domain", "")

        if not isinstance(gold, str):
            raise ValueError(
                f"gold_domain for id {row['id']} must be a string."
            )

        gold = gold.strip().lower()

        if not gold:
            raise ValueError(f"Missing gold_domain for id {row['id']}.")

        gold_secondary = row.get("gold_secondary_domain", "")

        if gold_secondary is None:
            gold_secondary = ""

        if not isinstance(gold_secondary, str):
            raise ValueError(
                f"gold_secondary_domain for id {row['id']} must be a string."
            )

        gold_secondary = gold_secondary.strip().lower()

        meta = metadata.get(str(row["id"]))

        if meta is None:
            raise ValueError(f"No metadata found for id {row['id']}.")

        if meta["measure"] != row["measure"]:
            raise ValueError(f"Measure mismatch for id {row['id']}.")

        predicted = str(meta["predicted_domain"]).strip().lower()

        predicted_secondary = meta.get("secondary_domain")

        if predicted_secondary is None:
            predicted_secondary = ""

        predicted_secondary = str(predicted_secondary).strip().lower()

        rows.append({
            "id": row["id"],
            "measure": row["measure"],
            "gold": gold,
            "gold_secondary": gold_secondary,
            "predicted": predicted,
            "predicted_secondary": predicted_secondary,
            "score": float(meta["score"]),
            "method": meta["method"],
            "confidence": meta.get("confidence"),
        })

    return rows


# ============================================================
# PRIMARY DOMAIN METRICS
# ============================================================

def classification_metrics(rows):
    domains = sorted(
        set(row["gold"] for row in rows)
        | set(row["predicted"] for row in rows)
    )

    confusion = {
        true_domain: {
            predicted_domain: 0
            for predicted_domain in domains
        }
        for true_domain in domains
    }

    for row in rows:
        confusion[row["gold"]][row["predicted"]] += 1

    total = len(rows)
    correct = sum(confusion[domain][domain] for domain in domains)
    accuracy = correct / total if total else 0.0

    per_domain = {}

    for domain in domains:
        tp = confusion[domain][domain]

        fp = sum(
            confusion[true_domain][domain]
            for true_domain in domains
            if true_domain != domain
        )

        fn = sum(
            confusion[domain][predicted_domain]
            for predicted_domain in domains
            if predicted_domain != domain
        )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        per_domain[domain] = {
            "support": tp + fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    macro_f1 = (
        sum(metrics["f1"] for metrics in per_domain.values()) / len(per_domain)
        if per_domain
        else 0.0
    )

    return {
        "n": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_domain": per_domain,
        "confusion_matrix": confusion,
    }


# ============================================================
# SCORE ANALYSIS
# ============================================================

def score_analysis(rows):
    buckets = [
        ("<0.50", 0.00, 0.50),
        ("0.50-0.60", 0.50, 0.60),
        ("0.60-0.70", 0.60, 0.70),
        ("0.70-0.80", 0.70, 0.80),
        ("0.80-0.90", 0.80, 0.90),
        ("0.90+", 0.90, 1.01),
    ]

    result = {}

    for name, low, high in buckets:
        subset = [
            row
            for row in rows
            if low <= row["score"] < high
        ]

        if not subset:
            result[name] = {
                "n": 0,
                "correct": 0,
                "incorrect": 0,
                "accuracy": None,
            }
            continue

        correct = sum(
            row["gold"] == row["predicted"]
            for row in subset
        )

        result[name] = {
            "n": len(subset),
            "correct": correct,
            "incorrect": len(subset) - correct,
            "accuracy": round(correct / len(subset), 4),
        }

    return result


# ============================================================
# METHOD ANALYSIS
# ============================================================

def method_analysis(rows):
    grouped = {}

    for row in rows:
        grouped.setdefault(row["method"], []).append(row)

    result = {}

    for method, subset in grouped.items():
        correct = sum(
            row["gold"] == row["predicted"]
            for row in subset
        )

        result[method] = {
            "n": len(subset),
            "correct": correct,
            "incorrect": len(subset) - correct,
            "accuracy": round(correct / len(subset), 4),
        }

    return result


# ============================================================
# PRIMARY ERRORS
# ============================================================

def primary_errors(rows):
    errors = []

    for row in rows:
        if row["gold"] == row["predicted"]:
            continue

        errors.append({
            "id": row["id"],
            "measure": row["measure"],
            "gold_domain": row["gold"],
            "predicted_domain": row["predicted"],
            "score": row["score"],
            "method": row["method"],
        })

    errors.sort(key=lambda x: x["score"], reverse=True)

    return errors


# ============================================================
# SECONDARY DOMAIN DIAGNOSTIC ANALYSIS
# ============================================================

def secondary_analysis(rows):
    gold_rows = [
        row
        for row in rows
        if row["gold_secondary"]
    ]

    predicted_rows = [
        row
        for row in rows
        if row["predicted_secondary"]
    ]

    exact_matches = [
        row
        for row in rows
        if (
            row["gold_secondary"]
            and row["predicted_secondary"]
            and row["gold_secondary"] == row["predicted_secondary"]
        )
    ]

    recovered_primary = [
        row
        for row in predicted_rows
        if (
            row["predicted_secondary"] == row["gold"]
            and row["predicted"] != row["gold"]
        )
    ]

    predicted_cases = []

    for row in predicted_rows:
        predicted_cases.append({
            "id": row["id"],
            "measure": row["measure"],
            "gold_domain": row["gold"],
            "predicted_domain": row["predicted"],
            "gold_secondary_domain": row["gold_secondary"],
            "predicted_secondary_domain": row["predicted_secondary"],
            "primary_correct": row["gold"] == row["predicted"],
            "secondary_exact_match": (
                bool(row["gold_secondary"])
                and row["gold_secondary"] == row["predicted_secondary"]
            ),
            "gold_primary_recovered_as_secondary": (
                row["predicted_secondary"] == row["gold"]
                and row["predicted"] != row["gold"]
            ),
            "method": row["method"],
            "score": row["score"],
        })

    total = len(rows)

    return {
        "n": total,
        "gold_secondary_count": len(gold_rows),
        "gold_secondary_rate": round(
            len(gold_rows) / total,
            4,
        ) if total else 0.0,
        "predicted_secondary_count": len(predicted_rows),
        "predicted_secondary_rate": round(
            len(predicted_rows) / total,
            4,
        ) if total else 0.0,
        "exact_secondary_matches": len(exact_matches),
        "gold_primary_recovered_as_secondary_count": len(recovered_primary),
        "gold_primary_recovered_as_secondary_rate": round(
            len(recovered_primary) / len(predicted_rows),
            4,
        ) if predicted_rows else 0.0,
        "predicted_secondary_cases": predicted_cases,
        "note": (
            "Secondary-domain assignment is treated as optional diagnostic "
            "output. Precision, recall and F1 are not used as main evaluation "
            "metrics because secondary annotation is more subjective and the "
            "classifier produces very few secondary predictions."
        ),
    }


# ============================================================
# SCORE
# ============================================================

def score():
    rows = read_rows()

    primary = classification_metrics(rows)

    report = {
        "primary_domain": primary,
        "score_analysis": score_analysis(rows),
        "method_analysis": method_analysis(rows),
        "primary_errors": primary_errors(rows),
        "secondary_domain_analysis": secondary_analysis(rows),
    }

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("STEP 7 EVALUATION")
    print("=" * 70)

    print()
    print("PRIMARY DOMAIN")
    print("-" * 70)

    print(f"N:         {primary['n']}")
    print(f"Correct:   {primary['correct']}")
    print(f"Incorrect: {primary['incorrect']}")
    print(f"Accuracy:  {primary['accuracy']:.1%}")
    print(f"Macro F1:  {primary['macro_f1']:.1%}")

    print()
    print("PER PRIMARY DOMAIN")
    print("-" * 70)

    for domain, metrics in primary["per_domain"].items():
        print(
            f"{domain:30s} "
            f"n={metrics['support']:3d} "
            f"P={metrics['precision']:.1%} "
            f"R={metrics['recall']:.1%} "
            f"F1={metrics['f1']:.1%}"
        )

    print()
    print("SCORE VS ACCURACY")
    print("-" * 70)

    for bucket, result in report["score_analysis"].items():
        if result["accuracy"] is None:
            print(f"{bucket:12s} n=0")
        else:
            print(
                f"{bucket:12s} "
                f"n={result['n']:3d} "
                f"accuracy={result['accuracy']:.1%}"
            )

    print()
    print("BY METHOD")
    print("-" * 70)

    for method, result in report["method_analysis"].items():
        print(
            f"{method:25s} "
            f"n={result['n']:3d} "
            f"accuracy={result['accuracy']:.1%}"
        )

    secondary = report["secondary_domain_analysis"]

    print()
    print("SECONDARY DOMAIN — DIAGNOSTIC")
    print("-" * 70)

    print(
        f"Gold secondary domains:      "
        f"{secondary['gold_secondary_count']}/{secondary['n']} "
        f"({secondary['gold_secondary_rate']:.1%})"
    )

    print(
        f"Predicted secondary domains: "
        f"{secondary['predicted_secondary_count']}/{secondary['n']} "
        f"({secondary['predicted_secondary_rate']:.1%})"
    )

    print(
        f"Exact secondary matches:     "
        f"{secondary['exact_secondary_matches']}"
    )

    print(
        f"Gold primary recovered as predicted secondary: "
        f"{secondary['gold_primary_recovered_as_secondary_count']}/"
        f"{secondary['predicted_secondary_count']} "
        f"({secondary['gold_primary_recovered_as_secondary_rate']:.1%})"
    )

    if secondary["predicted_secondary_cases"]:
        print()
        print("PREDICTED SECONDARY CASES")
        print("-" * 70)

        for case in secondary["predicted_secondary_cases"]:
            gold_secondary = case["gold_secondary_domain"] or "-"
            predicted_secondary = case["predicted_secondary_domain"] or "-"

            print(
                f"id={case['id']:3d} | "
                f"GOLD: primary={case['gold_domain']}, "
                f"secondary={gold_secondary} | "
                f"PREDICTED: primary={case['predicted_domain']}, "
                f"secondary={predicted_secondary}"
            )

            print(f"       {case['measure']}")

            if case["gold_primary_recovered_as_secondary"]:
                print(
                    "       NOTE: gold primary was recovered "
                    "as predicted secondary"
                )

    print()
    print(f"Report saved to: {REPORT_FILE}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quality evaluation for Step 7.")
    parser.add_argument("mode", choices=["prepare", "score"])
    args = parser.parse_args()

    if args.mode == "prepare":
        prepare()
    else:
        score()