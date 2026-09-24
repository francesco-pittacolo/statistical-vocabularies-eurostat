"""
Quality evaluation for Step 8 semantic relationships.

Usage:
    python evaluate/evaluate_step_8.py prepare
    python evaluate/evaluate_step_8.py check
    python evaluate/evaluate_step_8.py score
    python evaluate/evaluate_step_8.py judge

The Step 8 classifier produces six positive semantic relations.

The manual relationship evaluation focuses on the five specific relations
listed in CORE_RELATIONS. The generic "related" prediction is excluded from
the sampling frame, but manual annotation may still assign "related" or
"unrelated" when a specific predicted relation is not justified.
"""

import argparse
import json
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_OUTPUT_DIR = ROOT_DIR / "results" / "step_outputs" / "7605" / "step8"
EVALUATION_DIR = ROOT_DIR / "results" / "evaluation" / "step8"

RELATIONSHIPS_FILE = DATA_OUTPUT_DIR / "8_semantic_relationships_7605_results.json"


# ============================================================
# CONFIGURATION
# ============================================================

RELATIONS = {
    "same_concept",
    "synonym",
    "subtype",
    "complementary",
    "different_aspects",
    "related",
}

CORE_RELATIONS = {
    "same_concept",
    "synonym",
    "subtype",
    "complementary",
    "different_aspects",
}

GOLD_RELATIONS = RELATIONS | {"unrelated"}

SEED = 42
RELATIONSHIP_SAMPLE_SIZE = 100
JUDGE_SAMPLE_PER_METHOD = 10

ANNOTATION_FILE = EVALUATION_DIR / f"annotations_{RELATIONSHIP_SAMPLE_SIZE}.json"
METADATA_FILE = EVALUATION_DIR / f"metadata_{RELATIONSHIP_SAMPLE_SIZE}.json"
RELATION_RESULT_FILE = EVALUATION_DIR / f"evaluation_report_{RELATIONSHIP_SAMPLE_SIZE}.json"

CONSISTENCY_REPORT_FILE = EVALUATION_DIR / f"consistency_report_{RELATIONSHIP_SAMPLE_SIZE}.json"
JUDGE_RESULT_FILE = EVALUATION_DIR / f"llm_judge_report_{RELATIONSHIP_SAMPLE_SIZE}.json"

STOPWORDS = {
    "the", "of", "and", "or", "for", "in", "on", "to",
    "a", "an", "by", "from", "with", "at", "as", "is",
    "are", "was", "were", "be", "per",
    "de", "la", "le", "des", "du", "di",
    "del", "della", "dei", "degli",
}


# ============================================================
# UTILITIES
# ============================================================

def load_json(path):
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize(text):
    text = str(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text):
    return {
        token
        for token in normalize(text).split()
        if token not in STOPWORDS and len(token) >= 2
    }


def pair_key(a, b):
    return tuple(sorted((str(a), str(b))))


def validate_gold_relation(value, row_id):
    if value not in GOLD_RELATIONS:
        raise ValueError(
            f"Invalid gold_relation '{value}' for id {row_id}. "
            f"Allowed values: {sorted(GOLD_RELATIONS)}"
        )


def validate_predicted_relation(value, row_id):
    if value not in RELATIONS:
        raise ValueError(
            f"Invalid predicted relation '{value}' for id {row_id}. "
            f"Allowed values: {sorted(RELATIONS)}"
        )


def annotation_file_has_ground_truth(path):
    if not path.exists():
        return False

    annotation = load_json(path)

    if not isinstance(annotation, list):
        return False

    return any(
        str(row.get("gold_relation", "") or "").strip()
        for row in annotation
    )


def confirm_overwrite():
    if not annotation_file_has_ground_truth(ANNOTATION_FILE):
        return True

    print()
    print("WARNING: the Step 8 annotation file contains ground-truth labels:")
    print(f"  {ANNOTATION_FILE}")
    print()

    answer = input(
        "Regenerating this evaluation sample will overwrite these annotations. "
        "Continue? [y/N]: "
    ).strip().lower()

    return answer in {"y", "yes"}


def confirm_judge_overwrite():
    if not JUDGE_RESULT_FILE.exists():
        return True

    print()
    print("WARNING: the Step 8 LLM judge report already exists:")
    print(f"  {JUDGE_RESULT_FILE}")
    print()

    answer = input(
        "Running the judge again will overwrite the existing report. "
        "Continue? [y/N]: "
    ).strip().lower()

    return answer in {"y", "yes"}


# ============================================================
# CONSISTENCY CHECKS
# ============================================================

def run_checks():
    relationships = load_json(RELATIONSHIPS_FILE)

    if relationships is None:
        raise FileNotFoundError(RELATIONSHIPS_FILE)

    report = {}

    pair_keys = [
        pair_key(row["measure1"], row["measure2"])
        for row in relationships
    ]

    duplicates = [
        pair
        for pair, count in Counter(pair_keys).items()
        if count > 1
    ]

    report["duplicate_pairs"] = {
        "count": len(duplicates),
        "examples": duplicates[:10],
    }

    invalid_relations = [
        {
            "measure1": row.get("measure1"),
            "measure2": row.get("measure2"),
            "relation": row.get("relation"),
        }
        for row in relationships
        if row.get("relation") not in RELATIONS
    ]

    report["invalid_relation_labels"] = {
        "count": len(invalid_relations),
        "examples": invalid_relations[:10],
    }

    signature_violations = []

    for row in relationships:
        if row.get("relation") not in {"same_concept", "subtype"}:
            continue

        sig1 = set(row.get("exact_signature1", []))
        sig2 = set(row.get("exact_signature2", []))

        if sig1 and sig2 and sig1 != sig2:
            signature_violations.append({
                "measure1": row["measure1"],
                "measure2": row["measure2"],
                "relation": row["relation"],
                "method": row.get("method"),
            })

    report["exact_signature_violations"] = {
        "count": len(signature_violations),
        "examples": signature_violations[:10],
    }

    subtype_violations = []

    for row in relationships:
        if row.get("relation") != "subtype":
            continue

        evidence = row.get("evidence", {})
        contained = evidence.get("contained_measure")
        expanded = evidence.get("expanded_measure")

        if not contained or not expanded:
            continue

        if not tokens(contained).issubset(tokens(expanded)):
            subtype_violations.append({
                "measure1": row["measure1"],
                "measure2": row["measure2"],
                "method": row.get("method"),
            })

    report["subtype_containment_violations"] = {
        "count": len(subtype_violations),
        "examples": subtype_violations[:10],
    }

    relation_counts = Counter(
        row["relation"]
        for row in relationships
        if row.get("relation") in RELATIONS
    )

    total = len(relationships)

    report["relation_balance"] = {
        relation: {
            "count": relation_counts.get(relation, 0),
            "share": round(relation_counts.get(relation, 0) / total, 4) if total else 0.0,
        }
        for relation in sorted(RELATIONS)
    }

    fallback_rows = [
        row
        for row in relationships
        if str(row.get("method", "")).startswith("fallback_")
    ]

    report["fallback_relationships"] = {
        "count": len(fallback_rows),
        "share": round(len(fallback_rows) / total, 4) if total else 0.0,
        "by_method": dict(
            Counter(row.get("method", "unknown") for row in fallback_rows)
        ),
    }

    by_method = defaultdict(list)

    for row in relationships:
        by_method[row.get("method", "unknown")].append(row)

    method_profile = {}

    for method, rows in sorted(by_method.items()):
        confidences = [
            float(row["confidence"])
            for row in rows
            if row.get("confidence") is not None
        ]

        embeddings = [
            row.get("evidence", {}).get("embedding_similarity")
            for row in rows
        ]

        embeddings = [
            float(value)
            for value in embeddings
            if value is not None
        ]

        method_profile[method] = {
            "n": len(rows),
            "confidence_mean": (
                round(statistics.mean(confidences), 4)
                if confidences else None
            ),
            "embedding_similarity_mean": (
                round(statistics.mean(embeddings), 4)
                if embeddings else None
            ),
        }

    report["method_profile"] = method_profile

    save_json(CONSISTENCY_REPORT_FILE, report)

    print("=" * 70)
    print("STEP 8 CONSISTENCY CHECKS")
    print("=" * 70)
    print(f"Relationships: {len(relationships)}")
    print(f"Duplicate pairs: {report['duplicate_pairs']['count']}")
    print(f"Invalid relation labels: {report['invalid_relation_labels']['count']}")
    print(f"Qualifier violations: {report['exact_signature_violations']['count']}")
    print(
        "Subtype containment violations: "
        f"{report['subtype_containment_violations']['count']}"
    )
    print(f"Fallback relationships: {report['fallback_relationships']['count']}")
    print(f"\nSaved: {CONSISTENCY_REPORT_FILE}")


# ============================================================
# PREPARE RELATIONSHIP SAMPLE
# ============================================================

def generate_relationship_sample(relationships):
    """
    Generate a blind sample for relationship-quality evaluation.

    Only relationships predicted as one of the five specific semantic
    relations are eligible.

    The generic predicted relation "related" is excluded from the sample.
    Manual annotation may still assign "related" or "unrelated".

    Sampling is approximately balanced across the five specific relations.
    """

    rng = random.Random(SEED)

    eligible = [
        row
        for row in relationships
        if row.get("relation") in CORE_RELATIONS
    ]

    unique_eligible = {}

    for row in eligible:
        key = pair_key(row["measure1"], row["measure2"])

        if key not in unique_eligible:
            unique_eligible[key] = row

    eligible = sorted(
        unique_eligible.values(),
        key=lambda row: pair_key(row["measure1"], row["measure2"]),
    )

    if len(eligible) < RELATIONSHIP_SAMPLE_SIZE:
        raise ValueError(
            f"Only {len(eligible)} unique specific relationships are available, "
            f"but {RELATIONSHIP_SAMPLE_SIZE} are required."
        )

    by_relation = defaultdict(list)

    for row in eligible:
        by_relation[row["relation"]].append(row)

    selected = []
    selected_keys = set()

    base_target = RELATIONSHIP_SAMPLE_SIZE // len(CORE_RELATIONS)

    for relation in sorted(CORE_RELATIONS):
        rows = by_relation.get(relation, [])

        if not rows:
            continue

        take = min(base_target, len(rows))

        for row in rng.sample(rows, take):
            key = pair_key(row["measure1"], row["measure2"])

            if key in selected_keys:
                continue

            selected_keys.add(key)
            selected.append(row)

    if len(selected) < RELATIONSHIP_SAMPLE_SIZE:
        remaining = [
            row
            for row in eligible
            if pair_key(row["measure1"], row["measure2"]) not in selected_keys
        ]

        needed = RELATIONSHIP_SAMPLE_SIZE - len(selected)

        if len(remaining) < needed:
            raise ValueError(
                f"Not enough remaining specific relationships: "
                f"need {needed}, have {len(remaining)}."
            )

        for row in rng.sample(remaining, needed):
            key = pair_key(row["measure1"], row["measure2"])
            selected_keys.add(key)
            selected.append(row)

    if len(selected) != RELATIONSHIP_SAMPLE_SIZE:
        raise RuntimeError(
            f"Expected {RELATIONSHIP_SAMPLE_SIZE} relationship rows, "
            f"got {len(selected)}."
        )

    rng.shuffle(selected)

    annotation = []
    metadata = {}

    for i, row in enumerate(selected):
        annotation.append({
            "id": i,
            "measure1": row["measure1"],
            "measure2": row["measure2"],
            "gold_relation": "",
        })

        metadata[str(i)] = {
            "measure1": row["measure1"],
            "measure2": row["measure2"],
            "predicted_relation": row["relation"],
            "method": row.get("method", "unknown"),
            "confidence": row.get("confidence"),
            "evidence": row.get("evidence", {}),
        }

    save_json(ANNOTATION_FILE, annotation)
    save_json(METADATA_FILE, metadata)

    predicted_counts = Counter(row["relation"] for row in selected)

    print(f"Relationship sample: {ANNOTATION_FILE} ({len(annotation)} rows)")

    print()
    print("PREDICTED SPECIFIC RELATIONS")
    print("-" * 70)

    for relation in sorted(CORE_RELATIONS):
        print(f"{relation:25s} {predicted_counts.get(relation, 0):3d}")

    print()
    print("Allowed gold labels:")

    for relation in sorted(GOLD_RELATIONS):
        print(f"  {relation}")


# ============================================================
# PREPARE
# ============================================================

def prepare():
    if not confirm_overwrite():
        print("Step 8 evaluation sample preserved.")
        return

    relationships = load_json(RELATIONSHIPS_FILE)

    if relationships is None:
        raise FileNotFoundError(RELATIONSHIPS_FILE)

    print("=" * 70)
    print("STEP 8 PREPARE RELATIONSHIP SAMPLE")
    print("=" * 70)
    print()

    generate_relationship_sample(relationships)


# ============================================================
# READ RELATIONSHIP ANNOTATIONS
# ============================================================

def read_relationship_rows():
    annotation = load_json(ANNOTATION_FILE)
    metadata = load_json(METADATA_FILE)

    if annotation is None:
        raise FileNotFoundError(ANNOTATION_FILE)

    if metadata is None:
        raise FileNotFoundError(METADATA_FILE)

    rows = []

    for row in annotation:
        gold = row.get("gold_relation", "")

        if not isinstance(gold, str):
            raise ValueError(
                f"gold_relation for id {row['id']} must be a string."
            )

        gold = gold.strip()

        if not gold:
            raise ValueError(f"Missing gold_relation for id {row['id']}.")

        validate_gold_relation(gold, row["id"])

        meta = metadata.get(str(row["id"]))

        if meta is None:
            raise ValueError(f"No metadata found for id {row['id']}.")

        if (
            meta["measure1"] != row["measure1"]
            or meta["measure2"] != row["measure2"]
        ):
            raise ValueError(f"Measure mismatch for id {row['id']}.")

        predicted = meta["predicted_relation"]
        validate_predicted_relation(predicted, row["id"])

        rows.append({
            "id": row["id"],
            "measure1": row["measure1"],
            "measure2": row["measure2"],
            "gold": gold,
            "predicted": predicted,
            "method": meta.get("method", "unknown"),
            "confidence": meta.get("confidence"),
        })

    return rows


# ============================================================
# SCORE
# ============================================================

def score():
    """
    Evaluate relationship classification quality on a manually annotated
    sample of predicted Step 8 relationships.

    The sampling frame contains predictions from the five specific relation
    types. Gold annotation may also assign "related" or "unrelated".

    Precision, recall, F1 and macro-F1 are computed over the five specific
    relation types.
    """

    rows = read_relationship_rows()
    total = len(rows)

    unexpected_predictions = [
        row
        for row in rows
        if row["predicted"] not in CORE_RELATIONS
    ]

    if unexpected_predictions:
        raise ValueError(
            "Relationship-quality metadata contains predictions outside "
            f"CORE_RELATIONS. First invalid id: "
            f"{unexpected_predictions[0]['id']}."
        )

    valid_relationships = sum(
        row["gold"] != "unrelated"
        for row in rows
    )

    core_gold = sum(
        row["gold"] in CORE_RELATIONS
        for row in rows
    )

    generic_related_gold = sum(
        row["gold"] == "related"
        for row in rows
    )

    unrelated_gold = sum(
        row["gold"] == "unrelated"
        for row in rows
    )

    exact_correct = sum(
        row["gold"] == row["predicted"]
        for row in rows
    )

    valid_relationship_rate = valid_relationships / total if total else 0.0
    exact_accuracy = exact_correct / total if total else 0.0

    confusion = {
        gold: {
            predicted: 0
            for predicted in sorted(CORE_RELATIONS)
        }
        for gold in sorted(GOLD_RELATIONS)
    }

    for row in rows:
        confusion[row["gold"]][row["predicted"]] += 1

    per_relation = {}

    for relation in sorted(CORE_RELATIONS):
        tp = sum(
            row["gold"] == relation and row["predicted"] == relation
            for row in rows
        )

        fp = sum(
            row["gold"] != relation and row["predicted"] == relation
            for row in rows
        )

        fn = sum(
            row["gold"] == relation and row["predicted"] != relation
            for row in rows
        )

        support = sum(
            row["gold"] == relation
            for row in rows
        )

        predicted_support = sum(
            row["predicted"] == relation
            for row in rows
        )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        per_relation[relation] = {
            "support": support,
            "predicted_support": predicted_support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    supported_f1 = [
        metrics["f1"]
        for metrics in per_relation.values()
        if metrics["support"] > 0
    ]

    macro_f1 = (
        sum(supported_f1) / len(supported_f1)
        if supported_f1
        else 0.0
    )

    by_method_rows = defaultdict(list)

    for row in rows:
        by_method_rows[row["method"]].append(row)

    by_method = {}

    for method, subset in sorted(by_method_rows.items()):
        n = len(subset)

        valid = sum(
            row["gold"] != "unrelated"
            for row in subset
        )

        exact = sum(
            row["gold"] == row["predicted"]
            for row in subset
        )

        by_method[method] = {
            "n": n,
            "valid_relationship_rate": round(valid / n, 4),
            "exact_relation_accuracy": round(exact / n, 4),
        }

    by_predicted_relation = {}

    for relation in sorted(CORE_RELATIONS):
        subset = [
            row
            for row in rows
            if row["predicted"] == relation
        ]

        if not subset:
            continue

        n = len(subset)

        valid = sum(
            row["gold"] != "unrelated"
            for row in subset
        )

        exact = sum(
            row["gold"] == relation
            for row in subset
        )

        by_predicted_relation[relation] = {
            "n": n,
            "valid_relationship_rate": round(valid / n, 4),
            "exact_relation_accuracy": round(exact / n, 4),
        }

    errors = [
        {
            "id": row["id"],
            "measure1": row["measure1"],
            "measure2": row["measure2"],
            "gold_relation": row["gold"],
            "predicted_relation": row["predicted"],
            "method": row["method"],
        }
        for row in rows
        if row["gold"] != row["predicted"]
    ]

    report = {
        "n": total,
        "sampling_frame": sorted(CORE_RELATIONS),
        "valid_relationships": valid_relationships,
        "valid_relationship_rate": round(valid_relationship_rate, 4),
        "gold_specific_relationships": core_gold,
        "gold_generic_related": generic_related_gold,
        "gold_unrelated": unrelated_gold,
        "exact_relation_matches": exact_correct,
        "exact_relation_accuracy": round(exact_accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_relation": per_relation,
        "by_method": by_method,
        "by_predicted_relation": by_predicted_relation,
        "confusion_matrix": confusion,
        "errors": errors,
        "note": (
            "The evaluation sample contains predictions from the five specific "
            "semantic relation types. The generic 'related' prediction is "
            "excluded from sampling, while manual gold labels may still be "
            "'related' or 'unrelated'."
        ),
    }

    save_json(RELATION_RESULT_FILE, report)

    print("=" * 70)
    print("STEP 8 RELATIONSHIP EVALUATION")
    print("=" * 70)

    print(f"Sample size: {total}")
    print(
        f"Semantically valid pairs: {valid_relationships}/{total} "
        f"({valid_relationship_rate:.1%})"
    )
    print(f"Gold specific relations:  {core_gold}")
    print(f"Gold generic related:     {generic_related_gold}")
    print(f"Gold unrelated:           {unrelated_gold}")
    print(
        f"Exact relation matches:   {exact_correct}/{total} "
        f"({exact_accuracy:.1%})"
    )
    print(f"Macro F1:                 {macro_f1:.1%}")

    print()
    print("PER RELATION")
    print("-" * 70)

    for relation, metrics in per_relation.items():
        print(
            f"{relation:25s} "
            f"gold={metrics['support']:3d} "
            f"pred={metrics['predicted_support']:3d} "
            f"P={metrics['precision']:.1%} "
            f"R={metrics['recall']:.1%} "
            f"F1={metrics['f1']:.1%}"
        )

    print()
    print("BY CLASSIFICATION METHOD")
    print("-" * 70)

    for method, metrics in by_method.items():
        print(
            f"{method:40s} "
            f"n={metrics['n']:3d} "
            f"valid={metrics['valid_relationship_rate']:.1%} "
            f"exact={metrics['exact_relation_accuracy']:.1%}"
        )

    print(f"\nSaved: {RELATION_RESULT_FILE}")


# ============================================================
# OPTIONAL LLM JUDGE
# ============================================================

def run_judge():
    """
    Optional LLM-based proxy consistency check.

    This is diagnostic only and is not treated as manual ground truth.
    """

    if not confirm_judge_overwrite():
        print("Existing Step 8 LLM judge report preserved.")
        return

    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv(ROOT_DIR / ".env")

    api_key = os.getenv("API_KEY")

    if not api_key:
        print("API_KEY not set in .env.")
        return

    relationships = load_json(RELATIONSHIPS_FILE)

    if relationships is None:
        raise FileNotFoundError(RELATIONSHIPS_FILE)

    rule_based = [
        row
        for row in relationships
        if not row.get("llm", False)
    ]

    by_method = defaultdict(list)

    for row in rule_based:
        by_method[row.get("method", "unknown")].append(row)

    rng = random.Random(SEED)
    sample = []
    sampled_by_method = {}

    for method, rows in sorted(by_method.items()):
        rows = sorted(
            rows,
            key=lambda row: pair_key(row["measure1"], row["measure2"]),
        )

        selected = rng.sample(
            rows,
            min(JUDGE_SAMPLE_PER_METHOD, len(rows)),
        )

        sample.extend(selected)
        sampled_by_method[method] = {
            "sampled": len(selected),
            "available": len(rows),
        }

    client = Groq(api_key=api_key, max_retries=0)

    print("=" * 70)
    print("STEP 8 LLM JUDGE")
    print("=" * 70)
    print(f"Rule-based relationships: {len(rule_based)}")
    print(f"Classification methods:   {len(by_method)}")
    print(f"Sampled relationships:    {len(sample)}")
    print()
    print("SAMPLE BY METHOD")
    print("-" * 70)

    for method, values in sampled_by_method.items():
        print(
            f"{method:45s} "
            f"sampled={values['sampled']:2d} "
            f"available={values['available']:4d}"
        )

    print()
    print("RUNNING LLM JUDGE")
    print("-" * 70)

    agree = 0
    disagree = 0
    errors = 0
    results = []

    progress_width = 120

    def short_text(text, max_length=26):
        text = " ".join(str(text).split())

        if len(text) <= max_length:
            return text

        return text[:max_length - 3] + "..."

    def print_progress(message):
        if len(message) > progress_width:
            message = message[:progress_width - 3] + "..."

        print(
            f"\r{message:<{progress_width}}",
            end="",
            flush=True,
        )

    total = len(sample)

    for index, row in enumerate(sample, start=1):
        method = row.get("method", "unknown")
        relation = row["relation"]

        prompt = (
            "You are independently reviewing an automatic classification "
            "of the semantic relationship between two statistical measures.\n\n"
            f'Measure A: "{row["measure1"]}"\n'
            f'Measure B: "{row["measure2"]}"\n'
            f'Assigned relation: "{row["relation"]}"\n\n'
            "Allowed gold labels:\n"
            "- same_concept\n"
            "- synonym\n"
            "- subtype\n"
            "- complementary\n"
            "- different_aspects\n"
            "- related\n"
            "- unrelated\n\n"
            'Return JSON only: {"agree": true/false, '
            '"suggested_relation": "..."}'
        )

        response = None
        last_error = None

        for attempt in range(6):
            try:
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                break

            except Exception as exc:
                last_error = str(exc)

                if "429" not in str(exc) or attempt >= 5:
                    break

                delay = min(2 ** attempt * 2, 60)

                print_progress(
                    f"[{index:02d}/{total}] "
                    f'M1="{short_text(row["measure1"])}" | '
                    f'M2="{short_text(row["measure2"])}" | '
                    f"relation={relation} | retry={delay}s"
                )

                time.sleep(delay)

        if response is None:
            errors += 1

            results.append({
                "measure1": row["measure1"],
                "measure2": row["measure2"],
                "relation": row["relation"],
                "method": method,
                "agree": None,
                "suggested_relation": "",
                "error": last_error,
            })

            print_progress(
                f"[{index:02d}/{total}] "
                f'M1="{short_text(row["measure1"])}" | '
                f'M2="{short_text(row["measure2"])}" | '
                f"relation={relation} | agree=None"
            )

            continue

        try:
            parsed = json.loads(response.choices[0].message.content)

            agrees = bool(parsed.get("agree"))
            suggested = str(parsed.get("suggested_relation", "")).strip()

            if suggested:
                validate_gold_relation(suggested, "judge")

            if agrees:
                agree += 1
            else:
                disagree += 1

            results.append({
                "measure1": row["measure1"],
                "measure2": row["measure2"],
                "relation": row["relation"],
                "method": method,
                "agree": agrees,
                "suggested_relation": suggested,
            })

            result_text = f"agree={agrees}"

            if not agrees and suggested:
                result_text += f" -> {suggested}"

        except Exception as exc:
            errors += 1
            result_text = "agree=None"

            results.append({
                "measure1": row["measure1"],
                "measure2": row["measure2"],
                "relation": row["relation"],
                "method": method,
                "agree": None,
                "suggested_relation": "",
                "error": str(exc),
            })

        print_progress(
            f"[{index:02d}/{total}] "
            f'M1="{short_text(row["measure1"])}" | '
            f'M2="{short_text(row["measure2"])}" | '
            f"relation={relation} | {result_text}"
        )

    print()

    judged = agree + disagree
    agreement = agree / judged if judged else None

    report = {
        "n_sampled": len(sample),
        "n_judged": judged,
        "agree": agree,
        "disagree": disagree,
        "errors": errors,
        "agreement": round(agreement, 4) if agreement is not None else None,
        "results": results,
        "note": (
            "LLM judge results are diagnostic only and are not treated "
            "as human ground truth."
        ),
    }

    save_json(JUDGE_RESULT_FILE, report)

    print()
    print("SUMMARY")
    print("-" * 70)
    print(f"Sampled: {len(sample)}")
    print(f"Judged: {judged}")
    print(f"Agree: {agree}")
    print(f"Disagree: {disagree}")
    print(f"Errors: {errors}")

    if agreement is None:
        print("Agreement: N/A")
    else:
        print(f"Agreement: {agreement:.1%}")

    print(f"\nSaved: {JUDGE_RESULT_FILE}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quality evaluation for Step 8.")
    parser.add_argument("mode", choices=["prepare", "check", "score", "judge"])
    args = parser.parse_args()

    if args.mode == "prepare":
        prepare()
    elif args.mode == "check":
        run_checks()
    elif args.mode == "score":
        score()
    elif args.mode == "judge":
        run_judge()