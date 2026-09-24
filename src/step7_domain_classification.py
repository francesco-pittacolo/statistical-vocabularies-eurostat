import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config import DATASET, OUTPUT_DIR, REFERENCE_DIR, CACHE_DIR

INPUT_FILE = OUTPUT_DIR / f"6_V_classified_{DATASET}_results.json"

KEYWORDS_FILE = REFERENCE_DIR / "domains_keywords.json"
DESCRIPTION_FILE = REFERENCE_DIR / "domain_descriptions.json"

STEP7_OUTPUT_DIR = OUTPUT_DIR / "step7"

OUTPUT_DOMAINS = STEP7_OUTPUT_DIR / f"7_M_domains_{DATASET}_results.json"
OUTPUT_DETAILS = STEP7_OUTPUT_DIR / f"7_M_classification_details_{DATASET}_results.json"
OUTPUT_UNRESOLVED = STEP7_OUTPUT_DIR / f"7_M_unresolved_{DATASET}_results.json"
FINAL_OUTPUT = STEP7_OUTPUT_DIR / f"7_M_final_classification_{DATASET}_results.json"

GROQ_CACHE_FILE = CACHE_DIR / "7_M_groq_cache.json"

MODEL_NAME = "all-MiniLM-L6-v2"


# -------------------------
# Blended score weights
# -------------------------

DICT_WEIGHT = 0.4
EMB_WEIGHT = 0.6


# -------------------------
# Confidence thresholds
# -------------------------

DICT_CONFIDENT_THRESHOLD = 0.55
EMB_CONFIDENT_THRESHOLD = 0.45
DICTIONARY_MARGIN_THRESHOLD = 0.15
EMBEDDING_MARGIN_THRESHOLD = 0.08
SECONDARY_MARGIN = 0.15
SECONDARY_MIN_SCORE = 0.35
MIN_DICT_SCORE_TO_COUNT = 3

CALIBRATION_PERCENTILE = 0.95
MIN_HITS_FOR_CALIBRATION = 15


# -------------------------
# Groq
# -------------------------

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_TEMPERATURE = 0
WAITING_TIME = 0.5
MAX_RETRIES = 3


# -------------------------
# Global data
# -------------------------

KEYWORDS = {}
DESCRIPTIONS = {}

measures = []
domains = []

model = None
domain_embeddings = None

KEYWORD_PATTERNS = {}
MAX_DICT_SCORE = {}
DICT_CALIBRATION = {}

client = None


# -------------------------
# Helper functions
# -------------------------

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def raw_dictionary_score(domain, text_padded):
    return sum(
        weight
        for pattern, _, weight in KEYWORD_PATTERNS[domain]
        if pattern.search(text_padded)
    )


def compute_calibration():
    hits = {domain: [] for domain in domains}

    for measure in measures:
        padded = " " + measure.lower() + " "

        for domain in domains:
            score = raw_dictionary_score(domain, padded)

            if score > 0:
                hits[domain].append(score)

    calibration = {}

    for domain, scores in hits.items():
        if len(scores) >= MIN_HITS_FOR_CALIBRATION:
            scores_sorted = sorted(scores)
            idx = max(0, int(CALIBRATION_PERCENTILE * len(scores_sorted)) - 1)
            calibration[domain] = scores_sorted[idx]
        else:
            calibration[domain] = MAX_DICT_SCORE[domain]

        if calibration[domain] <= 0:
            calibration[domain] = 1

    return calibration


def build_embedding_text(measure):
    return measure


def dictionary_scores(text):
    padded = " " + text.lower() + " "
    raw = {}

    for domain, patterns in KEYWORD_PATTERNS.items():
        score = 0
        matches = []

        for pattern, word, weight in patterns:
            if pattern.search(padded):
                score += weight
                matches.append(word)

        raw[domain] = (score, matches)

    normalized = {}

    for domain, (score, matches) in raw.items():
        if score < MIN_DICT_SCORE_TO_COUNT:
            normalized[domain] = (0.0, matches)
            continue

        cap = DICT_CALIBRATION.get(domain, 1) or 1
        normalized[domain] = (min(score / cap, 1.0), matches)

    return normalized


def embedding_scores(text):
    embedding = model.encode([text], normalize_embeddings=True)
    similarities = cosine_similarity(embedding, domain_embeddings)[0]

    return {
        domain: float(max(0.0, min(1.0, similarities[i])))
        for i, domain in enumerate(domains)
    }


# -------------------------
# Blended classifier
# -------------------------

def blended_classify(measure):
    text = build_embedding_text(measure)

    dscores = dictionary_scores(measure)
    escores = embedding_scores(text)

    dictionary_ranked = sorted(
        ((domain, dscores[domain][0]) for domain in domains),
        key=lambda x: x[1],
        reverse=True
    )

    best_dictionary_domain, dictionary_best_score = dictionary_ranked[0]
    second_dictionary_domain, second_dictionary_score = dictionary_ranked[1]
    dictionary_margin = dictionary_best_score - second_dictionary_score
    dictionary_matches = dscores[best_dictionary_domain][1]

    embedding_ranked = sorted(
        ((domain, escores[domain]) for domain in domains),
        key=lambda x: x[1],
        reverse=True
    )

    best_embedding_domain, embedding_best_score = embedding_ranked[0]
    second_embedding_domain, second_embedding_score = embedding_ranked[1]
    embedding_margin = embedding_best_score - second_embedding_score

    blended = {
        domain: DICT_WEIGHT * dscores[domain][0] + EMB_WEIGHT * escores[domain]
        for domain in domains
    }

    blended_ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)
    best_blended_domain, best_blended_score = blended_ranked[0]

    signals_agree = best_dictionary_domain == best_embedding_domain

    dictionary_is_strong = dictionary_best_score >= DICT_CONFIDENT_THRESHOLD
    embedding_is_strong = embedding_best_score >= EMB_CONFIDENT_THRESHOLD

    dictionary_clear = dictionary_is_strong and dictionary_margin >= DICTIONARY_MARGIN_THRESHOLD
    embedding_clear = embedding_is_strong and embedding_margin >= EMBEDDING_MARGIN_THRESHOLD

    needs_llm = False
    decision_reason = ""
    final_domain = best_blended_domain

    if signals_agree:
        final_domain = best_dictionary_domain

        if dictionary_clear or embedding_clear:
            needs_llm = False
            decision_reason = "dictionary_embedding_agree"
        else:
            needs_llm = True
            decision_reason = "agreement_but_signals_not_clear"

    else:
        if dictionary_clear and embedding_clear:
            needs_llm = True
            decision_reason = "clear_dictionary_clear_embedding_conflict"

        elif embedding_clear:
            final_domain = best_embedding_domain
            needs_llm = False
            decision_reason = "clear_embedding_dictionary_not_clear"

        elif dictionary_clear:
            final_domain = best_dictionary_domain
            needs_llm = False
            decision_reason = "clear_dictionary_embedding_not_clear"

        else:
            needs_llm = True
            decision_reason = "neither_signal_clear"

    final_dictionary_score = dscores[final_domain][0]
    final_embedding_score = escores[final_domain]
    final_blended_score = DICT_WEIGHT * final_dictionary_score + EMB_WEIGHT * final_embedding_score

    secondary_domain = None

    for candidate_domain, candidate_score in blended_ranked:
        if candidate_domain == final_domain:
            continue

        if candidate_score >= SECONDARY_MIN_SCORE and final_blended_score - candidate_score <= SECONDARY_MARGIN:
            secondary_domain = candidate_domain
            break

    return {
        "domain": final_domain,
        "secondary_domain": secondary_domain,
        "confidence": round(final_blended_score, 3),
        "dictionary_domain": best_dictionary_domain,
        "dictionary_score": round(dictionary_best_score, 3),
        "dictionary_matches": dictionary_matches,
        "dictionary_second_domain": second_dictionary_domain,
        "dictionary_second_score": round(second_dictionary_score, 3),
        "dictionary_margin": round(dictionary_margin, 3),
        "embedding_domain": best_embedding_domain,
        "embedding_score": round(embedding_best_score, 3),
        "embedding_second_domain": second_embedding_domain,
        "embedding_second_score": round(second_embedding_score, 3),
        "embedding_margin": round(embedding_margin, 3),
        "blended_domain": best_blended_domain,
        "blended_score": round(best_blended_score, 3),
        "final_dictionary_score": round(final_dictionary_score, 3),
        "final_embedding_score": round(final_embedding_score, 3),
        "final_blended_score": round(final_blended_score, 3),
        "signals_agree": signals_agree,
        "dictionary_strong": dictionary_is_strong,
        "embedding_strong": embedding_is_strong,
        "dictionary_clear": dictionary_clear,
        "embedding_clear": embedding_clear,
        "decision_reason": decision_reason,
        "needs_llm": needs_llm,
    }


# -------------------------
# Groq helpers
# -------------------------

def generate_request_id(measure):
    key = measure.strip()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def classify_error(message):
    msg = str(message).lower()

    if "daily limit" in msg or "per day" in msg or "daily" in msg:
        return "daily_limit"

    if "rate limit" in msg or "tokens per minute" in msg or "tpm" in msg or "too many requests" in msg:
        return "retry"

    return "unknown"


def parse_llm_json(text):
    if text is None:
        return None

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return None

    return None


def build_groq_prompt(measure):
    domain_list = ", ".join(domains)

    return f"""Classify this statistical measure into exactly one domain.

Allowed domains:
{domain_list}

Measure:
{measure}

Rules:
1. Select exactly one domain from the allowed list.
2. Use the semantic meaning of the measure.
3. Do not invent a new domain.
4. Do not classify based only on one isolated word.
5. Prefer the most specific meaningful domain.
6. Return a confidence between 0 and 1.
7. Return ONLY valid JSON.

Return:
{{
    "domain": "selected domain",
    "confidence": 0.0,
    "explanation": "short explanation"
}}
"""


def get_groq_client():
    global client

    if client is not None:
        return client

    load_dotenv(ROOT_DIR / ".env")

    api_key = os.getenv("API_KEY")

    if not api_key:
        raise ValueError("API_KEY is not set in the .env file")

    client = Groq(api_key=api_key)

    return client


def call_groq(measure, model_name=GROQ_MODEL, temperature=GROQ_TEMPERATURE):
    groq_client = get_groq_client()
    prompt = build_groq_prompt(measure)

    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"}
    )

    usage = getattr(response, "usage", None)
    tokens = None

    if usage:
        tokens = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

    content = response.choices[0].message.content.strip()

    return content, tokens


def validate_groq_result(result):
    if not isinstance(result, dict):
        return False

    if "domain" not in result or "confidence" not in result:
        return False

    if result["domain"] not in domains:
        return False

    try:
        confidence = float(result["confidence"])
    except Exception:
        return False

    if not 0 <= confidence <= 1:
        return False

    return True


def groq_classifier(measure):
    for attempt in range(MAX_RETRIES):
        try:
            raw_output, tokens = call_groq(measure)
            result = parse_llm_json(raw_output)

            if not validate_groq_result(result):
                print(f"LLM invalid JSON/result: {measure}")

                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue

                return {"status": "json_error", "result": None, "tokens": tokens}

            return {"status": "ok", "result": result, "tokens": tokens}

        except Exception as e:
            action = classify_error(str(e))

            if action == "daily_limit":
                return {
                    "status": "daily_limit",
                    "result": None,
                    "tokens": None,
                    "error": str(e),
                }

            if action == "retry":
                wait = min(60, 2 ** attempt)
                print(f"LLM retry {attempt + 1}/{MAX_RETRIES} after {wait}s: {measure}")
                time.sleep(wait)
                continue

            return {
                "status": "error",
                "result": None,
                "tokens": None,
                "error": str(e),
            }

    return {"status": "error", "result": None, "tokens": None}


# -------------------------
# Cache
# -------------------------

def load_groq_cache():
    if not GROQ_CACHE_FILE.exists():
        return {}

    with open(GROQ_CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_groq_cache(groq_cache):
    with open(GROQ_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(groq_cache, f, indent=4, ensure_ascii=False)


# -------------------------
# Main
# -------------------------

def main():
    global KEYWORDS
    global DESCRIPTIONS
    global measures
    global domains
    global model
    global domain_embeddings
    global KEYWORD_PATTERNS
    global MAX_DICT_SCORE
    global DICT_CALIBRATION

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Step 6 output not found: {INPUT_FILE}\nRun Step 6 before Step 7.")

    if not KEYWORDS_FILE.exists():
        raise FileNotFoundError(f"Domain keywords file not found: {KEYWORDS_FILE}")

    if not DESCRIPTION_FILE.exists():
        raise FileNotFoundError(f"Domain descriptions file not found: {DESCRIPTION_FILE}")

    STEP7_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load data
    # -------------------------

    print("Loading data...")

    V = load_json(INPUT_FILE)
    KEYWORDS = load_json(KEYWORDS_FILE)
    DESCRIPTIONS = load_json(DESCRIPTION_FILE)

    M = V["M"]

    measures = list(M.keys())
    domains = list(DESCRIPTIONS.keys())

    print(f"Dataset: {DATASET}")
    print(f"Loaded {len(measures)} measures")
    print(f"Loaded {len(domains)} domains")

    # -------------------------
    # Embedding model
    # -------------------------

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    domain_texts = [DESCRIPTIONS[domain] for domain in domains]
    domain_embeddings = model.encode(domain_texts, normalize_embeddings=True)

    # -------------------------
    # Keyword patterns
    # -------------------------

    KEYWORD_PATTERNS = {
        domain: [
            (re.compile(r"(?<![a-z])" + re.escape(word.lower()) + r"(?![a-z])"), word, weight)
            for word, weight in keywords.items()
        ]
        for domain, keywords in KEYWORDS.items()
    }

    MAX_DICT_SCORE = {
        domain: sum(weight for _, _, weight in patterns)
        for domain, patterns in KEYWORD_PATTERNS.items()
    }

    # -------------------------
    # Dictionary calibration
    # -------------------------

    print("Calibrating dictionary normalization against corpus statistics...")

    DICT_CALIBRATION = compute_calibration()

    for domain in domains:
        print(
            f"  {domain:20s} "
            f"budget={MAX_DICT_SCORE[domain]:3d}  "
            f"calibrated_norm={DICT_CALIBRATION[domain]:3d}"
        )

    # -------------------------
    # Cache
    # -------------------------

    groq_cache = load_groq_cache()

    print(f"Loaded LLM cache: {len(groq_cache)} entries")

    # -------------------------
    # Containers
    # -------------------------

    domain_results = {domain: [] for domain in domains}
    details = {}
    groq_candidates = []

    statistics = {
        "dictionary_or_embedding": 0,
        "groq_cached": 0,
        "groq_new": 0,
        "groq_failed": 0,
    }

    decision_reasons = {}

    # -------------------------
    # Phase 1
    # -------------------------

    print()
    print(f"PHASE 1 | Blended dictionary + embedding classification | {len(measures)} measures")

    for i, measure in enumerate(measures, 1):
        result = blended_classify(measure)

        reason = result["decision_reason"]
        decision_reasons[reason] = decision_reasons.get(reason, 0) + 1

        shared_fields = {
            "secondary_domain": result["secondary_domain"],
            "confidence": result["confidence"],
            "dictionary_domain": result["dictionary_domain"],
            "dictionary_score": result["dictionary_score"],
            "dictionary_matches": result["dictionary_matches"],
            "dictionary_second_domain": result["dictionary_second_domain"],
            "dictionary_second_score": result["dictionary_second_score"],
            "dictionary_margin": result["dictionary_margin"],
            "embedding_domain": result["embedding_domain"],
            "embedding_score": result["embedding_score"],
            "embedding_second_domain": result["embedding_second_domain"],
            "embedding_second_score": result["embedding_second_score"],
            "embedding_margin": result["embedding_margin"],
            "blended_domain": result["blended_domain"],
            "blended_score": result["blended_score"],
            "final_dictionary_score": result["final_dictionary_score"],
            "final_embedding_score": result["final_embedding_score"],
            "final_blended_score": result["final_blended_score"],
            "signals_agree": result["signals_agree"],
            "dictionary_strong": result["dictionary_strong"],
            "embedding_strong": result["embedding_strong"],
            "dictionary_clear": result["dictionary_clear"],
            "embedding_clear": result["embedding_clear"],
            "decision_reason": result["decision_reason"],
        }

        if not result["needs_llm"]:
            domain_results[result["domain"]].append(measure)

            details[measure] = {
                "domain": result["domain"],
                "method": "dictionary+embedding",
                **shared_fields,
                "needs_llm": False,
            }

            statistics["dictionary_or_embedding"] += 1

        else:
            groq_candidates.append(measure)

            details[measure] = {
                "domain": None,
                "method": "llm_required",
                **shared_fields,
                "needs_llm": True,
            }

        if i % 250 == 0 or i == len(measures):
            print(
                f"Progress | {i}/{len(measures)} | "
                f"classified={statistics['dictionary_or_embedding']} | "
                f"remaining={len(groq_candidates)}"
            )

    print(
        f"PHASE 1 DONE | classified={statistics['dictionary_or_embedding']} | "
        f"LLM candidates={len(groq_candidates)}"
    )

    print()
    print("DECISION REASONS")

    for reason, count in sorted(decision_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason:60s} {count}")

    # -------------------------
    # Phase 2
    # -------------------------

    print()
    print(f"PHASE 2 | LLM preparation | candidates={len(groq_candidates)}")

    pending_groq = []
    cached_groq = []

    for measure in groq_candidates:
        request_id = generate_request_id(measure)

        if request_id in groq_cache:
            cached_groq.append((measure, request_id, groq_cache[request_id]))
        else:
            pending_groq.append((measure, request_id))

    print(
        f"LLM estimate | candidates={len(groq_candidates)} | "
        f"cached={len(cached_groq)} | new_requests={len(pending_groq)}"
    )

    # -------------------------
    # Cached LLM results
    # -------------------------

    for measure, request_id, cached in cached_groq:
        result = cached.get("result")

        if not validate_groq_result(result):
            pending_groq.append((measure, request_id))
            continue

        domain = result["domain"]
        confidence = float(result["confidence"])

        domain_results[domain].append(measure)

        details[measure]["domain"] = domain
        details[measure]["confidence"] = round(confidence, 3)
        details[measure]["method"] = "llm"
        details[measure]["explanation"] = result.get("explanation", "")
        details[measure]["request_id"] = request_id
        details[measure]["tokens"] = cached.get("tokens")

        statistics["groq_cached"] += 1

    print(f"LLM cache processed | restored={statistics['groq_cached']}")

    # -------------------------
    # New LLM requests
    # -------------------------

    groq_daily_limit_reached = False

    print(f"LLM requests starting | {len(pending_groq)} new requests")

    for i, (measure, request_id) in enumerate(pending_groq, 1):
        print(f"LLM request {i}/{len(pending_groq)} | {measure}", end="\r")

        response = groq_classifier(measure)
        status = response["status"]

        if status == "daily_limit":
            print("\nLLM daily limit reached. Stopping LLM phase.")
            groq_daily_limit_reached = True
            break

        if status != "ok":
            print(f"\nLLM failed | status={status} | measure={measure}")
            print("Error:", response.get("error"))
            statistics["groq_failed"] += 1
            continue

        result = response["result"]
        tokens = response.get("tokens")

        groq_cache[request_id] = {
            "measure": measure,
            "result": result,
            "tokens": tokens,
            "model": GROQ_MODEL,
            "timestamp": time.time(),
        }

        save_groq_cache(groq_cache)

        domain = result["domain"]
        confidence = float(result["confidence"])

        domain_results[domain].append(measure)

        details[measure]["domain"] = domain
        details[measure]["confidence"] = round(confidence, 3)
        details[measure]["method"] = "llm"
        details[measure]["explanation"] = result.get("explanation", "")
        details[measure]["request_id"] = request_id
        details[measure]["tokens"] = tokens

        statistics["groq_new"] += 1

        time.sleep(WAITING_TIME)

    # -------------------------
    # Unresolved
    # -------------------------

    unresolved = [
        measure
        for measure in measures
        if details.get(measure, {}).get("domain") is None
    ]

    # -------------------------
    # Sort domain results
    # -------------------------

    for domain in domain_results:
        domain_results[domain] = sorted(set(domain_results[domain]))

    # -------------------------
    # Save outputs
    # -------------------------

    with open(OUTPUT_DOMAINS, "w", encoding="utf-8") as f:
        json.dump(domain_results, f, indent=4, ensure_ascii=False)

    with open(OUTPUT_DETAILS, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=4, ensure_ascii=False)

    with open(OUTPUT_UNRESOLVED, "w", encoding="utf-8") as f:
        json.dump(unresolved, f, indent=4, ensure_ascii=False)

    final_results = {}

    for measure in measures:
        detail = details.get(measure, {})

        final_results[measure] = {
            "domain": detail.get("domain"),
            "secondary_domain": detail.get("secondary_domain"),
            "confidence": detail.get("confidence", 0),
            "method": detail.get("method", "unresolved"),
        }

    with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    # -------------------------
    # Summary
    # -------------------------

    total_measures = len(measures)

    dictionary_embedding_count = statistics["dictionary_or_embedding"]
    llm_count = statistics["groq_cached"] + statistics["groq_new"]

    domain_counts = {domain: len(domain_results[domain]) for domain in domains}

    dictionary_embedding_percentage = dictionary_embedding_count / total_measures * 100
    llm_percentage = llm_count / total_measures * 100

    print()
    print("========== STEP 7 SUMMARY ==========")
    print(f"Dataset:              {DATASET}")
    print(f"Total measures:       {total_measures}")
    print(f"Dictionary/embedding: {dictionary_embedding_count} ({dictionary_embedding_percentage:.2f}%)")
    print(f"LLM:                  {llm_count} ({llm_percentage:.2f}%)")
    print(f"LLM cached:           {statistics['groq_cached']}")
    print(f"LLM new:              {statistics['groq_new']}")
    print(f"LLM failed:           {statistics['groq_failed']}")
    print(f"Unresolved:           {len(unresolved)}")

    print()
    print("Domain distribution:")

    for domain in sorted(domain_counts):
        count = domain_counts[domain]
        percentage = count / total_measures * 100
        print(f"- {domain}: {count} ({percentage:.2f}%)")

    print(f"Total assigned: {sum(domain_counts.values())}")

    if groq_daily_limit_reached:
        print()
        print("WARNING: LLM daily limit reached. Unprocessed candidates remain unresolved.")

    print()
    print(f"Domains:    {OUTPUT_DOMAINS}")
    print(f"Details:    {OUTPUT_DETAILS}")
    print(f"Unresolved: {OUTPUT_UNRESOLVED}")
    print(f"Final:      {FINAL_OUTPUT}")
    print(f"Cache:      {GROQ_CACHE_FILE}")


if __name__ == "__main__":
    main()