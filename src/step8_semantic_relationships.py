import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ============================================================
# CONFIGURATION
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config import DATASET, OUTPUT_DIR, CACHE_DIR


INPUT_FILE = OUTPUT_DIR / f"6_V_classified_{DATASET}_results.json"
DOMAIN_FILE = OUTPUT_DIR / "step7" / f"7_M_final_classification_{DATASET}_results.json"

STEP8_OUTPUT_DIR = OUTPUT_DIR / "step8"

OUTPUT_RELATIONSHIPS = STEP8_OUTPUT_DIR / f"8_semantic_relationships_{DATASET}_results.json"
OUTPUT_CANDIDATES = STEP8_OUTPUT_DIR / f"8_semantic_candidates_{DATASET}_results.json"
OUTPUT_UNRESOLVED = STEP8_OUTPUT_DIR / f"8_semantic_unresolved_{DATASET}_results.json"
OUTPUT_STATS = STEP8_OUTPUT_DIR / f"8_semantic_statistics_{DATASET}_results.json"
OUTPUT_GRAPH = STEP8_OUTPUT_DIR / f"8_semantic_graph_{DATASET}_results.json"
OUTPUT_LEARNED_PATTERNS = STEP8_OUTPUT_DIR / f"8_learned_patterns_{DATASET}_results.json"

LLM_CACHE_FILE = CACHE_DIR / "8_llm_semantic_cache.json"

MODEL_NAME = "all-MiniLM-L6-v2"
USE_LLM = True

GROQ_MODEL = "openai/gpt-oss-120b"
ADD_EXPLANATION = False

LLM_BATCH_SIZE = 40
LLM_CONFIDENCE_THRESHOLD = 0.70

MAX_RETRIES = 3
WAITING_TIME = 0.5

RELATIONS = {
    "same_concept",
    "synonym",
    "subtype",
    "complementary",
    "different_aspects",
    "related",
}


# ------------------------------------------------------------
# Candidate generation
# ------------------------------------------------------------
# Keep the original embedding retrieval conservative so the LLM queue remains
# close to the previous implementation.
TOP_K_NEIGHBORS = 5
MIN_EMBEDDING_SIMILARITY = 0.55
SEMANTIC_CANDIDATE_THRESHOLD = 0.64

# Add a second lexical retrieval channel. These candidates improve coverage in
# OUTPUT_CANDIDATES, but TF-IDF-only pairs are not sent to the LLM unless they
# are also embedding candidates. Strong deterministic rules may still resolve
# them automatically.
LEXICAL_K_NEIGHBORS = 5
MIN_TFIDF_SIMILARITY = 0.18

# Preserve the original domain filter for embedding retrieval to avoid a large
# increase in ambiguous LLM cases. Lexical retrieval is not hard-filtered by
# domain because it is used mainly to improve candidate recall.
USE_DOMAIN_INFORMATION = True
USE_DOMAIN_FILTER_FOR_EMBEDDING = True
CROSS_DOMAIN_MIN_SIMILARITY = 0.84

# Kept only as a weak scoring signal for lexical candidates.
CROSS_DOMAIN_SOFT_THRESHOLD = 0.76
CROSS_DOMAIN_SCORE_PENALTY = 0.02

# ------------------------------------------------------------
# Same concept
# ------------------------------------------------------------
SAME_CONCEPT_MIN_SIM = 0.965
SAME_CONCEPT_MIN_LEX = 0.90
SAME_CONCEPT_MIN_OVERLAP = 0.88
REORDER_MIN_SIM = 0.965

# ------------------------------------------------------------
# Qualifier guard
# ------------------------------------------------------------
QUALIFIER_TERM_SIM_MAX = 0.78
QUALIFIER_NUMERIC_GUARD = True
QUALIFIER_SINGLETON_GUARD = True

# ------------------------------------------------------------
# Subtype
# ------------------------------------------------------------
SUBTYPE_MIN_CONTAINMENT = 0.82
SUBTYPE_MIN_SIM = 0.84
SUBTYPE_MAX_ADDED_TOKENS = 6
SUBTYPE_MIN_COMMON = 2
SUBTYPE_MIN_STRUCTURAL = 0.80

# ------------------------------------------------------------
# Learned substitutions
# ------------------------------------------------------------
MIN_COMMON_TOKENS = 2
LEARN_MIN_SUPPORT = 2
LEARN_MIN_TERM_SIM = 0.45
LEARN_MAX_TERM_SIM = 0.985
VARIANT_MIN_SUPPORT = 3
VARIANT_MIN_TERM_SIM = 0.84
VARIANT_MAX_TERM_SIM = 0.985
VARIANT_MIN_CONTEXT = 0.82
VARIANT_MIN_EMBED = 0.90
ASPECT_MIN_SIM = 0.84
ASPECT_MIN_STRUCTURAL = 0.72
CONSOLIDATION_MIN_SUPPORT = 2
CONSOLIDATION_MIN_CONTEXTS = 2
CONSOLIDATION_MIN_CONTEXT_SIM = 0.72

# ------------------------------------------------------------
# Structural aspect
# ------------------------------------------------------------
STRUCTURAL_MIN_SIM = 0.92
STRUCTURAL_MIN_OVERLAP = 0.55
STRUCTURAL_MAX_ADDED_TOKENS = 3
STRUCTURAL_MIN_COMMON = 2

# ------------------------------------------------------------
# Generic related
# ------------------------------------------------------------
RELATED_MIN_SCORE = 0.78
RELATED_MIN_EMBED = 0.76
RELATED_MIN_LEX = 0.10
RELATED_MIN_OVERLAP = 0.10

# Automatic acceptance threshold by relation.
# Generic `related` is deliberately never accepted automatically.
AUTO_CONFIDENCE_THRESHOLDS = {
    "same_concept": 0.85,
    "synonym": 0.88,
    "subtype": 0.85,
    "different_aspects": 0.88,
    "complementary": 1.01,
    "related": 1.01,
}

DEBUG = True
DEBUG_SAMPLE_PER_METHOD = 6
BALANCE_DOMINANT_SHARE = 0.60

STOPWORDS = {
    "the", "of", "and", "or", "for", "in", "on", "to", "a", "an", "by",
    "from", "with", "at", "as", "is", "are", "was", "were", "be", "per",
    "de", "la", "le", "des", "du", "di", "del", "della", "dei", "degli",
}


# ============================================================
# ENVIRONMENT
# ============================================================
client = None

# ============================================================
# UTILITIES
# ============================================================


def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def json_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    return obj


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2, ensure_ascii=False)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# VALIDATE PATHS
# ============================================================
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Step 6 output not found: {INPUT_FILE}\n"
        "Run Step 6 before Step 8."
    )

if USE_DOMAIN_INFORMATION and not DOMAIN_FILE.exists():
    raise FileNotFoundError(
        f"Step 7 output not found: {DOMAIN_FILE}\n"
        "Run Step 7 before Step 8."
    )

STEP8_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD INPUT
# ============================================================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
if not isinstance(data, dict) or "M" not in data or not isinstance(data["M"], dict):
    raise ValueError(f"{INPUT_FILE} does not contain a dictionary under 'M'")
measures = list(data["M"].keys())
measure_set = set(measures)
n = len(measures)
print(f"Measures={n}")

# ============================================================
# NORMALIZATION
# ============================================================


def normalize(text):
    text = str(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def conservative_token(word):
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ses"):
        return word
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokens(text):
    out = set()
    for token in normalize(text).split():
        if token in STOPWORDS or len(token) < 2:
            continue
        out.add(conservative_token(token))
    return out


def exact_signature(text):
    """Flat qualifier signature retained for output/debugging."""
    t = str(text)
    codes = set()
    for value in re.findall(r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)?(?![A-Za-z0-9])", t):
        codes.add(value.replace(",", ".").lower())
    for a, b in re.findall(
        r"(?<![A-Za-z0-9])([A-Za-z0-9]{1,3})-([A-Za-z0-9]{1,3})(?![A-Za-z0-9])",
        t,
    ):
        codes.add(f"{a}-{b}".lower())
    for code in re.findall(r"\(([A-Za-z%]{1,6})\)", t):
        codes.add(code.lower())
    for code in re.findall(r"[:-]\s*([A-Z]{1,6})(?=\s|$|[,.])", t):
        codes.add(code.lower())
    return codes


def exact_signature_categorized(text):
    """Qualifier values grouped by category for conflict detection."""
    t = str(text)
    sig = {}
    m = re.search(r"\bNUTS\s*([0-9])\b", t, re.IGNORECASE)
    if m:
        sig["nuts"] = m.group(1)
    m = re.search(r"\bRev\.?\s*([0-9]+(?:\.[0-9]+)?)\b", t, re.IGNORECASE)
    if m:
        sig["rev"] = m.group(1)
    m = re.search(r"\(([A-Za-z%]{1,6})\)", t)
    if m:
        sig["unit"] = m.group(1).lower()
    m = re.search(
        r"(?<![A-Za-z0-9])([A-Za-z0-9]{1,3})-([A-Za-z0-9]{1,3})(?![A-Za-z0-9])",
        t,
    )
    if m:
        sig["range"] = f"{m.group(1)}-{m.group(2)}".lower()
    m = re.search(r"[:-]\s*([A-Z]{1,6})(?=\s|$|[,.])", t)
    if m:
        sig["section"] = m.group(1).lower()
    return sig


def qualifier_conflict(text_a, text_b):
    sig_a = exact_signature_categorized(text_a)
    sig_b = exact_signature_categorized(text_b)
    for key in sig_a.keys() & sig_b.keys():
        if sig_a[key] != sig_b[key]:
            return True
    return False

normalized = [normalize(x) for x in measures]
measure_tokens = [tokens(x) for x in measures]
exact_signatures = [exact_signature(x) for x in measures]


def measure_core(text):
    """
    Return the central measured phenomenon, stripping common breakdown and
    presentation suffixes. Eurostat table titles very often use the form
    "<indicator> by <dimensions>"; changes after the first "by" usually alter
    the disaggregation rather than the underlying statistical concept.
    """
    core = normalize(text)
    # Remove common trailing frequency / presentation markers first.
    core = re.sub(
        r"\s*[-,:]?\s*(annual|monthly|quarterly|bi annual|biannual)\s+data.*$",
        "",
        core,
    ).strip()
    # The first 'by' normally separates the measure from its dimensions.
    core = re.split(r"\s+by\s+", core, maxsplit=1)[0].strip()
    # Remove a few presentation/unit suffixes that do not change the measure.
    core = re.sub(r"\s+(in|as)\s+percent(?:age)?$", "", core).strip()
    core = re.sub(r"\s+index$", "", core).strip()
    return core

measure_cores = [measure_core(x) for x in measures]
measure_core_tokens = [tokens(x) for x in measure_cores]


def has_by_breakdown(text):
    return bool(re.search(r"\s+by\s+", normalize(text)))


def same_indicator_different_breakdown(i, j, embedding_similarity):
    """Same measured indicator, different explicit 'by <dimension>' breakdown."""
    core_a = measure_cores[i]
    core_b = measure_cores[j]
    if not core_a or core_a != core_b:
        return False
    if not (has_by_breakdown(measures[i]) or has_by_breakdown(measures[j])):
        return False
    n_core = len(measure_core_tokens[i])
    # Multi-token cores are relatively safe.
    if n_core >= 2:
        return embedding_similarity >= 0.80
    # Cover one-word indicators such as Population / Employment more carefully.
    if n_core == 1 and len(core_a) >= 5:
        return embedding_similarity >= 0.88
    return False

# ============================================================
# DOMAIN MAPPING
# ============================================================


def extract_domain_mapping(obj):
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if (
                isinstance(value, dict)
                and isinstance(value.get("domain"), str)
                and value["domain"].strip()
            ):
                out[str(key)] = value["domain"].strip()
    return out

domain_mapping = {}
if USE_DOMAIN_INFORMATION and DOMAIN_FILE and os.path.exists(DOMAIN_FILE):
    domain_data = load_json(DOMAIN_FILE, {})
    raw_mapping = extract_domain_mapping(domain_data)
    domain_mapping = {
        key: value
        for key, value in raw_mapping.items()
        if key in measure_set
    }
print(f"Domains available for {len(domain_mapping)}/{n} measures")

# ============================================================
# EMBEDDINGS + TF-IDF
# ============================================================
print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(
    measures,
    normalize_embeddings=True,
    show_progress_bar=True,
)
embeddings = np.asarray(embeddings, dtype=np.float32)
print("Building TF-IDF matrix...")
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
)
tfidf_matrix = tfidf.fit_transform(normalized)

# ============================================================
# SIMILARITY FEATURES
# ============================================================


def lexical_similarity(i, j):
    return float(tfidf_matrix[i].dot(tfidf_matrix[j].T).toarray()[0][0])


def token_overlap(i, j):
    a, b = measure_tokens[i], measure_tokens[j]
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def token_containment(i, j):
    a, b = measure_tokens[i], measure_tokens[j]
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def sequence_similarity(i, j):
    return SequenceMatcher(None, normalized[i], normalized[j]).ratio()


def semantic_features(i, j, emb=None, cross_domain=False):
    if emb is None:
        emb = float(np.dot(embeddings[i], embeddings[j]))
    lex = lexical_similarity(i, j)
    overlap = token_overlap(i, j)
    containment = token_containment(i, j)
    sequence = sequence_similarity(i, j)
    score = (
        0.62 * emb
        + 0.16 * lex
        + 0.10 * overlap
        + 0.07 * containment
        + 0.05 * sequence
    )
    # Soft domain penalty only. Cross-domain pairs remain candidates.
    if cross_domain and emb < CROSS_DOMAIN_SOFT_THRESHOLD:
        score -= CROSS_DOMAIN_SCORE_PENALTY
    return {
        "embedding_similarity": float(emb),
        "lexical_similarity": float(lex),
        "token_overlap": float(overlap),
        "token_containment": float(containment),
        "sequence_similarity": float(sequence),
        "semantic_score": float(max(0.0, min(score, 1.0))),
    }

# ============================================================
# CANDIDATE GENERATION
# ============================================================
print("Generating candidates...")
candidate_map = {}


def _candidate_base(a, b, emb=None):
    if emb is None:
        emb = float(np.dot(embeddings[a], embeddings[b]))
    domain_a = domain_mapping.get(measures[a])
    domain_b = domain_mapping.get(measures[b])
    same_domain = (
        domain_a is not None
        and domain_b is not None
        and domain_a == domain_b
    )
    cross_domain = (
        domain_a is not None
        and domain_b is not None
        and domain_a != domain_b
    )
    return {
        "measure1": measures[a],
        "measure2": measures[b],
        "index1": a,
        "index2": b,
        "domain1": domain_a,
        "domain2": domain_b,
        "same_domain": bool(same_domain),
        "cross_domain": bool(cross_domain),
        **semantic_features(a, b, emb, cross_domain=cross_domain),
        "retrieval_sources": [],
    }

# ------------------------------------------------------------
# 1. Embedding nearest neighbours: same conservative retrieval as baseline

# ------------------------------------------------------------
for i in range(n):
    sims = np.dot(embeddings, embeddings[i])
    sims[i] = -1.0
    k = min(TOP_K_NEIGHBORS, n - 1)
    nearest = np.argpartition(sims, -k)[-k:]
    nearest = sorted(nearest, key=lambda x: float(sims[x]), reverse=True)
    for j in nearest:
        j = int(j)
        emb = float(sims[j])
        if emb < MIN_EMBEDDING_SIMILARITY:
            continue
        a, b = sorted((i, j))
        if a == b:
            continue
        domain_a = domain_mapping.get(measures[a])
        domain_b = domain_mapping.get(measures[b])
        same_domain = (
            domain_a is not None
            and domain_b is not None
            and domain_a == domain_b
        )
        if (
            USE_DOMAIN_FILTER_FOR_EMBEDDING
            and domain_mapping
            and domain_a is not None
            and domain_b is not None
            and not same_domain
            and emb < CROSS_DOMAIN_MIN_SIMILARITY
        ):
            continue
        key = (a, b)
        if key not in candidate_map:
            candidate_map[key] = _candidate_base(a, b, emb=emb)
        candidate_map[key]["retrieval_sources"].append("embedding_knn")

# ------------------------------------------------------------
# 2. TF-IDF nearest neighbours: improves recall without enlarging LLM queue

# ------------------------------------------------------------
if n > 1:
    lexical_k = min(LEXICAL_K_NEIGHBORS + 1, n)
    lexical_nn = NearestNeighbors(
        n_neighbors=lexical_k,
        metric="cosine",
        algorithm="brute",
    )
    lexical_nn.fit(tfidf_matrix)
    lexical_distances, lexical_indices = lexical_nn.kneighbors(tfidf_matrix)
    for i in range(n):
        for distance, j in zip(
            lexical_distances[i],
            lexical_indices[i],
        ):
            j = int(j)
            if j == i:
                continue
            tfidf_similarity = 1.0 - float(distance)
            if tfidf_similarity < MIN_TFIDF_SIMILARITY:
                continue
            a, b = sorted((i, j))
            if a == b:
                continue
            key = (a, b)
            if key not in candidate_map:
                candidate_map[key] = _candidate_base(a, b)
            if "tfidf_knn" not in candidate_map[key]["retrieval_sources"]:
                candidate_map[key]["retrieval_sources"].append("tfidf_knn")

candidates = list(candidate_map.values())
for candidate in candidates:
    sources = set(candidate.get("retrieval_sources", []))
    # Only embedding-retrieved candidates enter the normal semantic/LLM path.
    # TF-IDF-only candidates are retained for candidate recall and deterministic
    # specific rules, but never sent directly to the LLM.
    embedding_semantic = (
        "embedding_knn" in sources
        and candidate["semantic_score"] >= SEMANTIC_CANDIDATE_THRESHOLD
    )
    candidate["decision"] = (
        "related_candidate"
        if embedding_semantic
        else "retrieval_candidate"
    )
semantic_candidates = [
    c for c in candidates
    if c["decision"] == "related_candidate"
]
retrieval_only_candidates = [
    c for c in candidates
    if c["decision"] == "retrieval_candidate"
]
candidates.sort(key=lambda x: x["semantic_score"], reverse=True)
retrieval_counts = Counter(
    source
    for c in candidates
    for source in set(c.get("retrieval_sources", []))
)
both_sources = sum(
    1
    for c in candidates
    if {"embedding_knn", "tfidf_knn"}.issubset(
        set(c.get("retrieval_sources", []))
    )
)
print(
    f"Candidates={len(candidates)} | "
    f"Embedding semantic candidates={len(semantic_candidates)} | "
    f"retrieval-only candidates={len(retrieval_only_candidates)}"
)
print(
    f"Retrieval sources: embedding={retrieval_counts.get('embedding_knn', 0)} | "
    f"tfidf={retrieval_counts.get('tfidf_knn', 0)} | both={both_sources}"
)

# ============================================================
# STRUCTURAL FEATURES
# ============================================================


def difference_structure(i, j):
    a, b = measure_tokens[i], measure_tokens[j]
    common = a & b
    return {
        "common": common,
        "only_a": a - common,
        "only_b": b - common,
        "n_common": len(common),
        "n_only_a": len(a - common),
        "n_only_b": len(b - common),
    }


def structural_similarity(i, j):
    a, b = measure_tokens[i], measure_tokens[j]
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), len(b))


def numeric_tokens(text):
    return set(re.findall(r"\b\d+(?:[.,]\d+)?\b", normalize(text)))

# ============================================================
# TERM EMBEDDINGS FOR LEARNED SUBSTITUTIONS
# ============================================================
print("Encoding individual terms for learned substitutions...")
all_terms = sorted({term for token_set in measure_tokens for term in token_set})
term_embeddings = {}
if all_terms:
    term_array = model.encode(
        all_terms,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    term_embeddings = {
        term: np.asarray(vector, dtype=np.float32)
        for term, vector in zip(all_terms, term_array)
    }


def term_similarity(a, b):
    ea = term_embeddings.get(a)
    eb = term_embeddings.get(b)
    if ea is None or eb is None:
        return 0.0
    return float(np.dot(ea, eb))

# ============================================================
# QUALIFIER CHANGE DETECTION
# ============================================================


def changed_qualifier_detected(i, j):
    if qualifier_conflict(measures[i], measures[j]):
        return {
            "detected": True,
            "reason": "qualifier_category_conflict",
            "signature_a": exact_signature_categorized(measures[i]),
            "signature_b": exact_signature_categorized(measures[j]),
        }
    d = difference_structure(i, j)
    if QUALIFIER_NUMERIC_GUARD:
        numeric_a = numeric_tokens(measures[i])
        numeric_b = numeric_tokens(measures[j])
        if numeric_a != numeric_b and numeric_a and numeric_b:
            return {
                "detected": True,
                "reason": "numeric_token_changed",
                "numeric_a": sorted(numeric_a),
                "numeric_b": sorted(numeric_b),
            }
    if d["n_only_a"] == 1 and d["n_only_b"] == 1 and d["n_common"] >= 2:
        a = next(iter(d["only_a"]))
        b = next(iter(d["only_b"]))
        sim = term_similarity(a, b) if term_embeddings else 0.0
        if sim < QUALIFIER_TERM_SIM_MAX:
            return {
                "detected": True,
                "reason": "single_changed_token",
                "token_a": a,
                "token_b": b,
                "term_similarity": sim,
            }
    # A singleton addition/removal should not block subtype.
    return {"detected": False}

# ============================================================
# LEARN CONTEXTUAL SUBSTITUTIONS
# ============================================================
print("Learning contextual substitutions...")
substitution_examples = defaultdict(list)
for c in semantic_candidates:
    d = difference_structure(c["index1"], c["index2"])
    if (
        d["n_common"] < MIN_COMMON_TOKENS
        or d["n_only_a"] != 1
        or d["n_only_b"] != 1
    ):
        continue
    a = next(iter(d["only_a"]))
    b = next(iter(d["only_b"]))
    if a == b:
        continue
    if qualifier_conflict(c["measure1"], c["measure2"]):
        continue
    if a.isdigit() or b.isdigit():
        continue
    key = tuple(sorted((a, b)))
    substitution_examples[key].append({
        "candidate": c,
        "context": d["common"],
    })
learned_substitutions = {}
for pair, examples in substitution_examples.items():
    support = len(examples)
    if support < LEARN_MIN_SUPPORT:
        continue
    a, b = pair
    term_sim = term_similarity(a, b)
    if not (LEARN_MIN_TERM_SIM <= term_sim <= LEARN_MAX_TERM_SIM):
        continue
    contexts = [example["context"] for example in examples]
    unique_contexts = {tuple(sorted(context)) for context in contexts}
    context_scores = []
    for x in range(len(contexts)):
        for y in range(x + 1, len(contexts)):
            if contexts[x] and contexts[y]:
                context_scores.append(
                    len(contexts[x] & contexts[y])
                    / len(contexts[x] | contexts[y])
                )
    context_similarity = (
        float(np.mean(context_scores))
        if context_scores else 0.0
    )
    learned_substitutions[pair] = {
        "support": support,
        "unique_contexts": len(unique_contexts),
        "term_similarity": term_sim,
        "context_similarity": context_similarity,
    }
consolidated = {}
for pair, info in learned_substitutions.items():
    evidence_score = (
        0.35 * min(info["support"] / 10.0, 1.0)
        + 0.25 * min(info["unique_contexts"] / 5.0, 1.0)
        + 0.20 * info["context_similarity"]
        + 0.20 * info["term_similarity"]
    )
    consolidated[pair] = {
        **info,
        "evidence_score": float(evidence_score),
        "consolidated": (
            info["support"] >= CONSOLIDATION_MIN_SUPPORT
            and info["unique_contexts"] >= CONSOLIDATION_MIN_CONTEXTS
            and info["context_similarity"] >= CONSOLIDATION_MIN_CONTEXT_SIM
        ),
    }
learned_variants = {}
for pair, info in consolidated.items():
    if not info["consolidated"]:
        continue
    if info["support"] < VARIANT_MIN_SUPPORT:
        continue
    if not (
        VARIANT_MIN_TERM_SIM
        <= info["term_similarity"]
        <= VARIANT_MAX_TERM_SIM
    ):
        continue
    if info["context_similarity"] < VARIANT_MIN_CONTEXT:
        continue
    full_scores = [
        example["candidate"]["embedding_similarity"]
        for example in substitution_examples[pair]
    ]
    if not full_scores or float(np.mean(full_scores)) < VARIANT_MIN_EMBED:
        continue
    learned_variants[pair] = info
learned_dimensions = {}
for pair, info in consolidated.items():
    valid_examples = [
        example["candidate"]
        for example in substitution_examples[pair]
        if structural_similarity(
            example["candidate"]["index1"],
            example["candidate"]["index2"],
        ) >= ASPECT_MIN_STRUCTURAL
        and example["candidate"]["embedding_similarity"] >= ASPECT_MIN_SIM
    ]
    unique_contexts = {
        tuple(sorted(example["context"]))
        for example in substitution_examples[pair]
    }
    if len(valid_examples) < LEARN_MIN_SUPPORT or len(unique_contexts) < 2:
        continue
    learned_dimensions[pair] = {
        **info,
        "valid_examples": len(valid_examples),
        "unique_contexts": len(unique_contexts),
    }
save_json(
    OUTPUT_LEARNED_PATTERNS,
    {
        "learned_substitutions": {
            f"{a} <-> {b}": info
            for (a, b), info in learned_substitutions.items()
        },
        "learned_variants": {
            f"{a} <-> {b}": info
            for (a, b), info in learned_variants.items()
        },
        "learned_dimensions": {
            f"{a} <-> {b}": info
            for (a, b), info in learned_dimensions.items()
        },
    },
)
print(
    f"Observed substitutions={len(substitution_examples)} | "
    f"learned variants={len(learned_variants)} | "
    f"learned dimensions={len(learned_dimensions)}"
)

# ============================================================
# RULE CASCADE
# ============================================================


def same_token_set_reordered(i, j):
    if measure_tokens[i] != measure_tokens[j]:
        return False
    if normalized[i] == normalized[j]:
        return False
    return float(np.dot(embeddings[i], embeddings[j])) >= REORDER_MIN_SIM


def rule_same_concept(c):
    i, j = c["index1"], c["index2"]
    if normalized[i] == normalized[j]:
        return {
            "relation": "same_concept",
            "confidence": 0.999,
            "evidence": {"reason": "normalized_text_identity"},
        }
    if same_token_set_reordered(i, j):
        return {
            "relation": "same_concept",
            "confidence": 0.995,
            "evidence": {"reason": "token_reorder"},
        }
    # Explicit Eurostat-style "by <dimension>" suffixes describe a
    # disaggregation of the same measured indicator, not a subtype.
    if same_indicator_different_breakdown(i, j, c["embedding_similarity"]):
        both_have_breakdown = (
            has_by_breakdown(measures[i]) and has_by_breakdown(measures[j])
        )
        return {
            "relation": "same_concept",
            # Deliberately borderline: the shared core is strong evidence,
            # but the LLM decides whether the breakdown still represents
            # same_concept or a more specific semantic relation.
            "confidence": 0.840 if both_have_breakdown else 0.820,
            "evidence": {
                "reason": "same_measure_core_different_breakdown",
                "measure_core": measure_cores[i],
            },
        }
    # From this point onward, conservative qualifier guards remain useful for
    # fuzzy same-concept decisions.
    if qualifier_conflict(c["measure1"], c["measure2"]):
        return None
    a_tokens = measure_tokens[i]
    b_tokens = measure_tokens[j]
    strict_containment = (
        a_tokens != b_tokens
        and (a_tokens <= b_tokens or b_tokens <= a_tokens)
    )
    if not strict_containment:
        if (
            c["embedding_similarity"] >= SAME_CONCEPT_MIN_SIM
            and c["lexical_similarity"] >= SAME_CONCEPT_MIN_LEX
            and c["token_overlap"] >= SAME_CONCEPT_MIN_OVERLAP
        ):
            return {
                "relation": "same_concept",
                "confidence": 0.985,
                "evidence": {
                    "reason": "high_agreement_no_qualifier_conflict"
                },
            }
        if (
            c["embedding_similarity"] >= 0.985
            and c["sequence_similarity"] >= 0.965
            and c["token_overlap"] >= 0.90
        ):
            return {
                "relation": "same_concept",
                "confidence": 0.990,
                "evidence": {
                    "reason": "near_text_identity_no_qualifier_conflict"
                },
            }
    return None


def rule_subtype(c):
    i, j = c["index1"], c["index2"]
    a, b = measure_tokens[i], measure_tokens[j]
    if not a or not b or a == b:
        return None
    # A pure "by <dimension>" disaggregation is the same indicator,
    # not a genuinely narrower statistical concept.
    if same_indicator_different_breakdown(i, j, c["embedding_similarity"]):
        return None
    if c["embedding_similarity"] < SUBTYPE_MIN_SIM:
        return None
    containment = c["token_containment"]
    if containment < SUBTYPE_MIN_CONTAINMENT:
        return None
    common = a & b
    if len(common) < SUBTYPE_MIN_COMMON:
        return None
    if len(a) >= len(b):
        smaller, larger = b, a
        smaller_measure, larger_measure = c["measure2"], c["measure1"]
    else:
        smaller, larger = a, b
        smaller_measure, larger_measure = c["measure1"], c["measure2"]
    if not smaller.issubset(larger):
        return None
    added = larger - smaller
    if not added or len(added) > SUBTYPE_MAX_ADDED_TOKENS:
        return None
    if qualifier_conflict(c["measure1"], c["measure2"]):
        return None
    structural = structural_similarity(i, j)
    if structural < SUBTYPE_MIN_STRUCTURAL:
        return None
    confidence = min(
        0.95,
        0.72
        + 0.10 * containment
        + 0.08 * c["embedding_similarity"]
        + 0.05 * structural,
    )
    return {
        "relation": "subtype",
        "confidence": confidence,
        "evidence": {
            "contained_measure": smaller_measure,
            "expanded_measure": larger_measure,
            "common_tokens": sorted(common),
            "added_tokens": sorted(added),
            "token_containment": containment,
            "structural_similarity": structural,
        },
    }


def changed_substitution(c):
    d = difference_structure(c["index1"], c["index2"])
    if d["n_only_a"] != 1 or d["n_only_b"] != 1:
        return None
    a = next(iter(d["only_a"]))
    b = next(iter(d["only_b"]))
    return tuple(sorted((a, b)))


def rule_variant(c):
    pair = changed_substitution(c)
    if pair is None:
        return None
    info = learned_variants.get(pair)
    if info is None:
        return None
    if c["embedding_similarity"] < VARIANT_MIN_EMBED:
        return None
    if c["semantic_score"] < 0.82:
        return None
    confidence = min(
        0.97,
        0.78
        + 0.08 * min(info["term_similarity"], 1.0)
        + 0.06 * min(info["context_similarity"], 1.0)
        + 0.03 * min(info["support"] / 10.0, 1.0),
    )
    return {
        "relation": "synonym",
        "confidence": confidence,
        "evidence": {
            "term_a": pair[0],
            "term_b": pair[1],
            "support": info["support"],
            "unique_contexts": info["unique_contexts"],
            "context_similarity": info["context_similarity"],
            "term_similarity": info["term_similarity"],
        },
    }


def rule_dimension(c):
    pair = changed_substitution(c)
    if pair is None:
        return None
    d = difference_structure(c["index1"], c["index2"])
    if d["n_common"] < MIN_COMMON_TOKENS:
        return None
    info = learned_dimensions.get(pair)
    if info is None:
        return None
    if c["embedding_similarity"] < ASPECT_MIN_SIM:
        return None
    structural = structural_similarity(c["index1"], c["index2"])
    if structural < ASPECT_MIN_STRUCTURAL:
        return None
    confidence = min(
        0.96,
        0.72
        + 0.10 * c["embedding_similarity"]
        + 0.07 * structural
        + 0.04 * min(info["support"] / 8.0, 1.0),
    )
    return {
        "relation": "different_aspects",
        "confidence": confidence,
        "evidence": {
            "dimension_a": pair[0],
            "dimension_b": pair[1],
            "support": info["support"],
            "unique_contexts": info["unique_contexts"],
            "context_similarity": info["context_similarity"],
            "term_similarity": info["term_similarity"],
        },
    }


def rule_structural_aspect(c):
    i, j = c["index1"], c["index2"]
    d = difference_structure(i, j)
    if d["n_common"] < STRUCTURAL_MIN_COMMON:
        return None
    if d["n_only_a"] == 0 or d["n_only_b"] == 0:
        return None
    if (
        d["n_only_a"] > STRUCTURAL_MAX_ADDED_TOKENS
        or d["n_only_b"] > STRUCTURAL_MAX_ADDED_TOKENS
    ):
        return None
    if qualifier_conflict(c["measure1"], c["measure2"]):
        return None
    structural = structural_similarity(i, j)
    if structural < 0.86:
        return None
    if c["embedding_similarity"] < STRUCTURAL_MIN_SIM:
        return None
    if c["token_overlap"] < STRUCTURAL_MIN_OVERLAP:
        return None
    pair = changed_substitution(c)
    learned = pair in learned_dimensions if pair is not None else False
    if not learned and structural < 0.94:
        return None
    confidence = min(
        0.95,
        0.60
        + 0.18 * c["embedding_similarity"]
        + 0.12 * structural
        + 0.08 * c["token_overlap"],
    )
    return {
        "relation": "different_aspects",
        "confidence": confidence,
        "evidence": {
            "common_tokens": sorted(d["common"]),
            "only_measure1": sorted(d["only_a"]),
            "only_measure2": sorted(d["only_b"]),
            "structural_similarity": structural,
            "learned_support": learned,
        },
    }


def rule_related(c):
    if c["semantic_score"] < RELATED_MIN_SCORE:
        return None
    if c["embedding_similarity"] < RELATED_MIN_EMBED:
        return None
    if (
        c["lexical_similarity"] < RELATED_MIN_LEX
        and c["token_overlap"] < RELATED_MIN_OVERLAP
    ):
        return None
    confidence = min(
        0.88,
        0.50
        + 0.23 * c["semantic_score"]
        + 0.14 * c["embedding_similarity"]
        + 0.05 * max(c["lexical_similarity"], c["token_overlap"]),
    )
    return {
        "relation": "related",
        "confidence": confidence,
        "evidence": {
            "semantic_score": c["semantic_score"],
            "embedding_similarity": c["embedding_similarity"],
            "lexical_similarity": c["lexical_similarity"],
            "token_overlap": c["token_overlap"],
        },
    }


def classify_candidate(c):
    rules = (
        (rule_same_concept, None),
        (rule_subtype, "subtype"),
        (rule_variant, "learned_variant"),
        (rule_dimension, "learned_dimension"),
        (rule_structural_aspect, "structural_aspect"),
        (rule_related, "related"),
    )
    for rule_fn, base_method in rules:
        result = rule_fn(c)
        if result is None:
            continue
        if base_method is None:
            method = (
                "automatic_same_concept_"
                + result["evidence"]["reason"]
            )
        else:
            method = f"automatic_{base_method}"
        return {**result, "method": method}
    return None


def llm_candidate_is_worth_classifying(c, result):
    """Evidence-based LLM gating; this is not a request-count cap."""
    if result is not None and result["relation"] in {
        "same_concept", "synonym", "subtype", "different_aspects",
    }:
        return True
    emb = c["embedding_similarity"]
    score = c["semantic_score"]
    lexical_signal = max(c["lexical_similarity"], c["token_overlap"])
    if c.get("cross_domain", False):
        return emb >= 0.84 and score >= 0.76 and lexical_signal >= 0.08
    return emb >= 0.76 and score >= 0.74 and lexical_signal >= 0.10

print("Running rule cascade...")
auto_relationships = []
llm_candidates = []
candidate_only = []
rule_counts = Counter()
for c in candidates:
    result = classify_candidate(c)
    if result is None:
        c["auto_relation"] = None
        c["auto_confidence"] = None
        c["auto_method"] = None
        c["auto_evidence"] = None
    else:
        c["auto_relation"] = result["relation"]
        c["auto_confidence"] = float(result["confidence"])
        c["auto_method"] = result["method"]
        c["auto_evidence"] = result["evidence"]
    sources = set(c.get("retrieval_sources", []))
    is_embedding_semantic = (
        "embedding_knn" in sources
        and c["semantic_score"] >= SEMANTIC_CANDIDATE_THRESHOLD
    )
    if result is not None:
        threshold = AUTO_CONFIDENCE_THRESHOLDS.get(
            result["relation"],
            0.90,
        )
        if result["confidence"] >= threshold:
            auto_relationships.append(c)
            rule_counts[result["method"]] += 1
            continue
    # TF-IDF-only pairs remain in the candidate set for recall. Embedding
    # candidates reach the LLM only when there is enough semantic evidence.
    if is_embedding_semantic and llm_candidate_is_worth_classifying(c, result):
        llm_candidates.append(c)
    else:
        c["llm_skip_reason"] = (
            "insufficient_specific_semantic_evidence"
            if is_embedding_semantic
            else "retrieval_only_candidate"
        )
        candidate_only.append(c)
# Keep the existing variable name for the LLM/output sections below.
unresolved = llm_candidates
print(
    f"Resolved automatically={len(auto_relationships)} | "
    f"sent to LLM={len(llm_candidates)} | "
    f"candidate-only={len(candidate_only)}"
)

# ============================================================
# REQUEST IDS
# ============================================================


def request_id(a, b):
    a, b = sorted((a.strip(), b.strip()))
    raw = f"semrel-v6c-hybrid|{MODEL_NAME}|{GROQ_MODEL if USE_LLM else 'norun'}|{a}|{b}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

for c in candidates:
    c["request_id"] = request_id(c["measure1"], c["measure2"])

# ============================================================
# LLM PROMPT + PARSING
# ============================================================


def build_batch_prompt(batch):
    text_to_id = {}
    legend = []
    for pair in batch:
        for measure in (pair["measure1"], pair["measure2"]):
            if measure not in text_to_id:
                text_to_id[measure] = len(text_to_id) + 1
                legend.append(f'{text_to_id[measure]}: {measure}')
    pairs_block = "\n".join(
        f'{pair["candidate_id"]}: '
        f'{text_to_id[pair["measure1"]]}-{text_to_id[pair["measure2"]]}'
        for pair in batch
    )
    legend_block = "\n".join(legend)
    explanation_instruction = ""
    output_format = (
        '{"results":[{"pair_id":1,"relation":"subtype","confidence":0.9}]}'
    )
    if ADD_EXPLANATION:
        explanation_instruction = "\nGive an explanation of at most 15 words per pair."
        output_format = (
            '{"results":[{"pair_id":1,"relation":"subtype",'
            '"confidence":0.9,"explanation":"..."}]}'
        )
    return f"""Classify each pair of official statistical measures with exactly one label.
Use only the supplied measure names. Choose the label that best describes the semantic relation.
same_concept: same measured indicator/quantity. A 'by X' suffix normally changes
only the disaggregation, not the concept. Frequency, dimension order,
presentation, geography/NUTS level, or classification revision alone also do
not create a new concept when the measured indicator remains the same.
synonym: same concept expressed with substantially different terminology.
subtype: one measure is genuinely narrower in phenomenon, population, category,
or scope. Example: employment -> employment in manufacturing. A pure 'by X'
breakdown is not by itself a subtype.
different_aspects: same underlying phenomenon, but different component,
quantity, direction, characteristic, category, or outcome.
complementary: the measures are distinct statistical indicators that are not
aspects or subtypes of one another, but whose meanings are naturally connected
and useful together.
related: meaningful semantic relation not covered above. Shared broad topic
alone is insufficient.

Use related only if none of the five specific relations applies.
Measures:
{legend_block}
Pairs (pair_id: measure_idA-measure_idB):
{pairs_block}
{explanation_instruction}
Return ONLY valid JSON:
{output_format}
confidence must be in [0,1].
"""


def parse_json(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"**\\{**.***\\}**", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def validate_result(result, expected_ids):
    if not isinstance(result, dict):
        return {}, "response is not a dictionary"
    items = result.get("results")
    if not isinstance(items, list):
        return {}, "response has no results list"
    valid = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        if not {"pair_id", "relation", "confidence"}.issubset(item):
            continue
        try:
            pair_id = int(item["pair_id"])
            confidence = float(item["confidence"])
        except (TypeError, ValueError):
            continue
        relation = str(item["relation"]).strip()
        if pair_id not in expected_ids or pair_id in valid:
            continue
        if relation not in RELATIONS:
            continue
        if not 0.0 <= confidence <= 1.0:
            continue
        valid[pair_id] = {
            "pair_id": pair_id,
            "relation": relation,
            "confidence": confidence,
        }
        if ADD_EXPLANATION:
            valid[pair_id]["explanation"] = str(
                item.get("explanation", "")
            )
    if not valid:
        return {}, "no valid entries matched expected pair ids"
    return valid, None


def llm_same_concept_conflict(text_a, text_b):
    """
    Detect substantive qualifier conflicts for same_concept.

    NUTS level and classification revision differences are allowed because
    they may describe alternative breakdowns of the same underlying measure.
    """
    sig_a = exact_signature_categorized(text_a)
    sig_b = exact_signature_categorized(text_b)
    for key in {"unit", "range", "section"}:
        if key in sig_a and key in sig_b and sig_a[key] != sig_b[key]:
            return True
    return False


def llm_result_allowed(c, result):
    if not isinstance(result, dict):
        return False
    relation = result.get("relation")
    if relation == "same_concept":
        return not llm_same_concept_conflict(
            c["measure1"],
            c["measure2"],
        )
    if relation == "subtype":
        # Subtype remains more conservative: conflicting qualifiers can change
        # the population/category semantics rather than just presentation.
        return not qualifier_conflict(c["measure1"], c["measure2"])
    return True


def get_groq_client():
    global client

    if client is not None:
        return client

    from groq import Groq

    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("API_KEY")

    if not api_key:
        raise ValueError("API_KEY is not set in the .env file")

    client = Groq(api_key=api_key, timeout=60.0)
    return client


def call_llm(batch):
    groq_client = get_groq_client()
    prompt = build_batch_prompt(batch)
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=4096,
        reasoning_effort="low",
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content
    parsed = parse_json(text)
    expected_ids = {int(item["candidate_id"]) for item in batch}
    valid_entries, reason = validate_result(parsed, expected_ids)
    if not valid_entries:
        raise ValueError(f"Invalid LLM response: {reason}")
    usage = getattr(response, "usage", None)
    token_info = {}
    if usage:
        token_info = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
    return {"results": list(valid_entries.values())}, token_info

# ============================================================
# LLM PROCESSING
# ============================================================
llm_results = []
llm_stats = {
    "calls": 0,
    "cached_pairs": 0,
    "new_pairs": 0,
    "guard_rejected": 0,
    "below_confidence": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "daily_limit_reached": False,
}
cache = load_json(LLM_CACHE_FILE, {}) if USE_LLM else {}
pending = []
if USE_LLM:
    for c in unresolved:
        cached = cache.get(c["request_id"])
        if cached and isinstance(cached.get("result"), dict):
            result = cached["result"]
            if llm_result_allowed(c, result):
                c["llm_result"] = result
                c["llm_cached"] = True
                llm_results.append(c)
                llm_stats["cached_pairs"] += 1
            else:
                c["llm_result"] = result
                c["llm_cached"] = True
                c["guard_rejected"] = True
                llm_stats["guard_rejected"] += 1
        else:
            pending.append(c)
    total_pending = len(pending)
    total_llm_pairs = len(unresolved)
    total_batches = (
        (total_pending + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE
        if total_pending else 0
    )
    print()
    print(
        f"[LLM] total={total_llm_pairs} | "
        f"cached={llm_stats['cached_pairs']} | "
        f"pending={total_pending} | "
        f"batches={total_batches}"
    )
    for start in range(0, total_pending, LLM_BATCH_SIZE):
        if llm_stats["daily_limit_reached"]:
            break
        batch = pending[start:start + LLM_BATCH_SIZE]
        batch_number = start // LLM_BATCH_SIZE + 1
        processed_before = (
            llm_stats["cached_pairs"]
            + llm_stats["new_pairs"]
            + llm_stats["guard_rejected"]
        )
        print(
            f"\r[LLM] batch={batch_number}/{total_batches} | "
            f"calls={llm_stats['calls']} | "
            f"new={llm_stats['new_pairs']} | "
            f"cached={llm_stats['cached_pairs']} | "
            f"guard_rejected={llm_stats['guard_rejected']} | "
            f"processed={processed_before}/{total_llm_pairs}",
            end="",
            flush=True,
        )
        for candidate_id, candidate in enumerate(batch, start=1):
            candidate["candidate_id"] = candidate_id
        result = None
        token_info = None
        for attempt in range(MAX_RETRIES):
            try:
                result, token_info = call_llm(batch)
                break
            except Exception as exc:
                message = str(exc).lower()
                if any(key in message for key in ("daily limit", "per day", "tpd")):
                    llm_stats["daily_limit_reached"] = True
                    print("Daily LLM limit reached. Stopping fallback.")
                    break
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"LLM batch failed permanently: {exc}")
        if llm_stats["daily_limit_reached"]:
            break
        if result is None:
            continue
        llm_stats["calls"] += 1
        if token_info:
            llm_stats["prompt_tokens"] += token_info.get("prompt_tokens", 0)
            llm_stats["completion_tokens"] += token_info.get("completion_tokens", 0)
            llm_stats["total_tokens"] += token_info.get("total_tokens", 0)
        by_id = {
            int(item["pair_id"]): item
            for item in result["results"]
        }
        for c in batch:
            llm_result = by_id.get(int(c["candidate_id"]))
            if llm_result is None:
                continue
            c["llm_result"] = llm_result
            c["llm_cached"] = False
            if not llm_result_allowed(c, llm_result):
                c["guard_rejected"] = True
                llm_stats["guard_rejected"] += 1
                cache[c["request_id"]] = {
                    "measure1": c["measure1"],
                    "measure2": c["measure2"],
                    "result": llm_result,
                    "model": GROQ_MODEL,
                    "timestamp": time.time(),
                    "guard_rejected": True,
                }
                continue
            llm_results.append(c)
            llm_stats["new_pairs"] += 1
            cache[c["request_id"]] = {
                "measure1": c["measure1"],
                "measure2": c["measure2"],
                "result": llm_result,
                "model": GROQ_MODEL,
                "timestamp": time.time(),
                "guard_rejected": False,
            }
        save_json(LLM_CACHE_FILE, cache)
        processed_after = (
            llm_stats["cached_pairs"]
            + llm_stats["new_pairs"]
            + llm_stats["guard_rejected"]
        )
        print(
            f"\r[LLM] batch={batch_number}/{total_batches} | "
            f"calls={llm_stats['calls']} | "
            f"new={llm_stats['new_pairs']} | "
            f"cached={llm_stats['cached_pairs']} | "
            f"guard_rejected={llm_stats['guard_rejected']} | "
            f"processed={processed_after}/{total_llm_pairs}",
            end="",
            flush=True,
        )
        time.sleep(WAITING_TIME)
    print()
    print(
        f"[LLM] finished | calls={llm_stats['calls']} | "
        f"new={llm_stats['new_pairs']} | "
        f"cached={llm_stats['cached_pairs']} | "
        f"guard_rejected={llm_stats['guard_rejected']} | "
        f"daily_limit={llm_stats['daily_limit_reached']}"
    )

# ============================================================
# FINAL RELATIONSHIPS
# ============================================================
relationships = []
for c in auto_relationships:
    relationships.append({
        "request_id": c["request_id"],
        "measure1": c["measure1"],
        "measure2": c["measure2"],
        "relation": c["auto_relation"],
        "related": True,
        "confidence": round(float(c["auto_confidence"]), 4),
        "embedding_similarity": round(c["embedding_similarity"], 4),
        "semantic_score": round(c["semantic_score"], 4),
        "lexical_similarity": round(c["lexical_similarity"], 4),
        "token_overlap": round(c["token_overlap"], 4),
        "exact_signature1": sorted(exact_signatures[c["index1"]]),
        "exact_signature2": sorted(exact_signatures[c["index2"]]),
        "domain1": c.get("domain1"),
        "domain2": c.get("domain2"),
        "cross_domain": c.get("cross_domain", False),
        "evidence": c.get("auto_evidence", {}),
        "method": c["auto_method"],
        "llm": False,
        "llm_cached": False,
    })
accepted_llm_ids = set()
for c in llm_results:
    result = c.get("llm_result") or {}
    relation = result.get("relation")
    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if relation not in RELATIONS:
        continue
    if confidence < LLM_CONFIDENCE_THRESHOLD:
        llm_stats["below_confidence"] += 1
        continue
    accepted_llm_ids.add(c["request_id"])
    evidence = {}
    if ADD_EXPLANATION:
        evidence["explanation"] = result.get("explanation", "")
    relationships.append({
        "request_id": c["request_id"],
        "measure1": c["measure1"],
        "measure2": c["measure2"],
        "relation": relation,
        "related": True,
        "confidence": round(confidence, 4),
        "embedding_similarity": round(c["embedding_similarity"], 4),
        "semantic_score": round(c["semantic_score"], 4),
        "lexical_similarity": round(c["lexical_similarity"], 4),
        "token_overlap": round(c["token_overlap"], 4),
        "exact_signature1": sorted(exact_signatures[c["index1"]]),
        "exact_signature2": sorted(exact_signatures[c["index2"]]),
        "domain1": c.get("domain1"),
        "domain2": c.get("domain2"),
        "cross_domain": c.get("cross_domain", False),
        "evidence": evidence,
        "method": "llm_classification",
        "llm": True,
        "llm_cached": bool(c.get("llm_cached", False)),
    })

# ============================================================
# FINAL UNRESOLVED
# ============================================================
final_unresolved = []
auto_ids = {c["request_id"] for c in auto_relationships}
for c in semantic_candidates:
    request = c["request_id"]
    if request in auto_ids or request in accepted_llm_ids:
        continue
    reason = "unresolved"
    if c.get("guard_rejected"):
        reason = "llm_guard_rejected"
    elif c.get("llm_result") is not None:
        try:
            confidence = float(c["llm_result"].get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < LLM_CONFIDENCE_THRESHOLD:
            reason = "llm_below_confidence"
        else:
            reason = "llm_not_accepted"
    elif USE_LLM and llm_stats["daily_limit_reached"]:
        reason = "llm_daily_limit_not_processed"
    elif not USE_LLM:
        reason = "llm_disabled"
    elif c.get("auto_relation") == "related":
        reason = "generic_related_requires_llm"
    else:
        reason = "no_safe_rule"
    final_unresolved.append({
        "request_id": request,
        "measure1": c["measure1"],
        "measure2": c["measure2"],
        "reason": reason,
        "auto_relation": c.get("auto_relation"),
        "auto_confidence": c.get("auto_confidence"),
        "llm_result": c.get("llm_result"),
        "embedding_similarity": round(c["embedding_similarity"], 4),
        "semantic_score": round(c["semantic_score"], 4),
        "lexical_similarity": round(c["lexical_similarity"], 4),
        "token_overlap": round(c["token_overlap"], 4),
        "domain1": c.get("domain1"),
        "domain2": c.get("domain2"),
        "cross_domain": c.get("cross_domain", False),
    })

# ============================================================
# REMOVE DUPLICATES
# ============================================================
unique = {}
for relationship in relationships:
    rid = relationship["request_id"]
    if (
        rid not in unique
        or relationship["confidence"] > unique[rid]["confidence"]
    ):
        unique[rid] = relationship
relationships = list(unique.values())
relationships.sort(
    key=lambda x: (x["relation"], -x["confidence"], x["measure1"], x["measure2"])
)

# ============================================================
# OUTPUT STRUCTURES
# ============================================================
candidate_output = [
    {
        "request_id": c["request_id"],
        "measure1": c["measure1"],
        "measure2": c["measure2"],
        "domain1": c.get("domain1"),
        "domain2": c.get("domain2"),
        "same_domain": c.get("same_domain", False),
        "cross_domain": c.get("cross_domain", False),
        "retrieval_sources": sorted(set(c.get("retrieval_sources", []))),
        "embedding_similarity": round(c["embedding_similarity"], 4),
        "semantic_score": round(c["semantic_score"], 4),
        "lexical_similarity": round(c["lexical_similarity"], 4),
        "token_overlap": round(c["token_overlap"], 4),
        "token_containment": round(c["token_containment"], 4),
        "sequence_similarity": round(c["sequence_similarity"], 4),
        "decision": c["decision"],
    }
    for c in candidates
]
nodes = sorted({
    measure
    for relationship in relationships
    for measure in (relationship["measure1"], relationship["measure2"])
})
node_id = {measure: index for index, measure in enumerate(nodes)}
graph = {
    "nodes": [
        {
            "id": node_id[measure],
            "label": measure,
            "domain": domain_mapping.get(measure),
        }
        for measure in nodes
    ],
    "edges": [
        {
            "source": node_id[relationship["measure1"]],
            "target": node_id[relationship["measure2"]],
            "relation": relationship["relation"],
            "confidence": relationship["confidence"],
            "method": relationship["method"],
        }
        for relationship in relationships
    ],
}
relation_counts = Counter(r["relation"] for r in relationships)
method_counts = Counter(r["method"] for r in relationships)
unresolved_reason_counts = Counter(r["reason"] for r in final_unresolved)
stats = {
    "configuration": {
        "top_k_neighbors": TOP_K_NEIGHBORS,
        "min_embedding_similarity": MIN_EMBEDDING_SIMILARITY,
        "semantic_candidate_threshold": SEMANTIC_CANDIDATE_THRESHOLD,
        "lexical_k_neighbors": LEXICAL_K_NEIGHBORS,
        "min_tfidf_similarity": MIN_TFIDF_SIMILARITY,
        "cross_domain_min_similarity": CROSS_DOMAIN_MIN_SIMILARITY,
        "use_domain_information": USE_DOMAIN_INFORMATION,
        "domain_filter_mode": "hard_for_embedding_soft_for_tfidf",
        "cross_domain_soft_threshold": CROSS_DOMAIN_SOFT_THRESHOLD,
        "cross_domain_score_penalty": CROSS_DOMAIN_SCORE_PENALTY,
        "llm_confidence_threshold": LLM_CONFIDENCE_THRESHOLD,
        "generic_related_automatic": False,
    },
    "counts": {
        "measures": n,
        "all_candidates": len(candidates),
        "embedding_semantic_candidates": len(semantic_candidates),
        "retrieval_only_candidates": len(retrieval_only_candidates),
        "embedding_retrieved": retrieval_counts.get("embedding_knn", 0),
        "tfidf_retrieved": retrieval_counts.get("tfidf_knn", 0),
        "retrieved_by_both": both_sources,
        "resolved_automatically": len(auto_relationships),
        "sent_to_llm": len(llm_candidates),
        "candidate_only_after_rules": len(candidate_only),
        "llm_results_available": len(llm_results),
        "final_relationships": len(relationships),
        "unresolved": len(final_unresolved),
    },
    "relation_counts": dict(relation_counts),
    "method_counts": dict(method_counts),
    "unresolved_reason_counts": dict(unresolved_reason_counts),
    "rule_counts": dict(rule_counts),
    "llm_stats": llm_stats,
}

# ============================================================
# SAVE OUTPUTS
# ============================================================
save_json(OUTPUT_RELATIONSHIPS, relationships)
save_json(OUTPUT_CANDIDATES, candidate_output)
save_json(OUTPUT_UNRESOLVED, final_unresolved)
save_json(OUTPUT_GRAPH, graph)
save_json(OUTPUT_STATS, stats)

# ============================================================
# DEBUG SUMMARY
# ============================================================
if DEBUG:
    print("\n================ FINAL STATUS ================")
    print(f"Measures                 : {n}")
    print(f"Candidates               : {len(candidates)}")
    print(f"Semantic candidates      : {len(semantic_candidates)}")
    print(f"Retrieval-only candidates: {len(retrieval_only_candidates)}")
    print(f"Resolved by rules        : {len(auto_relationships)}")
    print(f"Sent to LLM             : {len(llm_candidates)}")
    print(f"LLM results available    : {len(llm_results)}")
    print(f"Final relationships      : {len(relationships)}")
    print(f"Unresolved               : {len(final_unresolved)}")
    print("\n================ RELATIONS ================")
    total_relationships = sum(relation_counts.values())
    for relation in sorted(RELATIONS):
        count = relation_counts.get(relation, 0)
        share = count / total_relationships if total_relationships else 0.0
        flag = ""
        if count == 0:
            flag = "  <-- no final examples"
        elif share > BALANCE_DOMINANT_SHARE:
            flag = "  <-- dominant relation; inspect thresholds"
        print(f"{relation:20s}: {count:6d} ({share:5.1%}){flag}")
    print("\n================ METHODS ================")
    for method, count in method_counts.most_common():
        print(f"{method:48s}: {count}")
    print("\n================ UNRESOLVED ================")
    for reason, count in unresolved_reason_counts.most_common():
        print(f"{reason:40s}: {count}")
    print("\nOutputs:")
    for path in (
        OUTPUT_RELATIONSHIPS,
        OUTPUT_CANDIDATES,
        OUTPUT_UNRESOLVED,
        OUTPUT_STATS,
        OUTPUT_GRAPH,
        OUTPUT_LEARNED_PATTERNS,
    ):
        print(f"  {path}")
print("\nDone.")
