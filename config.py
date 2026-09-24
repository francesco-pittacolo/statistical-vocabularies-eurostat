from pathlib import Path


# Dataset to process: "2000" or "7605"
DATASET = "2000"

if DATASET not in {"2000", "7605"}:
    raise ValueError("DATASET must be '2000' or '7605'")


ROOT_DIR = Path(__file__).resolve().parent

TABLE_DIR = ROOT_DIR / "data" / f"eurostat_{DATASET}_tables"
OUTPUT_DIR = ROOT_DIR / "results" / "step_outputs" / DATASET
REFERENCE_DIR = ROOT_DIR / "data" / "reference"
CACHE_DIR = ROOT_DIR / "data" / "cache"