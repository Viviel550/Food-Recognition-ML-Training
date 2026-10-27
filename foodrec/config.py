"""Stałe i ścieżki projektu."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
IMAGES_DIR = ARTIFACTS / "images"
FEATURES_DIR = ARTIFACTS / "features"
RESULTS_DIR = ARTIFACTS / "results"
METADATA_PATH = ARTIFACTS / "metadata.parquet"
MODEL_PATH = ARTIFACTS / "food_recognizer.joblib"

HF_DATASET = "Humanbased-AI/MM-Food-100K"
IMG_SIZE = 128  # rozmiar obrazów wejściowych sieci (piksele)
SEED = 42

NUTRITION_COLS = ["calories_kcal", "protein_g", "fat_g", "carbohydrate_g"]
# Dodatkowy cel regresji: suma wag składników z kolumny `portion_size`.
TARGET_COLS = NUTRITION_COLS + ["total_weight_g"]


def ensure_dirs() -> None:
    for d in (ARTIFACTS, IMAGES_DIR, FEATURES_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
