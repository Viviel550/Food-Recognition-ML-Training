"""Wczytywanie, czyszczenie, pobieranie obrazów i podział zbioru MM-Food-100K."""
from __future__ import annotations

import ast
import hashlib
import io
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from . import config

# ----------------------------------------------------------------------------
# Parsowanie pól tekstowych
# ----------------------------------------------------------------------------


def _as_obj(v):
    """Zwraca obiekt Pythona z wartości, która może być listą, dict-em lub stringiem JSON."""
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, dict)):
        return v
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(v)
            except Exception:
                continue
    return None


def normalize_text(s) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^\w\s-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _singular(word: str) -> str:
    """Prosta singularyzacja angielskiego rzeczownika (egg/eggs, tomato/tomatoes, berry/berries)."""
    if len(word) <= 3 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("oes", "ches", "shes", "xes")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def normalize_ingredient(s) -> str:
    """Małe litery, bez znaków specjalnych, ostatnie słowo w liczbie pojedynczej."""
    n = normalize_text(s)
    if not n:
        return n
    parts = n.split()
    parts[-1] = _singular(parts[-1])
    return " ".join(parts)


def parse_ingredients(v) -> list[str]:
    obj = _as_obj(v)
    if obj is None and isinstance(v, str):
        obj = v.split(",")
    if isinstance(obj, str):
        obj = [obj]
    if not isinstance(obj, (list, tuple)):
        return []
    out, seen = [], set()
    for x in obj:
        n = normalize_ingredient(x)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def parse_nutrition(v) -> dict[str, float]:
    obj = _as_obj(v)
    res = {c: np.nan for c in config.NUTRITION_COLS}
    if isinstance(obj, dict):
        for c in config.NUTRITION_COLS:
            try:
                res[c] = float(obj.get(c))
            except (TypeError, ValueError):
                pass
    return res


_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(kg|g|grams?|ml|l)\b", re.IGNORECASE)
_UNIT_TO_G = {"kg": 1000.0, "g": 1.0, "gram": 1.0, "grams": 1.0, "ml": 1.0, "l": 1000.0}  # 1 ml ~ 1 g


def parse_total_weight(v) -> float:
    """Sumuje wagi (g; ml traktowane jak g) z pola portion_size, np. ["eggs:100g", "milk:250ml"].

    Pozycje bez jednostki wagi/objętości (np. "10 pieces") są pomijane; jeśli nie ma żadnej – NaN.
    """
    obj = _as_obj(v)
    if isinstance(obj, str):
        obj = [obj]
    if isinstance(obj, dict):
        obj = [f"{k}:{val}" for k, val in obj.items()]
    if not isinstance(obj, (list, tuple)):
        return np.nan
    total, found = 0.0, False
    for item in obj:
        # bierzemy pierwszą wartość z jednostką wagi/objętości w danej pozycji
        m = _WEIGHT_RE.search(str(item))
        if m:
            total += float(m.group(1)) * _UNIT_TO_G[m.group(2).lower()]
            found = True
    return total if found else np.nan


# ----------------------------------------------------------------------------
# Wczytanie i przygotowanie tabeli
# ----------------------------------------------------------------------------


def load_raw() -> pd.DataFrame:
    """Pobiera tabelę (bez obrazów, tylko URL-e) z Hugging Face Hub."""
    from datasets import load_dataset

    ds = load_dataset(config.HF_DATASET, split="train")
    return ds.to_pandas()


def build_table(
    raw: pd.DataFrame,
    top_dishes: int = 100,
    top_ingredients: int = 150,
    min_per_class: int = 30,
    max_per_class: int = 400,
    min_food_prob: float = 0.5,
    camera_only: bool = False,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Czyszczenie, normalizacja i ograniczenie do najczęstszych klas/składników."""
    df = raw.copy()
    df = df.dropna(subset=["image_url", "dish_name"]).drop_duplicates("image_url")

    if "food_prob" in df.columns:
        df = df[df["food_prob"].fillna(0) >= min_food_prob]
    if camera_only and {"camera_or_phone_prob", "online_download_prob"} <= set(df.columns):
        df = df[df["camera_or_phone_prob"] >= df["online_download_prob"]]

    df["dish"] = df["dish_name"].map(normalize_text)
    df = df[df["dish"] != ""]

    # Najczęstsze dania (długi ogon odrzucamy).
    counts = df["dish"].value_counts()
    keep = counts[counts >= min_per_class].index[:top_dishes]
    df = df[df["dish"].isin(keep)]

    # Ograniczenie liczby próbek na klasę (mniejsze niezbalansowanie i rozmiar pobierania).
    df = df.sample(frac=1.0, random_state=seed)
    df = df.groupby("dish", group_keys=False).head(max_per_class)

    # Składniki: normalizacja i słownik top-K.
    df["ingredients"] = df["ingredients"].map(parse_ingredients)
    ing_counts = Counter(i for lst in df["ingredients"] for i in lst)
    vocab = {i for i, _ in ing_counts.most_common(top_ingredients)}
    df["ingredients"] = df["ingredients"].map(lambda lst: [i for i in lst if i in vocab])
    df = df[df["ingredients"].map(len) > 0]

    # Cele regresji.
    nutr = pd.DataFrame(list(df["nutritional_profile"].map(parse_nutrition)), index=df.index)
    for c in config.NUTRITION_COLS:
        df[c] = nutr[c]
    df["total_weight_g"] = df["portion_size"].map(parse_total_weight)
    df = df.dropna(subset=config.NUTRITION_COLS)
    df = df[(df["calories_kcal"] > 0) & (df["calories_kcal"] < 5000)]

    cols = ["image_url", "dish", "ingredients", *config.TARGET_COLS]
    for extra in ("food_type", "cooking_method"):
        if extra in df.columns:
            cols.append(extra)
    return df[cols].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Pobieranie obrazów
# ----------------------------------------------------------------------------


def _image_path(url: str, out_dir: Path) -> Path:
    return out_dir / (hashlib.md5(url.encode("utf-8")).hexdigest() + ".jpg")


def _download_one(args) -> bool:
    url, path, max_side = args
    if path.exists():
        return True
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        img.thumbnail((max_side, max_side))
        img.save(path, "JPEG", quality=90)
        return True
    except Exception:
        return False


def download_images(
    df: pd.DataFrame, out_dir: Path = config.IMAGES_DIR, max_side: int = 384, workers: int = 16
) -> pd.DataFrame:
    """Pobiera obrazy lokalnie; zwraca tylko wiersze z poprawnie pobranym plikiem."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [_image_path(u, out_dir) for u in df["image_url"]]
    jobs = [(u, p, max_side) for u, p in zip(df["image_url"], paths)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        ok = list(tqdm(ex.map(_download_one, jobs), total=len(jobs), desc="Pobieranie obrazów"))
    df = df.copy()
    df["image_path"] = [str(p) for p in paths]
    n_bad = len(df) - sum(ok)
    print(f"Pobrano {sum(ok)} / {len(df)} obrazów (martwe linki: {n_bad}).")
    return df[np.array(ok)].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Podział train / val / test
# ----------------------------------------------------------------------------


def assign_split(
    df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, seed: int = config.SEED,
    min_per_class: int = 10,
) -> pd.DataFrame:
    """Dodaje kolumnę `split` (stratyfikowany podział według dania)."""
    counts = df["dish"].value_counts()
    df = df[df["dish"].isin(counts[counts >= min_per_class].index)].reset_index(drop=True)
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=val_size + test_size, stratify=df["dish"], random_state=seed)
    va, te = train_test_split(
        tmp, test_size=test_size / (val_size + test_size), stratify=df["dish"].iloc[tmp], random_state=seed
    )
    df["split"] = "train"
    df.loc[va, "split"] = "val"
    df.loc[te, "split"] = "test"
    return df


# ----------------------------------------------------------------------------
# Obrazy -> tablice uint8 (cache w pamięci/dysku)
# ----------------------------------------------------------------------------


def to_array(img, size: int = config.IMG_SIZE) -> np.ndarray:
    """Ścieżka lub PIL.Image -> tablica uint8 (size, size, 3): skalowanie krótszego boku + kadr środkowy."""
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return np.asarray(img.crop((left, top, left + size, top + size)), dtype=np.uint8)


def load_or_build_array(paths: list[str], size: int = config.IMG_SIZE) -> np.ndarray:
    """Zwraca tablicę (N, size, size, 3) uint8; liczy ją raz i zapisuje w artifacts/."""
    f = config.ARTIFACTS / f"images_{size}.npz"
    key = hashlib.md5("\n".join(paths).encode("utf-8")).hexdigest()
    if f.exists():
        d = np.load(f)
        if str(d["key"]) == key:
            return d["images"]
    with ThreadPoolExecutor(max_workers=8) as ex:
        arr = np.stack(list(tqdm(ex.map(lambda p: to_array(p, size), paths), total=len(paths),
                                  desc="Przygotowanie tablicy obrazów")))
    np.savez_compressed(f, images=arr, key=key)
    return arr


# ----------------------------------------------------------------------------
# Pomocnicze funkcje dla etapów modelowania
# ----------------------------------------------------------------------------


def load_metadata() -> pd.DataFrame:
    if not config.METADATA_PATH.exists():
        raise FileNotFoundError("Brak metadata.parquet. Uruchom najpierw scripts/01_prepare_data.py")
    df = pd.read_parquet(config.METADATA_PATH)
    df["ingredients"] = df["ingredients"].map(list)
    return df


def regression_targets(df: pd.DataFrame) -> np.ndarray:
    """Macierz celów regresji (brakujące wartości = NaN, są maskowane w funkcji straty)."""
    return df[config.TARGET_COLS].to_numpy(dtype=float)
