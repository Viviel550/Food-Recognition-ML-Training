"""Etap 1: pobranie MM-Food-100K z Hugging Face, czyszczenie, pobranie obrazów, podział danych.

Przykład:
    python scripts/01_prepare_data.py --top-dishes 100 --max-per-class 300
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foodrec import config, data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-dishes", type=int, default=100, help="liczba najczęstszych dań (klas)")
    ap.add_argument("--top-ingredients", type=int, default=150, help="rozmiar słownika składników")
    ap.add_argument("--min-per-class", type=int, default=30)
    ap.add_argument("--max-per-class", type=int, default=400)
    ap.add_argument("--min-food-prob", type=float, default=0.5)
    ap.add_argument("--camera-only", action="store_true",
                    help="tylko zdjęcia zrobione aparatem (camera_or_phone_prob >= online_download_prob)")
    ap.add_argument("--max-side", type=int, default=384, help="maks. dłuższy bok zapisywanego obrazu")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    config.ensure_dirs()
    print("Wczytywanie danych z Hugging Face...")
    raw = data.load_raw()
    print(f"Surowa tabela: {len(raw)} wierszy, kolumny: {list(raw.columns)}")

    df = data.build_table(
        raw, top_dishes=args.top_dishes, top_ingredients=args.top_ingredients,
        min_per_class=args.min_per_class, max_per_class=args.max_per_class,
        min_food_prob=args.min_food_prob, camera_only=args.camera_only,
    )
    print(f"Po czyszczeniu: {len(df)} wierszy, {df['dish'].nunique()} dań")

    df = data.download_images(df, max_side=args.max_side, workers=args.workers)
    df = data.assign_split(df)
    df.to_parquet(config.METADATA_PATH, index=False)

    print(f"Zapisano {config.METADATA_PATH}")
    print(df["split"].value_counts().to_string())
    print("Przykładowe dania:", ", ".join(df["dish"].value_counts().index[:10]))


if __name__ == "__main__":
    main()
