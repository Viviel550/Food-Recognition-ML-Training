"""Etap 3: prototyp systemu – aplikacja Gradio lub tryb wiersza poleceń.

    python scripts/03_app.py                       # interfejs w przeglądarce
    python scripts/03_app.py --image zdjecie.jpg   # predykcja z linii poleceń
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foodrec import config  # noqa: E402
from foodrec.pipeline import FoodRecognizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(config.MODEL_PATH))
    ap.add_argument("--image", default=None, help="ścieżka do zdjęcia (tryb CLI)")
    ap.add_argument("--share", action="store_true", help="publiczny link Gradio")
    args = ap.parse_args()

    system = FoodRecognizer.load(args.model)

    if args.image:
        print(json.dumps(system.predict([args.image])[0], indent=2, ensure_ascii=False))
        return

    import gradio as gr

    def run(img):
        if img is None:
            return {}, "", {}
        r = system.predict([img])[0]
        return (dict(r["dish"]),
                ", ".join(f"{n} ({p:.0%})" for n, p in r["ingredients"]),
                r["nutrition"])

    gr.Interface(
        fn=run,
        inputs=gr.Image(type="pil", label="Zdjęcie posiłku"),
        outputs=[gr.Label(num_top_classes=3, label="Danie"),
                 gr.Textbox(label="Rozpoznane składniki"),
                 gr.JSON(label="Szacunkowe wartości odżywcze")],
        title="Rozpoznawanie jedzenia i składników",
        description="Własna sieć CNN trenowana od zera na MM-Food-100K. Wartości odżywcze są szacunkowe.",
        flagging_mode="never",
    ).launch(share=args.share)


if __name__ == "__main__":
    main()
