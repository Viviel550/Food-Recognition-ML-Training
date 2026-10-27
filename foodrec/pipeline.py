"""Gotowy system: zdjęcie -> danie, składniki, wartości odżywcze."""
from __future__ import annotations

import joblib
import numpy as np

from . import config, data
from .foodnet import FoodNetEstimator


class FoodRecognizer:
    """Łączy wytrenowaną sieć z nazwami klas i przetwarzaniem obrazów."""

    def __init__(self, estimator: FoodNetEstimator, dish_names, ingredient_names, target_names,
                 img_size: int = config.IMG_SIZE):
        self.estimator = estimator
        self.dish_names = list(dish_names)
        self.ingredient_names = list(ingredient_names)
        self.target_names = list(target_names)
        self.img_size = img_size

    def predict(self, images, top_k: int = 3) -> list[dict]:
        """`images`: lista ścieżek lub obiektów PIL.Image."""
        X = np.stack([data.to_array(im, self.img_size) for im in images])
        D = self.estimator.predict_proba(X)
        I = self.estimator.predict_ingredient_proba(X)
        N = self.estimator.predict_nutrition(X)
        thr = self.estimator.ingredient_threshold_

        results = []
        for d, i, n in zip(D, I, N):
            top = np.argsort(-d)[:top_k]
            chosen = np.where(i >= thr)[0]
            if len(chosen) == 0:
                chosen = np.array([int(np.argmax(i))])
            chosen = chosen[np.argsort(-i[chosen])]
            results.append({
                "dish": [(self.dish_names[j], float(d[j])) for j in top],
                "ingredients": [(self.ingredient_names[j], float(i[j])) for j in chosen],
                "nutrition": {name: round(float(v), 1) for name, v in zip(self.target_names, n)},
            })
        return results

    def save(self, path) -> None:
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path) -> "FoodRecognizer":
        return joblib.load(path)
