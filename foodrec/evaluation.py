"""Metryki i wizualizacje (matplotlib) – działają na gotowych predykcjach."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    top_k_accuracy_score,
)


def dish_metrics(y, proba, ks=(3, 5)) -> dict:
    pred = proba.argmax(axis=1)
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
    }
    for k in ks:
        if k < proba.shape[1]:
            out[f"top{k}_accuracy"] = top_k_accuracy_score(y, proba, k=k, labels=np.arange(proba.shape[1]))
    return out


def ingredient_metrics(Y, P, pred) -> dict:
    has_pos = Y.sum(axis=0) > 0
    return {
        "micro_f1": f1_score(Y, pred, average="micro", zero_division=0),
        "macro_f1": f1_score(Y, pred, average="macro", zero_division=0),
        "samples_f1": f1_score(Y, pred, average="samples", zero_division=0),
        "jaccard_samples": jaccard_score(Y, pred, average="samples", zero_division=0),
        "hamming_loss": hamming_loss(Y, pred),
        "mAP": average_precision_score(Y[:, has_pos], P[:, has_pos], average="macro"),
    }


def regression_metrics(Y, pred, names) -> dict:
    """Metryki liczone tylko na polach, w których wartość prawdziwa jest znana (nie-NaN)."""
    out = {}
    for j, n in enumerate(names):
        ok = np.isfinite(Y[:, j])
        out[f"{n}_mae"] = mean_absolute_error(Y[ok, j], pred[ok, j])
        out[f"{n}_rmse"] = float(np.sqrt(mean_squared_error(Y[ok, j], pred[ok, j])))
        out[f"{n}_r2"] = r2_score(Y[ok, j], pred[ok, j])
        # punkt odniesienia: stała predykcja = średnia
        out[f"{n}_mae_baseline"] = mean_absolute_error(Y[ok, j], np.full(ok.sum(), Y[ok, j].mean()))
    return out


# ----------------------------------------------------------------------------
# Wykresy
# ----------------------------------------------------------------------------


def plot_history(history: list[dict], path: Path) -> None:
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    if "val_loss" in history[0]:
        ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set_xlabel("epoka")
    ax[0].set_ylabel("strata")
    ax[0].legend()
    if "val_dish_acc" in history[0]:
        ax[1].plot(ep, [h["val_dish_acc"] for h in history])
        ax[1].set_xlabel("epoka")
        ax[1].set_ylabel("accuracy dania (val)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_ingredient_f1(Y, pred, names, path: Path, top: int = 30) -> None:
    order = np.argsort(-Y.sum(axis=0))[:top]
    f1 = [f1_score(Y[:, j], pred[:, j], zero_division=0) for j in order]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh([names[j] for j in order][::-1], f1[::-1])
    ax.set_xlabel("F1")
    ax.set_title(f"F1 dla {top} najczęstszych składników (zbiór testowy)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y, pred, classes, path: Path, top: int = 20) -> None:
    order = np.argsort(-np.bincount(y, minlength=len(classes)))[:top]
    cm = confusion_matrix(y, pred, labels=order, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=[classes[i] for i in order]).plot(
        ax=ax, xticks_rotation=90, colorbar=False, values_format=".1f")
    ax.set_title(f"Macierz pomyłek – {top} najczęstszych dań (znormalizowana)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_regression_scatter(Y, pred, names, path: Path) -> None:
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
    for ax, j, n in zip(np.atleast_1d(axes), range(len(names)), names):
        ok = np.isfinite(Y[:, j])
        ax.scatter(Y[ok, j], pred[ok, j], s=4, alpha=0.3)
        lim = max(Y[ok, j].max(), pred[ok, j].max())
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        ax.set_xlabel(f"{n} (prawda)")
        ax.set_ylabel(f"{n} (predykcja)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
