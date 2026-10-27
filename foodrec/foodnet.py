"""Własny model: wielozadaniowa sieć konwolucyjna uczona od zera (bez pretrenowanych wag).

Jedna sieć (rdzeń ResNet-podobny) ma trzy głowice:
  * danie            – klasyfikacja wieloklasowa (softmax),
  * składniki        – klasyfikacja wieloetykietowa (sigmoid, próg dobierany na zbiorze walidacyjnym),
  * wartości odżywcze – regresja (kalorie, białko, tłuszcz, węglowodany, waga porcji).

`FoodNetEstimator` jest estymatorem zgodnym z API scikit-learn (BaseEstimator, fit/predict/predict_proba).
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from torch import nn


# ----------------------------------------------------------------------------
# Architektura
# ----------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Blok rezydualny: conv3x3-BN-ReLU-conv3x3-BN + połączenie skrótowe."""

    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.short = None
        if stride != 1 or c_in != c_out:
            self.short = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + (x if self.short is None else self.short(x)))


class FoodNet(nn.Module):
    def __init__(self, n_dishes: int, n_ingredients: int, n_targets: int, width: int = 32,
                 dropout: float = 0.3):
        super().__init__()
        w = width
        layers = [nn.Conv2d(3, w, 3, 2, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
        c = w
        for c_out, stride in [(w, 1), (2 * w, 2), (4 * w, 2), (8 * w, 2)]:
            layers += [ResBlock(c, c_out, stride), ResBlock(c_out, c_out, 1)]
            c = c_out
        self.features = nn.Sequential(*layers)
        self.neck = nn.Sequential(nn.Linear(c, c), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.dish_head = nn.Linear(c, n_dishes)
        self.ingredient_head = nn.Linear(c, n_ingredients)
        self.nutrition_head = nn.Linear(c, n_targets)

    def forward(self, x):
        h = self.neck(self.features(x).mean(dim=(2, 3)))
        return self.dish_head(h), self.ingredient_head(h), self.nutrition_head(h)


# ----------------------------------------------------------------------------
# Augmentacja (na tensorach, wsadowo): skala, obrót, odbicie, przesunięcie, jasność/kontrast
# ----------------------------------------------------------------------------


def augment(x: torch.Tensor) -> torch.Tensor:
    B, dev = x.shape[0], x.device
    s = torch.empty(B, device=dev).uniform_(0.7, 1.0)
    ang = torch.empty(B, device=dev).uniform_(-0.17, 0.17)
    flip = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1
    tx = (torch.rand(B, device=dev) * 2 - 1) * (1 - s)
    ty = (torch.rand(B, device=dev) * 2 - 1) * (1 - s)
    cos, sin = torch.cos(ang) * s, torch.sin(ang) * s
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1),
                         torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    contrast = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
    bright = torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.15, 0.15)
    return x * contrast + bright


# ----------------------------------------------------------------------------
# Estymator zgodny ze scikit-learn
# ----------------------------------------------------------------------------


class FoodNetEstimator(BaseEstimator):
    """Wielozadaniowa sieć CNN trenowana od zera.

    fit(X, y, eval_set=None)
        X : uint8 (N, H, W, 3)
        y : dict z kluczami "dish" (int, N), "ingredients" (0/1, N x K), "nutrition" (N x T, NaN dozwolone)
        eval_set : krotka (X_val, y_val) – do wczesnego zatrzymania i doboru progu składników.
    """

    def __init__(self, width: int = 32, epochs: int = 30, batch_size: int = 64, lr: float = 2e-3,
                 weight_decay: float = 5e-2, dropout: float = 0.3, label_smoothing: float = 0.1,
                 loss_weights=(1.0, 5.0, 1.0), patience: int = 6, device: str | None = None,
                 random_state: int = 42, verbose: bool = True):
        self.width = width
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        self.loss_weights = loss_weights
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # --- pomocnicze -------------------------------------------------------
    def _dev(self) -> torch.device:
        return torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    @staticmethod
    def _tensor(xb: np.ndarray, dev) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(xb)).to(dev).permute(0, 3, 1, 2).float()
        return t / 127.5 - 1.0

    def _prep_targets(self, y):
        yd = torch.as_tensor(np.asarray(y["dish"]), dtype=torch.long)
        Yi = torch.as_tensor(np.asarray(y["ingredients"]), dtype=torch.float32)
        Z = (np.log1p(np.asarray(y["nutrition"], dtype=float)) - self.nut_mean_) / self.nut_std_
        mask = torch.as_tensor(np.isfinite(Z), dtype=torch.float32)
        Z = torch.as_tensor(np.nan_to_num(Z), dtype=torch.float32)
        return yd, Yi, Z, mask

    def _loss(self, out, tgt, smoothing: float):
        d, g, r = out
        yd, Yi, Z, mask = tgt
        w = self.loss_weights
        l_d = F.cross_entropy(d, yd, label_smoothing=smoothing)
        l_i = F.binary_cross_entropy_with_logits(g, Yi)
        l_r = (F.smooth_l1_loss(r, Z, reduction="none") * mask).sum() / mask.sum().clamp(min=1)
        return w[0] * l_d + w[1] * l_i + w[2] * l_r

    @torch.no_grad()
    def _forward(self, X, batch_size: int = 128):
        dev = self._dev()
        self.model_.to(dev).eval()
        outs = [[], [], []]
        for i in range(0, len(X), batch_size):
            o = self.model_(self._tensor(X[i:i + batch_size], dev))
            for k in range(3):
                outs[k].append(o[k].float().cpu())
        return tuple(torch.cat(o) for o in outs) if outs[0] else tuple(torch.empty(0) for _ in outs)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "model_" in state:
            state["model_"] = self.model_.to("cpu")
        return state

    # --- uczenie ----------------------------------------------------------
    def fit(self, X, y, eval_set=None):
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        dev = self._dev()

        n_dishes = int(np.max(y["dish"])) + 1
        n_ing = np.asarray(y["ingredients"]).shape[1]
        R = np.log1p(np.asarray(y["nutrition"], dtype=float))
        self.nut_mean_ = np.nanmean(R, axis=0)
        self.nut_std_ = np.nanstd(R, axis=0) + 1e-6
        self.classes_ = np.arange(n_dishes)
        self.ingredient_threshold_ = 0.3

        self.model_ = FoodNet(n_dishes, n_ing, R.shape[1], self.width, self.dropout).to(dev)
        tgt = self._prep_targets(y)
        val_tgt = self._prep_targets(eval_set[1]) if eval_set is not None else None

        n = len(X)
        steps = math.ceil(n / self.batch_size)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.epochs * steps)
        use_amp = dev.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        self.history_, best_loss, best_state, bad = [], np.inf, None, 0
        for ep in range(self.epochs):
            t0 = time.time()
            self.model_.train()
            perm = rng.permutation(n)
            run = 0.0
            for i in range(steps):
                idx = perm[i * self.batch_size:(i + 1) * self.batch_size]
                if len(idx) < 2:
                    continue
                xb = augment(self._tensor(X[idx], dev))
                tb = tuple(t[idx].to(dev) for t in tgt)
                with torch.autocast(device_type=dev.type, enabled=use_amp):
                    loss = self._loss(self.model_(xb), tb, self.label_smoothing)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                run += float(loss)
            rec = {"epoch": ep + 1, "train_loss": run / steps}

            if eval_set is not None:
                out = self._forward(eval_set[0])
                vt = tuple(t for t in val_tgt)
                rec["val_loss"] = float(self._loss(out, vt, 0.0))
                rec["val_dish_acc"] = float((out[0].argmax(1) == vt[0]).float().mean())
                crit = rec["val_loss"]
            else:
                crit = rec["train_loss"]
            self.history_.append(rec)
            if self.verbose:
                print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items())
                      + f"  ({time.time() - t0:.0f}s)")

            if crit < best_loss - 1e-4:
                best_loss, bad = crit, 0
                best_state = copy.deepcopy(self.model_.state_dict())
            else:
                bad += 1
                if bad >= self.patience:
                    if self.verbose:
                        print(f"Wczesne zatrzymanie po epoce {ep + 1}")
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        # Dobór progu dla składników (maksymalizacja micro-F1 na zbiorze walidacyjnym).
        if eval_set is not None:
            P = torch.sigmoid(self._forward(eval_set[0])[1]).numpy()
            Yv = np.asarray(eval_set[1]["ingredients"])
            grid = np.linspace(0.05, 0.6, 23)
            f1s = [f1_score(Yv, P >= t, average="micro", zero_division=0) for t in grid]
            self.ingredient_threshold_ = float(grid[int(np.argmax(f1s))])
        self.model_.to("cpu")
        return self

    # --- predykcja ---------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        """Prawdopodobieństwa dań (N, liczba_dań)."""
        return F.softmax(self._forward(X)[0], dim=1).numpy()

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def predict_ingredient_proba(self, X) -> np.ndarray:
        return torch.sigmoid(self._forward(X)[1]).numpy()

    def predict_ingredients(self, X) -> np.ndarray:
        P = self.predict_ingredient_proba(X)
        pred = (P >= self.ingredient_threshold_).astype(int)
        empty = pred.sum(axis=1) == 0  # zawsze co najmniej jeden składnik
        pred[empty, P[empty].argmax(axis=1)] = 1
        return pred

    def predict_nutrition(self, X) -> np.ndarray:
        Z = self._forward(X)[2].numpy()
        return np.clip(np.expm1(Z * self.nut_std_ + self.nut_mean_), 0, None)
