"""NumPy GRU with analytic full-BPTT gradients (H2 sequential state).

Single-layer GRU, one sequence at a time (histories here are <= ~200 events),
gradients accumulated across sequences per optimization step (full-batch Adam).

    r = sigmoid(x @ Wxr + h @ Whr + br)
    g = sigmoid(x @ Wxg + h @ Whg + bg)          (update gate)
    c = tanh(x @ Wxc + (r * h) @ Whc + bc)
    h' = (1 - g) * h + g * c

Gradients are analytic (no autograd); tests/test_h2.py checks them against
central finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _glorot(rng: np.random.Generator, n_in: int, n_out: int) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(2.0 / (n_in + n_out)), size=(n_in, n_out))


class Adam:
    """Minimal Adam for a dict of ndarray parameters (shared by GRU and world model)."""

    def __init__(self, params: dict[str, np.ndarray], lr: float, l2: float = 0.0,
                 clip_norm: float = 5.0):
        self.params = params
        self.lr, self.l2, self.clip_norm = lr, l2, clip_norm
        self._m = {k: np.zeros_like(v) for k, v in params.items()}
        self._v = {k: np.zeros_like(v) for k, v in params.items()}
        self._t = 0

    def step(self, grads: dict[str, np.ndarray]) -> None:
        gnorm = float(np.sqrt(sum(float((grads[k] ** 2).sum()) for k in grads)))
        scale = min(1.0, self.clip_norm / (gnorm + 1e-12))
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, g in grads.items():
            g = (g + self.l2 * self.params[k]) * scale
            self._m[k] = b1 * self._m[k] + (1.0 - b1) * g
            self._v[k] = b2 * self._v[k] + (1.0 - b2) * g**2
            mhat = self._m[k] / (1.0 - b1**self._t)
            vhat = self._v[k] / (1.0 - b2**self._t)
            self.params[k] = self.params[k] - self.lr * mhat / (np.sqrt(vhat) + eps)


@dataclass
class GRUConfig:
    n_in: int
    n_hidden: int
    lr: float = 1e-2
    l2: float = 1e-5
    n_epochs: int = 150
    clip_norm: float = 5.0
    seed: int = 0


class GRU:
    """GRU cell; `forward` returns hidden states and a per-step cache for `backward`."""

    def __init__(self, cfg: GRUConfig, rng: np.random.Generator):
        self.cfg = cfg
        dz, dx = cfg.n_hidden, cfg.n_in
        self.params: dict[str, np.ndarray] = {
            "Wxr": _glorot(rng, dx, dz), "Whr": _glorot(rng, dz, dz), "br": np.zeros(dz),
            "Wxg": _glorot(rng, dx, dz), "Whg": _glorot(rng, dz, dz),
            "bg": np.full(dz, -0.5),  # start near h' = h so early gradients flow
            "Wxc": _glorot(rng, dx, dz), "Whc": _glorot(rng, dz, dz), "bc": np.zeros(dz),
        }
        self.adam = Adam(self.params, cfg.lr, cfg.l2, cfg.clip_norm)

    def forward(self, X: np.ndarray, h0: np.ndarray):
        p = self.params
        cache = []
        h = h0
        for t in range(X.shape[0]):
            x = X[t]
            r = sigmoid(x @ p["Wxr"] + h @ p["Whr"] + p["br"])
            g = sigmoid(x @ p["Wxg"] + h @ p["Whg"] + p["bg"])
            c = np.tanh(x @ p["Wxc"] + (r * h) @ p["Whc"] + p["bc"])
            h2 = (1.0 - g) * h + g * c
            cache.append({"x": x, "h": h, "r": r, "g": g, "c": c, "h2": h2})
            h = h2
        return cache

    def backward(self, cache: list[dict], dH: np.ndarray) -> dict[str, np.ndarray]:
        """dH: (T, dz) gradient of the loss w.r.t. each step's output state."""
        p = self.params
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        dh_next = np.zeros(self.cfg.n_hidden)
        for t in reversed(range(len(cache))):
            st = cache[t]
            dh2 = dH[t] + dh_next
            h, r, g, c, x = st["h"], st["r"], st["g"], st["c"], st["x"]
            dg = dh2 * (c - h)
            dc = dh2 * g
            dh_direct = dh2 * (1.0 - g)
            dhc = dc * (1.0 - c**2)
            grads["Whc"] += np.outer(r * h, dhc)
            grads["bc"] += dhc
            drh = dhc @ p["Whc"].T
            dr = drh * h
            dh_r = drh * r
            dxr = dr * r * (1.0 - r)
            dxg = dg * g * (1.0 - g)
            grads["Wxr"] += np.outer(x, dxr)
            grads["Whr"] += np.outer(h, dxr)
            grads["br"] += dxr
            grads["Wxg"] += np.outer(x, dxg)
            grads["Whg"] += np.outer(h, dxg)
            grads["bg"] += dxg
            grads["Wxc"] += np.outer(x, dhc)
            dx = dxr @ p["Wxr"].T + dxg @ p["Wxg"].T + dhc @ p["Wxc"].T
            dh_next = dh_direct + dh_r + dxr @ p["Whr"].T + dxg @ p["Whg"].T
            st["dx"] = dx
        return grads

    def step(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        p = self.params
        r = sigmoid(x @ p["Wxr"] + h @ p["Whr"] + p["br"])
        g = sigmoid(x @ p["Wxg"] + h @ p["Whg"] + p["bg"])
        c = np.tanh(x @ p["Wxc"] + (r * h) @ p["Whc"] + p["bc"])
        return (1.0 - g) * h + g * c

    def apply_grads(self, grads: dict[str, np.ndarray]) -> None:
        self.adam.step(grads)
