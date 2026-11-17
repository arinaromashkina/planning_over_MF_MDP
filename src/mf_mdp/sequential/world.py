"""Sequential world model for the H2 line: feedback-conditioned dynamics over a GRU state.

Model (docs/H2_HYBRID_SEQUENTIAL_STATE_DESIGN.md §2):
    event x_t      = [E[a_{t-1}], y_{t-1}, log1p(dt)]        (E = item embeddings)
    h_t            = GRU(h_{t-1}, x_t)
    p(y_t | h_t,a) = sigmoid(w_r . [h_t, E_a, h_t * E_a] + b_r)
    hazard / next-item readout: same head form / ||W_r h_t - E[a_{t+1}]||^2

Factorization: p_theta(h_{t+1}, y_t | h_t, a_t) = p_phi(y_t | h_t, a_t) * I[h_{t+1} = g(h_t, a_t, y_t)]
with deterministic GRU g — the minimal feedback-conditioned dynamics.

Negative controls (training-time input corruption; the response head ALWAYS scores the
true (h_t, a_t), labels unchanged):
    clean        : no corruption
    shuffle      : permute event order within each trajectory (destroys order)
    no_response  : force y_{t-1} := 0 in the encoder input (destroys feedback conditioning)
    random_action: replace a_{t-1} in the encoder input by a random item (destroys action
                   conditioning of the transition)

No popularity or side features anywhere in this module by design (design doc §2.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mf_mdp.sequential.gru import GRU, GRUConfig, Adam

CONTROL_MODES = ("clean", "shuffle", "no_response", "random_action")


@dataclass
class SeqWorldConfig:
    emb_dim: int = 32
    hid_dim: int = 32
    lr: float = 1e-2
    l2: float = 1e-5
    n_epochs: int = 150
    w_next: float = 0.5  # weight of the next-item embedding regression
    w_hazard: float = 0.3
    seed: int = 0


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class SeqWorldModel:
    """GRU state + response/hazard heads + next-item readout, trained by teacher forcing."""

    def __init__(self, n_items: int, cfg: SeqWorldConfig = SeqWorldConfig()):
        self.cfg = cfg
        self.n_items = n_items
        rng = np.random.default_rng(cfg.seed)
        m, dz = cfg.emb_dim, cfg.hid_dim
        self.E = rng.normal(0.0, 0.05, size=(n_items + 1, m))  # last row = start pad
        self.gru = GRU(GRUConfig(n_in=m + 2, n_hidden=dz, lr=cfg.lr, l2=cfg.l2,
                                 n_epochs=cfg.n_epochs, seed=cfg.seed), rng)
        s = np.sqrt(1.0 / dz)
        self.w_resp = rng.normal(0.0, s, size=3 * dz)
        self.w_term = rng.normal(0.0, s, size=3 * dz)
        self.W_r = rng.normal(0.0, s, size=(dz, m))
        self.biases = {"b_resp": np.zeros(1), "b_term": np.zeros(1)}
        self.adam = Adam({"E": self.E, "W_r": self.W_r, "w_resp": self.w_resp,
                          "w_term": self.w_term, **self.biases}, cfg.lr, cfg.l2)
        self.loss_history: list[float] = []

    # ---------------- encoding ----------------

    def encode_events(self, events: list[tuple[int, int]], control: str, rng: np.random.Generator):
        """Teacher-forcing encode of one trajectory.

        Args:
            events: [(a_t, y_t), ...] with y_t in {0,1} (engage/skip).
            control: one of CONTROL_MODES.

        Returns:
            per-step dicts: x (gru input), h (pre-response state), a (true action),
            y (label), term (label), a_next (or None), a_in (action used in x).
        """
        if control not in CONTROL_MODES:
            raise ValueError(f"unknown control: {control}")
        ev = list(events)
        if control == "shuffle" and len(ev) > 1:
            ev = [ev[i] for i in rng.permutation(len(ev))]
        steps = []
        h = np.zeros(self.cfg.hid_dim)
        a_prev, y_prev = self.n_items, 0.0  # pad row
        for t, (a, y) in enumerate(ev):
            a_in = a_prev
            if control == "random_action":
                a_in = int(rng.integers(self.n_items))
            x = np.concatenate([self.E[a_in], [float(y_prev), np.log1p(1.0)]])
            cache = self.gru.forward(x[None, :], h)
            h = cache[-1]["h2"]
            steps.append({"x": x, "h": h.copy(), "a": int(a), "y": float(y),
                          "a_next": (int(ev[t + 1][0]) if t + 1 < len(ev) else None),
                          "a_in": int(a_in)})
            a_prev, y_prev = a, (0.0 if control == "no_response" else float(y))
        return steps

    # ---------------- heads ----------------

    def _phi(self, h: np.ndarray, a: int) -> np.ndarray:
        e = self.E[a]
        return np.concatenate([h, e, h * e])

    def response_prob(self, h: np.ndarray, a: int) -> float:
        return float(_sig(self.w_resp @ self._phi(h, a) + self.biases["b_resp"][0]))

    def hazard_prob(self, h: np.ndarray, a: int) -> float:
        return float(_sig(self.w_term @ self._phi(h, a) + self.biases["b_term"][0]))

    def readout(self, h: np.ndarray) -> np.ndarray:
        """User-vector readout W_r h (sequential-geometry user preference)."""
        return self.W_r @ h

    def score(self, h: np.ndarray, a: int) -> float:
        """Next-item score E_a . (W_r h) (SASRec-style dot scoring)."""
        return float(self.E[a] @ self.readout(h))

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.cfg.hid_dim)

    def step_state(self, h: np.ndarray, a: int, y: float, dt: float = 1.0) -> np.ndarray:
        """Feedback-conditioned transition g(h, a, y): single GRU step on event (a, y)."""
        x = np.concatenate([self.E[a], [float(y), np.log1p(dt)]])
        return self.gru.step(x, h)

    def state_of(self, events: list[tuple[int, int]]) -> np.ndarray:
        """Clean (uncorrupted) encode of observed history — used at decision time."""
        h = self.initial_state()
        for a, y in events:
            h = self.step_state(h, a, y)
        return h

    # ---------------- training ----------------

    def fit(self, trajectories: list[list[tuple[int, int, int]]], control: str = "clean") -> None:
        """Full-batch Adam training with analytic gradients.

        Args:
            trajectories: per user, a list of (a, y, term) steps.
            control: training-time input corruption (CONTROL_MODES).
        """
        rng = np.random.default_rng(self.cfg.seed + 7)
        c = self.cfg
        for _ in range(c.n_epochs):
            grads = {k: np.zeros_like(v) for k, v in self.adam.params.items()}
            dH_all: list[np.ndarray] = []
            a_ins_all: list[list[int]] = []
            loss = 0.0
            n_terms = 0
            caches: list[list[dict]] = []
            for traj in trajectories:
                events = [(a, y) for (a, y, _t) in traj]
                terms = [t for (_a, _y, t) in traj]
                steps = self.encode_events(events, control, rng)
                T = len(steps)
                dH = np.zeros((T, c.hid_dim))
                for t, st in enumerate(steps):
                    h, a, y = st["h"], st["a"], st["y"]
                    phi = self._phi(h, a)
                    p = float(_sig(self.w_resp @ phi + self.biases["b_resp"][0]))
                    loss += -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
                    d = p - y
                    grads["w_resp"] += d * phi
                    grads["b_resp"] += d
                    dH[t] += d * (self.w_resp[:c.hid_dim] + self.w_resp[2 * c.hid_dim:] * self.E[a])
                    grads["E"][a] += d * (self.w_resp[c.hid_dim:2 * c.hid_dim]
                                          + self.w_resp[2 * c.hid_dim:] * h)
                    tm = float(terms[t])
                    pt = float(_sig(self.w_term @ phi + self.biases["b_term"][0]))
                    loss += c.w_hazard * (-(tm * np.log(pt + 1e-12)
                                            + (1 - tm) * np.log(1 - pt + 1e-12)))
                    dt_ = c.w_hazard * (pt - tm)
                    grads["w_term"] += dt_ * phi
                    grads["b_term"] += dt_
                    dH[t] += dt_ * (self.w_term[:c.hid_dim] + self.w_term[2 * c.hid_dim:] * self.E[a])
                    grads["E"][a] += dt_ * (self.w_term[c.hid_dim:2 * c.hid_dim]
                                            + self.w_term[2 * c.hid_dim:] * h)
                    if st["a_next"] is not None:
                        an = st["a_next"]
                        diff = self.readout(h) - self.E[an]
                        loss += c.w_next * float(diff @ diff)
                        grads["W_r"] += 2.0 * c.w_next * np.outer(diff, h)
                        dH[t] += 2.0 * c.w_next * (self.W_r.T @ diff)
                        grads["E"][an] -= 2.0 * c.w_next * diff
                        n_terms += 1
                dH_all.append(dH)
                a_ins_all.append([st["a_in"] for st in steps])
                caches.append(self.gru.forward(np.stack([st["x"] for st in steps]),
                                               self.initial_state()))
            n_terms = max(n_terms, 1)
            loss /= n_terms
            self.loss_history.append(loss)
            gru_grads = {k: np.zeros_like(v) for k, v in self.gru.params.items()}
            for dH, a_ins, cache in zip(dH_all, a_ins_all, caches):
                gg = self.gru.backward(cache, dH)
                for k, v in gg.items():
                    gru_grads[k] += v
                for t, st in enumerate(cache):
                    grads["E"][a_ins[t]] += st["dx"][:c.emb_dim]
            self.gru.apply_grads(gru_grads)
            self.adam.step(grads)

    # ---------------- predictive metrics ----------------

    def predict_trajectory(self, events: list[tuple[int, int]]):
        """Teacher-forced (clean) per-step response probs and next-item scores."""
        rng = np.random.default_rng(0)
        steps = self.encode_events(events, "clean", rng)
        probs = np.array([self.response_prob(st["h"], st["a"]) for st in steps])
        scores = np.stack([self.E @ self.readout(st["h"]) for st in steps]) if steps else np.zeros((0, self.n_items))
        ys = np.array([st["y"] for st in steps])
        return probs, scores, ys
