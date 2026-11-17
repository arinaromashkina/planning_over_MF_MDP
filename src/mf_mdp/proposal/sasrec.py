"""SASRec (Kang & McAuley, ICDM'18; CE variant Wang et al., SIGIR'21) as a candidate
proposer for planning — NOT as a final recommender (docs/H3_LEARNED_ACTION_PROPOSALS_DESIGN.md).

Causal self-attention over item embeddings with learned positional embeddings, pre-LN
residual blocks, dropout, full-softmax next-item cross-entropy with tied input/output
embeddings. The catalog here is ~3.4k items, so full softmax is cheap and avoids
sampling bias. Torch is an OPTIONAL heavy dependency: this module is imported lazily
by callers; tests skip when torch is absent.

Indexing convention: item ids are the dataset ids 0..n_items-1 (shared with the MF
factors V). Padding id = n_items (extra embedding row, zeroed, excluded from scoring).
Sequences are LEFT-padded to maxlen so the last real token always sits at position
maxlen-1 and the next-item readout is the last position's hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:  # optional heavy dependency (docs/DECISIONS.md)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised on torch-less installs
    torch = None
    nn = None
    F = None
    _HAS_TORCH = False


@dataclass(frozen=True)
class SASRecConfig:
    n_items: int
    dim: int = 64
    n_layers: int = 2
    n_heads: int = 1
    maxlen: int = 200
    dropout: float = 0.2

    @property
    def pad(self) -> int:
        return self.n_items


if _HAS_TORCH:

    class _Block(nn.Module):
        """Pre-LN transformer block: causal MHSA + FFN, both with residuals."""

        def __init__(self, dim: int, n_heads: int, dropout: float):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
            self.ln1 = nn.LayerNorm(dim)
            self.qkv = nn.Linear(dim, 3 * dim)
            self.proj = nn.Linear(dim, dim)
            self.ln2 = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, 2 * dim),
                nn.GELU(),
                nn.Linear(2 * dim, dim),
            )
            self.attn_drop = nn.Dropout(dropout)
            self.drop = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor, causal: torch.Tensor, key_pad: torch.Tensor) -> torch.Tensor:
            B, L, D = x.shape
            h = self.ln1(x)
            qkv = self.qkv(h).view(B, L, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, hd)
            attn = (q @ k.transpose(-2, -1)) / self.head_dim**0.5
            attn = attn.masked_fill(causal[None, None, :, :], float("-inf"))
            attn = attn.masked_fill(key_pad[:, None, None, :], float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)  # all-pad query rows: no keys -> zero attn
            attn = self.attn_drop(attn)
            out = (attn @ v).transpose(1, 2).reshape(B, L, D)
            x = x + self.drop(self.proj(out))
            x = x + self.drop(self.ffn(self.ln2(x)))
            return x

    class SASRec(nn.Module):
        """Causal sequential encoder; next-item logits = h_last @ item_emb[:n_items].T."""

        def __init__(self, cfg: SASRecConfig):
            super().__init__()
            self.cfg = cfg
            self.item_emb = nn.Embedding(cfg.n_items + 1, cfg.dim, padding_idx=cfg.pad)
            self.pos_emb = nn.Embedding(cfg.maxlen, cfg.dim)
            self.emb_drop = nn.Dropout(cfg.dropout)
            self.blocks = nn.ModuleList(
                [_Block(cfg.dim, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)]
            )
            self.final_ln = nn.LayerNorm(cfg.dim)
            causal = torch.triu(
                torch.ones(cfg.maxlen, cfg.maxlen, dtype=torch.bool), diagonal=1
            )
            self.register_buffer("causal", causal, persistent=False)

        def encode(self, seqs: torch.Tensor) -> torch.Tensor:
            """(B, L) item ids -> (B, L, D) hidden states. Pad ids attend to nothing."""
            B, L = seqs.shape
            if L > self.cfg.maxlen:
                seqs = seqs[:, -self.cfg.maxlen :]
                L = self.cfg.maxlen
            pos = torch.arange(L, device=seqs.device)
            x = self.item_emb(seqs) + self.pos_emb(pos)[None, :, :]
            x = self.emb_drop(x)
            key_pad = seqs == self.cfg.pad  # (B, L) True where pad (excluded as keys)
            causal = self.causal[:L, :L]
            for blk in self.blocks:
                x = blk(x, causal, key_pad)
            return self.final_ln(x)

        def logits(self, seqs: torch.Tensor) -> torch.Tensor:
            """(B, L) item ids -> (B, n_items) next-item logits from the last position."""
            h = self.encode(seqs)[:, -1]
            return h @ self.item_emb.weight[: self.cfg.n_items].T

    def build_seq_array(
        histories: list[np.ndarray], n_items: int, maxlen: int
    ) -> np.ndarray:
        """Left-padded int64 array (n, maxlen); pad id = n_items. Empty history -> all pad."""
        n = len(histories)
        out = np.full((n, maxlen), n_items, dtype=np.int64)
        for i, h in enumerate(histories):
            h = np.asarray(h, dtype=np.int64).ravel()
            if h.size:
                out[i, -min(h.size, maxlen) :] = h[-maxlen:]
        return out

    def fit_sasrec(
        model: "SASRec",
        train_seqs: np.ndarray,
        valid_fn: Callable[["SASRec"], float],
        device: str = "cpu",
        lr: float = 1e-3,
        batch_size: int = 128,
        max_epochs: int = 200,
        patience: int = 20,
        seed: int = 0,
        eval_every: int = 1,
        verbose: bool = True,
    ) -> tuple["SASRec", dict]:
        """Full-softmax next-item CE with left-pad masking and early stopping.

        Loss positions with a pad INPUT are masked out (strictly cleaner than the
        reference implementation, which lets all-pad positions predict the first real
        item). Early stopping maximizes `valid_fn` (validation next-item Recall@10).
        Returns (best_model_by_valid, history dict). Model is moved to `device`.
        """
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
        pad = model.cfg.pad
        seqs_t = torch.as_tensor(train_seqs, dtype=torch.long, device=device)
        n = seqs_t.shape[0]
        best_score, best_state, bad_epochs = -np.inf, None, 0
        history: dict = {"valid_recall10": [], "train_loss": []}
        for epoch in range(1, max_epochs + 1):
            model.train()
            order = rng.permutation(n)
            tot_loss, tot_cnt = 0.0, 0
            for s in range(0, n, batch_size):
                batch = seqs_t[order[s : s + batch_size]]
                logits = model.encode(batch)[:, :-1] @ model.item_emb.weight[: pad].T
                targets = batch[:, 1:]
                loss_mask = batch[:, :-1] != pad  # inputs that are real tokens
                loss = F.cross_entropy(
                    logits.reshape(-1, pad), targets.reshape(-1),
                    ignore_index=pad, reduction="none",
                )
                loss = (loss * loss_mask.reshape(-1)).sum() / loss_mask.sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot_loss += float(loss) * int(loss_mask.sum())
                tot_cnt += int(loss_mask.sum())
            history["train_loss"].append(tot_loss / max(tot_cnt, 1))
            if epoch % eval_every == 0:
                score = float(valid_fn(model))
                history["valid_recall10"].append(score)
                if verbose:
                    print(
                        f"epoch {epoch:4d}  train_loss {history['train_loss'][-1]:.4f}"
                        f"  valid_recall10 {score:.4f}"
                    )
                if score > best_score:
                    best_score, bad_epochs = score, 0
                    best_state = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        history["best_valid_recall10"] = best_score
        return model, history

    @torch.no_grad()
    def score_histories(
        model: "SASRec", seqs: np.ndarray, device: str = "cpu", batch_size: int = 512
    ) -> np.ndarray:
        """(n, maxlen) padded histories -> (n, n_items) next-item logits (float32 numpy)."""
        model = model.to(device)
        model.eval()
        out = np.empty((seqs.shape[0], model.cfg.n_items), dtype=np.float32)
        t = torch.as_tensor(seqs, dtype=torch.long, device=device)
        for s in range(0, t.shape[0], batch_size):
            out[s : s + batch_size] = model.logits(t[s : s + batch_size]).float().cpu().numpy()
        return out


def recall_at_k_from_logits(logits: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Fraction of targets inside the row-wise top-k (no consumed masking: model-level metric)."""
    part = np.argpartition(-logits, k - 1, axis=1)[:, :k]
    hit = 0
    for i, t in enumerate(targets):
        hit += int(t in part[i])
    return hit / max(len(targets), 1)
