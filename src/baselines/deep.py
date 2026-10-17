"""Deep comparison baselines (PyTorch, runnable on CPU).

Uniform interface ``f(S_ref, S_eval) -> (scores_ref, scores_eval)``, where ``S`` is an
``(n, 2, L)`` fixed-length ``(g, p)`` two-channel sequence and a larger score means
more anomalous.

* :func:`autoencoder_scores` -- fully connected autoencoder, anomaly score = reconstruction MSE;
* :func:`aelstm_ga_scores` -- LSTM autoencoder whose hyper-parameters are searched by a **genetic algorithm** (GA);
* :func:`gru_bayesian_scores` -- GRU autoencoder with **MC dropout** Bayesian uncertainty,
  anomaly score = ``mean + beta * std`` (reconstruction error over several random forward passes);
* :func:`anogan_scores` -- AnoGAN: train a DCGAN on normals, then optimise the latent ``z`` per sample at test time,
  anomaly score = residual loss + discriminative feature loss;
* :func:`deep_svdd_scores` -- Deep SVDD (ICML 2018): the encoder is trained to minimise the latent hypersphere,
  anomaly score = distance to the hypersphere centre;
* :func:`anomaly_transformer_scores` -- Anomaly-Transformer: reconstruction error weighted by the
  **association discrepancy** between the learnable Gaussian prior and the series association,
  trained with the min--max two-backward strategy.

All models are trained only on the reference normal curves.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import (BASELINE_ANOGAN_EPOCHS, BASELINE_ANOGAN_LAMBDA,
                      BASELINE_ANOGAN_STEPS, BASELINE_ANOGAN_Z, BASELINE_AT_DMODEL,
                      BASELINE_AT_HEADS, BASELINE_AT_K, BASELINE_AT_LAYERS,
                      BASELINE_AT_TEMPERATURE, BASELINE_BATCH, BASELINE_EPOCHS,
                      BASELINE_GA_EPOCHS, BASELINE_GA_GENS, BASELINE_GA_POP,
                      BASELINE_GRU_DROPOUT, BASELINE_GRU_HIDDEN, BASELINE_HIDDEN,
                      BASELINE_LATENT, BASELINE_LR, BASELINE_MAE_BETA,
                      BASELINE_MC_PASSES, BASELINE_PATIENCE, BASELINE_SEQ_LEN, SEED)
from . import cache

__all__ = ["autoencoder_scores", "aelstm_ga_scores",
           "gru_bayesian_scores", "deep_svdd_scores", "anogan_scores",
           "anomaly_transformer_scores"]

#: Score cache type: (reference scores, evaluation scores)
Scores = Tuple[np.ndarray, np.ndarray]


Tensor = torch.Tensor
LOGGER = logging.getLogger("tepad")


# --------------------------------------------------------------------------- #
# Common utilities
# --------------------------------------------------------------------------- #
def _seed_torch(seed: int = SEED) -> None:
    """Fix the PyTorch / numpy random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def _split_fit_val(n: int, val_frac: float = 0.15, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Split the reference set into train/validation parts (validation drives early stopping)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    return idx[n_val:], idx[:n_val]


def _batches(n: int, batch: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Shuffle into batches and merge a trailing singleton batch into the previous one
    (avoids single-sample BatchNorm batches)."""
    idx = rng.permutation(n)
    out: List[np.ndarray] = []
    for s in range(0, n, batch):
        b = idx[s:s + batch]
        if len(b) == 1 and out:
            out[-1] = np.concatenate([out[-1], b])
        elif len(b):
            out.append(b)
    return out


def _as_seq(S: np.ndarray) -> Tensor:
    """``(n, 2, L)`` -> ``(n, L, 2)`` float32 tensor (the recurrent-network input layout)."""
    return torch.as_tensor(np.asarray(S, dtype=np.float32), dtype=torch.float32).transpose(1, 2)


def _standardize(S_ref: np.ndarray, S_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel z-score using **reference-set** statistics (fitted on the reference normals only, no leakage).

    Standardization removes the dominance of the overall daily amplitude in the
    reconstruction error, so the anomaly score focuses more on the profile shape.
    """
    ref = np.asarray(S_ref, dtype=float)
    ev = np.asarray(S_eval, dtype=float)
    flat = ref.transpose(0, 2, 1).reshape(-1, ref.shape[1])
    mu, sd = flat.mean(axis=0), flat.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)

    def _apply(S: np.ndarray) -> np.ndarray:
        b = np.asarray(S, dtype=float)
        f = b.transpose(0, 2, 1).reshape(-1, b.shape[1])
        return ((f - mu) / sd).reshape(b.shape[0], b.shape[2], b.shape[1]).transpose(0, 2, 1)

    return _apply(ref), _apply(ev)


def _as_flat(S: np.ndarray) -> Tensor:
    """``(n, 2, L)`` -> ``(n, 2L)`` float32 tensor."""
    return torch.as_tensor(np.asarray(S, dtype=np.float32), dtype=torch.float32).reshape(len(S), -1)


def _train_reconstruction(model: nn.Module, x_fit: Tensor, x_val: Tensor, *,
                          epochs: int = BASELINE_EPOCHS, lr: float = BASELINE_LR,
                          batch: int = BASELINE_BATCH,
                          patience: int = BASELINE_PATIENCE) -> nn.Module:
    """Generic reconstruction training: MSE objective + early stopping on the validation set,
    returning the weights with the lowest validation error."""
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    best, best_state, bad = float("inf"), None, 0
    for ep in range(int(epochs)):
        model.train()
        rng = np.random.default_rng(SEED + ep)
        for b in _batches(len(x_fit), batch, rng):
            xb = x_fit[b]
            loss = F.mse_loss(model(xb)[0], xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(F.mse_loss(model(x_val)[0], x_val))
        if val < best - 1e-9:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _recon_scores(model: nn.Module, x: Tensor) -> np.ndarray:
    """Per-sample reconstruction MSE (accepts ``(n, L, C)`` or ``(n, D)``)."""
    model.eval()
    with torch.no_grad():
        out = model(x)[0]
        dims = tuple(range(1, out.dim()))
        return ((x - out) ** 2).mean(dim=dims).numpy()


# --------------------------------------------------------------------------- #
# Weight / score cache (see cache.py)
# --------------------------------------------------------------------------- #
def _slug(name: str, S_ref: np.ndarray, S_eval: np.ndarray, params: Dict) -> str:
    """Cache file name: model name + hyper-parameter hash + data hash."""
    return cache.make_slug(name, params, cache.data_digest(S_ref, S_eval))


def _cached_scores(cache_dir: Optional[Path], slug: str, use_cache: bool) -> Optional[Scores]:
    """Return the cached scores on a hit, skipping training and scoring."""
    if not use_cache:
        return None
    hit = cache.load_scores(cache_dir, slug)
    if hit is not None:
        LOGGER.info("    [cache] hit %s (skipping training and scoring)", slug)
    return hit


def _restore(cache_dir: Optional[Path], slug: str, models: Dict[str, nn.Module],
             use_cache: bool) -> bool:
    """Try to initialise ``models`` from the cached weights; return ``True`` on success."""
    if not use_cache:
        return False
    ckpt = cache.load_checkpoint(cache_dir, slug)
    if not ckpt:
        return False
    try:
        for key, model in models.items():
            model.load_state_dict(ckpt["state_dicts"][key])
        return True
    except Exception:  # pragma: no cover - fall back to retraining on structure mismatch
        return False


def _save(scores: Scores, cache_dir: Optional[Path], slug: str, params: Dict) -> Scores:
    """Persist the score cache and return the values unchanged."""
    if cache.save_scores(cache_dir, slug, scores[0], scores[1], params) is not None:
        LOGGER.info("    [cache] saved weights/scores %s", slug)
    return scores


# --------------------------------------------------------------------------- #
# Autoencoders
# --------------------------------------------------------------------------- #
class MLPAE(nn.Module):
    """Fully connected autoencoder (input is the flattened ``2L``-dimensional vector)."""

    def __init__(self, in_dim: int, hidden: int = BASELINE_HIDDEN,
                 latent: int = BASELINE_LATENT):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, in_dim))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.enc(x)
        return self.dec(z), z


class RecurrentAE(nn.Module):
    """GRU / LSTM encoder-decoder autoencoder.

    The encoder takes the last hidden state as the latent vector; at every time step
    the decoder is fed ``[broadcast latent vector, learnable positional encoding]``,
    so it can reconstruct the **shape and phase** of the intra-day profile
    (broadcasting the latent vector alone would collapse the decoder to a constant output).
    """

    def __init__(self, cell: str = "gru", n_ch: int = 2, seq_len: int = BASELINE_SEQ_LEN,
                 hidden: int = BASELINE_GRU_HIDDEN, latent: int = BASELINE_LATENT,
                 dropout: float = 0.0):
        super().__init__()
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.enc = rnn(n_ch, hidden, batch_first=True)
        self.to_z = nn.Linear(hidden, latent)
        self.from_z = nn.Linear(latent, hidden)
        self.pos = nn.Parameter(torch.randn(1, int(seq_len), hidden) * 0.02)
        self.dec = rnn(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_ch)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        _, h = self.enc(x)
        h = h[0] if isinstance(h, tuple) else h
        z = self.drop(self.to_z(h[-1]))
        rep = self.drop(self.from_z(z)).unsqueeze(1) + self.pos
        dec, _ = self.dec(rep)
        return self.out(self.drop(dec)), z


def autoencoder_scores(S_ref: np.ndarray, S_eval: np.ndarray, *,
                       hidden: int = BASELINE_HIDDEN, latent: int = BASELINE_LATENT,
                       epochs: int = BASELINE_EPOCHS, lr: float = BASELINE_LR,
                       batch: int = BASELINE_BATCH, seed: int = SEED,
                       cache_dir: Optional[Path] = None,
                       use_cache: bool = True) -> Scores:
    """Autoencoder baseline: anomaly score = reconstruction MSE."""
    params = {"model": "autoencoder", "hidden": int(hidden), "latent": int(latent),
              "epochs": int(epochs), "lr": float(lr), "batch": int(batch), "seed": int(seed)}
    slug = _slug("autoencoder", S_ref, S_eval, params)
    hit = _cached_scores(cache_dir, slug, use_cache)
    if hit is not None:
        return hit

    _seed_torch(seed)
    S_ref, S_eval = _standardize(S_ref, S_eval)
    xr, xe = _as_flat(S_ref), _as_flat(S_eval)
    fit, val = _split_fit_val(len(xr), seed=seed)
    model = MLPAE(xr.shape[1], hidden=hidden, latent=latent)
    if not _restore(cache_dir, slug, {"ae": model}, use_cache):
        _train_reconstruction(model, xr[fit], xr[val], epochs=epochs, lr=lr, batch=batch)
        cache.save_checkpoint(cache_dir, slug, {"ae": model.state_dict()}, params)
    return _save((_recon_scores(model, xr), _recon_scores(model, xe)), cache_dir, slug, params)


# --------------------------------------------------------------------------- #
# AE-LSTM-GA: LSTM autoencoder with genetic-algorithm hyper-parameter search
# --------------------------------------------------------------------------- #
_GA_SPACE: Dict[str, list] = {"hidden": [16, 32, 64], "latent": [4, 8, 16], "lr": [1e-3, 3e-3]}


def _ga_evaluate(cfg: Dict, x_fit: Tensor, x_val: Tensor, epochs: int,
                 batch: int) -> float:
    """Individual fitness = validation reconstruction MSE (smaller is better)."""
    _seed_torch()
    model = RecurrentAE("lstm", seq_len=x_fit.size(1), hidden=int(cfg["hidden"]),
                        latent=int(cfg["latent"]))
    _train_reconstruction(model, x_fit, x_val, epochs=epochs, lr=float(cfg["lr"]),
                          batch=batch, patience=max(10, epochs // 3))
    model.eval()
    with torch.no_grad():
        return float(F.mse_loss(model(x_val)[0], x_val))


def _ga_search(x_fit: Tensor, x_val: Tensor, *, epochs: int = BASELINE_GA_EPOCHS,
               batch: int = BASELINE_BATCH, pop: int = BASELINE_GA_POP,
               gens: int = BASELINE_GA_GENS, seed: int = SEED) -> Dict:
    """Tournament genetic algorithm: elitism + uniform crossover + single-gene mutation."""
    rng = np.random.default_rng(seed)
    keys = list(_GA_SPACE)

    def random_gene() -> Dict:
        return {k: _GA_SPACE[k][int(rng.integers(len(_GA_SPACE[k])))] for k in keys}

    population = [random_gene() for _ in range(int(pop))]
    best_cfg, best_fit = dict(population[0]), float("inf")
    for _ in range(int(gens)):
        fitness = [_ga_evaluate(c, x_fit, x_val, epochs, batch) for c in population]
        order = np.argsort(fitness)
        if fitness[order[0]] < best_fit:
            best_fit, best_cfg = float(fitness[order[0]]), dict(population[order[0]])
        elites = [population[i] for i in order[:max(1, int(pop) // 2)]]
        nxt = [dict(elites[0])]
        while len(nxt) < int(pop):
            p1, p2 = elites[int(rng.integers(len(elites)))], elites[int(rng.integers(len(elites)))]
            child = {k: (p1[k] if rng.random() < 0.5 else p2[k]) for k in keys}
            if rng.random() < 0.4:                      # mutation
                k = keys[int(rng.integers(len(keys)))]
                child[k] = _GA_SPACE[k][int(rng.integers(len(_GA_SPACE[k])))]
            nxt.append(child)
        population = nxt
    return best_cfg


def aelstm_ga_scores(S_ref: np.ndarray, S_eval: np.ndarray, *,
                     epochs: int = BASELINE_EPOCHS, ga_epochs: int = BASELINE_GA_EPOCHS,
                     pop: int = BASELINE_GA_POP, gens: int = BASELINE_GA_GENS,
                     batch: int = BASELINE_BATCH, seed: int = SEED,
                     cache_dir: Optional[Path] = None,
                     use_cache: bool = True) -> Scores:
    """AE-LSTM-GA baseline: GA searches the LSTM autoencoder hyper-parameters, then scores with reconstruction MSE."""
    params = {"model": "ae_lstm_ga", "epochs": int(epochs), "ga_epochs": int(ga_epochs),
              "pop": int(pop), "gens": int(gens), "batch": int(batch), "seed": int(seed)}
    slug = _slug("ae_lstm_ga", S_ref, S_eval, params)
    hit = _cached_scores(cache_dir, slug, use_cache)
    if hit is not None:
        return hit

    _seed_torch(seed)
    S_ref, S_eval = _standardize(S_ref, S_eval)
    xr, xe = _as_seq(S_ref), _as_seq(S_eval)
    fit, val = _split_fit_val(len(xr), seed=seed)

    model, cfg, restored = None, None, False
    if use_cache:
        ckpt = cache.load_checkpoint(cache_dir, slug)
        cfg = (ckpt or {}).get("meta", {}).get("config")
        if cfg is not None:
            model = RecurrentAE("lstm", seq_len=xr.size(1), hidden=int(cfg["hidden"]),
                                latent=int(cfg["latent"]))
            restored = _restore(cache_dir, slug, {"lstm_ae": model}, use_cache)

    if not restored:
        cfg = _ga_search(xr[fit], xr[val], epochs=ga_epochs, batch=batch, pop=pop,
                         gens=gens, seed=seed)
        _seed_torch(seed)
        model = RecurrentAE("lstm", seq_len=xr.size(1), hidden=int(cfg["hidden"]),
                            latent=int(cfg["latent"]))
        _train_reconstruction(model, xr[fit], xr[val], epochs=epochs, lr=float(cfg["lr"]),
                              batch=batch)
        cache.save_checkpoint(cache_dir, slug, {"lstm_ae": model.state_dict()},
                              dict(params, config=cfg))
    return _save((_recon_scores(model, xr), _recon_scores(model, xe)), cache_dir, slug, params)


# --------------------------------------------------------------------------- #
# GRU-Bayesian: MC dropout uncertainty
# --------------------------------------------------------------------------- #
def gru_bayesian_scores(S_ref: np.ndarray, S_eval: np.ndarray, *,
                        hidden: int = BASELINE_GRU_HIDDEN, latent: int = BASELINE_LATENT,
                        dropout: float = BASELINE_GRU_DROPOUT,
                        epochs: int = BASELINE_EPOCHS, lr: float = BASELINE_LR,
                        batch: int = BASELINE_BATCH, passes: int = BASELINE_MC_PASSES,
                        beta: float = BASELINE_MAE_BETA, seed: int = SEED,
                        cache_dir: Optional[Path] = None,
                        use_cache: bool = True) -> Scores:
    """GRU-Bayesian baseline: ``passes`` random forward passes with dropout, anomaly score = ``mean + beta*std``."""
    params = {"model": "gru_bayesian", "hidden": int(hidden), "latent": int(latent),
              "dropout": float(dropout), "epochs": int(epochs), "lr": float(lr),
              "batch": int(batch), "passes": int(passes), "beta": float(beta),
              "seed": int(seed)}
    slug = _slug("gru_bayesian", S_ref, S_eval, params)
    hit = _cached_scores(cache_dir, slug, use_cache)
    if hit is not None:
        return hit

    _seed_torch(seed)
    S_ref, S_eval = _standardize(S_ref, S_eval)
    xr, xe = _as_seq(S_ref), _as_seq(S_eval)
    fit, val = _split_fit_val(len(xr), seed=seed)
    model = RecurrentAE("gru", seq_len=xr.size(1), hidden=hidden, latent=latent, dropout=dropout)
    if not _restore(cache_dir, slug, {"gru_ae": model}, use_cache):
        _train_reconstruction(model, xr[fit], xr[val], epochs=epochs, lr=lr, batch=batch)
        cache.save_checkpoint(cache_dir, slug, {"gru_ae": model.state_dict()}, params)

    def _mc(x: Tensor) -> np.ndarray:
        model.train()                                   # enable dropout to emulate posterior sampling
        with torch.no_grad():
            stack = torch.stack([((x - model(x)[0]) ** 2).mean(dim=(1, 2))
                                 for _ in range(int(passes))])
        mean, std = stack.mean(0), stack.std(0)
        return (mean + float(beta) * std).numpy()

    return _save((_mc(xr), _mc(xe)), cache_dir, slug, params)


# --------------------------------------------------------------------------- #
# LSTM-AE: deterministic LSTM autoencoder without hyper-parameter search
# --------------------------------------------------------------------------- #
def deep_svdd_scores(S_ref: np.ndarray, S_eval: np.ndarray, *,
                     hidden: int = BASELINE_GRU_HIDDEN,
                     latent: int = BASELINE_LATENT, epochs: int = BASELINE_EPOCHS,
                     lr: float = BASELINE_LR, batch: int = BASELINE_BATCH,
                     seed: int = SEED, cache_dir: Optional[Path] = None,
                     use_cache: bool = True) -> Scores:
    """Deep SVDD baseline (ICML 2018): anomaly score = distance to the hypersphere centre ``c``.

    Training minimises the mean squared radius ``mean(||phi(x) - c||^2)`` of the encoder
    output; the centre ``c`` is set from the mean latent representation after
    initialisation (center init of the original paper), and the bias term on ``c`` is
    frozen to avoid degenerate solutions.
    """
    params = {"model": "deep_svdd", "hidden": int(hidden), "latent": int(latent),
              "epochs": int(epochs), "lr": float(lr), "batch": int(batch),
              "seed": int(seed)}
    slug = _slug("deep_svdd", S_ref, S_eval, params)
    hit = _cached_scores(cache_dir, slug, use_cache)
    if hit is not None:
        return hit

    _seed_torch(seed)
    S_ref, S_eval = _standardize(S_ref, S_eval)
    xr, xe = _as_seq(S_ref).transpose(1, 2).contiguous(), \
        _as_seq(S_eval).transpose(1, 2).contiguous()
    model = DeepSVDDEncoder(n_ch=xr.size(1), seq_len=xr.size(2),
                            hidden=hidden, latent=latent)
    center_path = None
    if cache_dir is not None:
        center_path = Path(cache_dir) / (slug + ".center.npy")
    center = None
    if use_cache and center_path is not None and center_path.exists():
        try:
            center = np.load(center_path)
        except Exception:  # pragma: no cover - treat a corrupted cache as a miss
            center = None
    if center is None or not _restore(cache_dir, slug, {"encoder": model}, use_cache):
        _seed_torch(seed)
        model = DeepSVDDEncoder(n_ch=xr.size(1), seq_len=xr.size(2),
                               hidden=hidden, latent=latent)
        model.eval()
        with torch.no_grad():
            center = model(xr).mean(dim=0).numpy()
        if center_path is not None:
            np.save(center_path, center)
        center_tensor = torch.as_tensor(center, dtype=torch.float32)
        opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=1e-6)
        best, best_state, bad = float("inf"), None, 0
        for ep in range(int(epochs)):
            model.train()
            rng = np.random.default_rng(seed + ep)
            for b in _batches(len(xr), batch, rng):
                out = model(xr[b])
                loss = ((out - center_tensor) ** 2).sum(dim=1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val = float(((model(xr) - center_tensor) ** 2).sum(dim=1).mean())
            if val < best - 1e-9:
                best, bad = val, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= BASELINE_PATIENCE:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        cache.save_checkpoint(cache_dir, slug, {"encoder": model.state_dict()}, params)

    center_tensor = torch.as_tensor(center, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        scores = (((model(xr) - center_tensor) ** 2).sum(dim=1).sqrt().numpy(),
                  ((model(xe) - center_tensor) ** 2).sum(dim=1).sqrt().numpy())
    return _save(scores, cache_dir, slug, params)


# --------------------------------------------------------------------------- #
# AnoGAN
# --------------------------------------------------------------------------- #
class GANGenerator(nn.Module):
    """AnoGAN generator: ``z -> (2, L)`` with ``L = 16 * l0``."""

    def __init__(self, z_dim: int = BASELINE_ANOGAN_Z, n_ch: int = 2, base: int = 32,
                 l0: int = 6):
        super().__init__()
        self.base, self.l0 = base, l0
        self.fc = nn.Linear(z_dim, base * 8 * l0)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(base * 8, base * 4, 4, 2, 1), nn.BatchNorm1d(base * 4), nn.ReLU(True),
            nn.ConvTranspose1d(base * 4, base * 2, 4, 2, 1), nn.BatchNorm1d(base * 2), nn.ReLU(True),
            nn.ConvTranspose1d(base * 2, base, 4, 2, 1), nn.BatchNorm1d(base), nn.ReLU(True),
            nn.ConvTranspose1d(base, n_ch, 4, 2, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        h = self.fc(z).view(-1, self.base * 8, self.l0)
        return self.net(h)


class GANDiscriminator(nn.Module):
    """AnoGAN discriminator: returns the logit and the intermediate feature map (used for the discriminative feature loss)."""

    def __init__(self, n_ch: int = 2, base: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_ch, base, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv1d(base, base * 2, 4, 2, 1), nn.BatchNorm1d(base * 2), nn.LeakyReLU(0.2, True),
            nn.Conv1d(base * 2, base * 4, 4, 2, 1), nn.BatchNorm1d(base * 4), nn.LeakyReLU(0.2, True),
            nn.Conv1d(base * 4, base * 8, 4, 2, 1), nn.BatchNorm1d(base * 8), nn.LeakyReLU(0.2, True),
        )
        self.head = nn.Linear(base * 8, 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        f = self.features(x)
        return self.head(f.mean(dim=2)), f


def _train_gan(G: nn.Module, D: nn.Module, x: Tensor, *, epochs: int, batch: int,
               lr: float = 2e-4) -> None:
    """Standard DCGAN adversarial training (BCE loss)."""
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    z_dim = G.fc.in_features
    for ep in range(int(epochs)):
        rng = np.random.default_rng(SEED + ep)
        for b in _batches(len(x), batch, rng):
            xb = x[b]
            z = torch.randn(len(xb), z_dim)
            with torch.no_grad():
                fake = G(z)
            real_logit, _ = D(xb)
            fake_logit, _ = D(fake)
            loss_d = bce(real_logit, torch.ones_like(real_logit)) + \
                bce(fake_logit, torch.zeros_like(fake_logit))
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            fake_logit, _ = D(G(z))
            loss_g = bce(fake_logit, torch.ones_like(fake_logit))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()


def _anogan_score(G: nn.Module, D: nn.Module, x: Tensor, *, lam: float,
                  steps: int, z_dim: int, lr: float = 0.02) -> np.ndarray:
    """Optimise the latent codes ``z`` in a batch, returning ``(1-lam)*residual loss + lam*discriminative feature loss``."""
    G.eval()
    D.eval()
    z = torch.randn(len(x), z_dim, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    with torch.no_grad():
        _, f_real = D(x)
    loss = torch.zeros(len(x))
    for _ in range(int(steps)):
        x_hat = G(z)
        residual = (x - x_hat).abs().mean(dim=(1, 2))
        _, f_fake = D(x_hat)
        discrimination = (f_real - f_fake).abs().mean(dim=(1, 2))
        loss = (1.0 - float(lam)) * residual + float(lam) * discrimination
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    return loss.detach().numpy()


def anogan_scores(S_ref: np.ndarray, S_eval: np.ndarray, *,
                  z_dim: int = BASELINE_ANOGAN_Z, epochs: int = BASELINE_ANOGAN_EPOCHS,
                  steps: int = BASELINE_ANOGAN_STEPS, lam: float = BASELINE_ANOGAN_LAMBDA,
                  batch: int = BASELINE_BATCH, seed: int = SEED,
                  cache_dir: Optional[Path] = None, use_cache: bool = True) -> Scores:
    """AnoGAN baseline: train a DCGAN on the reference normals, then score via latent-optimisation residuals."""
    params = {"model": "anogan", "z_dim": int(z_dim), "epochs": int(epochs),
              "steps": int(steps), "lam": float(lam), "batch": int(batch), "seed": int(seed)}
    slug = _slug("anogan", S_ref, S_eval, params)
    hit = _cached_scores(cache_dir, slug, use_cache)
    if hit is not None:
        return hit

    _seed_torch(seed)
    S_ref, S_eval = _standardize(S_ref, S_eval)
    xr, xe = _as_seq(S_ref).transpose(1, 2).contiguous(), _as_seq(S_eval).transpose(1, 2).contiguous()
    G = GANGenerator(z_dim=z_dim, n_ch=xr.size(1))
    D = GANDiscriminator(n_ch=xr.size(1))
    if not _restore(cache_dir, slug, {"G": G, "D": D}, use_cache):
        _train_gan(G, D, xr, epochs=epochs, batch=batch)
        cache.save_checkpoint(cache_dir, slug, {"G": G.state_dict(), "D": D.state_dict()}, params)

    scores = (_anogan_score(G, D, xr, lam=lam, steps=steps, z_dim=z_dim),
              _anogan_score(G, D, xe, lam=lam, steps=steps, z_dim=z_dim))
    return _save(scores, cache_dir, slug, params)


# --------------------------------------------------------------------------- #
# AnomalyTransformer
# --------------------------------------------------------------------------- #
class AnomalyAttention(nn.Module):
    """Anomaly-Attention: series association ``softmax(QK^T/√d)`` vs a learnable Gaussian prior."""

    def __init__(self, d_model: int, n_heads: int, seq_len: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.h, self.dk = int(n_heads), int(d_model // n_heads)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.sigma = nn.Parameter(torch.ones(int(seq_len), int(n_heads)))

    def prior(self, length: int) -> Tensor:
        """Learnable Gaussian prior ``(h, L, L)``: ``softmax_j(-(i-j)^2 / 2σ_i^2)``."""
        idx = torch.arange(length, dtype=torch.float, device=self.sigma.device)
        dist2 = (idx[:, None] - idx[None, :]) ** 2
        sig = F.softplus(self.sigma[:length]).transpose(0, 1).clamp(min=1.0)
        return torch.softmax(-dist2[None] / (2.0 * sig[:, :, None] ** 2), dim=-1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        b, length, dim = x.shape

        def split(t: Tensor) -> Tensor:
            return t.view(b, length, self.h, self.dk).transpose(1, 2)

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        series = torch.softmax(q @ k.transpose(-1, -2) / (self.dk ** 0.5), dim=-1)
        prior = self.prior(length).unsqueeze(0).expand(b, -1, -1, -1)
        ctx = (series @ v).transpose(1, 2).reshape(b, length, dim)
        return self.proj(ctx), series, prior


class TransformerLayer(nn.Module):
    """Anomaly-Attention + feed-forward network (residual + LayerNorm)."""

    def __init__(self, d_model: int, n_heads: int, seq_len: int):
        super().__init__()
        self.attn = AnomalyAttention(d_model, n_heads, seq_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(),
                                nn.Linear(d_model * 4, d_model))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        attn_out, series, prior = self.attn(x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x, series, prior


class AnomalyTransformerNet(nn.Module):
    """Reconstruction Transformer encoder that emits the series association and prior at every layer."""

    def __init__(self, in_ch: int = 2, seq_len: int = 96, d_model: int = BASELINE_AT_DMODEL,
                 n_heads: int = BASELINE_AT_HEADS, n_layers: int = BASELINE_AT_LAYERS):
        super().__init__()
        self.embed = nn.Linear(in_ch, d_model)
        self.pos = nn.Parameter(torch.randn(1, int(seq_len), d_model) * 0.02)
        self.layers = nn.ModuleList([TransformerLayer(d_model, n_heads, seq_len)
                                     for _ in range(int(n_layers))])
        self.head = nn.Linear(d_model, in_ch)

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        h = self.embed(x.transpose(1, 2)) + self.pos
        series_list, prior_list = [], []
        for layer in self.layers:
            h, series, prior = layer(h)
            series_list.append(series)
            prior_list.append(prior)
        return self.head(h).transpose(1, 2), series_list, prior_list


def _kl_loss(p: Tensor, q: Tensor) -> Tensor:
    """KL term of the official implementation, ``mean(p * (log p - log q))`` (sum over the last two dims, then mean)."""
    p = p.clamp_min(1e-8)
    q = q.clamp_min(1e-8)
    return (p * (p.log() - q.log())).sum(-1).mean()


def _kl_per_position(series: Tensor, prior: Tensor) -> Tensor:
    """Per-timestep value of the symmetric KL, shape ``(B, L)`` (averaged over heads first)."""
    s = series.clamp_min(1e-8)
    p = prior.clamp_min(1e-8)
    kl = (p * (p.log() - s.log())).sum(-1) + (s * (s.log() - p.log())).sum(-1)
    return 0.5 * kl.mean(1)


def _train_anomaly_transformer(model: AnomalyTransformerNet, x_fit: Tensor, x_val: Tensor, *,
                               k: float, epochs: int, lr: float, batch: int,
                               patience: int) -> AnomalyTransformerNet:
    """min--max training: ``loss1 = rec - k*series`` and ``loss2 = rec + k*prior`` are each back-propagated once."""
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    best, best_state, bad = float("inf"), None, 0
    for ep in range(int(epochs)):
        model.train()
        rng = np.random.default_rng(SEED + ep)
        for b in _batches(len(x_fit), batch, rng):
            xb = x_fit[b]
            out, series_list, prior_list = model(xb)
            rec = F.mse_loss(out, xb)
            series_loss = torch.zeros(())
            prior_loss = torch.zeros(())
            for s, p in zip(series_list, prior_list):
                pn = p / p.sum(-1, keepdim=True).clamp_min(1e-8)
                series_loss = series_loss + _kl_loss(s, pn.detach()) + _kl_loss(pn.detach(), s)
                prior_loss = prior_loss + _kl_loss(pn, s.detach()) + _kl_loss(s.detach(), pn)
            n = max(len(series_list), 1)
            series_loss, prior_loss = series_loss / n, prior_loss / n
            opt.zero_grad()
            (rec - float(k) * series_loss).backward(retain_graph=True)
            (rec + float(k) * prior_loss).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(F.mse_loss(model(x_val)[0], x_val))
        if val < best - 1e-9:
            best, bad = val, 0
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _at_scores(model: AnomalyTransformerNet, x: Tensor, temperature: float) -> np.ndarray:
    """Association-discrepancy-weighted reconstruction error: ``Σ_t softmax(-AssDis_t) · rec_t``."""
    model.eval()
    with torch.no_grad():
        out, series_list, prior_list = model(x)
        rec = ((x - out) ** 2).mean(dim=1)                      # (B, L)
        metrics = []
        for s, p in zip(series_list, prior_list):
            pn = p / p.sum(-1, keepdim=True).clamp_min(1e-8)
            metrics.append(torch.softmax(-_kl_per_position(s, pn) / float(temperature), dim=1))
        return (torch.stack(metrics).mean(0) * rec).sum(dim=1).numpy()


def anomaly_transformer_scores(S_ref: np.ndarray, S_eval: np.ndarray, *,
                               d_model: int = BASELINE_AT_DMODEL, n_heads: int = BASELINE_AT_HEADS,
                               n_layers: int = BASELINE_AT_LAYERS, k: float = BASELINE_AT_K,
                               temperature: float = BASELINE_AT_TEMPERATURE,
                               epochs: int = BASELINE_EPOCHS, lr: float = BASELINE_LR,
                               batch: int = BASELINE_BATCH, seed: int = SEED,
                               cache_dir: Optional[Path] = None,
                               use_cache: bool = True) -> Scores:
    """AnomalyTransformer baseline (association discrepancy + reconstruction error)."""
    params = {"model": "anomaly_transformer", "d_model": int(d_model),
              "n_heads": int(n_heads), "n_layers": int(n_layers), "k": float(k),
              "temperature": float(temperature), "epochs": int(epochs), "lr": float(lr),
              "batch": int(batch), "seed": int(seed)}
    slug = _slug("anomaly_transformer", S_ref, S_eval, params)
    hit = _cached_scores(cache_dir, slug, use_cache)
    if hit is not None:
        return hit

    _seed_torch(seed)
    S_ref, S_eval = _standardize(S_ref, S_eval)
    xr, xe = _as_seq(S_ref).transpose(1, 2).contiguous(), _as_seq(S_eval).transpose(1, 2).contiguous()
    fit, val = _split_fit_val(len(xr), seed=seed)
    model = AnomalyTransformerNet(in_ch=xr.size(1), seq_len=xr.size(2), d_model=d_model,
                                  n_heads=n_heads, n_layers=n_layers)
    if not _restore(cache_dir, slug, {"at": model}, use_cache):
        _train_anomaly_transformer(model, xr[fit], xr[val], k=k, epochs=epochs, lr=lr,
                                   batch=batch, patience=BASELINE_PATIENCE)
        cache.save_checkpoint(cache_dir, slug, {"at": model.state_dict()}, params)

    scores = (_at_scores(model, xr, temperature), _at_scores(model, xe, temperature))
    return _save(scores, cache_dir, slug, params)


if __name__ == "__main__":  # pragma: no cover - minimal example
    rng = np.random.default_rng(0)
    ref = rng.random((40, 2, 32))
    ev = np.vstack([rng.random((10, 2, 32)), rng.random((10, 2, 32)) * 0.2])
    for fn in (autoencoder_scores, gru_bayesian_scores):
        a, b = fn(ref, ev, epochs=5)
        print(fn.__name__, a.shape, b.shape)
