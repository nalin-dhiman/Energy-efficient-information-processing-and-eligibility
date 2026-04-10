"""Part C: canonical, reviewer-proof metrics utilities.

This module is intentionally small, dependency-light, and explicit.
It is designed to prevent silent unit bugs (nats vs bits) and to
make reviewer queries easy to answer.

All information quantities are computed in base-2 (bits).

Definitions
-----------
Let S be a discrete stimulus label with empirical distribution p(s).
Let a decoder produce probabilities \hat{p}(s|x) for each observation x.

- Entropy:
    H(S) = -\sum_s p(s) log2 p(s)

- Cross-entropy (bits):
    CE = -E_{(x,s)} [ log2 \hat{p}(s|x) ]

- Decoder lower bound on mutual information:
    I_lb = max(0, H(S) - CE)

I_lb is a lower bound because the decoder class may be suboptimal.

Guardrails
----------
- We clip probabilities to [eps, 1-eps] to avoid log(0).
- If labels are imbalanced, H(S) is computed empirically.

"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class InfoMetrics:
    H_bits: float
    CE_bits: float
    I_lb_bits: float


def entropy_bits(labels: np.ndarray) -> float:
    """Empirical entropy H(S) in bits."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    vals, counts = np.unique(labels, return_counts=True)
    p = counts.astype(float) / counts.sum()
    # Avoid log(0) if a class is missing
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def cross_entropy_bits(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    """Cross-entropy in bits from predicted probabilities and true labels.

    Parameters
    ----------
    probs:
        Array of shape (n_samples, n_classes) with rows summing to ~1.
    labels:
        Array of shape (n_samples,) with integer labels in [0, n_classes-1].

    Returns
    -------
    CE_bits:
        Mean negative log2 probability of the true class.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D, got shape {probs.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(f"n_samples mismatch: probs {probs.shape[0]} vs labels {labels.shape[0]}")

    n_classes = probs.shape[1]
    if labels.min() < 0 or labels.max() >= n_classes:
        raise ValueError(f"labels out of range [0,{n_classes-1}]")

    # Clip to avoid log(0) and extreme CE explosions
    p = np.clip(probs, eps, 1.0 - eps)
    p_true = p[np.arange(labels.shape[0]), labels]
    return float((-np.log2(p_true)).mean())


def info_lower_bound_bits(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> InfoMetrics:
    """Compute (H, CE, I_lb) in bits."""
    H = entropy_bits(labels)
    CE = cross_entropy_bits(probs, labels, eps=eps)
    I_lb = max(0.0, H - CE)
    return InfoMetrics(H_bits=H, CE_bits=CE, I_lb_bits=I_lb)
