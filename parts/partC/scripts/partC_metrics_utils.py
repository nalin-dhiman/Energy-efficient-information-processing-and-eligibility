

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class InfoMetrics:
    H_bits: float
    CE_bits: float
    I_lb_bits: float


def entropy_bits(labels: np.ndarray) -> float:

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    vals, counts = np.unique(labels, return_counts=True)
    p = counts.astype(float) / counts.sum()
    # Avoid log(0) if a class is missing
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def cross_entropy_bits(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
   
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


    p = np.clip(probs, eps, 1.0 - eps)
    p_true = p[np.arange(labels.shape[0]), labels]
    return float((-np.log2(p_true)).mean())


def info_lower_bound_bits(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> InfoMetrics:

    H = entropy_bits(labels)
    CE = cross_entropy_bits(probs, labels, eps=eps)
    I_lb = max(0.0, H - CE)
    return InfoMetrics(H_bits=H, CE_bits=CE, I_lb_bits=I_lb)
