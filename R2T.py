#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import psycopg2
import math
import numpy as np
import pandas as pd
from typing import List, Literal, Dict, Any, Tuple, Optional
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
import os
import scipy.optimize as opt
import math,random


# In[ ]:


AggType = Literal["count", "sum", "min", "max"]


# In[ ]:


def _laplace(scale: float, rng: Optional[np.random.Generator] = None) -> float:
    #if scale is non-positive, return 0 to avoid invalid noise generation
    if scale <= 0:
        return 0.0
    #use a local rng instance for reproducibility
    rng = rng or np.random.default_rng()
    u = rng.uniform(-0.5, 0.5)
    return -scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))



def _geom_grid(gs_q: int) -> List[int]:
    #construct a geometric grid of candidate truncation thresholds (tau values)
    #the grid is descending and includes gs_q explicitly as a candidate
    if gs_q < 1:
        raise ValueError("GS_Q must be >= 1.")
    vals = []
    v = 1
    while v <= gs_q:
        vals.append(v)
        v <<= 1
    if vals[-1] != gs_q:
        vals.append(gs_q)
    vals = sorted(set(vals), reverse=True)
    return vals

def _truncate_sja(contrib: np.ndarray, tau: int, agg: AggType) -> float:
    #compute the truncated query answer for a given tau under different aggregation
    #for count/sum, clip to [0,tau] to enforce non-negativity and bounded sensitivity
    #for min/max, clip to [-tau,tau] to bound the range before taking min/max
    tau = int(tau)
    if tau < 0:
        raise ValueError("tau must be non-negative.")
    if agg in ("count", "sum"):
        clipped = np.clip(contrib, 0, tau)
        return float(clipped.sum())
    elif agg == "min":
        clipped = np.clip(contrib, -tau, tau)
        return float(clipped.min()) if clipped.size > 0 else 0.0
    elif agg == "max":
        clipped = np.clip(contrib, -tau, tau)
        return float(clipped.max()) if clipped.size > 0 else 0.0
    else:
        raise ValueError(f"Unsupported agg: {agg}")


def r2t_instance_optimal(
    contrib: List[float],
    agg: AggType = "count",
    epsilon: float = 1.0,
    beta: float = 0.05,
    gs_q: int = 100,
    rng: Optional[np.random.Generator] = None,
    return_diagnostics: bool = True
) -> Dict[str, Any]:
    # R2T-style instance-optimal truncation selection for DP query evaluation.
    # 1) Define a candidate grid of truncation thresholds tau.
    # 2) For each tau, compute a truncated answer Q_trunc(tau).
    # 3) Add Laplace noise calibrated to tau (noise_scale), subtract a bias/penalty term.
    # 4) Track the best (largest) debiased noisy score across tau values.
    # 5) Return the final DP estimate and (optionally) diagnostics for analysis.

    rng = rng or np.random.default_rng()
    contrib = np.array(contrib, dtype=float)
    #compute the non-private true answer under the chosen aggregation
    #for count/sum, treat negative contributions as 0 to align with clipping rule
    if agg in ("count", "sum"):
        true_answer = float(np.maximum(0.0, contrib).sum())
    elif agg == "min":
        true_answer = float(np.min(contrib)) if contrib.size > 0 else 0.0
    elif agg == "max":
        true_answer = float(np.max(contrib)) if contrib.size > 0 else 0.0
    else:
        raise ValueError(f"Unsupported agg: {agg}")
    #candidate truncation thresholds, ordered from large to small
    tau_grid = _geom_grid(gs_q) 
    #approximate the number of grid levels used in the algorithm
    L = max(1, int(math.ceil(math.log2(gs_q))))
    #precompute log terms that repeatedly appear in the noise calibration and penalties
    ln_G = math.log(max(gs_q, 2))
    lnln_over_beta = math.log(max(math.log(max(gs_q, 2)), 1e-12) / max(beta, 1e-12))
    #track the best debiased noisy value encountered so far
    best_dp = -float("inf")
     
    details: List[Dict[str, Any]] = []

    for tau in tau_grid:
        noise_scale = (ln_G * tau) / max(epsilon, 1e-12)
        penalty = (ln_G * lnln_over_beta * tau) / max(epsilon, 1e-12)

        q_trunc = _truncate_sja(contrib, tau=tau, agg=agg)
        q_noisy = q_trunc + _laplace(noise_scale, rng=rng) - penalty

        beta_tau = beta / max(1, L)
        tail_up = noise_scale * math.log(2.0 / max(beta_tau, 1e-12))
        optimistic_upper = q_trunc + tail_up - penalty

        cand = q_noisy
        if cand > best_dp:
            best_dp = cand

        details.append({
            "tau": tau,
            "Q_trunc": q_trunc,
            "noise_scale": noise_scale,
            "penalty": penalty,
            "q_noisy": q_noisy,
            "optimistic_upper": optimistic_upper,
            "best_so_far": best_dp
        })

        if optimistic_upper <= best_dp:
            pass

    if agg in ("count", "sum"):
        q_tau0 = 0.0
    elif agg == "min":
        q_tau0 = true_answer
    else:
        q_tau0 = -float("inf")

    dp_answer = max(best_dp, q_tau0)

    out = {
        "noise_answer": float(dp_answer),
        "true_answer": float(true_answer),
    }
    if return_diagnostics:
        out["details"] = details
        out["tau_grid_desc"] = tau_grid
        out["params"] = {
            "epsilon": epsilon,
            "beta": beta,
            "GS_Q": gs_q,
            "agg": agg,
            "ln_G": ln_G,
            "lnln_over_beta": lnln_over_beta
        }
    return out

def r2t_count(per_entity_counts: List[int], **kwargs) -> Dict[str, Any]:
    return r2t_instance_optimal(per_entity_counts, agg="count", **kwargs)

def r2t_sum(per_entity_sums: List[float], **kwargs) -> Dict[str, Any]:
    return r2t_instance_optimal(per_entity_sums, agg="sum", **kwargs)

def r2t_min(per_entity_mins: List[float], **kwargs) -> Dict[str, Any]:
    return r2t_instance_optimal(per_entity_mins, agg="min", **kwargs)

def r2t_max(per_entity_maxs: List[float], **kwargs) -> Dict[str, Any]:
    return r2t_instance_optimal(per_entity_maxs, agg="max", **kwargs)


# In[ ]:




