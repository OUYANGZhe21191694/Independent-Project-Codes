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


def lp_baseline_sum(
    contrib,
    epsilon=1.0,
    tau=None,
    gs_q=100,
    rng=None,
    return_details=False
):
    # LP-based baseline for SUM under differential privacy.
    # 1)Optionally sample a truncation threshold tau from a geometric grid up to gs_q.
    # 2)Clip per-entity contributions to [0, tau] to bound sensitivity.
    # 3)Solve two trivial LPs to get bounds (lb/ub) under the same box constraints.
    # 4)Release ub + Laplace(tau/epsilon) as a private answer (optimistic upper-bounded estimate + noise).
    
    rng = rng or np.random.default_rng()
    contrib = np.array(contrib, float)
    #non-private sum of raw contributions
    true_answer = float(contrib.sum())

    #If tau not provided, draw tau from {1,2,4,...,<=gs_q} plus gs_q if not a power of 2.
    if tau is None:
        vals = []
        v = 1
        while v <= gs_q:
            vals.append(v)
            v <<= 1
        if vals[-1] != gs_q:
            vals.append(gs_q)
        tau = rng.choice(vals)
    #Clip contributions to enforce bounded sensitivity and non-negativity.
    clipped = np.clip(contrib, 0, tau)

    n = len(clipped)

    c_max = -np.ones(n)  
    bounds = [(0, tau)] * n 

    res_max = opt.linprog(c=c_max, bounds=bounds, method='highs')
    if not res_max.success:
        raise RuntimeError("LP max failed: " + res_max.message)
    ub = -res_max.fun 

    c_min = np.ones(n) 
    res_min = opt.linprog(c=c_min, bounds=bounds, method='highs')
    if not res_min.success:
        raise RuntimeError("LP min failed: " + res_min.message)
    lb = res_min.fun
    #add laplace noise calibrated to sensitivity tau for SUM after clipping
    noise = rng.laplace(scale=tau / max(epsilon, 1e-12))
    #release an upper-bound-based estimate plus noise.
    noisy_answer = ub + noise

    out = {
        "noise_answer": float(noisy_answer),
        "true_answer": true_answer,
        "tau": tau,
        "lower_bound": float(lb),
        "upper_bound": float(ub),
    }
    if return_details:
        out["clipped_sum"] = float(clipped.sum())
        out["noise"] = noise
    return out

def lp_baseline_count(
    contrib,
    epsilon=1.0,
    tau=None,  
    gs_q=1,   
    rng=None,
    return_details=False
):
    rng = rng or np.random.default_rng()
    contrib = np.array(contrib, float)
    #non-private count of items.
    true_answer = float(len(contrib))
    #add Laplace noise calibrated to count sensitivity (typically 1).
    noise = rng.laplace(scale=gs_q / max(epsilon, 1e-12))
    noisy_answer = true_answer + noise

    out = {
        "noise_answer": float(noisy_answer),
        "true_answer": true_answer,
        "tau": gs_q,
    }
    if return_details:
        out["noise"] = noise
    return out