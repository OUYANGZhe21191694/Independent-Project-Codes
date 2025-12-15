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


#baseline
#naive truncation+smooth sensitivity
def laplace(scale):
    #draw Laplace(0,scale) noise for differential privacy
    return np.random.laplace(loc=0.0, scale=scale)

def sample_theta_nt(gs_q):
    #build a geometric candidate set
    thetas = [2**i for i in range(int(math.log2(gs_q)) + 1)]
    #randomly choose one threshold (naive truncation step)
    return random.choice(thetas)

def NT_sum(contrib, epsilon, gs_p):
    #sample truncation threshold theta, controls both clipping and noise magnitude
    theta = sample_theta_nt(gs_p)
    #clip each user's contribution to enforce bounded sensitivity
    contrib_clip = np.clip(contrib, 0, theta)
    #compute the clipped sum as the deterministic component
    true_sum = contrib_clip.sum()
    #add laplace noise calibrated to sensitivity and privacy budget epsilon
    noise = laplace(scale=theta / epsilon)
    #organize the output
    answer = {}
    answer['true_answer'] = float(true_sum)
    answer['noise_answer'] = float(true_sum+noise)
    answer['theta'] = float(theta)
    return answer


