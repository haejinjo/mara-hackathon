"""
hashsim.kernels
Standard step-response kernels g(t) with range [0, 1].
All must satisfy g(0)=0, lim_{t→∞} g(t)=1.
"""

import numpy as np

def linear(t, t_ramp):
    return np.clip(t / t_ramp, 0.0, 1.0)

def quadratic(t, t_ramp):
    s = np.clip(t / t_ramp, 0.0, 1.0)
    return s**2

def exp1(t, tau):
    return 1.0 - np.exp(-t / tau)

def logistic(t, k, t_half):
    return 1.0 / (1.0 + np.exp(-k * (t - t_half)))

# registry makes string lookup convenient
REGISTRY = {
    "linear":    linear,
    "quadratic": quadratic,
    "exp":       exp1,
    "logistic":  logistic,
}
