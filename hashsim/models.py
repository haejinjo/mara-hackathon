"""
hashsim.model
Core class that learns ΔH/ΔP from baseline data and returns
hash-rate step-response functions for arbitrary power steps.
"""

from functools import partial
import numpy as np
from numpy.typing import ArrayLike
from .kernels import REGISTRY as _K
from typing import Callable


class LagModel:
    """
    Fit once, then generate hashrate-vs-time functions for any
    instantaneous power increase expressed in *percent* of the
    baseline power P₀.

    Example
    -------
    >>> model = LagModel.fit(t, P, H)
    >>> h_fun = model(10, 80, kernel="logistic", k=0.25, t_half=40)
    >>> h_of_t = h_fun(np.linspace(0, 180, 181))
    """

    # ----------------------------------------------------------------
    # constructor is private; use the class-method LagModel.fit (...)
    # ----------------------------------------------------------------
    def __init__(self, P0: float, H0: float, k_PH: float):
        self._P0  = P0          # last baseline power
        self._H0  = H0          # last baseline hashrate
        self._kPH = k_PH        # slope ΔH/ΔP

    # ----------------------------------------------------------------
    # public factory — learns k from baseline flat segment
    # ----------------------------------------------------------------
    @classmethod
    def fit(cls,
            t: ArrayLike,
            P: ArrayLike,
            H: ArrayLike) -> "LagModel":
        """
        Parameters
        ----------
        t, P, H : 1-D arrays of the same length.
        Only the *flat* tail is required; linear regression estimates k.

        Returns
        -------
        LagModel
        """
        P = np.asarray(P, dtype=float)
        H = np.asarray(H, dtype=float)

        if P.ndim != 1 or H.ndim != 1 or P.size != H.size:
            raise ValueError("t, P, H must be 1-D arrays of equal length")

        # linear regression H = k*P + c
        k, _ = np.polyfit(P, H, deg=1)
        return cls(P[-1], H[-1], k)

    # ----------------------------------------------------------------
    # __call__  ➜  returns a step-response *function*
    # ----------------------------------------------------------------
    def __call__(self,
                 pct_start: float,
                 pct_end: float,
                 *,
                 kernel: str | Callable = "exp" ,
                 **kernel_pars):
        """
        Build h(t) for an instantaneous power step.

        Parameters
        ----------
        pct_start, pct_end : floats
            Percentages of the baseline power P₀ (e.g. 10, 80).
        kernel : str or callable
            Name in hashsim.kernels.REGISTRY, or any g(t, **pars).
        kernel_pars : dict
            Extra parameters consumed by the chosen kernel.

        Returns
        -------
        h(t) : callable
            A function of elapsed time [same unit as baseline `t`].
        """
        if callable(kernel):
            g = partial(kernel, **kernel_pars)
        else:
            try:
                g0 = _K[kernel]
            except KeyError as exc:
                raise ValueError(f"unknown kernel '{kernel}'") from exc
            g = partial(g0, **kernel_pars)

        ΔP    = (pct_end - pct_start) / 100.0 * self._P0
        ΔH_ss = self._kPH * ΔP                      # steady-state gain
        H0    = self._H0

        def h(t):
            t = np.asarray(t, dtype=float)
            return H0 + ΔH_ss * g(t)

        return h
