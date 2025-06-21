# hashprice_lib.py
"""Light‑weight library for forecasting Bitcoin hash‑price
================================================================
Provides a single public class `HashPricePredictor` that loads a GRU
checkpoint (``best_gru.pt`` by default) and returns hash‑price forecasts
for arbitrary horizons.

Import example
--------------
```python
from hashprice_lib import HashPricePredictor
import numpy as np

predictor = HashPricePredictor("best_gru.pt", device="cpu")

price_30 = np.random.default_rng(0).uniform(40, 60, 30).tolist()
y_next   = predictor.predict(price_30, fng_value=57.0,
                            sent_class="Greed", temp_mean=15.2,
                            days_ahead=1)
print(y_next)
```
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import torch
from torch import nn

# ─────────────────────────────────────────────────────────────────────────────
# 1. constants (must match the training script)
# ─────────────────────────────────────────────────────────────────────────────
LOOK_BACK      = 30           # history window length
HIDDEN_SIZE    = 32           # GRU hidden dimension
BTC_CONVERSION = 80_000.0     # usd → btc conversion rule used in training

# Sentiment classes used during training (order matters for one‑hot)
CATEG_LABELS = [
    "Extreme_Fear", "Fear", "Neutral", "Greed", "Extreme_Greed"
]

# Means / stds of the three exogenous numeric vars learned by StandardScaler
_EXO_MEAN = np.array([1.25e-3, 50.0, 15.0], dtype=np.float32)
_EXO_STD  = np.array([2.0e-4, 20.0, 5.0], dtype=np.float32)

# Input feature dimension: 1 raw price + 3 z‑scored vars + 5 one‑hot = 9
D_IN = 1 + 3 + len(CATEG_LABELS)

# ─────────────────────────────────────────────────────────────────────────────
# 2. network definition (same layout as training)
# ─────────────────────────────────────────────────────────────────────────────
class _HashPriceGRU(nn.Module):
    def __init__(self, d_in: int = D_IN, hidden: int = HIDDEN_SIZE):
        super().__init__()
        self.gru  = nn.GRU(d_in, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 30, d)
        _, h = self.gru(x)
        return self.head(h.squeeze(0))

# ─────────────────────────────────────────────────────────────────────────────
# 3. preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _one_hot(label: str) -> np.ndarray:
    vec = np.zeros(len(CATEG_LABELS), dtype=np.float32)
    if label in CATEG_LABELS:
        vec[CATEG_LABELS.index(label)] = 1.0
    return vec


def _build_row(price: float, fng: float, sent_cls: str, temp: float) -> np.ndarray:
    """Return a single (d,) feature vector prepared exactly like training."""
    # raw price remains unscaled
    price_arr = np.array([price], dtype=np.float32)

    # three exogenous vars → z‑scores
    exo      = np.array([price / BTC_CONVERSION, fng, temp], dtype=np.float32)
    exo_z    = (exo - _EXO_MEAN) / _EXO_STD

    return np.concatenate([price_arr, exo_z, _one_hot(sent_cls)])

# ─────────────────────────────────────────────────────────────────────────────
# 4. public predictor class
# ─────────────────────────────────────────────────────────────────────────────
class HashPricePredictor:
    """GRU‑based hash‑price forecasting utility."""

    def __init__(self, checkpoint: str | Path = "best_gru.pt", *, device: str = "cpu"):
        self.device = torch.device(device)
        self.model  = _HashPriceGRU().to(self.device)
        state       = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    # ---------------------------------------------------------------------
    def predict(
        self,
        price_window: Sequence[float],
        *,
        fng_value: float,
        sent_class: str,
        temp_mean: float,
        days_ahead: int = 1,
        weather_series: Optional[Sequence[float]] = None,
        sent_series: Optional[Sequence[float]] = None,
    ) -> float:
        """Forecast hash‑price *days_ahead*.

        Parameters
        ----------
        price_window   : 30 floats (oldest→newest) of hp_hash_usd.
        fng_value      : numeric Fear‑and‑Greed of the latest day.
        sent_class     : sentiment class label of the latest day.
        temp_mean      : mean temperature of the latest day.
        days_ahead     : positive integer horizon.
        weather_series : optional 30‑element temperature history.
        sent_series    : optional 30‑element numeric sentiment history.

        Returns
        -------
        float – predicted hp_hash_usd at t + days_ahead.
        """
        if len(price_window) != LOOK_BACK:
            raise ValueError("price_window must contain exactly 30 values")

        # default exogenous histories: repeat latest value across window
        if weather_series is None:
            weather_series = [temp_mean] * LOOK_BACK
        if sent_series is None:
            sent_series = [fng_value] * LOOK_BACK
        if not (len(weather_series) == len(sent_series) == LOOK_BACK):
            raise ValueError("weather_series and sent_series must each be length 30")

        # build initial (30, d) feature matrix
        window = np.vstack([
            _build_row(price_window[i], sent_series[i], sent_class, weather_series[i])
            for i in range(LOOK_BACK)
        ])

        # recursive forecasting
        win = window.copy()
        for _ in range(days_ahead):
            x = torch.tensor(win[None, :, :], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                pred = self.model(x).item()
            # append synthetic next day and drop oldest
            new_row = _build_row(pred, fng_value, sent_class, temp_mean)
            win = np.vstack([win[1:], new_row])
        return pred

# ─────────────────────────────────────────────────────────────────────────────
# 5. self‑test (run only when executed as a script)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy.random as npr

    ckpt = Path("best_gru.pt")
    if not ckpt.exists():
        raise SystemExit("Checkpoint 'best_gru.pt' not found in current directory.")

    price_hist = 50 + npr.randn(LOOK_BACK)  # synthetic around 50 USD
    predictor  = HashPricePredictor(ckpt, device="cpu")
    demo_pred  = predictor.predict(price_hist, fng_value=60.0, sent_class="Greed", temp_mean=16.0, days_ahead=2)
    print(f"Demo 2‑day forecast: {demo_pred:.2f}")
