# src/training/early_stopping.py
"""Early stopping callback with exponential moving average smoothing."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping with exponential moving average (EMA) smoothing.

    Monitors a metric and stops training when the smoothed metric stops improving
    for a specified number of epochs (patience).

    Uses EMA to reduce noise from metric oscillations:
    - ema = alpha * current_value + (1 - alpha) * previous_ema
    - Higher alpha (0.7): more responsive to recent values, smoother tracking
    - Lower alpha (0.3): more stable, ignores short-term fluctuations

    Separately tracks raw best for checkpoint saving (actual best seen).

    Supports both "max" mode (higher is better, e.g., macro_f1) and
    "min" mode (lower is better, e.g., validation loss).

    Attributes:
        patience: Number of epochs to wait without improvement.
        metric: Name of the metric being monitored (for logging).
        mode: "max" (higher better) or "min" (lower better).
        alpha: EMA weight for current value (default: 0.7).
        best_raw: Best raw (unsmoothed) metric value.
        best_ema: Best EMA value.
        ema: Current EMA value.
        best_epoch: Epoch when best EMA was achieved.
        counter: Number of epochs since last improvement.
    """

    def __init__(
        self,
        patience: int = 8,
        metric: str = "macro_f1",
        mode: str = "max",
        ema_alpha: float = 0.7,
        min_delta: float = 1e-4,
    ):
        """
        Initialize EarlyStopping with EMA smoothing.

        Args:
            patience: Epochs to wait after last improvement (increased from 5 to 8).
            metric: Name of metric being tracked (for logging only).
            mode: "max" for F1/AUC, "min" for loss.
            ema_alpha: Weight for current value in EMA (0.7 = 70% current, 30% history).
                      Higher alpha = more responsive; lower = smoother.
            min_delta: Minimum improvement to count as a new best.

        Raises:
            ValueError: If mode is not "max" or "min".
        """
        if mode not in ["max", "min"]:
            raise ValueError(f"mode must be 'max' or 'min', got {mode}")
        if not (0.0 <= ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in [0, 1], got {ema_alpha}")

        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.alpha = ema_alpha
        self.min_delta = min_delta

        self.best_raw: Optional[float] = None   # Best raw (unsmoothed) metric
        self.best_ema: Optional[float] = None   # Best EMA value
        self.ema: Optional[float] = None        # Current EMA
        self.best_epoch: Optional[int] = None
        self.counter = 0
        self.should_stop = False

        logger.info(
            f"Initialized EarlyStopping with EMA smoothing "
            f"(metric={metric}, mode={mode}, patience={patience}, "
            f"ema_alpha={ema_alpha}, min_delta={min_delta})"
        )

    def __call__(self, raw_value: float, epoch: int) -> bool:
        """
        Update EMA and check if training should stop.

        Args:
            raw_value: Current raw (unsmoothed) metric value.
            epoch: Current epoch number.

        Returns:
            True if training should stop, False otherwise.
        """
        # Update EMA
        if self.ema is None:
            self.ema = raw_value
        else:
            self.ema = self.alpha * raw_value + (1 - self.alpha) * self.ema

        is_improvement = self._is_improvement(self.ema)

        if is_improvement:
            self.best_ema = self.ema
            self.best_epoch = epoch
            self.counter = 0
            logger.info(
                f"[Epoch {epoch}] {self.metric} improved (EMA): "
                f"{self.best_ema:.6f} ✓ | raw: {raw_value:.6f}"
            )
        else:
            self.counter += 1
            logger.info(
                f"[Epoch {epoch}] {self.metric}: raw={raw_value:.6f}, "
                f"ema={self.ema:.6f} | patience {self.counter}/{self.patience}"
            )

        # Always track raw best separately (for checkpoint saving)
        if self.best_raw is None or self._is_improvement_raw(raw_value):
            self.best_raw = raw_value

        self.should_stop = (self.counter >= self.patience)

        if self.should_stop:
            logger.warning(
                f"Early stopping triggered at epoch {epoch} after {self.patience} epochs "
                f"without improvement. Best {self.metric} (EMA): {self.best_ema:.6f} "
                f"at epoch {self.best_epoch}"
            )

        return self.should_stop

    def _is_improvement(self, value: float) -> bool:
        """Check if EMA value is an improvement."""
        if self.best_ema is None:
            return True
        if self.mode == "max":
            return value > self.best_ema + self.min_delta
        return value < self.best_ema - self.min_delta

    def _is_improvement_raw(self, value: float) -> bool:
        """Check if raw value is an improvement."""
        if self.best_raw is None:
            return True
        if self.mode == "max":
            return value > self.best_raw + self.min_delta
        return value < self.best_raw - self.min_delta

    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_raw = self.best_ema = self.ema = None
        self.best_epoch = None
        self.counter = 0
        self.should_stop = False
        logger.debug("Reset EarlyStopping state")

    @property
    def status(self) -> str:
        """Get status summary string."""
        return (
            f"EarlyStopping | metric={self.metric} | "
            f"best_raw={self.best_raw:.4f} | "
            f"best_ema={self.best_ema:.4f} | "
            f"best_epoch={self.best_epoch} | "
            f"counter={self.counter}/{self.patience} | "
            f"current_ema={self.ema:.4f}"
        )
