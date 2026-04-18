# src/training/early_stopping.py
"""Early stopping callback for training."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping callback to prevent overfitting.

    Monitors a metric and stops training when the metric stops improving
    for a specified number of epochs (patience).

    Supports both "max" mode (higher is better, e.g., macro_f1) and
    "min" mode (lower is better, e.g., validation loss).

    Attributes:
        patience: Number of epochs to wait without improvement.
        metric: Name of the metric being monitored (for logging).
        mode: "max" (higher better) or "min" (lower better).
        best_value: Best metric value observed so far.
        best_epoch: Epoch when best value was achieved.
        counter: Number of epochs since last improvement.
    """

    def __init__(
        self, patience: int = 5, metric: str = "macro_f1", mode: str = "max"
    ):
        """
        Initialize EarlyStopping.

        Args:
            patience: Number of epochs without improvement before stopping (default: 5).
            metric: Name of the metric to monitor (default: "macro_f1").
            mode: "max" if higher is better, "min" if lower is better (default: "max").

        Raises:
            ValueError: If mode is not "max" or "min".
        """
        if mode not in ["max", "min"]:
            raise ValueError(f"mode must be 'max' or 'min', got {mode}")

        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.counter = 0

        logger.info(
            f"Initialized EarlyStopping "
            f"(metric={metric}, mode={mode}, patience={patience})"
        )

    def __call__(self, value: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value.
            epoch: Current epoch number.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_value is None:
            # First call: initialize best value
            self.best_value = value
            self.best_epoch = epoch
            logger.info(
                f"[Epoch {epoch}] Initial {self.metric}: {value:.6f}"
            )
            return False

        # Check if current value is an improvement
        is_improvement = False
        if self.mode == "max":
            is_improvement = value > self.best_value
        else:  # mode == "min"
            is_improvement = value < self.best_value

        if is_improvement:
            # Reset counter and update best value
            self.counter = 0
            self.best_value = value
            self.best_epoch = epoch
            logger.info(
                f"[Epoch {epoch}] {self.metric} improved to {value:.6f} ✓"
            )
            return False
        else:
            # Increment counter
            self.counter += 1
            logger.info(
                f"[Epoch {epoch}] {self.metric}: {value:.6f} "
                f"(patience {self.counter}/{self.patience})"
            )

            # Check if patience exceeded
            if self.counter >= self.patience:
                logger.warning(
                    f"Early stopping triggered after {self.patience} epochs without improvement. "
                    f"Best {self.metric}: {self.best_value:.6f} at epoch {self.best_epoch}"
                )
                return True

            return False

    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_value = None
        self.best_epoch = None
        self.counter = 0
        logger.debug("Reset EarlyStopping state")

    @property
    def best_value_property(self) -> Optional[float]:
        """Get the best metric value observed."""
        return self.best_value

    @property
    def best_epoch_property(self) -> Optional[int]:
        """Get the epoch when best value was achieved."""
        return self.best_epoch

    @property
    def patience_counter(self) -> int:
        """Get the current patience counter."""
        return self.counter
