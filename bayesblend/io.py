from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from cmdstanpy import CmdStanMCMC


@dataclass
class Draws:
    log_lik: np.ndarray | None = None
    post_pred: np.ndarray | None = None

    def __post_init__(self):
        if self.log_lik is None and self.post_pred is None:
            raise ValueError(
                "At least one of `log_lik` or `post_pred` must be specified."
            )
        if self.log_lik is not None and self.post_pred is not None:
            if self.log_lik.shape != self.post_pred.shape:
                raise ValueError("`log_lik` and `post_pred` are different dimensions.")

    @cached_property
    def log_lik_2d(self) -> np.ndarray[Any, Any] | None:
        if self.log_lik is None:
            return None
        if self.log_lik.ndim == 2:
            return self.log_lik
        return self.log_lik.reshape(
            (self.log_lik.shape[0], np.prod(self.log_lik.shape[1:]))
        )

    @cached_property
    def post_pred_2d(self) -> np.ndarray[Any, Any] | None:
        if self.post_pred is None:
            return None
        if self.post_pred.ndim == 2:
            return self.post_pred
        return self.post_pred.reshape(
            (self.post_pred.shape[0], np.prod(self.post_pred.shape[1:]))
        )

    @classmethod
    def from_cmdstanpy(
        cls,
        fit: CmdStanMCMC,
        log_lik_name: str = "log_lik",
        post_pred_name: str = "post_pred",
    ) -> Draws:
        samples = {
            var: fit.stan_variable(var) for var in [log_lik_name, post_pred_name]
        }
        return cls(**samples)
