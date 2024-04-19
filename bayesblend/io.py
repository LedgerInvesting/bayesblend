from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Generator, Tuple

import arviz as az
import numpy as np
from cmdstanpy import CmdStanMCMC


@dataclass
class Draws:
    """Dataclass to extract and store posterior draws for use in BayesBlend models.

    The Draws dataclass is the core underlying data representation used by BayesBlend
    models for both fitting and blending.

    Attributes:
        log_lik: Array of posterior log likelihood samples for a model. Can be any
            dimension so long as the first dimenion is the MCMC/posterior sample index.
        post_pred: Array of posterior predictive samples from a model. Can be any
            dimension so long as the first dimenion is the MCMC/posterior sample index.

    Yields:
        Iterating over Draws objects will return {"log_lik": np.array} and
            {"post_pred": np.array} attributes for convenience.
    """

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

    def __iter__(self) -> Generator:
        for attr in [{"log_lik": self.log_lik_2d}, {"post_pred": self.post_pred_2d}]:
            for par, samples in attr.items():
                yield par, samples

    @property
    def n_samples(self) -> int:
        if self.log_lik is not None:
            return self.log_lik.shape[0]
        elif self.post_pred is not None:
            return self.post_pred.shape[0]
        else:
            return 0

    @property
    def n_datapoints(self) -> int:
        if self.log_lik_2d is not None:
            return self.log_lik_2d.shape[1]
        elif self.post_pred_2d is not None:
            return self.post_pred_2d.shape[1]
        else:
            return 0

    @property
    def shape(self) -> Tuple[int, ...]:
        return (
            self.log_lik.shape
            if self.log_lik is not None
            else self.post_pred.shape if self.post_pred is not None else ()
        )

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

    @cached_property
    def lpd(self) -> np.ndarray | None:
        if self.log_lik_2d is None:
            return None
        else:
            return np.apply_along_axis(compute_lpd, 0, self.log_lik_2d)

    @classmethod
    def from_cmdstanpy(
        cls,
        fit: CmdStanMCMC,
        log_lik_name: str = "log_lik",
        post_pred_name: str = "post_pred",
    ) -> Draws:
        samples = {
            "log_lik": fit.stan_variable(log_lik_name),
            "post_pred": fit.stan_variable(post_pred_name),
        }
        return cls(**samples)

    @classmethod
    def from_arviz(
        cls,
        fit: az.InferenceData,
        log_lik_name: str = "log_lik",
        post_pred_name: str = "post_pred",
    ) -> Draws:
        posterior = fit.to_dict()

        try:
            ll = posterior["log_likelihood"][log_lik_name]
        except KeyError:
            ll = posterior["posterior"][log_lik_name]
        except Exception:
            raise

        try:
            post_pred = posterior["posterior_predictive"][post_pred_name]
        except KeyError:
            post_pred = posterior["posterior"][post_pred_name]
        except Exception:
            raise

        samples = {"log_lik": np.vstack(ll), "post_pred": np.vstack(post_pred)}
        return cls(**samples)

    @classmethod
    def from_lpd(cls, lpd: np.ndarray, post_pred: np.ndarray | None = None) -> Draws:
        shape = (1, *lpd.shape) if post_pred is None else post_pred.shape
        lpd_full = np.full(shape, lpd.reshape((1, *lpd.shape)))
        return cls(log_lik=lpd_full, post_pred=post_pred)

    def to_arviz(self, dims: Tuple[int, int, int] | None = None) -> az.InferenceData:
        log_lik = {}
        post_pred = {}
        if self.log_lik is not None:
            log_lik = {
                "log_lik": self.log_lik.reshape(
                    self.log_lik.shape if dims is None else dims
                )
            }
        if self.post_pred is not None:
            post_pred = {
                "post_pred": self.post_pred.reshape(
                    self.post_pred.shape if dims is None else dims
                )
            }
        return az.from_dict(log_likelihood=log_lik, posterior_predictive=post_pred)


def compute_lpd(log_lik: np.ndarray) -> np.ndarray:
    max_ll = log_lik.max()
    return max_ll + np.log(np.mean(np.exp(log_lik - max_ll)))
