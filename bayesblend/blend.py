from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


def blend_draws(
    draws: Dict[str, Dict[str, np.ndarray]],
    weights: Dict[str, float],
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:

    np.random.seed(seed)

    M = len(draws)
    S = np.shape(list(list(draws.values())[0].values())[0])[0]
    N = np.shape(list(list(draws.values())[0].values()))[-1]

    # ensure same number of posterior samples
    for k, v in draws.items():
        for par, s in v.items():
            if np.shape(s)[0] != S:
                raise ValueError(
                    "The number of MCMC samples in `draws` is not consistent"
                )

    # use array to deal with multiple possible weights across observations
    weight_array = np.concatenate(list(weights.values())).T
    draws_idx_list = [
        np.random.choice(list(range(M)), S, p=weights) for weights in weight_array
    ]

    if len(draws_idx_list) != 1 and len(draws_idx_list) != N:
        raise ValueError(
            "Dimensions of `weights` do not match those of `draws`. Either a single "
            "set of weights should be supplied that will be applied to all observations "
            "in `draws`, or exactly one set of weights for each observation in `draws` "
            "should be supplied for pointwise blending."
        )

    if len(draws_idx_list) == 1:
        draws_idx_list = draws_idx_list * N

    blend = defaultdict(List[float])  # type: ignore
    blend_idx = {i: j for i, j in zip(draws.keys(), range(M))}
    for k, v in draws.items():
        blend_id = blend_idx[k]
        for par, s in v.items():
            blended_list = [
                list(s[draws_idx == blend_id, idx])
                for idx, draws_idx in enumerate(draws_idx_list)
            ]
            if par in blend:
                curr = blend[par]
                blend[par] = [c + b for c, b in zip(curr, blended_list)]  # type: ignore
            else:
                blend[par] = blended_list  # type: ignore

    return {par: np.asarray(blend[par]).T for par in blend}
