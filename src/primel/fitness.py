from typing import Tuple, List

import numpy as np

from .samplers import ImportanceSampler
from .distributions import Distribution


def induced_kl_divergence(
    f_vals: np.ndarray,
    sampler: ImportanceSampler,
    reference_dist: Distribution,
    lambda_: float = 1.0,
    exponent: float = 1.0,
    mean_center_on: str | List[str] | None = None,
) -> float:
    f_vals = f_vals.astype(np.float64)
    epsilon = 1e-10

    if mean_center_on is not None:
        if isinstance(mean_center_on, str):
            mean_center_on = [mean_center_on]

        center_mask = np.zeros(len(f_vals), dtype=bool)
        for name in mean_center_on:
            try:
                idx = sampler.name_map[name]
                start, end = sampler.range_map[idx]
                center_mask[start:end] = True
            except KeyError:
                raise ValueError(
                    f"Component ''{name}'' provided in `mean_center_on` was not found in the sampler."
                )

        if np.any(center_mask):
            centering_mean = np.mean(f_vals[center_mask])
            f_vals = f_vals - centering_mean

    candidate_dist_unnorm = np.exp(-lambda_ * np.abs(f_vals) ** exponent)
    reference_dist_unnorm = reference_dist.pdf(sampler.samples)

    candidate_dist_norm = candidate_dist_unnorm / (
        np.sum(candidate_dist_unnorm) + epsilon
    )
    reference_dist_norm = reference_dist_unnorm / (
        np.sum(reference_dist_unnorm) + epsilon
    )

    log_dist_ratio = np.log(
        (reference_dist_norm + epsilon) / (candidate_dist_norm + epsilon)
    )

    fitness_per_sample = sampler.weights * reference_dist_norm * log_dist_ratio

    return np.sum(fitness_per_sample).item()


def train_val_variance_split(
    f_vals: np.ndarray,
    sampler: ImportanceSampler,
    train_component_names: str | List[str],
    mean_center: bool = True,
) -> Tuple[float, float]:
    if isinstance(train_component_names, str):
        train_component_names = [train_component_names]

    train_mask = np.zeros(len(f_vals), dtype=bool)
    for name in train_component_names:
        try:
            train_idx = sampler.name_map[name]
            start, end = sampler.range_map[train_idx]
            train_mask[start:end] = True
        except KeyError:
            raise ValueError(
                f"Component ''{name}'' given for `train_component_names` was not found in the sampler."
            )

    if not np.any(train_mask):
        raise ValueError(
            "No training samples found for the given `train_component_names`."
        )

    if mean_center:
        train_mean = np.mean(f_vals[train_mask])
        f_vals = f_vals - train_mean

    var_train = np.var(f_vals[train_mask]).item()
    var_total = np.var(f_vals).item()

    return var_total, var_train
