from typing import List, Tuple, Protocol, Self

import numpy as np
from scipy.stats import qmc

from .modeling import Distribution, MixtureModel


class Sampler(Protocol):
    def sample(
        self: Self,
        dist: Distribution,
        n_samples: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


class RandomSampler:
    def sample(
        self: Self,
        dist: Distribution,
        n_samples: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        return dist.sample(n_samples, random_state)


class StratifiedSampler:
    def sample(
        self: Self,
        dist: MixtureModel,
        n_samples: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(random_state)

        samples_per_component = rng.multinomial(n_samples, dist.weights)
        component_random_states = rng.integers(0, 2**31, size=len(dist.components))

        samples_list: List[np.ndarray] = []
        for count, component, component_rs in zip(
            samples_per_component,
            dist.components,
            component_random_states,
        ):
            if count == 0:
                continue

            samples_list.append(
                qmc.LatinHypercube(
                    d=component.shape[0],
                    rng=rng,
                ).random(n=count)
            )


def importance_sampling(
    target_dist: Distribution,
    proposed_dist: Distribution,
    n_samples: int,
    sampler: Sampler,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = sampler.sample(proposed_dist, n_samples, random_state)
    target_densities = target_dist.pdf(samples)
    proposed_densities = proposed_dist.pdf(samples)

    sampling_weights = np.divide(
        1.0,
        proposed_densities,
        out=np.zeros_like(target_densities),
        where=proposed_densities != 0,
    )

    return samples, target_densities, sampling_weights
