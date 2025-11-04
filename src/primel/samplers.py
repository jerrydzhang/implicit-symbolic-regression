import dataclasses
from dataclasses import InitVar, dataclass, field
from typing import Dict, List, Protocol, Self, Sequence, Tuple, TypeVar

import numpy as np
from scipy.stats import qmc

from .distributions import Distribution, MixtureModel, MultivariateUniform

__all__ = [
    "Sampler",
    "RandomSampler",
    "StratifiedSampler",
    "LHSampler",
    "ImportanceSampler",
]

T_Distribution = TypeVar("T_Distribution", bound=Distribution)


@dataclass
class Sampler(Protocol[T_Distribution]):
    dist: T_Distribution

    def sample(
        self: Self,
        n_samples: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


@dataclass
class RandomSampler:
    dist: Distribution

    def sample(
        self: Self,
        n_samples: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        return self.dist.sample(n_samples, random_state)


@dataclass
class StratifiedSampler:
    dist: MixtureModel

    def sample(
        self: Self,
        n_samples: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(random_state)

        samples_per_component = rng.multinomial(
            n_samples,
            self.dist._weights,
        )
        component_random_states = rng.integers(
            0,
            2**31,
            size=len(self.dist.components),
        )

        samples_list: List[np.ndarray] = []
        for count, component, component_rs in zip(
            samples_per_component,
            self.dist.components,
            component_random_states,
        ):
            if count == 0:
                continue

            samples_list.append(
                component.sample(
                    count,
                    random_state=component_rs,
                )
            )

        return np.vstack(samples_list)


@dataclass
class LHSampler:
    dist: MultivariateUniform

    def sample(
        self: Self,
        n_samples: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=self.dist.shape[1], rng=random_state)
        samples = sampler.random(n=n_samples)
        scaled_samples = qmc.scale(samples, self.dist.lower, self.dist.upper)
        return scaled_samples


SamplerEntry = Tuple[str, Sampler, int]
WeightedSamplerEntry = Tuple[str, Sampler, int, float]


@dataclass
class ImportanceSampler:
    sampler_entries: InitVar[Sequence[SamplerEntry | WeightedSamplerEntry]]

    name_map: Dict[str, int] = field(init=False)
    range_map: List[Tuple[int, int]] = field(init=False)

    samples: np.ndarray = field(init=False)
    weights: np.ndarray = field(init=False)
    samplers: List[Sampler] = field(init=False)

    random_state: int | None = None

    def __post_init__(
        self: Self,
        sampler_entries: Sequence[SamplerEntry | WeightedSamplerEntry],
    ):
        self.samplers = []
        self.name_map: Dict[str, int] = {}
        self.range_map: List[Tuple[int, int]] = []

        samples_list: List[np.ndarray] = []
        weights_list: List[np.ndarray] = []

        pointer = 0
        for i, entry in enumerate(sampler_entries):
            if len(entry) == 4:
                name, sampler, n_samples, weight = entry
            elif len(entry) == 3:
                name, sampler, n_samples = entry
                weight = 1 / n_samples if n_samples > 0 else 0
            else:
                raise ValueError(
                    "Sampler entry must be of the form "
                    "(name, sampler, n_samples) or "
                    "(name, sampler, n_samples, weight)."
                )

            self.samplers.append(sampler)

            if n_samples > 0:
                samples = sampler.sample(n_samples, random_state=self.random_state)
                samples_list.append(samples)
                weights_list.append(np.full(n_samples, weight))

            self.range_map.append((pointer, pointer + n_samples))
            self.name_map[name] = i
            pointer += n_samples

        if samples_list:
            self.samples = np.vstack(samples_list)
            self.weights = np.hstack(weights_list)
            self.weights = self.weights / np.sum(self.weights)
        else:
            self.samples = np.array([])
            self.weights = np.array([])
            if sampler_entries:
                try:
                    d = sampler_entries[0][1].dist.shape[1]
                    self.samples = np.empty((0, d))
                except Exception:
                    pass  # keep samples as empty array

    def get_samples(self: Self, name: str) -> np.ndarray:
        if name not in self.name_map:
            raise ValueError(f"Sampler with name '{name}' not found.")

        index = self.name_map[name]
        start, end = self.range_map[index]
        return self.samples[start:end]

    def get_weights(self: Self, name: str) -> np.ndarray:
        if name not in self.name_map:
            raise ValueError(f"Sampler with name '{name}' not found.")

        index = self.name_map[name]
        start, end = self.range_map[index]
        return self.weights[start:end]

    def reweight_by_dist(self: Self, dist: Distribution) -> None:
        pdf_values = dist.pdf(self.samples)
        mid = (np.max(pdf_values) - np.min(pdf_values)) / 2.0
