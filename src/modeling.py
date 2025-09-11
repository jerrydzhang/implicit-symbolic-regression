from dataclasses import dataclass, field, InitVar
from typing import Self, Protocol
import warnings
from functools import cached_property

import numpy as np
from scipy.stats import gaussian_kde
from scipy.special import ndtr
from scipy.optimize import brentq


def _validate_query_points(query_points: np.ndarray, n_features: int):
    if query_points.ndim == 1:
        query_points = query_points.reshape(1, -1)

    if query_points.shape[1] != n_features:
        raise ValueError(
            "n_features of query_points must match n_features of the distribution",
        )
    return query_points


class Distribution(Protocol):
    shape: tuple[int, ...]

    def pdf(self: Self, query_points: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def cdf(self: Self, query_points: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def ppf(self: Self, query_points: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample(
        self: Self,
        n_samples: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


@dataclass
class WeightedKDE:
    shape: tuple[int, ...] = field(init=False)

    X: np.ndarray
    bandwidth: float
    _weights: np.ndarray = field(init=False)

    _gaussian_kde: gaussian_kde = field(init=False, repr=False)

    weights: InitVar[np.ndarray | None] = None

    def __post_init__(self: Self, initial_weights: np.ndarray | None):
        self.shape = self.X.shape

        if initial_weights is None:
            self._weights = np.ones(len(self.X)) / len(self.X)
        else:
            self._weights = initial_weights / np.sum(initial_weights)

        self._gaussian_kde = gaussian_kde(
            dataset=self.X.T,
            weights=self._weights,
            bw_method=self.bandwidth,
        )

    def pdf(self: Self, query_points: np.ndarray) -> np.ndarray:
        query_points = _validate_query_points(query_points, self.shape[1])
        return self._gaussian_kde.evaluate(query_points.T)

    def cdf(self: Self, query_points: np.ndarray) -> np.ndarray:
        cdf = tuple(
            ndtr(np.ravel(item - self.X) / np.sqrt(self.bandwidth)).dot(self._weights)
            for item in query_points
        )

        return np.array(cdf).T

    def sample(
        self: Self,
        n_samples: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        return self._gaussian_kde.resample(n_samples, seed=random_state).T


@dataclass
class MultivariateUniform:
    shape: tuple[int, ...] = field(init=False)

    upper: np.ndarray = field(init=False)
    lower: np.ndarray = field(init=False)

    X: InitVar[np.ndarray]
    margins: InitVar[float | np.ndarray]
    non_negative: InitVar[bool] = True

    def __post_init__(
        self: Self,
        X: np.ndarray,
        margins: float | np.ndarray,
        non_negative: bool,
    ):
        self.shape = X.shape

        if isinstance(margins, float):
            margins = np.full(X.shape[1], margins)

        self.lower = np.min(X, axis=0) - margins
        self.upper = np.max(X, axis=0) + margins

        if non_negative:
            self.lower = np.maximum(self.lower, 0)

        if np.any(self.lower >= self.upper):
            warnings.warn(
                "Some lower bounds are greater than or equal to upper bounds. "
                "This Potentially will result in a zero-volume space.",
                UserWarning,
            )

        self.upper = np.maximum(self.upper, self.lower)

    def pdf(self: Self, query_points: np.ndarray) -> np.ndarray:
        query_points = _validate_query_points(query_points, self.shape[1])

        volume = np.prod(self.upper - self.lower)
        if volume == 0:
            warnings.warn(
                "The volume of the defined space is zero. "
                "This will lead to undefined behavior.",
                UserWarning,
            )
            return np.zeros(len(query_points))

        in_bounds = np.all(
            (query_points >= self.lower) & (query_points <= self.upper),
            axis=1,
        )

        uniform_density = 1.0 / volume
        pdf_values = np.where(in_bounds, uniform_density, 0.0)

        return pdf_values

    def cdf(self: Self, query_points: np.ndarray) -> np.ndarray:
        query_points = _validate_query_points(query_points, self.shape[1])

        volume = np.prod(self.upper - self.lower)
        if volume == 0:
            warnings.warn(
                "The volume of the defined space is zero. "
                "This will lead to undefined behavior.",
                UserWarning,
            )
            return np.zeros(len(query_points))

        intersection_lengths = np.maximum(
            0, np.minimum(query_points, self.upper) - self.lower
        )
        intersection_volume = np.prod(intersection_lengths, axis=1)

        return intersection_volume / volume

    def sample(
        self: Self,
        n_samples: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        samples = rng.uniform(
            low=self.lower, high=self.upper, size=(n_samples, self.shape[1])
        )
        return samples


@dataclass
class MixtureModel:
    shape: tuple[int, ...] = field(init=False)

    components: list[Distribution]
    weights: np.ndarray = field(init=False)

    initial_weights: InitVar[np.ndarray | None] = None

    def __post_init__(self: Self, initial_weights: np.ndarray | None):
        self.shape = self.components[0].shape

        if initial_weights is None:
            self.weights = np.ones(len(self.components)) / len(self.components)
        else:
            self.weights = initial_weights / np.sum(initial_weights)

    def pdf(self: Self, query_points: np.ndarray) -> np.ndarray:
        if query_points.ndim == 1:
            query_points = query_points.reshape(1, -1)

        pdf_values = np.sum(
            np.dot(
                np.asarray([c.pdf(query_points) for c in self.components]),
                self.weights,
            ),
            axis=0,
        )

        return pdf_values

    def cdf(self: Self, query_points: np.ndarray) -> np.ndarray:
        if query_points.ndim == 1:
            query_points = query_points.reshape(1, -1)

        cdf_values = np.sum(
            np.dot(
                np.asarray([c.cdf(query_points) for c in self.components]),
                self.weights,
            ),
            axis=0,
        )

        return cdf_values

    def sample(
        self: Self,
        n_samples: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(random_state)

        component_index_counts = rng.multinomial(n_samples, self.weights)
        component_random_states = rng.integers(0, 2**31, size=len(self.components))

        samples = []
        for count, component, component_rs in zip(
            component_index_counts,
            self.components,
            component_random_states,
        ):
            if count == 0:
                continue

            samples.append(component.sample(count, random_state=component_rs))

        return np.vstack(samples)
