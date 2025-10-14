from dataclasses import dataclass, field, InitVar
from typing import Self, Protocol, Tuple
import warnings

import numpy as np
from primel.patch_kde import PicklableGaussianKDE as gaussian_kde
from scipy.spatial.distance import cdist
from scipy.optimize import nnls

__all__ = [
    "Distribution",
    "Empirical",
    "GaussianKDE",
    "MultivariateUniform",
    "MixtureModel",
]


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

    def sample(
        self: Self,
        n_samples: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


@dataclass
class Empirical:
    """
    A distribution based on a fixed dataset.
    Sampling from this distribution means drawing samples from the dataset.
    """

    data: np.ndarray
    shape: tuple[int, ...] = field(init=False)

    def __post_init__(self: Self):
        self.shape = self.data.shape

    def pdf(self: Self, query_points: np.ndarray) -> np.ndarray:
        """
        The PDF of a pure empirical distribution is a sum of Dirac delta functions,
        which is not practical for numerical computation. A smoothed version of this
        is the WeightedKDE. This method is not implemented as it's not well-defined
        for discrete points.
        """
        raise NotImplementedError(
            "PDF is not well-defined for an empirical distribution of discrete points. "
            "Consider using a WeightedKDE for a smoothed density approximation."
        )

    def sample(
        self: Self,
        n_samples: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        """
        Draw samples from the dataset.
        If n_samples is greater than the dataset size, it samples with replacement.
        Otherwise, it samples without replacement.
        """
        rng = np.random.default_rng(random_state)
        num_data_points = len(self.data)

        replace = n_samples > num_data_points

        indices = rng.choice(num_data_points, size=n_samples, replace=replace)
        return self.data[indices]


@dataclass
class GaussianKDE:
    shape: tuple[int, ...] = field(init=False)

    X: np.ndarray
    bandwidth: float
    weights: np.ndarray = field(init=False)

    _gaussian_kde: gaussian_kde = field(init=False, repr=False)

    def __post_init__(self: Self):
        self.shape = self.X.shape

        squared_diff = cdist(self.X, self.X, metric="sqeuclidean")

        gamma = 1 / (2 * self.bandwidth**2)
        G = np.exp(-gamma * squared_diff)
        ones = np.ones(self.X.shape[0])

        weights, _ = nnls(G, ones)
        self.weights = weights / np.sum(weights)

        self._gaussian_kde = gaussian_kde(
            dataset=self.X.T,
            weights=self.weights,
            bw_method=self.bandwidth,
        )

    @classmethod
    def from_weights(
        cls: type[Self],
        X: np.ndarray,
        bandwidth: float,
        initial_weights: np.ndarray,
    ) -> type[Self]:
        cls.shape = X.shape
        cls.X = X
        cls.bandwidth = bandwidth

        cls.weights = initial_weights / np.sum(initial_weights)
        cls._gaussian_kde = gaussian_kde(
            dataset=X.T,
            weights=cls.weights,
            bw_method=bandwidth,
        )

        return cls

    def pdf(self: Self, query_points: np.ndarray) -> np.ndarray:
        query_points = _validate_query_points(query_points, self.shape[1])
        return self._gaussian_kde.evaluate(query_points.T)

    def sample(
        self: Self,
        n_samples: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        return self._gaussian_kde.resample(n_samples, seed=random_state).T


@dataclass
class MultivariateUniform:
    shape: Tuple[int, ...] = field(init=False)

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
    _weights: np.ndarray = field(init=False)

    weights: InitVar[np.ndarray | None] = None

    def __post_init__(self: Self, weights: np.ndarray | None):
        self.shape = self.components[0].shape

        if weights is None:
            self._weights = np.ones(len(self.components)) / len(self.components)
        else:
            self._weights = weights / np.sum(weights)

    def pdf(self: Self, query_points: np.ndarray) -> np.ndarray:
        if query_points.ndim == 1:
            query_points = query_points.reshape(1, -1)

        pdf_values = np.zeros(query_points.shape[0])
        for weight, component in zip(self._weights, self.components):
            pdf_values += weight * component.pdf(query_points)

        return pdf_values

    def sample(
        self: Self,
        n_samples: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(random_state)

        component_index_counts = rng.multinomial(n_samples, self._weights)
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
