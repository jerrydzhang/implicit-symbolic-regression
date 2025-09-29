import numpy as np
import pytest
from primel.distributions import (
    Empirical,
    GaussianKDE,
    MultivariateUniform,
    MixtureModel,
)


@pytest.fixture
def sample_data():
    return np.array([[1, 2], [3, 4], [5, 7]])


def test_empirical_distribution(sample_data):
    dist = Empirical(data=sample_data)
    assert dist.shape == sample_data.shape

    # Test sampling without replacement
    samples_less = dist.sample(2, random_state=42)
    assert samples_less.shape == (2, 2)
    assert len(np.unique(samples_less, axis=0)) == 2

    # Test sampling with replacement
    samples_more = dist.sample(5, random_state=42)
    assert samples_more.shape == (5, 2)

    with pytest.raises(NotImplementedError):
        dist.pdf(sample_data)


def test_gaussian_kde(sample_data):
    dist = GaussianKDE(X=sample_data, bandwidth=0.5)
    assert dist.shape == sample_data.shape
    assert np.isclose(np.sum(dist.weights), 1.0)

    query_points = np.array([[1.1, 2.1], [3.1, 4.1]])
    pdf_values = dist.pdf(query_points)
    assert pdf_values.shape == (2,)
    assert np.all(pdf_values >= 0)

    samples = dist.sample(10, random_state=42)
    assert samples.shape == (10, 2)


def test_gaussian_kde_from_weights(sample_data):
    initial_weights = np.array([0.1, 0.5, 0.4])
    dist = GaussianKDE.from_weights(
        X=sample_data, bandwidth=0.5, initial_weights=initial_weights
    )
    assert dist.shape == sample_data.shape
    assert np.allclose(dist.weights, initial_weights / np.sum(initial_weights))


def test_multivariate_uniform(sample_data):
    dist = MultivariateUniform(X=sample_data, margins=0.1, non_negative=True)
    assert dist.shape == sample_data.shape
    assert np.all(dist.lower >= 0)

    # Test pdf
    in_bounds_points = np.array([[1.0, 2.0]])
    out_of_bounds_points = np.array([[10.0, 10.0]])
    pdf_in = dist.pdf(in_bounds_points)
    pdf_out = dist.pdf(out_of_bounds_points)
    assert pdf_in[0] > 0
    assert pdf_out[0] == 0

    # Test sampling
    samples = dist.sample(10, random_state=42)
    assert samples.shape == (10, 2)
    assert np.all(samples >= dist.lower)
    assert np.all(samples <= dist.upper)


def test_mixture_model(sample_data):
    dist1 = MultivariateUniform(X=sample_data, margins=0.1)
    dist2 = MultivariateUniform(X=sample_data, margins=0.2)

    # Test with no weights provided
    mixture = MixtureModel(components=[dist1, dist2])
    assert mixture.shape == sample_data.shape
    assert np.allclose(mixture._weights, np.array([0.5, 0.5]))

    # Test with weights provided
    weights = np.array([0.3, 0.7])
    mixture = MixtureModel(components=[dist1, dist2], weights=weights)
    assert np.allclose(mixture._weights, weights)

    # Test pdf
    query_points = np.array([[1.0, 2.0]])
    pdf_values = mixture.pdf(query_points)
    print(pdf_values)
    assert pdf_values.shape == (1,)

    # Test sampling
    samples = mixture.sample(10, random_state=42)
    assert samples.shape == (10, 2)
