import numpy as np
import pytest
from primel.distributions import Empirical, MixtureModel, MultivariateUniform
from primel.samplers import (
    RandomSampler,
    StratifiedSampler,
    LHSampler,
    ImportanceSampler,
)


@pytest.fixture
def sample_data():
    return np.array([[1, 2], [3, 4], [5, 6]])


@pytest.fixture
def empirical_dist(sample_data):
    return Empirical(data=sample_data)


@pytest.fixture
def uniform_dist(sample_data):
    return MultivariateUniform(X=sample_data, margins=0.1)


@pytest.fixture
def mixture_dist(sample_data):
    dist1 = MultivariateUniform(X=sample_data, margins=0.1)
    dist2 = MultivariateUniform(X=sample_data, margins=0.2)
    return MixtureModel(components=[dist1, dist2], weights=np.array([0.4, 0.6]))


def test_random_sampler(empirical_dist):
    sampler = RandomSampler(dist=empirical_dist)
    samples = sampler.sample(5, random_state=42)
    assert samples.shape == (5, 2)


def test_stratified_sampler(mixture_dist):
    sampler = StratifiedSampler(dist=mixture_dist)
    samples = sampler.sample(10, random_state=42)
    assert samples.shape == (10, 2)


def test_lh_sampler(uniform_dist):
    sampler = LHSampler(dist=uniform_dist)
    samples = sampler.sample(10, random_state=42)
    assert samples.shape == (10, 2)
    assert np.all(samples >= uniform_dist.lower)
    assert np.all(samples <= uniform_dist.upper)


def test_importance_sampler(empirical_dist, uniform_dist):
    sampler_entries = [
        ("empirical", RandomSampler(empirical_dist), 10),
        ("uniform", RandomSampler(uniform_dist), 20, 0.5),
    ]

    sampler = ImportanceSampler(sampler_entries=sampler_entries)

    assert list(sampler.name_map.keys()) == ["empirical", "uniform"]
    assert sampler.range_map == [(0, 10), (10, 30)]
    assert sampler.samples.shape == (30, 2)
    assert sampler.weights.shape == (30,)

    emp_samples = sampler.get_samples("empirical")
    assert emp_samples.shape == (10, 2)

    uni_weights = sampler.get_weights("uniform")
    assert uni_weights.shape == (20,)
    assert np.allclose(uni_weights, 0.5)

    with pytest.raises(ValueError):
        sampler.get_samples("nonexistent")

    with pytest.raises(ValueError):
        sampler.get_weights("nonexistent")
