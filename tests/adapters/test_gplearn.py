import numpy as np
import pytest

# Skip all tests in this module if gplearn is not installed
gplearn = pytest.importorskip("gplearn")

from gplearn.fitness import _Fitness

from primel.adapters.gplearn import GPLearnAdapter, convert_tree
from primel.distributions import MultivariateUniform
from primel.samplers import ImportanceSampler, RandomSampler


@pytest.fixture
def sampler():
    dist = MultivariateUniform(X=np.random.rand(10, 2), margins=0.1)
    sampler_entries = [("all", RandomSampler(dist), 10)]
    return ImportanceSampler(sampler_entries=sampler_entries)


@pytest.fixture
def reference_dist():
    return MultivariateUniform(X=np.random.rand(10, 2), margins=0.1)


def test_gplearn_adapter_init(sampler, reference_dist):
    adapter = GPLearnAdapter(
        sampler=sampler,
        reference_distribution=reference_dist,
    )
    assert adapter.lambda_ == 1.0


def test_gplearn_adapter_get_fitness(sampler, reference_dist):
    adapter = GPLearnAdapter(
        sampler=sampler,
        reference_distribution=reference_dist,
    )
    fitness_function = adapter.get_fitness()
    assert isinstance(fitness_function, _Fitness)
    assert not fitness_function.greater_is_better


def test_gplearn_build_tree():
    pass
