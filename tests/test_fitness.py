import numpy as np
import pytest
from primel.fitness import induced_kl_divergence, train_val_variance_split
from primel.samplers import ImportanceSampler, RandomSampler
from primel.distributions import MultivariateUniform, Empirical

@pytest.fixture
def f_vals():
    return np.random.rand(30)

@pytest.fixture
def sampler():
    dist1 = MultivariateUniform(X=np.random.rand(10, 2), margins=0.1)
    dist2 = MultivariateUniform(X=np.random.rand(20, 2), margins=0.1)
    sampler_entries = [
        ("train", RandomSampler(dist1), 10),
        ("val", RandomSampler(dist2), 20),
    ]
    return ImportanceSampler(sampler_entries=sampler_entries)

@pytest.fixture
def reference_dist():
    return MultivariateUniform(X=np.random.rand(30, 2), margins=0.1)

def test_induced_kl_divergence(f_vals, sampler, reference_dist):
    fitness = induced_kl_divergence(f_vals, sampler, reference_dist)
    assert isinstance(fitness, float)

    fitness_centered = induced_kl_divergence(
        f_vals, sampler, reference_dist, mean_center_on="train"
    )
    assert isinstance(fitness_centered, float)

    with pytest.raises(ValueError):
        induced_kl_divergence(
            f_vals, sampler, reference_dist, mean_center_on="nonexistent"
        )

def test_train_val_variance_split(f_vals, sampler):
    var_train, var_val = train_val_variance_split(
        f_vals, sampler, train_component_names="train"
    )
    assert isinstance(var_train, float)
    assert isinstance(var_val, float)

    var_train_centered, var_val_centered = train_val_variance_split(
        f_vals, sampler, train_component_names=["train"], mean_center=True
    )
    assert isinstance(var_train_centered, float)
    assert isinstance(var_val_centered, float)

    with pytest.raises(ValueError):
        train_val_variance_split(f_vals, sampler, train_component_names="nonexistent")

    # Test case where no training samples are found
    dist = Empirical(data=np.array([]))
    empty_sampler_entries = [("train", RandomSampler(dist), 0)]
    sampler_no_train = ImportanceSampler(sampler_entries=empty_sampler_entries)
    with pytest.raises(ValueError):
        train_val_variance_split(np.array([]), sampler_no_train, train_component_names="train")
