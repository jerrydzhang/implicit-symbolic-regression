from typing import Self, Any, Dict
from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from gplearn.genetic import SymbolicRegressor

from jernerics.experiment import Experiment
from primel.samplers import (
    ImportanceSampler,
    RandomSampler,
    LHSampler,
)
from primel.distributions import (
    Empirical,
    GaussianKDE,
    MultivariateUniform,
)
from primel.early_stopping import EarlyStopping
from primel.adapters.gplearn import GPLearnAdapter

DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class GpLearnExperiment(Experiment):
    random_state: int

    def setup_data(self: Self, config: Dict[str, Any]) -> np.ndarray:
        data_path = DATA_DIR / config["data_file"]
        data = np.loadtxt(data_path, delimiter=",")
        return data

    def save_model(self: Self, result_path: Path, model: SymbolicRegressor) -> None:
        model_path = result_path / f"{self.task_id}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    def train(self: Self, data: np.ndarray, config: dict) -> SymbolicRegressor:
        empirical_dist = Empirical(data=data)
        gaussian_kde_dist = GaussianKDE(
            X=data, bandwidth=config["model__kde_bandwidth"]
        )
        uniform_dist = MultivariateUniform(X=data, margins=0.1, non_negative=False)

        sampler_entries = [
            ("train", RandomSampler(empirical_dist), config["model__n_train"]),
            ("kde", RandomSampler(gaussian_kde_dist), config["model__n_kde"]),
            ("uniform", LHSampler(uniform_dist), config["model__n_uniform"]),
        ]

        sampler = ImportanceSampler(
            sampler_entries=sampler_entries, random_state=self.random_state
        )
        samples = sampler.samples
        early_stopping = EarlyStopping(
            sampler=sampler,
            train_X="train",
            total_variance_threshold=1e-4,
            training_variance_threshold=1e-6,
        )
        adapter = GPLearnAdapter(
            sampler=sampler,
            reference_distribution=gaussian_kde_dist,
            early_stopping=early_stopping,
            mean_center_on="train",
            lambda_=1.0,
            exponent=1.0,
        )

        model = SymbolicRegressor(
            metric=adapter.get_fitness(),
            population_size=config["model__population_size"],
            generations=config["model__generations"],
            parsimony_coefficient=config["model__parsimony_coefficient"],
            stopping_criteria=0.0,
            verbose=1,
            random_state=self.random_state,
        )

        model.fit(samples, np.zeros(samples.shape[0]))
        return model

    def evaluate(self: Self, model: SymbolicRegressor, data: np.ndarray) -> dict:
        kl_div = model.score(data, np.zeros(data.shape[0]))
        return {"kl_divergence": kl_div}


def get_experiment(config: dict) -> Experiment:
    return ExampleExperiment(**config)
