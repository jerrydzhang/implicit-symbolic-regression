from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Self

import numpy as np
from gplearn.functions import make_function
from jernerics.experiment import Experiment

from primel.adapters.gplearn import GPLearnAdapter, ImplicitSymbolicRegressor
from primel.distributions import (
    Empirical,
    GaussianKDE,
    MultivariateUniform,
)
from primel.early_stopping import EarlyStopping
from primel.fitness import induced_kl_divergence
from primel.samplers import (
    ImportanceSampler,
    LHSampler,
    RandomSampler,
)

DATA_DIR = Path(__file__).parent.parent / "data"


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(x2) > 1e-10, np.divide(x1, x2), 1.0)


def _protected_log(x1):
    """Closure of log for zero and negative arguments."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(x1) > 1e-10, np.log(np.abs(x1)), 0.0)


_div2 = make_function(function=_protected_division, name="div", arity=2)
_log1 = make_function(function=_protected_log, name="log", arity=1)


def _fit_model(
    model: ImplicitSymbolicRegressor,
    sampler: ImportanceSampler,
) -> ImplicitSymbolicRegressor:
    model.fit(sampler.samples, np.zeros(sampler.samples.shape[0]))
    return model


@dataclass
class GpLearnExperiment(Experiment):
    random_state: int
    metrics: dict[str, Any] = field(default_factory=dict)

    def setup_data(self: Self, config: Dict[str, Any]) -> np.ndarray:
        data_path = DATA_DIR / config["data_file"]
        data = np.loadtxt(data_path, delimiter=",")
        return data

    def save_model(
        self: Self, result_path: Path, model: ImplicitSymbolicRegressor
    ) -> None:
        pass
        # model_path = result_path / f"{self.task_id}_model.pkl"
        # with open(model_path, "wb") as f:
        #     pickle.dump(model, f)

    def train(self: Self, data: np.ndarray, config: dict) -> None:
        empirical_dist = Empirical(data=data)

        gaussian_kde_dist = GaussianKDE(
            X=data, bandwidth=config["model__kde_bandwidth"]
        )

        uniform_dist = MultivariateUniform(X=data, margins=0.1, non_negative=False)

        sampler_entries = [
            ("train", RandomSampler(empirical_dist), data.shape[0]),
            ("kde", RandomSampler(gaussian_kde_dist), config["model__n_kde"]),
            ("uniform", LHSampler(uniform_dist), config["model__n_uniform"]),
        ]

        sampler = ImportanceSampler(
            sampler_entries=sampler_entries, random_state=self.random_state
        )

        random_states = np.random.RandomState(self.random_state).randint(
            low=0,
            high=10000,
            size=config["n_trials"],
        )
        self.parameters["random_states"] = random_states.tolist()
        models = []
        for rs in random_states:
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

            model = ImplicitSymbolicRegressor(
                function_set=[
                    "add",
                    "sub",
                    "mul",
                    _div2,
                    "sin",
                    "cos",
                    "tan",
                    "sqrt",
                    _log1,
                ],
                adapter=adapter,
                # population_size=config["model__population_size"],
                # generations=config["model__generations"],
                # parsimony_coefficient=config["model__parsimony_coefficient"],
                max_length=100,
                population_size=1000,
                generations=100,
                parsimony_coefficient="auto",
                verbose=1,
                low_memory=True,
                random_state=rs,
            )

            models.append(model)

        for model in models:
            model.fit(sampler.samples, np.zeros(sampler.samples.shape[0]))

            self.metrics[f"trial_{len(self.metrics)}"] = self._compute_metrics(
                model, sampler, gaussian_kde_dist
            )

        # with ProcessPoolExecutor() as executor:
        #     results = executor.map(
        #         _fit_model,
        #         models,
        #         [sampler] * len(models),
        #     )
        #
        # models = list(results)
        # self.metrics = {
        #     f"trial_{i}": self._compute_metrics(model, sampler, gaussian_kde_dist)
        #     for i, model in enumerate(models)
        # }

    def _compute_metrics(
        self: Self,
        model: ImplicitSymbolicRegressor,
        sampler: ImportanceSampler,
        ref_dist: GaussianKDE,
    ) -> dict:
        f_vals = model.predict(sampler.samples)
        kl_div = induced_kl_divergence(
            f_vals,
            sampler,
            ref_dist,
            mean_center_on="train",
        )

        return {
            "success": model.early_stopped,
            "kl_divergence": kl_div,
            "equation": model._program.__str__(),
        }

    def evaluate(
        self: Self,
        model: ImplicitSymbolicRegressor,
        data: np.ndarray,
        config: dict,
    ) -> dict:
        if type(self.parameters["random_states"]) is not list:
            raise ValueError("random_states should be a list")

        total_kl_div = 0.0
        num_success = 0
        successful_random_states: List[int] = []
        for i in range(config["n_trials"]):
            trial_metrics = self.metrics.get(f"trial_{i}", {})
            total_kl_div += trial_metrics["kl_divergence"]
            if trial_metrics["success"]:
                num_success += 1
                successful_random_states.append(self.parameters["random_states"][i])

        total_kl_div /= config["n_trials"]

        self.metrics["total"] = {
            "num_success": num_success,
            "successful_random_states": successful_random_states,
            "kl_divergence": total_kl_div,
        }
        return self.metrics


def get_experiment(config: dict) -> Experiment:
    return GpLearnExperiment(**config)
