from typing import Self
from dataclasses import dataclass
import warnings

import numpy as np
from gplearn.fitness import make_fitness

from primel.samplers import ImportanceSampler
from primel.distributions import Distribution
from primel.fitness import induced_kl_divergence
from primel.early_stopping import EarlyStopping

# Patch gplearn until https://github.com/trevorstephens/gplearn/issues/303
# is resolved
from gplearn.genetic import BaseSymbolic
from sklearn.utils.validation import validate_data

BaseSymbolic._validate_data = lambda self, *args, **kwargs: validate_data(
    self,
    *args,
    **kwargs,
)


@dataclass
class GPLearnAdapter:
    sampler: ImportanceSampler
    reference_distribution: Distribution
    early_stopping: EarlyStopping | None = None

    # Controls if early stopping is active to allow computing the actual
    # kl value
    eval: bool = False

    lambda_: float = 1.0
    exponent: float = 1.0
    mean_center_on: str | list[str] | None = None

    # Required due to gplearn running a dry run to validate fitness function
    dry_run: bool = True

    def get_fitness(self):
        def _induced_kl_divergence(
            _y: np.ndarray,
            y_pred: np.ndarray,
            _w: np.ndarray,
        ) -> float:
            if self.dry_run:
                return 0.0

            if len(y_pred) != len(self.sampler.samples):
                raise ValueError(
                    (
                        f"Number of predictions ({len(y_pred)}) does not "
                        f"match number of samples ({len(self.sampler.samples)})."
                    )
                )

            score = induced_kl_divergence(
                f_vals=y_pred,
                sampler=self.sampler,
                reference_dist=self.reference_distribution,
                lambda_=self.lambda_,
                exponent=self.exponent,
                mean_center_on=self.mean_center_on,
            )

            if self.early_stopping is not None:
                if self.early_stopping.check(y_pred):
                    # print(
                    #     "Early stopping criteria met. Stopping GP evolution.",
                    # )
                    score = 0.0

            return score

        fitness = make_fitness(
            function=_induced_kl_divergence,
            greater_is_better=False,
        )

        self.dry_run = False

        return fitness
