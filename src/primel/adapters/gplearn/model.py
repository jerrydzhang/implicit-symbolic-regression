import itertools
from time import time
from warnings import warn

import numpy as np
from gplearn._program import _Program
from gplearn.fitness import _Fitness, _fitness_map
from gplearn.functions import _Function, _function_map
from gplearn.functions import sig1 as sigmoid
from gplearn.genetic import MAX_INT, BaseSymbolic, SymbolicRegressor, _parallel_evolve
from gplearn.utils import _partition_estimators, check_random_state
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import (
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.exceptions import NotFittedError
from sklearn.utils import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight, check_array, validate_data

from primel.tree import ExpressionTree, Node

from .adapter import GPLearnAdapter

# Patch gplearn until https://github.com/trevorstephens/gplearn/issues/303
# is resolved
BaseSymbolic._validate_data = lambda self, *args, **kwargs: validate_data(  # type: ignore
    self,
    *args,
    **kwargs,
)


def _build_tree(program: list) -> ExpressionTree:
    """Build an ExpressionTree from a gplearn program representation.

    Parameters
    ----------
    program : list
        The gplearn program representation as a list of nodes.

    Returns
    -------
    ExpressionTree
        The constructed ExpressionTree.

    """
    nodes = []
    for node in program:
        if isinstance(node, _Function):
            nodes.append(Node(name=node.name, value=node.function, arity=node.arity))
        elif isinstance(node, int):
            nodes.append(
                Node(name=f"x{node}", value=lambda x, n=node: x[:, n], arity=0)
            )
        elif isinstance(node, float):
            nodes.append(Node(name="constant", value=node, arity=0))

    return ExpressionTree.init_from_list(nodes)


class ImplicitSymbolicRegressor(SymbolicRegressor):
    def __init__(
        self,
        *,
        population_size=1000,
        generations=20,
        tournament_size=20,
        const_range=(-1.0, 1.0),
        init_depth=(2, 6),
        init_method="half and half",
        function_set=("add", "sub", "mul", "div"),
        adapter: GPLearnAdapter,
        parsimony_coefficient=0.001,
        p_crossover=0.9,
        p_subtree_mutation=0.01,
        p_hoist_mutation=0.01,
        p_point_mutation=0.01,
        p_point_replace=0.05,
        max_samples=1.0,
        feature_names=None,
        warm_start=False,
        low_memory=False,
        n_jobs=1,
        verbose=0,
        random_state=None,
    ):
        super().__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=0.0,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            metric=adapter.get_fitness(),
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )
        self.adapter = adapter
        self.early_stopped = False

    def fit(self, X, y, sample_weight=None):
        """Fit the Genetic Program according to X, y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.

        Returns
        -------
        self : object
            Returns self.

        """
        random_state = check_random_state(self.random_state)

        # Check arrays
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if isinstance(self, ClassifierMixin):
            X, y = self._validate_data(X, y, y_numeric=False)
            check_classification_targets(y)

            if self.class_weight:
                if sample_weight is None:
                    sample_weight = 1.0
                # modify the sample weights with the corresponding class weight
                sample_weight = sample_weight * compute_sample_weight(
                    self.class_weight, y
                )

            self.classes_, y = np.unique(y, return_inverse=True)
            n_trim_classes = np.count_nonzero(np.bincount(y, sample_weight))
            if n_trim_classes != 2:
                raise ValueError(
                    "y contains %d class after sample_weight "
                    "trimmed classes with zero weights, while 2 "
                    "classes are required." % n_trim_classes
                )
            self.n_classes_ = len(self.classes_)

        else:
            X, y = self._validate_data(X, y, y_numeric=True)

        hall_of_fame = self.hall_of_fame
        if hall_of_fame is None:
            hall_of_fame = self.population_size
        if hall_of_fame > self.population_size or hall_of_fame < 1:
            raise ValueError(
                "hall_of_fame (%d) must be less than or equal to "
                "population_size (%d)." % (self.hall_of_fame, self.population_size)
            )
        n_components = self.n_components
        if n_components is None:
            n_components = hall_of_fame
        if n_components > hall_of_fame or n_components < 1:
            raise ValueError(
                "n_components (%d) must be less than or equal to "
                "hall_of_fame (%d)." % (self.n_components, self.hall_of_fame)
            )

        self._function_set = []
        for function in self.function_set:
            if isinstance(function, str):
                if function not in _function_map:
                    raise ValueError(
                        "invalid function name %s found in `function_set`." % function
                    )
                self._function_set.append(_function_map[function])
            elif isinstance(function, _Function):
                self._function_set.append(function)
            else:
                raise ValueError(
                    "invalid type %s found in `function_set`." % type(function)
                )
        if not self._function_set:
            raise ValueError("No valid functions found in `function_set`.")

        # For point-mutation to find a compatible replacement node
        self._arities = {}
        for function in self._function_set:
            arity = function.arity
            self._arities[arity] = self._arities.get(arity, [])
            self._arities[arity].append(function)

        if isinstance(self.metric, _Fitness):
            self._metric = self.metric
        elif isinstance(self, RegressorMixin):
            if self.metric not in (
                "mean absolute error",
                "mse",
                "rmse",
                "pearson",
                "spearman",
            ):
                raise ValueError("Unsupported metric: %s" % self.metric)
            self._metric = _fitness_map[self.metric]
        elif isinstance(self, ClassifierMixin):
            if self.metric != "log loss":
                raise ValueError("Unsupported metric: %s" % self.metric)
            self._metric = _fitness_map[self.metric]
        elif isinstance(self, TransformerMixin):
            if self.metric not in ("pearson", "spearman"):
                raise ValueError("Unsupported metric: %s" % self.metric)
            self._metric = _fitness_map[self.metric]

        self._method_probs = np.array(
            [
                self.p_crossover,
                self.p_subtree_mutation,
                self.p_hoist_mutation,
                self.p_point_mutation,
            ]
        )
        self._method_probs = np.cumsum(self._method_probs)

        if self._method_probs[-1] > 1:
            raise ValueError(
                "The sum of p_crossover, p_subtree_mutation, "
                "p_hoist_mutation and p_point_mutation should "
                "total to 1.0 or less."
            )

        if self.init_method not in ("half and half", "grow", "full"):
            raise ValueError(
                "Valid program initializations methods include "
                '"grow", "full" and "half and half". Given %s.' % self.init_method
            )

        if not (
            (isinstance(self.const_range, tuple) and len(self.const_range) == 2)
            or self.const_range is None
        ):
            raise ValueError("const_range should be a tuple with length two, or None.")

        if not isinstance(self.init_depth, tuple) or len(self.init_depth) != 2:
            raise ValueError("init_depth should be a tuple with length two.")
        if self.init_depth[0] > self.init_depth[1]:
            raise ValueError(
                "init_depth should be in increasing numerical "
                "order: (min_depth, max_depth)."
            )

        if self.feature_names is not None:
            if self.n_features_in_ != len(self.feature_names):
                raise ValueError(
                    "The supplied `feature_names` has different "
                    "length to n_features. Expected %d, got %d."
                    % (self.n_features_in_, len(self.feature_names))
                )
            for feature_name in self.feature_names:
                if not isinstance(feature_name, str):
                    raise ValueError(
                        "invalid type %s found in `feature_names`." % type(feature_name)
                    )

        if self.transformer is not None:
            if isinstance(self.transformer, _Function):
                self._transformer = self.transformer
            elif self.transformer == "sigmoid":
                self._transformer = sigmoid
            else:
                raise ValueError(
                    "Invalid `transformer`. Expected either "
                    '"sigmoid" or _Function object, got %s' % type(self.transformer)
                )
            if self._transformer.arity != 1:
                raise ValueError(
                    "Invalid arity for `transformer`. Expected 1, "
                    "got %d." % (self._transformer.arity)
                )

        params = self.get_params()
        params["_metric"] = self._metric
        if hasattr(self, "_transformer"):
            params["_transformer"] = self._transformer
        else:
            params["_transformer"] = None
        params["function_set"] = self._function_set
        params["arities"] = self._arities
        params["method_probs"] = self._method_probs

        if not self.warm_start or not hasattr(self, "_programs"):
            # Free allocated memory, if any
            self._programs = []
            self.run_details_ = {
                "generation": [],
                "average_length": [],
                "average_fitness": [],
                "best_length": [],
                "best_fitness": [],
                "best_oob_fitness": [],
                "generation_time": [],
            }

        prior_generations = len(self._programs)
        n_more_generations = self.generations - prior_generations

        if n_more_generations < 0:
            raise ValueError(
                "generations=%d must be larger or equal to "
                "len(_programs)=%d when warm_start==True"
                % (self.generations, len(self._programs))
            )
        elif n_more_generations == 0:
            fitness = [program.raw_fitness_ for program in self._programs[-1]]
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new programs."
            )

        if self.warm_start:
            # Generate and discard seeds that would have been produced on the
            # initial fit call.
            for i in range(len(self._programs)):
                _ = random_state.randint(MAX_INT, size=self.population_size)

        if self.verbose:
            # Print header fields
            self._verbose_reporter()

        for gen in range(prior_generations, self.generations):
            start_time = time()

            if gen == 0:
                parents = None
            else:
                parents = self._programs[gen - 1]

            # Parallel loop
            n_jobs, n_programs, starts = _partition_estimators(
                self.population_size, self.n_jobs
            )
            seeds = random_state.randint(MAX_INT, size=self.population_size)

            population = Parallel(n_jobs=n_jobs, verbose=int(self.verbose > 1))(
                delayed(_parallel_evolve)(
                    n_programs[i],
                    parents,
                    X,
                    y,
                    sample_weight,
                    seeds[starts[i] : starts[i + 1]],
                    params,
                )
                for i in range(n_jobs)
            )

            # Reduce, maintaining order across different n_jobs
            population = list(itertools.chain.from_iterable(population))

            fitness = [program.raw_fitness_ for program in population]
            length = [program.length_ for program in population]

            parsimony_coefficient = None
            if self.parsimony_coefficient == "auto":
                parsimony_coefficient = np.cov(length, fitness)[1, 0] / np.var(length)
            for program in population:
                program.fitness_ = program.fitness(parsimony_coefficient)

            self._programs.append(population)

            # Remove old programs that didn't make it into the new population.
            if not self.low_memory:
                for old_gen in np.arange(gen, 0, -1):
                    indices = []
                    for program in self._programs[old_gen]:
                        if program is not None:
                            for idx in program.parents:
                                if "idx" in idx:
                                    indices.append(program.parents[idx])
                    indices = set(indices)
                    for idx in range(self.population_size):
                        if idx not in indices:
                            self._programs[old_gen - 1][idx] = None
            elif gen > 0:
                # Remove old generations
                self._programs[gen - 1] = None

            # Record run details
            if self._metric.greater_is_better:
                best_program = population[np.argmax(fitness)]
            else:
                best_program = population[np.argmin(fitness)]

            self.run_details_["generation"].append(gen)
            self.run_details_["average_length"].append(np.mean(length))
            self.run_details_["average_fitness"].append(np.mean(fitness))
            self.run_details_["best_length"].append(best_program.length_)
            self.run_details_["best_fitness"].append(best_program.raw_fitness_)
            oob_fitness = np.nan
            if self.max_samples < 1.0:
                oob_fitness = best_program.oob_fitness_
            self.run_details_["best_oob_fitness"].append(oob_fitness)
            generation_time = time() - start_time
            self.run_details_["generation_time"].append(generation_time)

            if self.verbose:
                self._verbose_reporter(self.run_details_)

            # NOTE: custom early stopping check

            # Check for early stopping
            if self.adapter.early_stopping is not None:
                # Evaluate best program on all samples
                simplified_tree = _build_tree(best_program.program)
                y_pred = simplified_tree.evaluate(X)
                if self.adapter.early_stopping.check(y_pred):
                    self.early_stopped = True
                    if self.verbose:
                        print(f"Early stopping triggered at generation {gen}.")
                    break

        if isinstance(self, TransformerMixin):
            # Find the best individuals in the final generation
            fitness = np.array(fitness)
            if self._metric.greater_is_better:
                hall_of_fame = fitness.argsort()[::-1][: self.hall_of_fame]
            else:
                hall_of_fame = fitness.argsort()[: self.hall_of_fame]
            evaluation = np.array(
                [gp.execute(X) for gp in [self._programs[-1][i] for i in hall_of_fame]]
            )
            if self.metric == "spearman":
                evaluation = np.apply_along_axis(rankdata, 1, evaluation)

            with np.errstate(divide="ignore", invalid="ignore"):
                correlations = np.abs(np.corrcoef(evaluation))
            np.fill_diagonal(correlations, 0.0)
            components = list(range(self.hall_of_fame))
            indices = list(range(self.hall_of_fame))
            # Iteratively remove least fit individual of most correlated pair
            while len(components) > self.n_components:
                most_correlated = np.unravel_index(
                    np.argmax(correlations), correlations.shape
                )
                # The correlation matrix is sorted by fitness, so identifying
                # the least fit of the pair is simply getting the higher index
                worst = max(most_correlated)
                components.pop(worst)
                indices.remove(worst)
                correlations = correlations[:, indices][indices, :]
                indices = list(range(len(components)))
            self._best_programs = [
                self._programs[-1][i] for i in hall_of_fame[components]
            ]

        else:
            # Find the best individual in the final generation
            if self._metric.greater_is_better:
                self._program = self._programs[-1][np.argmax(fitness)]
            else:
                self._program = self._programs[-1][np.argmin(fitness)]

        return self
