import copy
import pathlib
import yaml
import numpy as np
from scipy.stats import qmc

# --- Configuration ---
OUTPUT_FILE = "zhong_noiseless.yaml"
BASE_CONFIG_FILE = "gplearn_header.yaml"
N_EXPERIMENTS = 60
DATA_FILES = [f"zhong/f{n:02d}/f{n:02d}.csv" for n in range(1, 24)]

PARAM_RANGES = {
    "model__kde_bandwidth": (1e-6, 1e6),
    "model__n_kde": (5, 500),
    "model__n_uniform": (5, 500),
    # GpLearn specific parameters
    "model__population_size": (500, 1000),
    "model__generations": (20, 100),
    "model__parsimony_coefficient": (1e-5, 1e-1),
}
LOG_PARAMS = {"model__kde_bandwidth", "model__parsimony_coefficient"}
INT_PARAMS = {
    "model__n_kde",
    "model__n_uniform",
    "model__population_size",
    "model__generations",
}


def generate_samples(n_samples, param_ranges, log_params, int_params):
    """Generates hyperparameter samples using Latin Hypercube Sampling."""
    all_names = list(param_ranges.keys())
    log_names = [name for name in all_names if name in log_params]
    linear_names = [name for name in all_names if name not in log_params]

    def sample_group(names, is_log_scale):
        if not names:
            return np.array([[] for _ in range(n_samples)])

        bounds = [param_ranges[name] for name in names]
        l_bounds = [b[0] for b in bounds]
        u_bounds = [b[1] for b in bounds]

        if is_log_scale:
            l_bounds = np.log10(l_bounds)
            u_bounds = np.log10(u_bounds)

        sampler = qmc.LatinHypercube(d=len(names))
        samples = sampler.random(n=n_samples)
        scaled_samples = qmc.scale(samples, l_bounds, u_bounds)

        return 10**scaled_samples if is_log_scale else scaled_samples

    scaled_log_samples = sample_group(log_names, is_log_scale=True)
    scaled_linear_samples = sample_group(linear_names, is_log_scale=False)

    experiments = []
    for i in range(n_samples):
        params = {}
        log_vals = dict(zip(log_names, scaled_log_samples[i]))
        linear_vals = dict(zip(linear_names, scaled_linear_samples[i]))

        for name, value in log_vals.items():
            params[name] = float(value)

        for name, value in linear_vals.items():
            params[name] = int(round(value)) if name in int_params else float(value)

        experiments.append({"parameters": params})
    return experiments


def main():
    """Generates and saves the YAML configuration for experiments."""
    script_dir = pathlib.Path(__file__).parent
    base_config_path = script_dir.parent / BASE_CONFIG_FILE
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    sampled_params = generate_samples(
        N_EXPERIMENTS, PARAM_RANGES, LOG_PARAMS, INT_PARAMS
    )

    final_params = []
    for data_file in DATA_FILES:
        for sample in sampled_params:
            param_copy = copy.deepcopy(sample)
            param_copy["parameters"]["data_file"] = data_file
            final_params.append(param_copy)

    print(f"Generating {len(final_params)} experiments...")
    config["experiments"] = final_params
    output_yaml_path = script_dir.parent / OUTPUT_FILE
    with open(output_yaml_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)


if __name__ == "__main__":
    main()
