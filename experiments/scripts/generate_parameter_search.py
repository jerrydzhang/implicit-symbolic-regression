import pathlib

import numpy as np
import yaml
from scipy.stats import qmc

out = "gplearn_wide_search.yaml"
file = "gplearn_header.yaml"
n_experiments = 100
param_ranges = {
    "model__kde_bandwidth": (1e-6, 1e6),
    "model__n_train": (10, 90),
    "model__n_kde": (5, 500),
    "model__n_uniform": (5, 500),
    # gplearn specific
    "model__population_size": (500, 1000),
    "model__generations": (20, 100),
    "model__parsimony_coefficient": (1e-5, 1e-1),
}
log_params = ["model__kde_bandwidth"]
int_params = ["model__n_train", "model__n_kde", "model__n_uniform"]

inital_yaml = pathlib.Path(__file__).parent.parent / file

with open(inital_yaml, "r") as f:
    base_config = yaml.safe_load(f)

# Get the linear and log bounds separately
linear_bounds = []
log_bounds = []
param_names = list(param_ranges.keys())

for name in param_names:
    l_bound, u_bound = param_ranges[name]
    if name in log_params:
        log_bounds.append((np.log10(l_bound), np.log10(u_bound)))
    else:
        linear_bounds.append((l_bound, u_bound))

# Create separate samplers for log and linear parameters
sampler_log = qmc.LatinHypercube(d=len(log_params))
sampler_linear = qmc.LatinHypercube(d=len(param_names) - len(log_params))

# Sample in log space and then scale
log_sample = sampler_log.random(n=n_experiments)
log_sample_scaled = qmc.scale(
    log_sample,
    l_bounds=[v[0] for v in log_bounds],
    u_bounds=[v[1] for v in log_bounds],
)

# Sample in linear space and then scale
if linear_bounds:
    linear_sample = sampler_linear.random(n=n_experiments)
    linear_sample_scaled = qmc.scale(
        linear_sample,
        l_bounds=[v[0] for v in linear_bounds],
        u_bounds=[v[1] for v in linear_bounds],
    )

print(f"Generating {n_experiments} configurations in {out}")

sampled_params = []
for i in range(n_experiments):
    single_param = {"parameters": {}}
    log_values = 10 ** log_sample_scaled[i]
    linear_values = linear_sample_scaled[i] if linear_bounds else []

    log_idx, linear_idx = 0, 0
    for name in param_names:
        if name in log_params:
            single_param["parameters"][name] = float(log_values[log_idx])
            log_idx += 1
        else:
            if name in int_params:
                single_param["parameters"][name] = int(round(linear_values[linear_idx]))
            else:
                single_param["parameters"][name] = float(linear_values[linear_idx])
            linear_idx += 1

    sampled_params.append(single_param)

config = base_config.copy()
config["experiments"] = sampled_params
output_yaml = pathlib.Path(__file__).parent.parent / out
with open(output_yaml, "w") as f:
    yaml.dump(config, f)
