import numpy as np
import csv
from pathlib import Path

random_state = 42
noise = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

root_dir = Path(__file__).parent
data_dirs = [dir for dir in root_dir.iterdir() if dir.is_dir()]

for data_dir in data_dirs:
    base_data_path = data_dir / f"{data_dir.name}.csv"
    data = np.loadtxt(base_data_path, delimiter=",")
    for n in noise:
        noisy_data = data + np.random.RandomState(random_state).normal(0, n, data.shape)
        noisy_data_path = (
            data_dir / f"{data_dir.name}_noise_{int(np.abs(np.log10(n)).item())}.csv"
        )
        with open(noisy_data_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(noisy_data)
