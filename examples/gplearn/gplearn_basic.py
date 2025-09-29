import numpy as np
import matplotlib.pyplot as plt

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

def generate_circle_data(n_points, noise):
    """Generate noisy circle data."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    X += np.random.normal(0, noise, X.shape)
    return X


random_state = 42
n_train = 200

train_data = generate_circle_data(200, 0.01)

empirical_dist = Empirical(data=train_data)
gaussian_kde_dist = GaussianKDE(X=train_data, bandwidth=0.1)
uniform_dist = MultivariateUniform(X=train_data, margins=0.1)

sampler_entries = [
    ("train", RandomSampler(empirical_dist), n_train),
    ("kde", RandomSampler(gaussian_kde_dist), n_train // 2),
    ("uniform", LHSampler(uniform_dist), n_train),
]

sampler = ImportanceSampler(sampler_entries=sampler_entries)

# --- Plotting Code ---

print("Visualizing the reference distribution density on the sampler points...")

# The reference distribution is gaussian_kde_dist
# Calculate the probability density for each point in the sampler
pdf_values = gaussian_kde_dist.pdf(sampler.samples)

# Create the plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    sampler.samples[:, 0],
    sampler.samples[:, 1],
    c=pdf_values,
    cmap='viridis',
    alpha=0.7
)

# Add a colorbar to show the probability density mapping
cbar = plt.colorbar(scatter)
cbar.set_label("Probability Density (from Reference KDE)")

# Add labels and title
plt.title("Importance Sampler Points Colored by Reference Distribution Density")
plt.xlabel("X0")
plt.ylabel("X1")
plt.grid(True, linestyle='--', alpha=0.6)

# Save the plot to a file
output_filename = "sampler_reference_plot.png"
plt.savefig(output_filename)

print(f"Plot saved to {output_filename}")