import numpy as np

# Define a list of equations with their
# corresponding variable ranges and step sizes
equations = [
    (lambda x1, x2: x1**2 + x2**2 - 1, (-1.0, 1.0, 0.05)),
    (lambda x1, x2: -(x1**2) + x2**2 - 1, (-1.0, 1.0, 0.05)),
    (lambda x1, x2: x1**3 - x1 - x2**2 + 1, (-1.0, 1.0, 0.05)),
    (lambda x1, x2, x3: x1**2 + x2**2 + x3**2 - 1, (-0.9, 0.9, ???), (-1.0, 1.0,???)),
    (lambda x1, x2: x1**2 + x2**2 - 25, (-5.0, 5.0, 0.24)),
    (lambda x1, x2: 4 * x1**2 + x2**2 - 25,
    (lambda x1, x2: 4 * x1**2 + 9 * x2**2 - 25,
    (lambda x1, x2: x1**2 + x2**2 - 100,
    (lambda x1, x2: x1**4 + x2**2 - 1, (-1.0, 1.0), 0.05),
    (lambda x1, x2: x1**6 + x2**2 - 1, (-1.0, 1.0), 0.05),
    (lambda x1, x2: x1**6 + x2**4 - 1, (-1.0, 1.0), 0.05),
    (lambda x1, x2, x3: x1 * x2 + x2 * x3 + x1 * x3 - 1,
    (lambda x1, x2, x3, x4: x1**2 + x2**2 + x3**2 + x4**2 - 1,
    (lambda x1, x2, x3: x1 * x2 * x3 + x1**2 + x2**2 * x3 - 1,
    (lambda x1, x2, x3, x4: x1**2 + x2**2 + x3 + x4 - 1,
    (lambda x1, x2: ((x1 - x2) / x2) - x2 - 1,
    (lambda x1, x2: (x1 / x2 - x1) - x2 - 1,
    (lambda x1, x2: np.sin(x1) + np.cos(x2) - 1,
    (lambda x1, x2: 4 * np.sin(x1) * np.cos(x2) - 1,
    (lambda x1, x2: np.sin(x1) + np.sin(x2 + x1**2) - 1,
    (lambda x1, x2, x3: np.log(x1) + np.log(x2) + x3,
]
