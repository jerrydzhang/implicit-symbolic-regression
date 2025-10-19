from dataclasses import dataclass, field
from typing import List, Self
import numpy as np

from .samplers import ImportanceSampler
from .fitness import train_val_variance_split


@dataclass
class EarlyStopping:
    sampler: ImportanceSampler
    train_X: str | List[str]
    total_variance_threshold: float
    training_variance_threshold: float

    history: List[bool] = field(init=False)

    def __post_init__(self):
        self.history = []
        if isinstance(self.train_X, str):
            self.train_X = [self.train_X]

    def check(self: Self, y: np.ndarray) -> bool:
        varience = train_val_variance_split(y, self.sampler, self.train_X)
        result = (
            varience[0] > self.total_variance_threshold
            and varience[1] < self.training_variance_threshold
        )
        # print(f"Early stopping check: {varience}, stop: {result}")
        self.history.append(result)
        return result
