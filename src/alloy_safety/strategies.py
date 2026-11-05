"""Model selection strategies for alloy agents."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from inspect_ai.model import Model


class SelectionStrategy(ABC):
    """Base class for model selection strategies."""

    @abstractmethod
    def select(self, turn: int, models: list[Model]) -> tuple[int, Model]:
        """Select a model for the given turn.

        Returns:
            Tuple of (model_index, model).
        """
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class RoundRobin(SelectionStrategy):
    """Deterministic alternation between models."""

    def select(self, turn: int, models: list[Model]) -> tuple[int, Model]:
        idx = turn % len(models)
        return idx, models[idx]

    def name(self) -> str:
        return "round_robin"


class Roulette(SelectionStrategy):
    """Weighted random selection per turn."""

    def __init__(self, weights: list[float] | None = None, seed: int | None = None):
        self.weights = weights
        self._rng = random.Random(seed)

    def select(self, turn: int, models: list[Model]) -> tuple[int, Model]:
        if self.weights:
            (idx,) = self._rng.choices(range(len(models)), weights=self.weights, k=1)
        else:
            idx = self._rng.randint(0, len(models) - 1)
        return idx, models[idx]

    def name(self) -> str:
        return "roulette"


class FixedSequence(SelectionStrategy):
    """User-defined repeating sequence of model indices."""

    def __init__(self, sequence: list[int]):
        if not sequence:
            raise ValueError("Sequence must be non-empty")
        self.sequence = sequence

    def select(self, turn: int, models: list[Model]) -> tuple[int, Model]:
        idx = self.sequence[turn % len(self.sequence)]
        return idx, models[idx]

    def name(self) -> str:
        seq_str = "".join(str(i) for i in self.sequence)
        return f"fixed_{seq_str}"
