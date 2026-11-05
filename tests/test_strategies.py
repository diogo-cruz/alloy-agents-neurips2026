"""Tests for model selection strategies."""

from unittest.mock import MagicMock

import pytest

from alloy_safety.strategies import FixedSequence, RoundRobin, Roulette


def _mock_models(n: int) -> list:
    """Create mock model objects."""
    return [MagicMock(name=f"model_{i}") for i in range(n)]


class TestRoundRobin:
    def test_alternates_two_models(self):
        strategy = RoundRobin()
        models = _mock_models(2)

        results = [strategy.select(t, models) for t in range(6)]
        indices = [r[0] for r in results]

        assert indices == [0, 1, 0, 1, 0, 1]

    def test_cycles_three_models(self):
        strategy = RoundRobin()
        models = _mock_models(3)

        results = [strategy.select(t, models) for t in range(6)]
        indices = [r[0] for r in results]

        assert indices == [0, 1, 2, 0, 1, 2]

    def test_returns_correct_model(self):
        strategy = RoundRobin()
        models = _mock_models(2)

        _, model = strategy.select(0, models)
        assert model is models[0]

        _, model = strategy.select(1, models)
        assert model is models[1]


class TestRoulette:
    def test_deterministic_with_seed(self):
        models = _mock_models(2)

        strategy1 = Roulette(seed=42)
        strategy2 = Roulette(seed=42)

        results1 = [strategy1.select(t, models)[0] for t in range(10)]
        results2 = [strategy2.select(t, models)[0] for t in range(10)]

        assert results1 == results2

    def test_different_seeds_differ(self):
        models = _mock_models(2)

        strategy1 = Roulette(seed=42)
        strategy2 = Roulette(seed=123)

        results1 = [strategy1.select(t, models)[0] for t in range(20)]
        results2 = [strategy2.select(t, models)[0] for t in range(20)]

        # Very unlikely to be identical with different seeds over 20 trials
        assert results1 != results2

    def test_weighted_selection(self):
        models = _mock_models(2)
        # Heavily weight toward model 0
        strategy = Roulette(weights=[0.99, 0.01], seed=42)

        results = [strategy.select(t, models)[0] for t in range(100)]
        count_0 = results.count(0)

        # With 99% weight, should overwhelmingly select model 0
        assert count_0 > 80


class TestFixedSequence:
    def test_repeats_sequence(self):
        strategy = FixedSequence(sequence=[0, 0, 1])
        models = _mock_models(2)

        results = [strategy.select(t, models)[0] for t in range(9)]

        assert results == [0, 0, 1, 0, 0, 1, 0, 0, 1]

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError):
            FixedSequence(sequence=[])
