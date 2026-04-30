"""Tests for AFML Ch. 10 bet sizing from probability."""

from __future__ import annotations

import math

import pytest

from src.risk.bet_sizing import bet_size_from_probability


class TestBetSizeFromProbability:
    def test_below_min_p_returns_zero(self) -> None:
        assert bet_size_from_probability(0.50, min_p=0.55) == 0.0
        assert bet_size_from_probability(0.54, min_p=0.55) == 0.0

    def test_exactly_min_p_returns_zero(self) -> None:
        assert bet_size_from_probability(0.55, min_p=0.55) == 0.0

    def test_p_one_saturates_to_max(self) -> None:
        assert bet_size_from_probability(1.0, min_p=0.55, max_scale=1.0) == 1.0

    def test_monotone_increasing_above_cutoff(self) -> None:
        ps = [0.56, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 0.99]
        sizes = [bet_size_from_probability(p, min_p=0.55) for p in ps]
        for a, b in zip(sizes, sizes[1:]):
            assert a < b

    def test_returns_in_range(self) -> None:
        for p in [0.56, 0.60, 0.70, 0.85, 0.95]:
            s = bet_size_from_probability(p, min_p=0.55, max_scale=1.0)
            assert 0.0 <= s <= 1.0

    def test_known_value_p_equal_0_75(self) -> None:
        # m = 2 * Phi(z) - 1 where z = (p - 0.5) / sqrt(p(1-p))
        # For p=0.75: z = 0.25 / sqrt(0.1875) ≈ 0.5774
        # Phi(0.5774) ≈ 0.7181; m ≈ 0.4362
        s = bet_size_from_probability(0.75, min_p=0.55)
        assert s == pytest.approx(0.4362, abs=0.01)

    def test_max_scale_caps_output(self) -> None:
        s = bet_size_from_probability(0.99, min_p=0.55, max_scale=0.5)
        assert s <= 0.5

    def test_rejects_probabilities_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            bet_size_from_probability(1.5)
        with pytest.raises(ValueError):
            bet_size_from_probability(-0.1)

    def test_rejects_invalid_min_p(self) -> None:
        with pytest.raises(ValueError):
            bet_size_from_probability(0.7, min_p=1.5)
        with pytest.raises(ValueError):
            bet_size_from_probability(0.7, min_p=-0.1)

    def test_rejects_non_positive_max_scale(self) -> None:
        with pytest.raises(ValueError):
            bet_size_from_probability(0.7, max_scale=0)
        with pytest.raises(ValueError):
            bet_size_from_probability(0.7, max_scale=-1.0)

    def test_handles_p_at_half(self) -> None:
        # With min_p <= 0.5, p=0.5 gives z=0, m=0 (no edge)
        s = bet_size_from_probability(0.5, min_p=0.5)
        # p=0.5 with min_p=0.5 is at cutoff → 0 (cutoff uses strict >)
        assert s == 0.0
