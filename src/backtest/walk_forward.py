"""Walk-forward analysis: chronological train→test rolling windows.

Walk-forward is a complement to CPCV, not a replacement. CPCV tests
robustness to regime selection (which time slices to evaluate); walk-forward
tests robustness to chronological evolution (does the strategy keep working
as the calendar advances?). Different stress, same goal: avoid mistaking an
overfit point estimate for evidence of edge.

Output: a list of ``WalkForwardWindow`` objects compatible with the same
runner used for CPCV (it just sees a different list of test_indices and
train_indices per "path").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardWindow:
    train_indices: list[int]
    test_indices: list[int]


def generate_walk_forward_windows(
    n_observations: int,
    train_window: int,
    test_window: int,
    step: int,
) -> list[WalkForwardWindow]:
    """Build chronological train/test windows.

    Each window has:
        - train: ``[start, start+train_window)``
        - test:  ``[start+train_window, start+train_window+test_window)``

    Successive windows advance ``start`` by ``step``.
    """
    if train_window < 0:
        raise ValueError("train_window must be non-negative")
    if test_window <= 0:
        raise ValueError("test_window must be positive")
    if step <= 0:
        raise ValueError("step must be positive")

    windows: list[WalkForwardWindow] = []
    start = 0
    while start + train_window + test_window <= n_observations:
        train_indices = list(range(start, start + train_window))
        test_indices = list(range(
            start + train_window, start + train_window + test_window
        ))
        windows.append(WalkForwardWindow(
            train_indices=train_indices, test_indices=test_indices,
        ))
        start += step

    return windows
