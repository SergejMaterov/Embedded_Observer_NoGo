import os
import sys

import numpy as np
import pytest

# make the repository root importable (embedded_observer_checks.py lives there)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import embedded_observer_checks as ec  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_rng():
    """Every test gets its own seeded generator, independent of test order."""
    ec.rng = np.random.default_rng(12345)
    yield
