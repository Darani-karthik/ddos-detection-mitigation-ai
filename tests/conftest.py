import pytest

from ddos_ai.data import FlowData, build_simulated


@pytest.fixture(scope="session")
def small_df():
    """A small simulated data set: about 28 scenarios of 40 s, every scenario kind present."""
    return build_simulated(n_scenarios=30, seed=7, duration=40, cache_dir=None)


@pytest.fixture(scope="session")
def small_fd(small_df):
    return FlowData.from_frame(small_df)
