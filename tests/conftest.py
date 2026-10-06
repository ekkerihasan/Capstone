"""Test fixtures built on the shared phantom in `src.phantom`.

The generator used to live here. It moved to `src/phantom.py` so the demo
pipeline, the notebooks and the tests all exercise the *same* synthetic data —
a phantom that only pytest can build is one the review demo cannot use.
"""
import pytest

from src.phantom import RAW_SPACING, SHAPE, write_phantom_dataset  # noqa: F401  (re-exported)


@pytest.fixture(scope="session")
def msd_phantom(tmp_path_factory):
    """Returns the root of a directory that looks like Task07_Pancreas."""
    root = tmp_path_factory.mktemp("Task07_Phantom")
    return write_phantom_dataset(root, n_cases=6, include_junk=True)
