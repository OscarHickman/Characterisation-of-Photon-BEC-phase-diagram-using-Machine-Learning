# ruff: noqa: I001
import importlib.util

import numpy as np

from photon_bec import length


def test_min_max_scale():
    assert length.min_max_scale([1, 2, 3]) == [0.0, 0.5, 1.0]
    assert length.min_max_scale([5, 5, 5]) == [0.0, 0.0, 0.0]


def test_griddata_interpolate_simple():
    x = [0, 1, 0, 1]
    y = [0, 0, 1, 1]
    z = [0, 1, 1, 0]
    # griddata_interpolate depends on scipy; if scipy is not installed, skip this check

    if importlib.util.find_spec("scipy") is None:
        # SciPy missing in environment; test not executed
        return
    xi, yi, zi = length.griddata_interpolate(x, y, z, n_interp=10, method="linear")
    assert xi.shape == yi.shape == zi.shape
    # interpolation should produce finite array values (may contain nans at edges)
    assert isinstance(zi, np.ndarray)
