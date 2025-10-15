from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray


def min_max_scale(values: Iterable[Union[float, int]]) -> List[float]:
    """Scale a list of numeric values to the [0,1] range using min-max scaling.

    If all values are equal, returns a list of zeros.
    """
    vals = np.array(list(values), dtype=float)
    vmin = vals.min()
    vmax = vals.max()
    if vmax == vmin:
        return [0.0 for _ in vals]
    return ((vals - vmin) / (vmax - vmin)).tolist()


def griddata_interpolate(
    x: Iterable[Union[float, int]],
    y: Iterable[Union[float, int]],
    z: Iterable[Union[float, int]],
    n_interp: int = 200,
    method: str = "linear",
) -> Tuple[ndarray, ndarray, ndarray]:
    """Interpolate scattered data (x,y,z) onto a regular grid and return (xi, yi, zi).

    Returns:
        xi, yi, zi : meshgrid arrays
    """
    try:
        import scipy.interpolate
    except Exception:
        raise
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    xi_vals = np.linspace(x.min(), x.max(), n_interp)
    yi_vals = np.linspace(y.min(), y.max(), n_interp)
    xi, yi = np.meshgrid(xi_vals, yi_vals)
    zi = scipy.interpolate.griddata((x, y), z, (xi, yi), method=method)
    return xi, yi, zi


def prepare_length_features(raw_lengths: Iterable[float]) -> np.ndarray:
    """Scale raw length values to [0,1] and return as a 2D column vector."""
    scaled = min_max_scale(raw_lengths)
    return np.array(scaled, dtype=float).reshape(-1, 1)


def build_regressor(input_dim: int, hidden_units: int = 64, lr: float = 1e-3):
    """Build and return a compiled Keras regressor model.

    TensorFlow is imported inside the function so the module can be imported
    without TensorFlow installed when only helper functions are used.
    """
    try:
        # ty may not resolve the compiled tensorflow package; silence
        # type checker for this import
        from tensorflow.keras import layers, models, optimizers  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("TensorFlow is required to build/train models") from exc

    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(hidden_units, activation="relu"))
    model.add(layers.Dense(hidden_units // 2, activation="relu"))
    model.add(layers.Dense(1, activation="linear"))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"]
    )
    return model


def run_length_regression(
    model, X: np.ndarray, y: np.ndarray, epochs: int = 30, batch_size: int = 32
):
    return model.fit(X, y, epochs=epochs, batch_size=batch_size)


def create_length_surface(
    x_vals: Sequence[float],
    y_vals: Sequence[float],
    z_vals: Sequence[float],
    n_interp: int = 200,
):
    """Create interpolated grid arrays using `griddata_interpolate` helper.

    Returns: xi, yi, zi
    """
    return griddata_interpolate(x_vals, y_vals, z_vals, n_interp=n_interp)
