from typing import Iterable, List, Union

import numpy as np
from numpy import ndarray


def bits_to_int(bits: Iterable[Union[int, bool]]) -> int:
    """Convert an iterable of bits (0/1 or truthy/falsy) to an integer.

    Example: [1,0,1] -> 5 (binary 101)
    """
    # Ensure bits are ints 0/1 and join into binary string
    s = "".join(str(int(bool(b))) for b in bits)
    if s == "":
        return 0
    return int(s, 2)


def threshold_modes(
    mode_values: Iterable[Union[float, int]], threshold: float
) -> List[int]:
    """Given a sequence of mode values, return a binary list indicating which modes
    exceed the threshold.

    Args:
        mode_values: iterable of numeric mode populations
        threshold: numeric threshold

    Returns:
        list of int (0/1)
    """
    return [1 if float(v) >= threshold else 0 for v in mode_values]


def to_numpy_array(X: Iterable[Iterable[Union[float, int]]]) -> ndarray:
    """Helper: ensure an input 2D list-like becomes a numpy array (float).
    Keeps it small and testable.
    """
    return np.array(X, dtype=float)


def load_csv_as_matrix(path: str, delimiter: str = ",") -> np.ndarray:
    """Load a CSV file into a numpy 2D array of floats.

    Thin wrapper around numpy loading to keep callers simple.
    """
    return np.loadtxt(path, delimiter=delimiter, dtype=float)


def build_simple_classifier(input_dim: int, hidden_units: int = 64, lr: float = 1e-3):
    """Build and return a compiled Keras model for binary classification.

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
    # For binary-phase classification we emit a single logit
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def prepare_binary_targets(
    modes: Iterable[Iterable[float]], threshold: float
) -> np.ndarray:
    """Convert per-sample mode populations into integer labels.

    Each sample's mode population vector is thresholded to bits and then
    converted to an integer using bit encoding.
    """
    labels = []
    for mode_vals in modes:
        binary = threshold_modes(mode_vals, threshold)
        labels.append(bits_to_int(binary))
    return np.array(labels, dtype=int)


def train_classifier(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
    validation_split: float = 0.2,
    verbose: int = 1,
):
    """Train a compiled Keras model on X,y and return the history object.

    The function expects model to be pre-compiled.
    """
    return model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
    )


def predict_probs(model, X: np.ndarray) -> np.ndarray:
    """Return prediction probabilities from the model (shape: (n_samples,))."""
    preds = model.predict(X)
    return preds.ravel()


def simple_pipeline(
    X: Iterable[Iterable[float]],
    modes: Iterable[Iterable[float]],
    threshold: float = 0.5,
    epochs: int = 20,
):
    """A convenience function that runs the full simple binary pipeline.

    - converts inputs to numpy arrays
    - constructs labels from modes using thresholding
    - builds, trains, and returns a trained model

    Returns: (model, history, X_arr, y_arr)
    """
    X_arr = to_numpy_array(X)
    y = prepare_binary_targets(modes, threshold)
    # Ensure y is 0/1 for binary_crossentropy: collapse multi-bit integers into 0/1
    # by thresholding integer > 0
    y_binary = (y > 0).astype(int)
    model = build_simple_classifier(input_dim=X_arr.shape[1])
    history = train_classifier(model, X_arr, y_binary, epochs=epochs)
    return model, history, X_arr, y_binary
