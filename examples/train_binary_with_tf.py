"""Train a tiny TensorFlow model using the package helpers.

This example requires TensorFlow to be installed. It creates synthetic data,
builds a small classifier via `photon_bec.binary.build_simple_classifier`,
trains for a few epochs, and prints the final evaluation.
"""

import sys
from pathlib import Path

import numpy as np

try:
    from photon_bec.binary import build_simple_classifier, train_classifier
except Exception:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from photon_bec.binary import build_simple_classifier, train_classifier


def make_synthetic_dataset(n_samples: int = 200, n_features: int = 8):
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)
    # create simple labels from a linear combination
    weights = rng.randn(n_features)
    logits = X.dot(weights)
    y = (logits > 0).astype(int)
    return X, y


def main():
    X, y = make_synthetic_dataset(300, 16)
    model = build_simple_classifier(input_dim=X.shape[1], hidden_units=32)
    train_classifier(
        model,
        X,
        y,
        epochs=5,
        batch_size=32,
        validation_split=0.1,
        verbose=2,
    )
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"Final loss={loss:.4f} acc={acc:.4f}")


if __name__ == "__main__":
    main()
