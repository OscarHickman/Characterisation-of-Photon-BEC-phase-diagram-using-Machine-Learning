import importlib.util
import sys

import numpy as np
import pytest

# Ensure src/ is importable when running tests directly
sys.path.insert(0, "src")

TF = importlib.util.find_spec("tensorflow") is not None


@pytest.mark.skipif(not TF, reason="TensorFlow not installed")
def test_small_training_loop():
    """Small integration test that trains a tiny model for a couple of epochs.

    This test is skipped when TensorFlow is not installed in the environment.
    """
    from photon_bec.binary import build_simple_classifier, train_classifier

    # tiny synthetic dataset
    rng = np.random.RandomState(1)
    X = rng.randn(64, 8)
    weights = rng.randn(8)
    y = (X.dot(weights) > 0).astype(int)

    model = build_simple_classifier(input_dim=X.shape[1], hidden_units=16)
    history = train_classifier(
        model, X, y, epochs=2, batch_size=16, validation_split=0.1, verbose=0
    )
    assert hasattr(history, "history")
