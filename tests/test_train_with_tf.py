import importlib.util
import sys

import pytest

sys.path.insert(0, "src")

TF = importlib.util.find_spec("tensorflow") is not None


def pytest_configure(config):
    if not TF:
        config.pluginmanager.get_plugin("terminalreporter").section(
            "skipped", "TensorFlow not installed: skipping TF tests"
        )


@pytest.mark.skipif(not TF, reason="TensorFlow not installed")
def test_small_training_loop():
    import numpy as np

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
