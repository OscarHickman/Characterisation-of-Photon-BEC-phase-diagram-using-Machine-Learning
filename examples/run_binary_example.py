"""Example: run the binary pipeline helpers with synthetic data.

This script demonstrates how to import and use the canonical functions
from `photon_bec.binary`.

Note: The example avoids building/training a Keras model so it runs quickly
without TensorFlow installed. If you want to run training, call
`build_simple_classifier` and `train_classifier` after installing TensorFlow.
"""

try:
    from photon_bec.binary import (
        bits_to_int,
        prepare_binary_targets,
        threshold_modes,
        to_numpy_array,
    )
except Exception:
    # Fallback when running the script directly: add `src` to sys.path and retry
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from photon_bec.binary import (
        bits_to_int,
        prepare_binary_targets,
        threshold_modes,
        to_numpy_array,
    )

# Synthetic features: 6 samples, 4 features each
X = [[0.1 * i + j * 0.01 for j in range(4)] for i in range(6)]
# Synthetic mode populations (3 modes per sample)
modes = [
    [0.6, 0.1, 0.0],
    [0.0, 0.8, 0.0],
    [0.9, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.3, 0.3, 0.3],
    [1.0, 0.5, 0.0],
]

print("X ->", to_numpy_array(X).shape)
print("bits_to_int example ->", bits_to_int([1, 0, 1]))
print("threshold_modes example ->", threshold_modes(modes[0], 0.5))
print("prepare_binary_targets ->", prepare_binary_targets(modes, threshold=0.5))
