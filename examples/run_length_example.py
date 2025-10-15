"""Example: use length helpers with synthetic data.

Demonstrates `min_max_scale` and `prepare_length_features`.
"""

try:
    from photon_bec.length import min_max_scale, prepare_length_features
except Exception:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from photon_bec.length import min_max_scale, prepare_length_features

raw = [2.0, 3.0, 4.0, 2.5]
print("min_max_scale ->", min_max_scale(raw))
print("prepare_length_features ->", prepare_length_features(raw))
