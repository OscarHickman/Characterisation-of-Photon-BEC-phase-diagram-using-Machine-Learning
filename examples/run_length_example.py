"""Example: use length helpers with synthetic data.

Demonstrates `min_max_scale` and `prepare_length_features`.
"""

from photon_bec.length import min_max_scale, prepare_length_features

raw = [2.0, 3.0, 4.0, 2.5]
print("min_max_scale ->", min_max_scale(raw))
print("prepare_length_features ->", prepare_length_features(raw))
