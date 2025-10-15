from photon_bec import binary


def test_bits_to_int():
    assert binary.bits_to_int([1, 0, 1]) == 5
    assert binary.bits_to_int([]) == 0
    assert binary.bits_to_int([0, 0, 0]) == 0
    assert binary.bits_to_int([1]) == 1


def test_threshold_modes():
    vals = [0.1, 5.0, 10.0, 0.0]
    thr = 5.0
    assert binary.threshold_modes(vals, thr) == [0, 1, 1, 0]
