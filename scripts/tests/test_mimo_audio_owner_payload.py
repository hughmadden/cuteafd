"""CPU contracts for retained serving-library audio payload diagnostics."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('audio_owner_payload', ROOT / 'scripts/qualify/mimo_v2/audio-owner-payload.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_bf16_round_matches_serving_nearest_even():
    values = np.array([0x3f800000, 0x80000000, 0x3f808000, 0x3f818000], dtype=np.uint32).view(np.float32)
    assert MODULE.bf16_round(values).tolist() == [0x3f80, 0x8000, 0x3f80, 0x3f82]


def test_difference_locations_and_signed_ulp_distribution():
    expected = np.array([[0x3f80, 0xbf80, 0], [0x4000, 0xc000, 0x8000]], dtype=np.uint16)
    actual = np.array([[0x3f81, 0xbf81, 0], [0x4002, 0xc000, 0x8000]], dtype=np.uint16)
    result = MODULE.differences(actual, expected)
    assert result['different_elements'] == 3
    assert result['positions_row_dim'] == [[0, 0], [0, 1], [1, 0]]
    assert result['ulp_histogram'] == {'1': 2, '2': 1}
    assert result['max_bf16_ulp'] == 2
    assert result['max_abs'] == .03125
    assert result['different_elements_per_row'] == [2, 1]


def test_identical_payload_and_invalid_inputs():
    bits = np.array([[0x3f80, 0xbf80]], dtype=np.uint16)
    result = MODULE.differences(bits, bits)
    assert result['different_elements'] == result['max_abs'] == result['max_bf16_ulp'] == 0
    assert result['positions_row_dim'] == []
    with pytest.raises(ValueError, match='shape/dtype'):
        MODULE.differences(bits, bits.astype(np.uint32))
    with pytest.raises(ValueError, match='nonfinite'):
        MODULE.differences(np.array([[0x7f80, 0]], dtype=np.uint16), bits)
