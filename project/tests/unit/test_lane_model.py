import pytest
from core.models.lane import Lane

def test_lane_model():
    lane = Lane(coeffs=[1, 0, 0])
    assert hasattr(lane, 'coeffs')
    assert isinstance(lane.coeffs, list) or isinstance(lane.coeffs, (tuple, type(None)))
    assert lane.lane_type == 'center'
