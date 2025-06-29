import pytest
from core.algorithms.lane_fitting import fit_lane_polynomial
import numpy as np

def test_fit_lane_polynomial_basic():
    # y = x^2
    contour_points = np.array([[0,0],[1,1],[2,4],[3,9]])
    coeffs = fit_lane_polynomial(contour_points, degree=2)
    assert coeffs is not None
    assert len(coeffs) == 3
    # y = ax^2 + bx + c, test polinomun uydurulması
