class LaneModel:
    def __init__(self, coeffs, lane_type='center'):
        self.coeffs = coeffs  # Polynomial coefficients
        self.lane_type = lane_type

    def predict_x(self, y):
        # y: float or np.array
        import numpy as np
        return np.polyval(self.coeffs, y)

Lane = LaneModel  # Eski importlar için alias
