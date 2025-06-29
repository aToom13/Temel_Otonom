import numpy as np
import cv2

def fit_lane_polynomial(contour_points, degree=2):
    # contour_points: Nx2 numpy array
    if len(contour_points) < degree + 1:
        return None
    x = contour_points[:, 0]
    y = contour_points[:, 1]
    coeffs = np.polyfit(y, x, degree)
    return coeffs

def draw_lane_poly(image, coeffs, color=(0,255,0), thickness=2):
    h = image.shape[0]
    y_vals = np.linspace(0, h-1, h)
    x_vals = np.polyval(coeffs, y_vals)
    pts = np.array([np.stack([x_vals, y_vals], axis=1)], dtype=np.int32)
    cv2.polylines(image, pts, isClosed=False, color=color, thickness=thickness)
    return image
