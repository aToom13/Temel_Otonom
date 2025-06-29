def get_dynamic_roi(image_shape, roi_params):
    h, w = image_shape[:2]
    top = int(h * roi_params.get('top', 0.6))
    bottom = int(h * roi_params.get('bottom', 0.9))
    left = int(w * roi_params.get('left', 0.3))
    right = int(w * roi_params.get('right', 0.7))
    return (top, bottom, left, right)
