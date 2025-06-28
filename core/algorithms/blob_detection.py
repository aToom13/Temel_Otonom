import cv2
import numpy as np

def detect_blobs(image, min_area=100):
    # image: single channel (grayscale or binary)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = min_area
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    return keypoints
