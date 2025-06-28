import numpy as np

def analyze_histogram(image, threshold=0.5):
    # image: single channel
    hist = np.histogram(image, bins=256, range=(0,255))[0]
    total = hist.sum()
    if total == 0:
        return 0
    peak = np.argmax(hist)
    ratio = hist[peak] / total
    return ratio > threshold
