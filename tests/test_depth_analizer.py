import pytest
from modules.depth_analizer import DepthAnalyzer
import numpy as np

def test_depth_analyzer_initialization():
    analyzer = DepthAnalyzer()
    assert analyzer is not None

def test_depth_analyzer_analyze_depth_data():
    analyzer = DepthAnalyzer()
    # Simulate a depth map (e.g., all values are 2000mm = 2 meter, engel yok)
    dummy_depth_map = np.full((480, 640), 2000, dtype=np.uint16)
    results = analyzer.analyze(dummy_depth_map)

    assert isinstance(results, dict)
    assert "obstacle_detected" in results
    assert "distance" in results
    assert "status" in results
    assert results["obstacle_detected"] == False  # 2m'de engel yok kabulü

def test_depth_analyzer_analyze_rgb_data_no_zed():
    analyzer = DepthAnalyzer()
    # Simulate an RGB frame when Zed is not connected
    dummy_rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = analyzer.analyze(dummy_rgb_frame)

    assert isinstance(results, dict)
    assert "obstacle_detected" in results
    assert results["obstacle_detected"] == False
    assert "Depth analysis not possible" in results["status"]
