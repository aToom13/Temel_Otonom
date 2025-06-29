import pytest
from flask import Flask
from web_interface.app import create_app

def test_api_status():
    app = create_app()
    client = app.test_client()
    response = client.get('/api/status')
    assert response.status_code == 200
    data = response.get_json()
    # 'status' anahtarı yerine gerçek anahtarları kontrol et
    assert 'arduino_status' in data
    assert 'detection_results' in data
    assert 'lane_results' in data
    assert 'obstacle_results' in data
    assert 'direction_data' in data
