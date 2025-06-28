from flask import Blueprint, jsonify
import main as processing_main

api = Blueprint('api', __name__)

@api.route('/api/status')
def api_status():
    try:
        status_data = {
            "zed_camera_status": getattr(processing_main, 'zed_camera_status', None),
            "arduino_status": getattr(processing_main, 'arduino_status', None),
            "detection_results": getattr(processing_main, 'detection_results', None),
            "lane_results": getattr(processing_main, 'lane_results', None),
            "obstacle_results": getattr(processing_main, 'obstacle_results', None),
            "direction_data": getattr(processing_main, 'direction_data', None)
        }
        return jsonify(status_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
