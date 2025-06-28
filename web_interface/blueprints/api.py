from flask import Blueprint, jsonify
import main as processing_main

api = Blueprint('api', __name__)

@api.route('/api/status')
def api_status():
    try:
        status_data = processing_main.get_system_status()
        return jsonify(status_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/api/camera/switch_to_zed')
def switch_to_zed():
    try:
        if processing_main.camera_manager:
            success = processing_main.camera_manager.switch_to_zed_if_available()
            return jsonify({
                "success": success,
                "message": "Switched to ZED camera" if success else "ZED camera not available"
            })
        else:
            return jsonify({"success": False, "message": "Camera manager not initialized"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@api.route('/api/imu/data')
def get_imu_data():
    try:
        if processing_main.camera_manager and processing_main.camera_manager.has_imu_capability():
            imu_data = processing_main.camera_manager.get_imu_data()
            return jsonify(imu_data)
        else:
            return jsonify({"error": "IMU not available"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/api/safety/emergency_stop', methods=['POST'])
def emergency_stop():
    try:
        processing_main.safety_monitor.trigger_emergency_stop("Manual emergency stop via API")
        return jsonify({"success": True, "message": "Emergency stop activated"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@api.route('/api/safety/reset', methods=['POST'])
def reset_emergency():
    try:
        processing_main.safety_monitor.reset_emergency_stop()
        return jsonify({"success": True, "message": "Emergency stop reset"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500