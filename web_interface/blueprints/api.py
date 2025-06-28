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

@api.route('/api/camera/switch_to_zed', methods=['POST'])
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

@api.route('/api/lidar/data')
def get_lidar_data():
    try:
        if processing_main.lidar_processor:
            lidar_data = processing_main.lidar_processor.get_scan_data_for_visualization()
            return jsonify(lidar_data)
        else:
            return jsonify({"error": "LiDAR not available"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/api/lidar/status')
def get_lidar_status():
    try:
        if processing_main.lidar_processor:
            status = processing_main.lidar_processor.get_status()
            return jsonify(status)
        else:
            return jsonify({"error": "LiDAR not available"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/api/lidar/start', methods=['POST'])
def start_lidar():
    try:
        if processing_main.lidar_processor:
            if not processing_main.lidar_processor.is_connected:
                success = processing_main.lidar_processor.connect()
                if not success:
                    return jsonify({"success": False, "message": "Failed to connect to LiDAR"}), 500
            
            success = processing_main.lidar_processor.start_scanning()
            return jsonify({
                "success": success,
                "message": "LiDAR scanning started" if success else "Failed to start LiDAR scanning"
            })
        else:
            return jsonify({"success": False, "message": "LiDAR processor not initialized"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@api.route('/api/lidar/stop', methods=['POST'])
def stop_lidar():
    try:
        if processing_main.lidar_processor:
            processing_main.lidar_processor.stop_scanning()
            return jsonify({"success": True, "message": "LiDAR scanning stopped"})
        else:
            return jsonify({"success": False, "message": "LiDAR processor not initialized"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

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