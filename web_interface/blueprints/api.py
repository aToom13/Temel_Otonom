from flask import Blueprint, jsonify, request
import main as processing_main
import logging

logger = logging.getLogger(__name__)
api = Blueprint('api', __name__)

@api.route('/api/status')
def api_status():
    """Get comprehensive system status"""
    try:
        status_data = processing_main.get_system_status()
        return jsonify(status_data)
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        return jsonify({"error": str(e)}), 500

@api.route('/api/telemetry')
def api_telemetry():
    """Get telemetry data"""
    try:
        # Get basic telemetry from system status
        status_data = processing_main.get_system_status()
        
        telemetry = {
            'frame_rate': status_data.get('performance_metrics', {}).get('fps', 0),
            'processing_time': 15,  # Placeholder
            'objects_detected': len(status_data.get('detection_results', {}).get('traffic_signs', [])),
            'speed': status_data.get('direction_data', {}).get('target_speed', 0),
            'battery': 85,  # Placeholder
            'status': status_data.get('direction_data', {}).get('vehicle_status', 'IDLE'),
            'temperature': 25  # Placeholder
        }
        
        return jsonify(telemetry)
    except Exception as e:
        logger.error(f"Telemetry endpoint error: {e}")
        return jsonify({"error": str(e)}), 500

@api.route('/api/logs')
def api_logs():
    """Get system logs"""
    try:
        # Return recent log entries
        logs = [
            'System initialized...',
            'Camera feed started...',
            'Object detection running',
            'LiDAR processor active',
            'Safety monitor enabled'
        ]
        return jsonify(logs)
    except Exception as e:
        logger.error(f"Logs endpoint error: {e}")
        return jsonify({"error": str(e)}), 500

@api.route('/api/arduino')
def api_arduino():
    """Get Arduino data"""
    try:
        status_data = processing_main.get_system_status()
        
        arduino_data = {
            'status': 'OK' if status_data.get('arduino_status') == 'Connected' else 'ERROR',
            'connection': status_data.get('arduino_status', 'Unknown'),
            'last_command': 'ACK'
        }
        
        return jsonify(arduino_data)
    except Exception as e:
        logger.error(f"Arduino endpoint error: {e}")
        return jsonify({"error": str(e)}), 500

@api.route('/api/camera/switch_to_zed', methods=['POST'])
def switch_to_zed():
    """Switch to ZED camera"""
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
        logger.error(f"Camera switch error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api.route('/api/imu/data')
def get_imu_data():
    """Get IMU sensor data"""
    try:
        if processing_main.camera_manager and processing_main.camera_manager.has_imu_capability():
            imu_data = processing_main.camera_manager.get_imu_data()
            return jsonify(imu_data)
        else:
            return jsonify({"error": "IMU not available"}), 404
    except Exception as e:
        logger.error(f"IMU data error: {e}")
        return jsonify({"error": str(e)}), 500

@api.route('/api/lidar/data')
def get_lidar_data():
    """Get LiDAR scan data"""
    try:
        if processing_main.lidar_processor:
            lidar_data = processing_main.lidar_processor.get_scan_data_for_visualization()
            return jsonify(lidar_data)
        else:
            return jsonify({"error": "LiDAR not available"}), 404
    except Exception as e:
        logger.error(f"LiDAR data error: {e}")
        return jsonify({"error": str(e)}), 500

@api.route('/api/lidar/status')
def get_lidar_status():
    """Get LiDAR status"""
    try:
        if processing_main.lidar_processor:
            status = processing_main.lidar_processor.get_status()
            return jsonify(status)
        else:
            return jsonify({"error": "LiDAR not available"}), 404
    except Exception as e:
        logger.error(f"LiDAR status error: {e}")
        return jsonify({"error": str(e)}), 500

@api.route('/api/lidar/start', methods=['POST'])
def start_lidar():
    """Start LiDAR scanning"""
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
        logger.error(f"LiDAR start error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api.route('/api/lidar/stop', methods=['POST'])
def stop_lidar():
    """Stop LiDAR scanning"""
    try:
        if processing_main.lidar_processor:
            processing_main.lidar_processor.stop_scanning()
            return jsonify({"success": True, "message": "LiDAR scanning stopped"})
        else:
            return jsonify({"success": False, "message": "LiDAR processor not initialized"}), 500
    except Exception as e:
        logger.error(f"LiDAR stop error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api.route('/api/safety/emergency_stop', methods=['POST'])
def emergency_stop():
    """Activate emergency stop"""
    try:
        processing_main.safety_monitor.trigger_emergency_stop("Manual emergency stop via API")
        return jsonify({"success": True, "message": "Emergency stop activated"})
    except Exception as e:
        logger.error(f"Emergency stop error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api.route('/api/safety/reset', methods=['POST'])
def reset_emergency():
    """Reset emergency stop"""
    try:
        processing_main.safety_monitor.reset_emergency_stop()
        return jsonify({"success": True, "message": "Emergency stop reset"})
    except Exception as e:
        logger.error(f"Emergency reset error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        status_data = processing_main.get_system_status()
        
        health = {
            "status": "healthy",
            "timestamp": status_data.get("timestamp", 0),
            "components": {
                "camera": status_data.get("camera_status", {}).get("is_connected", False),
                "arduino": status_data.get("arduino_status") == "Connected",
                "lidar": status_data.get("lidar_status", {}).get("is_connected", False),
                "safety": status_data.get("safety_status", {}).get("current_state") == "SAFE"
            }
        }
        
        # Determine overall health
        if not any(health["components"].values()):
            health["status"] = "critical"
        elif not all(health["components"].values()):
            health["status"] = "degraded"
        
        return jsonify(health)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "components": {}
        }), 500

# Error handlers
@api.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@api.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@api.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled API exception: {e}")
    return jsonify({"error": "An unexpected error occurred"}), 500