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

@api.route('/api/simulation/data')
def api_simulation_data():
    """Get comprehensive simulation data for 3D visualization"""
    try:
        status_data = processing_main.get_system_status()
        
        # Enhanced simulation data with 3D coordinates and vectors
        simulation_data = {
            'vehicle': {
                'position': {'x': 0, 'y': 0, 'z': 0},
                'heading': status_data.get('imu_data', {}).get('heading_degrees', 0),
                'speed': status_data.get('direction_data', {}).get('target_speed', 0),
                'acceleration': {
                    'x': status_data.get('imu_data', {}).get('acceleration_magnitude', 0),
                    'y': 0,
                    'z': 0
                },
                'orientation': {
                    'roll': status_data.get('imu_data', {}).get('roll_degrees', 0),
                    'pitch': status_data.get('imu_data', {}).get('pitch_degrees', 0),
                    'yaw': status_data.get('imu_data', {}).get('heading_degrees', 0)
                }
            },
            'lanes': process_lane_data_3d(status_data.get('lane_results', {})),
            'obstacles': process_obstacle_data_3d(status_data.get('obstacle_results', {})),
            'lidar_points': process_lidar_data_3d(status_data.get('lidar_results', {})),
            'traffic_signs': process_traffic_signs_3d(status_data.get('detection_results', {})),
            'planned_path': generate_planned_path_3d(status_data.get('direction_data', {})),
            'safety_zones': status_data.get('lidar_results', {}).get('safety_zones', {
                'immediate': 0.5,
                'warning': 1.0,
                'caution': 2.0
            }),
            'environment': {
                'weather': 'clear',
                'lighting': 'daylight',
                'road_type': 'highway'
            },
            'timestamp': status_data.get('timestamp', 0)
        }
        
        return jsonify(simulation_data)
    except Exception as e:
        logger.error(f"Simulation data endpoint error: {e}")
        return jsonify({"error": str(e)}), 500

def process_lane_data_3d(lane_results):
    """Process lane data for 3D visualization"""
    if not lane_results or not lane_results.get('lanes'):
        return []
    
    lanes_3d = []
    for i, lane in enumerate(lane_results['lanes']):
        lane_3d = {
            'id': f"lane_{i}",
            'type': lane.get('lane_type', 'unknown'),
            'confidence': lane.get('confidence', 0.8),
            'points': generate_lane_points_3d(lane, i),
            'curvature': lane.get('curvature', 0),
            'width': 3.5,  # Standard lane width in meters
            'color': get_lane_color(lane.get('lane_type', 'unknown'))
        }
        lanes_3d.append(lane_3d)
    
    return lanes_3d

def generate_lane_points_3d(lane, lane_index):
    """Generate 3D points for lane visualization"""
    points = []
    lane_offset = (lane_index - 1) * 3.5  # 3.5m lane width
    curvature = lane.get('curvature', 0)
    
    for i in range(51):  # 0 to 100m ahead
        y = i * 2  # 2m intervals
        x = lane_offset + curvature * y * y * 0.001
        z = 0  # Road level
        
        points.append({
            'x': x,
            'y': y,
            'z': z,
            'confidence': lane.get('confidence', 0.8)
        })
    
    return points

def get_lane_color(lane_type):
    """Get color for lane type"""
    colors = {
        'left': '#ffffff',
        'right': '#ffffff',
        'center': '#ffff00',
        'unknown': '#888888'
    }
    return colors.get(lane_type, '#ffffff')

def process_obstacle_data_3d(obstacle_results):
    """Process obstacle data for 3D visualization"""
    if not obstacle_results or not obstacle_results.get('obstacles'):
        return []
    
    obstacles_3d = []
    for i, obs in enumerate(obstacle_results['obstacles']):
        obstacle_3d = {
            'id': f"obstacle_{i}",
            'position': {
                'x': obs.get('center', {}).get('x', 0),
                'y': obs.get('center', {}).get('y', 0),
                'z': obs.get('center', {}).get('z', 0)
            },
            'size': obs.get('size', [2, 2, 1.5]),
            'confidence': obs.get('confidence', 0.8),
            'type': 'unknown',
            'velocity': {'x': 0, 'y': 0, 'z': 0},  # Static for now
            'threat_level': calculate_threat_level(obs)
        }
        obstacles_3d.append(obstacle_3d)
    
    return obstacles_3d

def process_lidar_data_3d(lidar_results):
    """Process LiDAR data for 3D visualization"""
    if not lidar_results or not lidar_results.get('points'):
        return []
    
    # Limit points for performance
    points = lidar_results['points'][:1000]
    
    lidar_points_3d = []
    for point in points:
        point_3d = {
            'position': {
                'x': point.get('x', 0),
                'y': point.get('y', 0),
                'z': 0  # 2D LiDAR, so Z is always 0
            },
            'intensity': point.get('quality', 200),
            'distance': point.get('distance', 0),
            'angle': point.get('angle', 0)
        }
        lidar_points_3d.append(point_3d)
    
    return lidar_points_3d

def process_traffic_signs_3d(detection_results):
    """Process traffic sign data for 3D visualization"""
    if not detection_results or not detection_results.get('traffic_signs'):
        return []
    
    signs_3d = []
    for i, sign in enumerate(detection_results['traffic_signs']):
        sign_3d = {
            'id': f"sign_{i}",
            'position': {
                'x': (i - len(detection_results['traffic_signs'])/2) * 5,  # Spread signs
                'y': 15 + i * 5,  # Place ahead of vehicle
                'z': 2  # Sign height
            },
            'type': sign.get('label', 'unknown'),
            'confidence': sign.get('confidence', 0.8),
            'size': {'width': 1, 'height': 1},
            'message': sign.get('label', 'Unknown Sign')
        }
        signs_3d.append(sign_3d)
    
    return signs_3d

def generate_planned_path_3d(direction_data):
    """Generate planned path for 3D visualization"""
    if not direction_data:
        return []
    
    path_points = []
    steering_angle = direction_data.get('steering_angle', 0) * 3.14159 / 180  # Convert to radians
    speed = direction_data.get('target_speed', 0)
    
    for i in range(21):  # 0 to 40m ahead
        t = i * 2  # 2m intervals
        
        # Simple path prediction based on steering angle
        if abs(steering_angle) > 0.01:
            # Curved path
            radius = 10 / abs(steering_angle)  # Simple radius calculation
            angle = t / radius
            x = radius * (1 - np.cos(angle)) * (1 if steering_angle > 0 else -1)
            y = radius * np.sin(angle)
        else:
            # Straight path
            x = 0
            y = t
        
        path_point = {
            'position': {'x': x, 'y': y, 'z': 0},
            'speed': max(0, speed * (1 - t * 0.01)),  # Gradual speed reduction
            'confidence': max(0.3, 1.0 - t * 0.03)  # Decreasing confidence with distance
        }
        path_points.append(path_point)
    
    return path_points

def calculate_threat_level(obstacle):
    """Calculate threat level for obstacle"""
    distance = obstacle.get('center', {}).get('y', 10)
    confidence = obstacle.get('confidence', 0.5)
    
    if distance < 5 and confidence > 0.8:
        return 'high'
    elif distance < 10 and confidence > 0.6:
        return 'medium'
    else:
        return 'low'

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

# Import numpy for mathematical operations
import numpy as np