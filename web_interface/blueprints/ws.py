from flask import Blueprint
from flask_socketio import SocketIO, emit, disconnect
import main as processing_main
import threading
import base64
import cv2
import numpy as np
import time
import logging
import json

logger = logging.getLogger(__name__)
ws = Blueprint('ws', __name__)
socketio = SocketIO()

# Global variables for WebSocket management
connected_clients = set()
broadcast_thread = None
broadcast_active = False

@ws.route('/ws_status')
def ws_status():
    """WebSocket status endpoint"""
    return {
        'connected_clients': len(connected_clients),
        'broadcast_active': broadcast_active
    }

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    try:
        connected_clients.add(request.sid)
        logger.info(f"Client connected: {request.sid}")
        
        # Send initial connection response
        emit('connection_response', {
            'status': 'connected',
            'message': 'WebSocket connection established',
            'timestamp': time.time()
        })
        
        # Send initial system status
        try:
            status_data = processing_main.get_system_status()
            emit('system_status', status_data)
        except Exception as e:
            logger.error(f"Failed to send initial status: {e}")
        
        # Start broadcast thread if not already running
        global broadcast_thread, broadcast_active
        if not broadcast_active:
            broadcast_active = True
            broadcast_thread = threading.Thread(target=broadcast_system_data, daemon=True)
            broadcast_thread.start()
            logger.info("Started WebSocket broadcast thread")
            
    except Exception as e:
        logger.error(f"Connection error: {e}")
        emit('error', {'message': 'Connection failed'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    try:
        connected_clients.discard(request.sid)
        logger.info(f"Client disconnected: {request.sid}")
        
        # Stop broadcast thread if no clients
        global broadcast_active
        if len(connected_clients) == 0:
            broadcast_active = False
            logger.info("Stopped WebSocket broadcast thread - no clients")
            
    except Exception as e:
        logger.error(f"Disconnection error: {e}")

@socketio.on('status_request')
def handle_status_request():
    """Handle status request from client"""
    try:
        status_data = processing_main.get_system_status()
        emit('status_update', status_data)
    except Exception as e:
        logger.error(f"Status request error: {e}")
        emit('error', {'message': 'Failed to get status'})

@socketio.on('imu_request')
def handle_imu_request():
    """Handle IMU data request"""
    try:
        if processing_main.camera_manager and processing_main.camera_manager.has_imu_capability():
            imu_data = processing_main.camera_manager.get_imu_data()
            emit('imu_update', imu_data)
        else:
            emit('error', {'message': 'IMU not available'})
    except Exception as e:
        logger.error(f"IMU request error: {e}")
        emit('error', {'message': 'Failed to get IMU data'})

@socketio.on('lidar_request')
def handle_lidar_request():
    """Handle LiDAR data request"""
    try:
        if processing_main.lidar_processor:
            lidar_data = processing_main.lidar_processor.get_scan_data_for_visualization()
            emit('lidar_update', lidar_data)
        else:
            emit('error', {'message': 'LiDAR not available'})
    except Exception as e:
        logger.error(f"LiDAR request error: {e}")
        emit('error', {'message': 'Failed to get LiDAR data'})

@socketio.on('emergency_stop')
def handle_emergency_stop():
    """Handle emergency stop request"""
    try:
        processing_main.safety_monitor.trigger_emergency_stop("Emergency stop via WebSocket")
        emit('emergency_response', {'status': 'activated', 'message': 'Emergency stop activated'})
        
        # Broadcast to all clients
        socketio.emit('safety_alert', {
            'type': 'emergency_stop',
            'message': 'Emergency stop activated',
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"Emergency stop error: {e}")
        emit('error', {'message': 'Failed to activate emergency stop'})

@socketio.on('reset_emergency')
def handle_reset_emergency():
    """Handle emergency reset request"""
    try:
        processing_main.safety_monitor.reset_emergency_stop()
        emit('emergency_response', {'status': 'reset', 'message': 'Emergency stop reset'})
        
        # Broadcast to all clients
        socketio.emit('safety_alert', {
            'type': 'emergency_reset',
            'message': 'Emergency stop reset',
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"Emergency reset error: {e}")
        emit('error', {'message': 'Failed to reset emergency stop'})

def broadcast_system_data():
    """Broadcast system data to all connected clients"""
    last_status_broadcast = 0
    last_imu_broadcast = 0
    last_lidar_broadcast = 0
    last_video_broadcast = 0
    last_pointcloud_broadcast = 0
    
    while broadcast_active and len(connected_clients) > 0:
        try:
            current_time = time.time()
            
            # Broadcast system status every 1 second
            if current_time - last_status_broadcast >= 1.0:
                try:
                    status_data = processing_main.get_system_status()
                    socketio.emit('system_status', status_data)
                    last_status_broadcast = current_time
                except Exception as e:
                    logger.error(f"Status broadcast error: {e}")
            
            # Broadcast camera frame every 0.067 s (~15 FPS)
            if current_time - last_video_broadcast >= 0.067:
                try:
                    if (processing_main and hasattr(processing_main, 'processed_frame') and
                            processing_main.processed_frame is not None):
                        with processing_main.processed_frame_lock:
                            frame = processing_main.processed_frame.copy()
                        # JPEG encode with moderate quality for lower bandwidth
                        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if success:
                            frame_b64 = base64.b64encode(buffer).decode('utf-8')
                            socketio.emit('camera_frame', {'frame': frame_b64})
                            last_video_broadcast = current_time
                except Exception as e:
                    logger.debug(f"Video broadcast error: {e}")

            # Broadcast IMU data every 0.1 seconds (10 Hz)
            if current_time - last_imu_broadcast >= 0.1:
                try:
                    if (processing_main.camera_manager and 
                        processing_main.camera_manager.has_imu_capability()):
                        imu_data = processing_main.camera_manager.get_imu_data()
                        socketio.emit('imu_update', imu_data)
                        last_imu_broadcast = current_time
                except Exception as e:
                    logger.debug(f"IMU broadcast error: {e}")
            
            # Broadcast pointcloud every 0.1 seconds (10 Hz)
            if current_time - last_pointcloud_broadcast >= 0.1:
                try:
                    if (hasattr(processing_main, 'processed_depth') and
                            processing_main.processed_depth is not None):
                        with processing_main.processed_depth_lock:
                            depth_map = processing_main.processed_depth.copy()
                        h, w = depth_map.shape[:2]
                        # Intrinsic approximation
                        fx = fy = 700.0
                        cx = w / 2.0
                        cy = h / 2.0
                        # Flatten depth map and sample indices where depth > 0
                        flat_depth = depth_map.flatten()
                        valid_indices = np.where(flat_depth > 0)[0]
                        if len(valid_indices) > 0:
                            sample_size = min(2048, len(valid_indices))
                            sample_idx = np.random.choice(valid_indices, sample_size, replace=False)
                            points = []
                            for idx in sample_idx:
                                z = float(flat_depth[idx])
                                v = idx // w
                                u = idx % w
                                x = (u - cx) * z / fx
                                y = (v - cy) * z / fy
                                points.append([x, y, z])
                            socketio.emit('pointcloud', {'points': points})
                            last_pointcloud_broadcast = current_time
                except Exception as e:
                    logger.debug(f"Pointcloud broadcast error: {e}")
            
            # Broadcast LiDAR data every 0.2 seconds (5 Hz)
            if current_time - last_lidar_broadcast >= 0.2:
                try:
                    if processing_main.lidar_processor:
                        lidar_data = processing_main.lidar_processor.get_scan_data_for_visualization()
                        socketio.emit('lidar_update', lidar_data)
                        last_lidar_broadcast = current_time
                except Exception as e:
                    logger.debug(f"LiDAR broadcast error: {e}")
            
            time.sleep(0.05)  # 20 Hz loop
            
        except Exception as e:
            logger.error(f"Broadcast loop error: {e}")
            time.sleep(1.0)
    
    logger.info("WebSocket broadcast thread stopped")

@socketio.on_error_default
def default_error_handler(e):
    """Default error handler for WebSocket events"""
    logger.error(f"WebSocket error: {e}")
    emit('error', {'message': 'An error occurred'})

# Import request for access to session ID
from flask import request