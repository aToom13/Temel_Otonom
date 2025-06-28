from flask import Blueprint
from flask_socketio import SocketIO, emit
import main as processing_main

ws = Blueprint('ws', __name__)
socketio = SocketIO()

@ws.route('/ws_status')
def ws_status():
    # Placeholder HTTP endpoint (for test)
    return 'WebSocket endpoint is ready.'

@socketio.on('status_request')
def handle_status_request():
    status_data = {
        "zed_camera_status": processing_main.zed_camera_status,
        "arduino_status": processing_main.arduino_status,
        "detection_results": processing_main.detection_results,
        "lane_results": processing_main.lane_results,
        "obstacle_results": processing_main.obstacle_results,
        "direction_data": processing_main.direction_data
    }
    emit('status_update', status_data)
