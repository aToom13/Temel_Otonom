<<<<<<< HEAD
import sys
import json
from flask import Flask, jsonify, Response
from flask_cors import CORS
import psutil
import cv2
import numpy as np
from flask_socketio import SocketIO, emit

# Try to import pyzed, if not available, set a flag
try:
    import pyzed.sl as sl
    zed_sdk_available = True
except ImportError:
    zed_sdk_available = False

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/api/system_info')
def system_info():
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    return jsonify({
        'cpu_percent': cpu_percent,
        'memory_total': memory.total,
        'memory_available': memory.available,
        'memory_used': memory.used,
        'memory_percent': memory.percent
    })

@app.route('/api/zed_status')
def zed_status():
    if not zed_sdk_available:
        return jsonify({
            'connected': False,
            'error': 'ZED SDK not installed'
        })

    # Try to open the camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    err = zed.open(init_params)
    if err == sl.ERROR_CODE.SUCCESS:
        zed.close()
        return jsonify({'connected': True})
    else:
        return jsonify({
            'connected': False,
            'error': str(err)
        })

@app.route('/api/arduino')
def arduino():
    return jsonify({"status": "Arduino bağlantısı henüz kurulmadı"})

@app.route('/api/telemetry')
def telemetry():
    return jsonify({"status": "Telemetri verileri henüz hazır değil"})

@app.route('/api/logs')
def logs():
    return jsonify({"logs": [], "message": "Log sistemi henüz entegre edilmedi"})

def generate_frames():
    if not zed_sdk_available:
        # Create a black frame
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ret, jpeg = cv2.imencode('.jpg', black_frame)
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        return

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        # Return an error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "ZED Camera Error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        ret, jpeg = cv2.imencode('.jpg', error_frame)
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        return

    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()
            # Convert to JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                break
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            break
    zed.close()

@app.route('/api/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    print('İstemci bağlandı')
    emit('connection_response', {'data': 'Bağlantı başarılı'})

@socketio.on('disconnect')
def handle_disconnect():
    print('İstemci bağlantısı kesildi')

if __name__ == '__main__':
    socketio.run(app, debug=True)
=======
import subprocess
import sys
import os
import time

ORCHESTATOR_CMD = [sys.executable, 'orchestator.py']

print("[RUNNER] Orchestator başlatılıyor...")
orchestator_proc = subprocess.Popen(ORCHESTATOR_CMD)

try:
    orchestator_proc.wait()
except KeyboardInterrupt:
    print("\n[RUNNER] Kapatılıyor...")
    orchestator_proc.terminate()
    sys.exit(0)
>>>>>>> 99224143311a21e90a259e80c2e07249bbd7c822
