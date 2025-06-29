from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import base64
import time
import threading
import cv2
import os

app = Flask(__name__)
CORS(app)  # React frontend ile iletişim için
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/api/telemetry', methods=['GET'])
def get_telemetry():
    # Örnek telemetri verisi
    return jsonify({
        'frame_rate': 30,
        'processing_time': 15,
        'objects_detected': 4
    })

@app.route('/api/logs', methods=['GET'])
def get_logs():
    # Örnek log verisi
    return jsonify([
        'System initialized...',
        'Camera feed started...',
        'Object detection running'
    ])

@app.route('/api/arduino', methods=['GET'])
def get_arduino_data():
    # Örnek Arduino verisi
    return jsonify({
        'status': 'OK',
        'connection': 'Stable',
        'last_command': 'ACK'
    })

# Örnek: Gerçek zamanlı kamera frame'i gönderen event
def send_camera_frame():
    # Birden fazla kamera indeksini dene
    cap = None
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"Kamera {i} indeksinde bulundu")
                    break
                else:
                    cap.release()
                    cap = None
        except Exception as e:
            print(f"Kamera {i} açılamadı: {e}")
            if cap:
                cap.release()
            cap = None
    
    if not cap or not cap.isOpened():
        print("Hiçbir kamera bulunamadı!")
        return
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                continue
            # Görüntüyü JPEG'e çevir
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('camera_frame', {'frame': jpg_as_text})
            time.sleep(0.1)  # 10 FPS
        except Exception as e:
            print(f"Frame capture error: {e}")
            time.sleep(1)

def send_log_updates():
    log_path = os.path.join(os.path.dirname(__file__), '../../logs/dursun.log')
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if line:
                    socketio.emit('log_update', {'log': line.strip()})
                else:
                    time.sleep(0.2)
    except Exception as e:
        print(f'Log izleme hatası: {e}')

@socketio.on('connect')
def handle_connect():
    emit('message', {'data': 'WebSocket bağlantısı kuruldu.'})

if __name__ == '__main__':
    # Kamera frame'lerini gönderen thread başlatılıyor
    threading.Thread(target=send_camera_frame, daemon=True).start()
    # Log güncellemelerini gönderen thread başlatılıyor
    threading.Thread(target=send_log_updates, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)