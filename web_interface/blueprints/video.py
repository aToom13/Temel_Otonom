from flask import Blueprint, Response
import cv2
import main as processing_main
import time

video = Blueprint('video', __name__)

def generate_frames():
    while True:
        try:
            if processing_main.processed_frame is not None:
                ret, buffer = cv2.imencode('.jpg', processing_main.processed_frame)
                frame = buffer.tobytes()
                yield (b'--frame\n'
                       b'Content-Type: image/jpeg\n\n' + frame + b'\n')
            else:
                time.sleep(0.1)
        except Exception:
            time.sleep(0.5)

@video.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
