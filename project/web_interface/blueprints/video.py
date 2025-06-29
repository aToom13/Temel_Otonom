from flask import Blueprint, Response
import cv2
import main as processing_main
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)
video = Blueprint('video', __name__)

def generate_frames():
    """Generate video frames for streaming"""
    frame_count = 0
    last_frame_time = time.time()
    
    while True:
        try:
            current_time = time.time()
            
            # Get processed frame from main processing
            if (processing_main and 
                hasattr(processing_main, 'processed_frame') and 
                processing_main.processed_frame is not None):
                
                with processing_main.processed_frame_lock:
                    frame = processing_main.processed_frame.copy()
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    frame_count += 1
                    
                    # Log frame rate every 30 frames
                    if frame_count % 30 == 0:
                        fps = 30 / (current_time - last_frame_time)
                        logger.debug(f"Video stream FPS: {fps:.1f}")
                        last_frame_time = current_time
                else:
                    logger.warning("Failed to encode frame")
                    time.sleep(0.1)
            else:
                # Generate placeholder frame when no camera data
                placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder_frame, "No Camera Data", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', placeholder_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.1)  # 10 FPS for placeholder
                
        except Exception as e:
            logger.error(f"Video stream error: {e}")
            time.sleep(0.5)

@video.route('/video_feed')
def video_feed():
    """Video streaming route"""
    try:
        return Response(
            generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        logger.error(f"Video feed error: {e}")
        return "Video feed error", 500

@video.route('/api/video/status')
def video_status():
    """Get video stream status"""
    try:
        if (processing_main and 
            hasattr(processing_main, 'camera_manager') and 
            processing_main.camera_manager):
            
            camera_status = processing_main.camera_manager.get_camera_status()
            return {
                "status": "active" if camera_status.is_connected else "inactive",
                "camera_type": camera_status.camera_type,
                "resolution": camera_status.resolution,
                "fps": camera_status.fps
            }
        else:
            return {
                "status": "inactive",
                "camera_type": "None",
                "resolution": [0, 0],
                "fps": 0
            }
    except Exception as e:
        logger.error(f"Video status error: {e}")
        return {"error": str(e)}, 500