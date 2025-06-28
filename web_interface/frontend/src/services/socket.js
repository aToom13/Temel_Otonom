import { io } from 'socket.io-client';

const SOCKET_URL = 'http://localhost:5000';
const socket = io(SOCKET_URL, { transports: ['websocket'] });

export default socket;

// Kamera frame'lerini dinlemek için yardımcı fonksiyon
export function subscribeCameraFrame(callback) {
  socket.on('camera_frame', (data) => {
    if (data && data.frame) {
      callback(data.frame);
    }
  });
}

export function unsubscribeCameraFrame() {
  socket.off('camera_frame');
}

// Log güncellemelerini dinlemek için yardımcı fonksiyon
export function subscribeLogUpdate(callback) {
  socket.on('log_update', (data) => {
    if (data && data.log) {
      callback(data.log);
    }
  });
}

export function unsubscribeLogUpdate() {
  socket.off('log_update');
}
