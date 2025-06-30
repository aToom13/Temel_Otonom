import { io } from 'socket.io-client';

const SOCKET_URL = process.env.REACT_APP_SOCKET_URL || 'http://localhost:5000';

class SocketService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
    this.callbacks = new Map();
  }

  connect() {
    if (this.socket && this.isConnected) {
      return this.socket;
    }

    this.socket = io(SOCKET_URL, {
      transports: ['websocket', 'polling'],
      timeout: 5000,
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: this.reconnectDelay
    });

    this.socket.on('connect', () => {
      console.log('Socket connected');
      this.isConnected = true;
      this.reconnectAttempts = 0;
    });

    this.socket.on('disconnect', (reason) => {
      console.log('Socket disconnected:', reason);
      this.isConnected = false;
    });

    this.socket.on('connect_error', (error) => {
      console.error('Socket connection error:', error);
      this.isConnected = false;
      this.reconnectAttempts++;
      
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        console.error('Max reconnection attempts reached');
      }
    });

    this.socket.on('reconnect', (attemptNumber) => {
      console.log('Socket reconnected after', attemptNumber, 'attempts');
      this.isConnected = true;
      this.reconnectAttempts = 0;
    });

    return this.socket;
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.isConnected = false;
    }
  }

  subscribe(event, callback) {
    if (!this.socket) {
      this.connect();
    }

    // Store callback for cleanup
    if (!this.callbacks.has(event)) {
      this.callbacks.set(event, new Set());
    }
    this.callbacks.get(event).add(callback);

    this.socket.on(event, callback);
  }

  unsubscribe(event, callback = null) {
    if (!this.socket) return;

    if (callback) {
      this.socket.off(event, callback);
      if (this.callbacks.has(event)) {
        this.callbacks.get(event).delete(callback);
      }
    } else {
      this.socket.off(event);
      this.callbacks.delete(event);
    }
  }

  emit(event, data) {
    if (!this.socket || !this.isConnected) {
      console.warn('Socket not connected, cannot emit event:', event);
      return false;
    }

    this.socket.emit(event, data);
    return true;
  }

  getConnectionStatus() {
    return {
      isConnected: this.isConnected,
      reconnectAttempts: this.reconnectAttempts,
      maxReconnectAttempts: this.maxReconnectAttempts
    };
  }
}

// Create singleton instance
const socketService = new SocketService();

// Auto-connect on import
socketService.connect();

export default socketService;

// Legacy API for backward compatibility
export function subscribeCameraFrame(callback) {
  socketService.subscribe('camera_frame', (data) => {
    if (data && data.frame) {
      callback(data.frame);
    }
  });
}

export function unsubscribeCameraFrame() {
  socketService.unsubscribe('camera_frame');
}

export function subscribeLogUpdate(callback) {
  socketService.subscribe('log_update', (data) => {
    if (data && data.log) {
      callback(data.log);
    }
  });
}

export function unsubscribeLogUpdate() {
  socketService.unsubscribe('log_update');
}

// New API methods
export function subscribeSystemStatus(callback) {
  socketService.subscribe('system_status', callback);
}

export function unsubscribeSystemStatus() {
  socketService.unsubscribe('system_status');
}

export function subscribeIMUData(callback) {
  socketService.subscribe('imu_update', callback);
}

export function unsubscribeIMUData() {
  socketService.unsubscribe('imu_update');
}

// Pointcloud helpers
export function subscribePointcloud(callback) {
  socketService.subscribe('pointcloud', callback);
}

export function unsubscribePointcloud() {
  socketService.unsubscribe('pointcloud');
}

export function subscribeLidarData(callback) {
  socketService.subscribe('lidar_update', callback);
}

export function unsubscribeLidarData() {
  socketService.unsubscribe('lidar_update');
}

export function subscribeSafetyAlert(callback) {
  socketService.subscribe('safety_alert', callback);
}

export function unsubscribeSafetyAlert() {
  socketService.unsubscribe('safety_alert');
}