// API service layer for centralized backend communication
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // System status endpoints
  async getSystemStatus() {
    return this.request('/api/status');
  }

  async getTelemetry() {
    return this.request('/api/telemetry');
  }

  async getLogs() {
    return this.request('/api/logs');
  }

  async getArduinoData() {
    return this.request('/api/arduino');
  }

  // Camera endpoints
  async switchToZed() {
    return this.request('/api/camera/switch_to_zed', { method: 'POST' });
  }

  // IMU endpoints
  async getIMUData() {
    return this.request('/api/imu/data');
  }

  // LiDAR endpoints
  async getLidarData() {
    return this.request('/api/lidar/data');
  }

  async getLidarStatus() {
    return this.request('/api/lidar/status');
  }

  async startLidar() {
    return this.request('/api/lidar/start', { method: 'POST' });
  }

  async stopLidar() {
    return this.request('/api/lidar/stop', { method: 'POST' });
  }

  // Safety endpoints
  async emergencyStop() {
    return this.request('/api/safety/emergency_stop', { method: 'POST' });
  }

  async resetEmergency() {
    return this.request('/api/safety/reset', { method: 'POST' });
  }
}

export const apiService = new ApiService();

// Legacy exports for backward compatibility
export const fetchTelemetry = () => apiService.getTelemetry();
export const fetchLogs = () => apiService.getLogs();
export const fetchArduinoData = () => apiService.getArduinoData();