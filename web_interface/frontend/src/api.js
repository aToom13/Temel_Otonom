import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

export const fetchTelemetry = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/telemetry`);
    return response.data;
  } catch (error) {
    console.error('Telemetri verileri alınamadı:', error);
    return null;
  }
};

export const fetchLogs = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/logs`);
    return response.data;
  } catch (error) {
    console.error('Loglar alınamadı:', error);
    return [];
  }
};

export const fetchArduinoData = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/arduino`);
    return response.data;
  } catch (error) {
    console.error('Arduino verileri alınamadı:', error);
    return null;
  }
};
