import React, { useState, useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import './RealTimeDashboard.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const RealTimeDashboard = () => {
  const [telemetryData, setTelemetryData] = useState({
    speed: [],
    steering: [],
    obstacles: [],
    performance: [],
    timestamps: []
  });

  const [systemMetrics, setSystemMetrics] = useState({
    cpu: 0,
    memory: 0,
    gpu: 0,
    fps: 0,
    latency: 0
  });

  const [safetyStatus, setSafetyStatus] = useState({
    state: 'SAFE',
    emergencyStop: false,
    componentHealth: {},
    recentEvents: []
  });

  const maxDataPoints = 50;
  const updateInterval = useRef(null);

  useEffect(() => {
    // Start real-time data updates
    updateInterval.current = setInterval(fetchRealTimeData, 200); // 5 FPS

    return () => {
      if (updateInterval.current) {
        clearInterval(updateInterval.current);
      }
    };
  }, []);

  const fetchRealTimeData = async () => {
    try {
      // Fetch telemetry data
      const response = await fetch('/api/status');
      const data = await response.json();

      const timestamp = new Date().toLocaleTimeString();

      // Update telemetry data
      setTelemetryData(prev => {
        const newData = { ...prev };
        
        // Add new data points
        if (data.direction_data) {
          newData.speed.push(data.direction_data.target_speed || 0);
          newData.steering.push(data.direction_data.steering_angle || 0);
        } else {
          newData.speed.push(0);
          newData.steering.push(0);
        }

        newData.obstacles.push(data.obstacle_results?.obstacle_detected ? 1 : 0);
        newData.timestamps.push(timestamp);

        // Limit data points
        Object.keys(newData).forEach(key => {
          if (newData[key].length > maxDataPoints) {
            newData[key] = newData[key].slice(-maxDataPoints);
          }
        });

        return newData;
      });

      // Update system metrics (simulated for now)
      setSystemMetrics({
        cpu: Math.random() * 100,
        memory: Math.random() * 100,
        gpu: Math.random() * 100,
        fps: 25 + Math.random() * 10,
        latency: 20 + Math.random() * 30
      });

      // Update safety status
      setSafetyStatus({
        state: 'SAFE',
        emergencyStop: false,
        componentHealth: {
          camera: data.zed_camera_status === 'Connected' ? 'HEALTHY' : 'ERROR',
          arduino: data.arduino_status === 'Connected' ? 'HEALTHY' : 'ERROR',
          processing: 'HEALTHY'
        },
        recentEvents: []
      });

    } catch (error) {
      console.error('Failed to fetch real-time data:', error);
    }
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 0 // Disable animations for real-time updates
    },
    scales: {
      x: {
        display: false
      },
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        ticks: {
          color: '#e0e0e0'
        }
      }
    },
    plugins: {
      legend: {
        labels: {
          color: '#e0e0e0'
        }
      }
    },
    elements: {
      line: {
        tension: 0.4
      },
      point: {
        radius: 0
      }
    }
  };

  const speedChartData = {
    labels: telemetryData.timestamps,
    datasets: [
      {
        label: 'Speed (km/h)',
        data: telemetryData.speed,
        borderColor: '#4CAF50',
        backgroundColor: 'rgba(76, 175, 80, 0.1)',
        fill: true
      }
    ]
  };

  const steeringChartData = {
    labels: telemetryData.timestamps,
    datasets: [
      {
        label: 'Steering Angle (°)',
        data: telemetryData.steering,
        borderColor: '#2196F3',
        backgroundColor: 'rgba(33, 150, 243, 0.1)',
        fill: true
      }
    ]
  };

  const obstacleChartData = {
    labels: telemetryData.timestamps,
    datasets: [
      {
        label: 'Obstacles Detected',
        data: telemetryData.obstacles,
        borderColor: '#FF5722',
        backgroundColor: 'rgba(255, 87, 34, 0.1)',
        fill: true,
        stepped: true
      }
    ]
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'HEALTHY':
      case 'SAFE':
        return '#4CAF50';
      case 'WARNING':
        return '#FF9800';
      case 'ERROR':
      case 'CRITICAL':
        return '#F44336';
      default:
        return '#757575';
    }
  };

  return (
    <div className="dashboard-container">
      {/* Header */}
      <div className="dashboard-header">
        <h2>Real-Time System Dashboard</h2>
        <div className="safety-indicator">
          <div 
            className={`safety-light ${safetyStatus.state.toLowerCase()}`}
            style={{ backgroundColor: getStatusColor(safetyStatus.state) }}
          />
          <span>System Status: {safetyStatus.state}</span>
        </div>
      </div>

      {/* Main Content */}
      <div className="dashboard-content">
        {/* Left Panel - Charts */}
        <div className="charts-panel">
          <div className="chart-row">
            <div className="chart-container">
              <h4>Vehicle Speed</h4>
              <div className="chart-wrapper">
                <Line data={speedChartData} options={chartOptions} />
              </div>
            </div>
            <div className="chart-container">
              <h4>Steering Angle</h4>
              <div className="chart-wrapper">
                <Line data={steeringChartData} options={chartOptions} />
              </div>
            </div>
          </div>
          
          <div className="chart-row">
            <div className="chart-container">
              <h4>Obstacle Detection</h4>
              <div className="chart-wrapper">
                <Line data={obstacleChartData} options={chartOptions} />
              </div>
            </div>
            <div className="chart-container">
              <h4>System Performance</h4>
              <div className="performance-metrics">
                <div className="metric">
                  <span className="metric-label">CPU:</span>
                  <div className="metric-bar">
                    <div 
                      className="metric-fill" 
                      style={{ 
                        width: `${systemMetrics.cpu}%`,
                        backgroundColor: systemMetrics.cpu > 80 ? '#F44336' : '#4CAF50'
                      }}
                    />
                  </div>
                  <span className="metric-value">{systemMetrics.cpu.toFixed(1)}%</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Memory:</span>
                  <div className="metric-bar">
                    <div 
                      className="metric-fill" 
                      style={{ 
                        width: `${systemMetrics.memory}%`,
                        backgroundColor: systemMetrics.memory > 80 ? '#F44336' : '#4CAF50'
                      }}
                    />
                  </div>
                  <span className="metric-value">{systemMetrics.memory.toFixed(1)}%</span>
                </div>
                <div className="metric">
                  <span className="metric-label">FPS:</span>
                  <span className="metric-value large">{systemMetrics.fps.toFixed(1)}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Latency:</span>
                  <span className="metric-value large">{systemMetrics.latency.toFixed(0)}ms</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel - System Status */}
        <div className="status-panel">
          <div className="status-section">
            <h4>Component Health</h4>
            <div className="component-list">
              {Object.entries(safetyStatus.componentHealth).map(([component, status]) => (
                <div key={component} className="component-status">
                  <div 
                    className="status-indicator"
                    style={{ backgroundColor: getStatusColor(status) }}
                  />
                  <span className="component-name">{component.toUpperCase()}</span>
                  <span className="component-status-text">{status}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="status-section">
            <h4>Current Values</h4>
            <div className="current-values">
              <div className="value-item">
                <span className="value-label">Speed:</span>
                <span className="value-number">
                  {telemetryData.speed[telemetryData.speed.length - 1] || 0} km/h
                </span>
              </div>
              <div className="value-item">
                <span className="value-label">Steering:</span>
                <span className="value-number">
                  {telemetryData.steering[telemetryData.steering.length - 1] || 0}°
                </span>
              </div>
              <div className="value-item">
                <span className="value-label">Obstacles:</span>
                <span className="value-number">
                  {telemetryData.obstacles[telemetryData.obstacles.length - 1] || 0}
                </span>
              </div>
            </div>
          </div>

          <div className="status-section">
            <h4>Emergency Controls</h4>
            <div className="emergency-controls">
              <button 
                className="emergency-button"
                onClick={() => {
                  // Implement emergency stop
                  console.log('Emergency stop triggered');
                }}
              >
                🛑 EMERGENCY STOP
              </button>
              <button 
                className="reset-button"
                onClick={() => {
                  // Implement system reset
                  console.log('System reset triggered');
                }}
              >
                🔄 RESET SYSTEM
              </button>
            </div>
          </div>

          <div className="status-section">
            <h4>Recent Events</h4>
            <div className="events-list">
              {safetyStatus.recentEvents.length === 0 ? (
                <div className="no-events">No recent events</div>
              ) : (
                safetyStatus.recentEvents.map((event, index) => (
                  <div key={index} className="event-item">
                    <span className="event-time">{event.timestamp}</span>
                    <span className="event-description">{event.description}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RealTimeDashboard;