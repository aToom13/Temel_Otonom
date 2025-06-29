import React, { useState, useEffect } from 'react';
import './index.css';
import { apiService } from './api';
import VideoStream from './components/VideoStream';
import EnhancedVideoStream from './components/EnhancedVideoStream';
import ProcessingVisualization from './components/ProcessingVisualization';
import RealTimeDashboard from './components/RealTimeDashboard';
import LidarVisualization from './components/LidarVisualization';
import BirdEyeSimulation from './components/BirdEyeSimulation';
import { subscribeLogUpdate, unsubscribeLogUpdate } from './services/socket';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [logs, setLogs] = useState([]);
  const [telemetry, setTelemetry] = useState({});
  const [arduinoData, setArduinoData] = useState({});
  const [systemStatus, setSystemStatus] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('connecting');

  useEffect(() => {
    const loadData = async () => {
      try {
        setConnectionStatus('connecting');
        
        // Fetch main system status
        const statusData = await apiService.getSystemStatus();
        setSystemStatus(statusData);
        
        // Update telemetry data from system status
        if (statusData.direction_data) {
          setTelemetry({
            frame_rate: statusData.performance_metrics?.fps || 0,
            objects_detected: statusData.detection_results?.traffic_signs?.length || 0,
            processing_time: 15, // Placeholder
            speed: statusData.direction_data.target_speed || 0,
            battery: 85, // Placeholder
            status: statusData.direction_data.vehicle_status || 'IDLE',
            temperature: 25 // Placeholder
          });
        }
        
        // Update Arduino data
        setArduinoData({
          status: 'OK',
          connection: statusData.arduino_status === 'Connected' ? 'Stable' : 'Disconnected',
          last_command: 'ACK'
        });

        // Try to fetch additional telemetry data
        try {
          const telemetryData = await apiService.getTelemetry();
          setTelemetry(prev => ({ ...prev, ...telemetryData }));
        } catch (error) {
          console.warn('Telemetry endpoint not available:', error);
        }

        // Try to fetch logs
        try {
          const logsData = await apiService.getLogs();
          if (Array.isArray(logsData)) {
            setLogs(logsData);
          }
        } catch (error) {
          console.warn('Logs endpoint not available:', error);
        }

        setConnectionStatus('connected');
      } catch (error) {
        console.error('Failed to fetch data:', error);
        setConnectionStatus('error');
      }
    };
    
    loadData();
    
    // Update data every 2 seconds for non-real-time tabs
    const interval = setInterval(loadData, 2000);
    
    // WebSocket log updates
    const handleLog = (log) => setLogs((prev) => [...prev.slice(-99), log]);
    subscribeLogUpdate(handleLog);
    
    return () => {
      clearInterval(interval);
      unsubscribeLogUpdate();
    };
  }, []);

  const tabs = [
    { id: 'dashboard', label: 'Real-Time Dashboard' },
    { id: 'simulation', label: '3D Bird Eye Simulation' },
    { id: 'raw', label: 'Raw Camera' },
    { id: 'strip', label: 'Lane Detection' },
    { id: 'mark', label: 'Object Detection' },
    { id: 'depth', label: 'Depth Analysis' },
    { id: 'lidar', label: 'LiDAR Visualization' },
    { id: 'combined', label: 'Combined Results' }
  ];

  const renderConnectionStatus = () => {
    const statusConfig = {
      connecting: { color: '#FF9800', text: 'Connecting...' },
      connected: { color: '#4CAF50', text: 'Connected' },
      error: { color: '#F44336', text: 'Connection Error' }
    };

    const config = statusConfig[connectionStatus];
    
    return (
      <div className="connection-status" style={{ 
        position: 'fixed', 
        top: 10, 
        right: 10, 
        background: 'rgba(0,0,0,0.8)', 
        padding: '8px 12px', 
        borderRadius: '4px',
        color: config.color,
        fontSize: '0.9rem',
        zIndex: 1000
      }}>
        <span style={{ 
          display: 'inline-block', 
          width: '8px', 
          height: '8px', 
          borderRadius: '50%', 
          backgroundColor: config.color, 
          marginRight: '8px' 
        }}></span>
        {config.text}
      </div>
    );
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return (
          <div className="dashboard-container">
            <RealTimeDashboard systemStatus={systemStatus} />
            <div className="right-panel">
              {renderDashboardTelemetry()}
            </div>
          </div>
        );
      
      case 'simulation':
        return <BirdEyeSimulation />;
      
      case 'raw':
        return (
          <div className="camera-feed">
            <EnhancedVideoStream 
              showOverlays={false}
              className="raw-camera-stream"
            />
          </div>
        );
      
      case 'strip':
        return (
          <div className="camera-feed">
            <ProcessingVisualization processingType="lanes" />
          </div>
        );
      
      case 'mark':
        return (
          <div className="camera-feed">
            <ProcessingVisualization processingType="objects" />
          </div>
        );
      
      case 'depth':
        return (
          <div className="camera-feed">
            <ProcessingVisualization processingType="obstacles" />
          </div>
        );
      
      case 'lidar':
        return <LidarVisualization />;
      
      case 'combined':
        return (
          <div className="camera-feed">
            <ProcessingVisualization processingType="combined" />
          </div>
        );
      
      default:
        return <div>Tab content not found</div>;
    }
  };

  const renderDashboardTelemetry = () => (
    <div className="telemetry-grid">
      <div className="telemetry-card">
        <h4>Speed</h4>
        <div className="value">
          {systemStatus?.direction_data?.target_speed || telemetry.speed || '0'}
          <span className="unit"> km/h</span>
        </div>
      </div>
      <div className="telemetry-card">
        <h4>Battery</h4>
        <div className="value">{telemetry.battery || '0'}<span className="unit">%</span></div>
      </div>
      <div className="telemetry-card">
        <h4>Status</h4>
        <div className="value">
          {systemStatus?.direction_data?.vehicle_status || telemetry.status || 'IDLE'}
        </div>
      </div>
      <div className="telemetry-card">
        <h4>Temperature</h4>
        <div className="value">{telemetry.temperature || '0'}<span className="unit">°C</span></div>
      </div>
    </div>
  );

  return (
    <div className="app-container">
      {renderConnectionStatus()}
      
      <div className="main-panels">
        <div className="left-panel">
          <div className="tabs">
            {tabs.map(tab => (
              <button
                key={tab.id}
                className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>
          <div className="tab-content">
            {renderTabContent()}
          </div>
        </div>

        {activeTab !== 'dashboard' && activeTab !== 'lidar' && activeTab !== 'simulation' && (
          <div className="right-panel">
            <h3>System Status</h3>
            <div className="telemetry-section">
              <h4>Processing Performance</h4>
              <div className="data-row">
                Frame Rate: 
                <span>{systemStatus?.performance_metrics?.fps?.toFixed(1) || 'N/A'} FPS</span>
              </div>
              <div className="data-row">
                Objects Detected: 
                <span>{systemStatus?.detection_results?.traffic_signs?.length || 'N/A'}</span>
              </div>
              <div className="data-row">
                Processing Time: 
                <span>{telemetry.processing_time || 'N/A'}ms</span>
              </div>
              <div className="data-row">
                Memory Usage: 
                <span>1.2GB/4GB</span>
              </div>
            </div>
            
            <div className="telemetry-section">
              <h4>Hardware Status</h4>
              <div className="data-row">
                Camera: 
                <span className={`status-${systemStatus?.camera_status?.is_connected ? 'connected' : 'disconnected'}`}>
                  {systemStatus?.camera_status?.camera_type || 'Disconnected'}
                </span>
              </div>
              <div className="data-row">
                LiDAR: 
                <span className={`status-${systemStatus?.lidar_status?.is_connected ? 'connected' : 'disconnected'}`}>
                  {systemStatus?.lidar_status?.is_connected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              <div className="data-row">
                Arduino: 
                <span className={`status-${systemStatus?.arduino_status === 'Connected' ? 'connected' : 'disconnected'}`}>
                  {systemStatus?.arduino_status || 'Unknown'}
                </span>
              </div>
              <div className="data-row">
                Last Command: 
                <span>{arduinoData.last_command || 'N/A'}</span>
              </div>
            </div>
            
            <div className="telemetry-section">
              <h4>Safety Status</h4>
              <div className="data-row">
                System State: 
                <span className={`status-${systemStatus?.safety_status?.current_state?.toLowerCase() || 'unknown'}`}>
                  {systemStatus?.safety_status?.current_state || 'UNKNOWN'}
                </span>
              </div>
              <div className="data-row">
                Emergency Stop: 
                <span className={`status-${systemStatus?.safety_status?.emergency_stop_active ? 'active' : 'inactive'}`}>
                  {systemStatus?.safety_status?.emergency_stop_active ? 'Active' : 'Inactive'}
                </span>
              </div>
              <div className="data-row">
                Watchdog: 
                <span className="status-active">Active</span>
              </div>
            </div>
            
            <div className="telemetry-section">
              <h4>Navigation & IMU</h4>
              <div className="data-row">
                Speed: 
                <span>{systemStatus?.direction_data?.target_speed || 0} km/h</span>
              </div>
              <div className="data-row">
                Steering: 
                <span>{systemStatus?.direction_data?.steering_angle || 0}°</span>
              </div>
              <div className="data-row">
                Lane Offset: 
                <span>{systemStatus?.lane_results?.lane_center_offset?.toFixed(2) || '0.00'}m</span>
              </div>
              <div className="data-row">
                IMU Heading: 
                <span>{systemStatus?.imu_data?.heading_degrees?.toFixed(1) || 'N/A'}°</span>
              </div>
              {systemStatus?.imu_data && (
                <>
                  <div className="data-row">
                    Vehicle Speed: 
                    <span>{systemStatus.imu_data.speed_kmh?.toFixed(1) || 'N/A'} km/h</span>
                  </div>
                  <div className="data-row">
                    Motion Confidence: 
                    <span>{(systemStatus.imu_data.motion_confidence * 100)?.toFixed(0) || 'N/A'}%</span>
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </div>

      {activeTab !== 'dashboard' && activeTab !== 'lidar' && activeTab !== 'simulation' && (
        <div className="bottom-panel">
          <h3>System Logs</h3>
          <div className="log-output">
            {logs.length === 0 ? (
              <div className="log-entry">No logs available...</div>
            ) : (
              logs.map((log, index) => (
                <div key={index} className="log-entry">
                  {typeof log === 'string' ? log : JSON.stringify(log)}
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;