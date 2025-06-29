import React, { useState, useEffect } from 'react';
import './index.css';
import { apiService } from './api';
import VideoStream from './components/VideoStream';
import EnhancedVideoStream from './components/EnhancedVideoStream';
import ProcessingVisualization from './components/ProcessingVisualization';
import RealTimeDashboard from './components/RealTimeDashboard';
import LidarVisualization from './components/LidarVisualization';
import BirdEyeSimulation from './components/BirdEyeSimulation';
import ErrorBoundary from './utils/errorBoundary';
import { subscribeLogUpdate, unsubscribeLogUpdate } from './services/socket';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [logs, setLogs] = useState([]);
  const [telemetry, setTelemetry] = useState({});
  const [arduinoData, setArduinoData] = useState({});
  const [systemStatus, setSystemStatus] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        setConnectionStatus('connecting');
        setError(null);
        
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
        setError(error.message || 'Bağlantı hatası');
      }
    };
    
    loadData();
    
    // Update data every 2 seconds for non-real-time tabs
    const interval = setInterval(loadData, 2000);
    
    // WebSocket log updates
    const handleLog = (log) => setLogs((prev) => [...prev.slice(-99), log]);
    
    try {
      subscribeLogUpdate(handleLog);
    } catch (error) {
      console.warn('WebSocket log subscription failed:', error);
    }
    
    return () => {
      clearInterval(interval);
      try {
        unsubscribeLogUpdate();
      } catch (error) {
        console.warn('WebSocket unsubscribe failed:', error);
      }
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
      connecting: { color: '#FF9800', text: 'Bağlanıyor...' },
      connected: { color: '#4CAF50', text: 'Bağlandı' },
      error: { color: '#F44336', text: error || 'Bağlantı Hatası' }
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
    try {
      switch (activeTab) {
        case 'dashboard':
          return (
            <div className="dashboard-container">
              <ErrorBoundary>
                <RealTimeDashboard systemStatus={systemStatus} />
              </ErrorBoundary>
              <div className="right-panel">
                {renderDashboardTelemetry()}
              </div>
            </div>
          );
        
        case 'simulation':
          return (
            <ErrorBoundary>
              <BirdEyeSimulation />
            </ErrorBoundary>
          );
        
        case 'raw':
          return (
            <div className="camera-feed">
              <ErrorBoundary>
                <EnhancedVideoStream 
                  showOverlays={false}
                  className="raw-camera-stream"
                />
              </ErrorBoundary>
            </div>
          );
        
        case 'strip':
          return (
            <div className="camera-feed">
              <ErrorBoundary>
                <ProcessingVisualization processingType="lanes" />
              </ErrorBoundary>
            </div>
          );
        
        case 'mark':
          return (
            <div className="camera-feed">
              <ErrorBoundary>
                <ProcessingVisualization processingType="objects" />
              </ErrorBoundary>
            </div>
          );
        
        case 'depth':
          return (
            <div className="camera-feed">
              <ErrorBoundary>
                <ProcessingVisualization processingType="obstacles" />
              </ErrorBoundary>
            </div>
          );
        
        case 'lidar':
          return (
            <ErrorBoundary>
              <LidarVisualization />
            </ErrorBoundary>
          );
        
        case 'combined':
          return (
            <div className="camera-feed">
              <ErrorBoundary>
                <ProcessingVisualization processingType="combined" />
              </ErrorBoundary>
            </div>
          );
        
        default:
          return <div>Tab içeriği bulunamadı</div>;
      }
    } catch (error) {
      console.error('Tab content render error:', error);
      return (
        <div className="error-content">
          <h3>İçerik Yükleme Hatası</h3>
          <p>Bu sekme içeriği yüklenirken bir hata oluştu.</p>
          <button onClick={() => window.location.reload()}>
            Sayfayı Yenile
          </button>
        </div>
      );
    }
  };

  const renderDashboardTelemetry = () => (
    <div className="telemetry-grid">
      <div className="telemetry-card">
        <h4>Hız</h4>
        <div className="value">
          {systemStatus?.direction_data?.target_speed || telemetry.speed || '0'}
          <span className="unit"> km/h</span>
        </div>
      </div>
      <div className="telemetry-card">
        <h4>Batarya</h4>
        <div className="value">{telemetry.battery || '0'}<span className="unit">%</span></div>
      </div>
      <div className="telemetry-card">
        <h4>Durum</h4>
        <div className="value">
          {systemStatus?.direction_data?.vehicle_status || telemetry.status || 'IDLE'}
        </div>
      </div>
      <div className="telemetry-card">
        <h4>Sıcaklık</h4>
        <div className="value">{telemetry.temperature || '0'}<span className="unit">°C</span></div>
      </div>
    </div>
  );

  return (
    <ErrorBoundary>
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
              <h3>Sistem Durumu</h3>
              <div className="telemetry-section">
                <h4>İşleme Performansı</h4>
                <div className="data-row">
                  Frame Rate: 
                  <span>{systemStatus?.performance_metrics?.fps?.toFixed(1) || 'N/A'} FPS</span>
                </div>
                <div className="data-row">
                  Algılanan Nesneler: 
                  <span>{systemStatus?.detection_results?.traffic_signs?.length || 'N/A'}</span>
                </div>
                <div className="data-row">
                  İşleme Süresi: 
                  <span>{telemetry.processing_time || 'N/A'}ms</span>
                </div>
                <div className="data-row">
                  Bellek Kullanımı: 
                  <span>1.2GB/4GB</span>
                </div>
              </div>
              
              <div className="telemetry-section">
                <h4>Donanım Durumu</h4>
                <div className="data-row">
                  Kamera: 
                  <span className={`status-${systemStatus?.camera_status?.is_connected ? 'connected' : 'disconnected'}`}>
                    {systemStatus?.camera_status?.camera_type || 'Bağlı Değil'}
                  </span>
                </div>
                <div className="data-row">
                  LiDAR: 
                  <span className={`status-${systemStatus?.lidar_status?.is_connected ? 'connected' : 'disconnected'}`}>
                    {systemStatus?.lidar_status?.is_connected ? 'Bağlı' : 'Bağlı Değil'}
                  </span>
                </div>
                <div className="data-row">
                  Arduino: 
                  <span className={`status-${systemStatus?.arduino_status === 'Connected' ? 'connected' : 'disconnected'}`}>
                    {systemStatus?.arduino_status || 'Bilinmiyor'}
                  </span>
                </div>
                <div className="data-row">
                  Son Komut: 
                  <span>{arduinoData.last_command || 'N/A'}</span>
                </div>
              </div>
              
              <div className="telemetry-section">
                <h4>Güvenlik Durumu</h4>
                <div className="data-row">
                  Sistem Durumu: 
                  <span className={`status-${systemStatus?.safety_status?.current_state?.toLowerCase() || 'unknown'}`}>
                    {systemStatus?.safety_status?.current_state || 'BİLİNMİYOR'}
                  </span>
                </div>
                <div className="data-row">
                  Acil Durdurma: 
                  <span className={`status-${systemStatus?.safety_status?.emergency_stop_active ? 'active' : 'inactive'}`}>
                    {systemStatus?.safety_status?.emergency_stop_active ? 'Aktif' : 'Pasif'}
                  </span>
                </div>
                <div className="data-row">
                  Watchdog: 
                  <span className="status-active">Aktif</span>
                </div>
              </div>
              
              <div className="telemetry-section">
                <h4>Navigasyon & IMU</h4>
                <div className="data-row">
                  Hız: 
                  <span>{systemStatus?.direction_data?.target_speed || 0} km/h</span>
                </div>
                <div className="data-row">
                  Direksiyon: 
                  <span>{systemStatus?.direction_data?.steering_angle || 0}°</span>
                </div>
                <div className="data-row">
                  Şerit Sapması: 
                  <span>{systemStatus?.lane_results?.lane_center_offset?.toFixed(2) || '0.00'}m</span>
                </div>
                <div className="data-row">
                  IMU Yönü: 
                  <span>{systemStatus?.imu_data?.heading_degrees?.toFixed(1) || 'N/A'}°</span>
                </div>
                {systemStatus?.imu_data && (
                  <>
                    <div className="data-row">
                      Araç Hızı: 
                      <span>{systemStatus.imu_data.speed_kmh?.toFixed(1) || 'N/A'} km/h</span>
                    </div>
                    <div className="data-row">
                      Hareket Güveni: 
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
            <h3>Sistem Logları</h3>
            <div className="log-output">
              {logs.length === 0 ? (
                <div className="log-entry">Loglar mevcut değil...</div>
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
    </ErrorBoundary>
  );
}

export default App;