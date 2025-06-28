import React, { useState, useEffect } from 'react';
import './index.css';
import { fetchTelemetry, fetchLogs, fetchArduinoData } from './api';
import VideoStream from './components/VideoStream';
import RealTimeDashboard from './components/RealTimeDashboard';
import { subscribeLogUpdate, unsubscribeLogUpdate } from './services/socket';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [logs, setLogs] = useState([]);
  const [telemetry, setTelemetry] = useState({});
  const [arduinoData, setArduinoData] = useState({});

  useEffect(() => {
    const loadData = async () => {
      const telemetryData = await fetchTelemetry();
      const logsData = await fetchLogs();
      const arduinoData = await fetchArduinoData();
      
      if (telemetryData) setTelemetry(telemetryData);
      if (logsData) setLogs(logsData);
      if (arduinoData) setArduinoData(arduinoData);
    };
    
    loadData();
    
    // Update data every 5 seconds for non-real-time tabs
    const interval = setInterval(loadData, 5000);
    
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
    { id: 'raw', label: 'Raw Camera' },
    { id: 'strip', label: 'Lane Detection' },
    { id: 'mark', label: 'Object Detection' },
    { id: 'depth', label: 'Depth Analysis' },
    { id: 'combined', label: 'Combined Results' }
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <RealTimeDashboard />;
      
      case 'raw':
        return (
          <div className="camera-feed">
            <VideoStream />
          </div>
        );
      
      case 'strip':
        return (
          <div className="camera-feed">
            <div className="feed-placeholder">
              <h3>Lane Detection Results</h3>
              <p>Enhanced lane detection with temporal consistency</p>
              <VideoStream />
              <div className="overlay-info">
                <div className="info-item">
                  <span>Detection Quality:</span>
                  <span className="value">85%</span>
                </div>
                <div className="info-item">
                  <span>Lane Departure:</span>
                  <span className="value warning">Warning</span>
                </div>
                <div className="info-item">
                  <span>Road Curvature:</span>
                  <span className="value">0.02 rad/m</span>
                </div>
              </div>
            </div>
          </div>
        );
      
      case 'mark':
        return (
          <div className="camera-feed">
            <div className="feed-placeholder">
              <h3>Object Detection Results</h3>
              <p>YOLO v8 traffic sign and vehicle detection</p>
              <VideoStream />
              <div className="overlay-info">
                <div className="info-item">
                  <span>Objects Detected:</span>
                  <span className="value">{telemetry.objects_detected || 0}</span>
                </div>
                <div className="info-item">
                  <span>Processing Time:</span>
                  <span className="value">{telemetry.processing_time || 0}ms</span>
                </div>
                <div className="info-item">
                  <span>Confidence:</span>
                  <span className="value">92%</span>
                </div>
              </div>
            </div>
          </div>
        );
      
      case 'depth':
        return (
          <div className="camera-feed">
            <div className="feed-placeholder">
              <h3>Depth Analysis Results</h3>
              <p>3D obstacle detection and spatial mapping</p>
              <div className="depth-visualization">
                <div className="depth-info">
                  <h4>Depth Map Status</h4>
                  <div className="status-grid">
                    <div className="status-item">
                      <span>ZED Camera:</span>
                      <span className="status-indicator connected">Connected</span>
                    </div>
                    <div className="status-item">
                      <span>Depth Quality:</span>
                      <span className="value">High</span>
                    </div>
                    <div className="status-item">
                      <span>3D Points:</span>
                      <span className="value">15,432</span>
                    </div>
                    <div className="status-item">
                      <span>Obstacles:</span>
                      <span className="value">2 detected</span>
                    </div>
                  </div>
                </div>
                <div className="depth-map-placeholder">
                  <p>3D Point Cloud Visualization</p>
                  <div className="point-cloud-viz">
                    {/* Placeholder for 3D visualization */}
                    <div className="viz-grid">
                      {Array.from({ length: 100 }, (_, i) => (
                        <div 
                          key={i} 
                          className="viz-point" 
                          style={{
                            opacity: Math.random(),
                            backgroundColor: `hsl(${Math.random() * 60 + 200}, 70%, 50%)`
                          }}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
      
      case 'combined':
        return (
          <div className="camera-feed">
            <div className="combined-overlay">
              <div className="overlay-container">
                <VideoStream />
                <div className="overlay-layer lane-overlay">
                  <h4>Lane Detection</h4>
                  <div className="visualization-content">
                    <svg className="overlay-svg" viewBox="0 0 640 480">
                      {/* Lane lines */}
                      <path d="M 100 480 Q 150 300 200 100" stroke="#bb86fc" strokeWidth="3" fill="none" />
                      <path d="M 440 480 Q 490 300 540 100" stroke="#bb86fc" strokeWidth="3" fill="none" />
                      {/* Center line */}
                      <path d="M 320 480 L 370 100" stroke="#4CAF50" strokeWidth="2" fill="none" strokeDasharray="10,5" />
                    </svg>
                  </div>
                </div>
                <div className="overlay-layer object-overlay">
                  <h4>Object Detection</h4>
                  <div className="visualization-content">
                    <svg className="overlay-svg" viewBox="0 0 640 480">
                      {/* Bounding boxes */}
                      <rect x="300" y="200" width="80" height="60" stroke="#4CAF50" strokeWidth="2" fill="none" />
                      <text x="305" y="195" fill="#4CAF50" fontSize="12">Stop Sign (95%)</text>
                      <rect x="450" y="300" width="100" height="80" stroke="#FF9800" strokeWidth="2" fill="none" />
                      <text x="455" y="295" fill="#FF9800" fontSize="12">Vehicle (87%)</text>
                    </svg>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
      
      default:
        return <div>Tab content not found</div>;
    }
  };

  return (
    <div className="app-container">
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

        {activeTab !== 'dashboard' && (
          <div className="right-panel">
            <h3>System Status</h3>
            <div className="telemetry-section">
              <h4>Processing Performance</h4>
              <div className="data-row">Frame Rate: <span>{telemetry.frame_rate || 'N/A'} FPS</span></div>
              <div className="data-row">Objects Detected: <span>{telemetry.objects_detected || 'N/A'}</span></div>
              <div className="data-row">Processing Time: <span>{telemetry.processing_time || 'N/A'}ms</span></div>
              <div className="data-row">Memory Usage: <span>1.2GB/4GB</span></div>
            </div>
            
            <div className="telemetry-section">
              <h4>Hardware Status</h4>
              <div className="data-row">ZED Camera: <span className="status-connected">Connected</span></div>
              <div className="data-row">Arduino: <span className="status-connected">{arduinoData.connection || 'N/A'}</span></div>
              <div className="data-row">Last Command: <span>{arduinoData.last_command || 'N/A'}</span></div>
            </div>
            
            <div className="telemetry-section">
              <h4>Safety Status</h4>
              <div className="data-row">System State: <span className="status-safe">SAFE</span></div>
              <div className="data-row">Emergency Stop: <span className="status-inactive">Inactive</span></div>
              <div className="data-row">Watchdog: <span className="status-active">Active</span></div>
            </div>
            
            <div className="telemetry-section">
              <h4>Navigation</h4>
              <div className="data-row">Speed: <span>25 km/h</span></div>
              <div className="data-row">Steering: <span>+5°</span></div>
              <div className="data-row">Lane Offset: <span>-0.1m</span></div>
              <div className="data-row">Next Turn: <span>200m</span></div>
            </div>
          </div>
        )}
      </div>

      {activeTab !== 'dashboard' && (
        <div className="bottom-panel">
          <h3>System Logs</h3>
          <div className="log-output">
            {logs.map((log, index) => (
              <div key={index} className="log-entry">
                {log}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;