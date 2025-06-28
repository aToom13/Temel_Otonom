import React, { useState, useEffect } from 'react';
import './index.css';
import { fetchTelemetry, fetchLogs, fetchArduinoData } from './api';
import VideoStream from './components/VideoStream';
import { subscribeLogUpdate, unsubscribeLogUpdate } from './services/socket';

function App() {
  const [activeTab, setActiveTab] = useState('raw');
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
    // Her 5 saniyede bir verileri güncelle
    const interval = setInterval(loadData, 5000);
    // WebSocket ile log güncellemelerini dinle
    const handleLog = (log) => setLogs((prev) => [...prev.slice(-99), log]);
    subscribeLogUpdate(handleLog);
    return () => {
      clearInterval(interval);
      unsubscribeLogUpdate();
    };
  }, []);

  const tabs = [
    { id: 'raw', label: 'Raw Camera' },
    { id: 'strip', label: 'Strip Results' },
    { id: 'mark', label: 'Mark Results' },
    { id: 'depth', label: 'Depth Results' },
    { id: 'combined', label: 'Combined Results' }
  ];

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
            {tabs.map(tab => (
              activeTab === tab.id && (
                <div key={tab.id} className="camera-feed">
                  <div className="feed-placeholder">
                    {tab.id === 'raw' ? (
                      <VideoStream />
                    ) : tab.id === 'combined' ? (
                      <div className="combined-overlay">
                        <div className="overlay-container">
                          <div className="overlay-layer">
                            <h4>Strip Detection</h4>
                            <div className="visualization-content">
                              {/* Strip visualization will appear here */}
                            </div>
                          </div>
                          <div className="overlay-layer">
                            <h4>Mark Detection</h4>
                            <div className="visualization-content">
                              {/* Mark visualization will appear here */}
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      `${tab.label} will appear here`
                    )}
                  </div>
                </div>
              )
            ))}
          </div>
        </div>

        <div className="right-panel">
          <h3>Telemetry Data</h3>
          <div className="telemetry-section">
            <h4>Processing Results</h4>
            <div className="data-row">Frame Rate: <span>{telemetry.frame_rate || 'N/A'} FPS</span></div>
            <div className="data-row">Objects Detected: <span>{telemetry.objects_detected || 'N/A'}</span></div>
            <div className="data-row">Processing Time: <span>{telemetry.processing_time || 'N/A'}ms</span></div>
          </div>
          <div className="telemetry-section">
            <h4>Arduino Communication</h4>
            <div className="data-row">Last Sent: <span>{arduinoData.status || 'N/A'}</span></div>
            <div className="data-row">Last Received: <span>{arduinoData.last_command || 'N/A'}</span></div>
            <div className="data-row">Connection: <span>{arduinoData.connection || 'N/A'}</span></div>
          </div>
          <div className="telemetry-section">
            <h4>System Status</h4>
            <div className="data-row">CPU Usage: <span>42%</span></div>
            <div className="data-row">Memory: <span>1.2GB/4GB</span></div>
          </div>
        </div>
      </div>

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
    </div>
  );
}

export default App;