import React, { useState, useEffect, useRef, useCallback } from 'react';
import { apiService } from '../api';
import './LidarVisualization.css';

const LidarVisualization = () => {
  const [lidarData, setLidarData] = useState(null);
  const [lidarStatus, setLidarStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    // Fetch LiDAR data periodically
    const fetchLidarData = async () => {
      try {
        setError(null);
        
        const [dataResponse, statusResponse] = await Promise.all([
          apiService.getLidarData().catch(() => null),
          apiService.getLidarStatus().catch(() => null)
        ]);

        if (dataResponse) {
          setLidarData(dataResponse);
        }

        if (statusResponse) {
          setLidarStatus(statusResponse);
        }
      } catch (error) {
        console.error('Failed to fetch LiDAR data:', error);
        setError('Failed to connect to LiDAR service');
      }
    };

    // Initial fetch
    fetchLidarData();

    // Set up periodic updates
    const interval = setInterval(fetchLidarData, 200); // 5 Hz

    return () => clearInterval(interval);
  }, []);

  const drawLidarData = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !lidarData) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const scale = Math.min(width, height) / 24; // 12m max range, scale to fit

    // Clear canvas
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    drawGrid(ctx, centerX, centerY, scale);

    // Draw safety zones
    if (lidarData.safety_zones) {
      drawSafetyZones(ctx, centerX, centerY, scale, lidarData.safety_zones);
    }

    // Draw LiDAR points
    if (lidarData.points && lidarData.points.length > 0) {
      drawLidarPoints(ctx, centerX, centerY, scale, lidarData.points);
    }

    // Draw obstacles
    if (lidarData.obstacles && lidarData.obstacles.length > 0) {
      drawObstacles(ctx, centerX, centerY, scale, lidarData.obstacles);
    }

    // Draw robot/vehicle
    drawVehicle(ctx, centerX, centerY);

    // Draw scan line (rotating indicator)
    if (lidarStatus?.is_scanning) {
      drawScanLine(ctx, centerX, centerY, scale);
    }
  }, [lidarData, lidarStatus]);

  useEffect(() => {
    if (lidarData && canvasRef.current) {
      drawLidarData();
    }
  }, [lidarData, drawLidarData]);

  const drawGrid = (ctx, centerX, centerY, scale) => {
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;

    // Concentric circles (distance rings)
    for (let r = 1; r <= 12; r++) {
      ctx.beginPath();
      ctx.arc(centerX, centerY, r * scale, 0, 2 * Math.PI);
      ctx.stroke();
    }

    // Radial lines (angle markers)
    for (let angle = 0; angle < 360; angle += 30) {
      const rad = (angle * Math.PI) / 180;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(
        centerX + Math.cos(rad) * 12 * scale,
        centerY + Math.sin(rad) * 12 * scale
      );
      ctx.stroke();
    }

    // Distance labels
    ctx.fillStyle = '#666';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    for (let r = 2; r <= 12; r += 2) {
      ctx.fillText(`${r}m`, centerX, centerY - r * scale + 4);
    }
  };

  const drawSafetyZones = (ctx, centerX, centerY, scale, safetyZones) => {
    const zones = [
      { name: 'caution', radius: safetyZones.caution, color: 'rgba(255, 255, 0, 0.1)' },
      { name: 'warning', radius: safetyZones.warning, color: 'rgba(255, 165, 0, 0.2)' },
      { name: 'immediate', radius: safetyZones.immediate, color: 'rgba(255, 0, 0, 0.3)' }
    ];

    zones.forEach(zone => {
      ctx.fillStyle = zone.color;
      ctx.beginPath();
      ctx.arc(centerX, centerY, zone.radius * scale, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  const drawLidarPoints = (ctx, centerX, centerY, scale, points) => {
    points.forEach(point => {
      const x = centerX + point.x * scale;
      const y = centerY - point.y * scale; // Flip Y axis

      // Color based on distance and quality
      const intensity = Math.min(255, point.quality || 200);
      const distance = point.distance;
      
      let color;
      if (distance < 1) {
        color = `rgb(255, ${intensity}, ${intensity})`; // Red for close
      } else if (distance < 3) {
        color = `rgb(255, 255, ${intensity})`; // Yellow for medium
      } else {
        color = `rgb(${intensity}, 255, ${intensity})`; // Green for far
      }

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  const drawObstacles = (ctx, centerX, centerY, scale, obstacles) => {
    obstacles.forEach(obstacle => {
      const x = centerX + obstacle.center_x * scale;
      const y = centerY - obstacle.center_y * scale; // Flip Y axis
      const size = obstacle.size * scale;

      // Draw obstacle circle
      ctx.strokeStyle = '#ff4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, size / 2, 0, 2 * Math.PI);
      ctx.stroke();

      // Draw confidence indicator
      ctx.fillStyle = `rgba(255, 68, 68, ${obstacle.confidence})`;
      ctx.beginPath();
      ctx.arc(x, y, size / 2, 0, 2 * Math.PI);
      ctx.fill();

      // Draw obstacle info
      ctx.fillStyle = '#fff';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(
        `${obstacle.confidence.toFixed(2)}`,
        x,
        y + 4
      );
    });
  };

  const drawVehicle = (ctx, centerX, centerY) => {
    // Draw vehicle as a triangle pointing up
    ctx.fillStyle = '#00ff00';
    ctx.beginPath();
    ctx.moveTo(centerX, centerY - 10);
    ctx.lineTo(centerX - 8, centerY + 8);
    ctx.lineTo(centerX + 8, centerY + 8);
    ctx.closePath();
    ctx.fill();

    // Draw vehicle outline
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
  };

  const drawScanLine = (ctx, centerX, centerY, scale) => {
    // Rotating scan line to show LiDAR is active
    const angle = (Date.now() / 10) % 360; // Rotate every 3.6 seconds
    const rad = (angle * Math.PI) / 180;

    ctx.strokeStyle = 'rgba(0, 255, 255, 0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(
      centerX + Math.cos(rad) * 12 * scale,
      centerY + Math.sin(rad) * 12 * scale
    );
    ctx.stroke();
  };

  const handleStartLidar = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const result = await apiService.startLidar();
      console.log('LiDAR start result:', result);
    } catch (error) {
      console.error('Failed to start LiDAR:', error);
      setError('Failed to start LiDAR');
    } finally {
      setIsLoading(false);
    }
  };

  const handleStopLidar = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const result = await apiService.stopLidar();
      console.log('LiDAR stop result:', result);
    } catch (error) {
      console.error('Failed to stop LiDAR:', error);
      setError('Failed to stop LiDAR');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="lidar-visualization">
      <div className="lidar-header">
        <h3>RPLIDAR A1 2D Visualization</h3>
        <div className="lidar-controls">
          <button 
            className="control-button start"
            onClick={handleStartLidar}
            disabled={lidarStatus?.is_scanning || isLoading}
          >
            {isLoading ? 'Starting...' : 'Start Scan'}
          </button>
          <button 
            className="control-button stop"
            onClick={handleStopLidar}
            disabled={!lidarStatus?.is_scanning || isLoading}
          >
            {isLoading ? 'Stopping...' : 'Stop Scan'}
          </button>
        </div>
      </div>

      {error && (
        <div className="error-message" style={{
          background: '#f44336',
          color: 'white',
          padding: '10px',
          margin: '10px',
          borderRadius: '4px'
        }}>
          {error}
        </div>
      )}

      <div className="lidar-content">
        <div className="lidar-canvas-container">
          <canvas
            ref={canvasRef}
            width={600}
            height={600}
            className="lidar-canvas"
          />
        </div>

        <div className="lidar-info">
          <div className="status-section">
            <h4>LiDAR Status</h4>
            <div className="status-grid">
              <div className="status-item">
                <span>Connection:</span>
                <span className={`status-indicator ${lidarStatus?.is_connected ? 'connected' : 'disconnected'}`}>
                  {lidarStatus?.is_connected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              <div className="status-item">
                <span>Scanning:</span>
                <span className={`status-indicator ${lidarStatus?.is_scanning ? 'active' : 'inactive'}`}>
                  {lidarStatus?.is_scanning ? 'Active' : 'Inactive'}
                </span>
              </div>
              <div className="status-item">
                <span>Scan Frequency:</span>
                <span className="value">{lidarStatus?.scan_frequency?.toFixed(1) || 0} Hz</span>
              </div>
              <div className="status-item">
                <span>Total Scans:</span>
                <span className="value">{lidarStatus?.scan_count || 0}</span>
              </div>
            </div>
          </div>

          <div className="data-section">
            <h4>Current Scan Data</h4>
            <div className="data-grid">
              <div className="data-item">
                <span>Points:</span>
                <span className="value">{lidarData?.total_points || 0}</span>
              </div>
              <div className="data-item">
                <span>Obstacles:</span>
                <span className="value">{lidarData?.obstacles?.length || 0}</span>
              </div>
              <div className="data-item">
                <span>Timestamp:</span>
                <span className="value">
                  {lidarData?.timestamp ? new Date(lidarData.timestamp * 1000).toLocaleTimeString() : 'N/A'}
                </span>
              </div>
            </div>
          </div>

          <div className="safety-section">
            <h4>Safety Zones</h4>
            <div className="safety-zones">
              <div className="zone immediate">
                <span>Immediate:</span>
                <span>{lidarData?.safety_zones?.immediate || 0.5}m</span>
              </div>
              <div className="zone warning">
                <span>Warning:</span>
                <span>{lidarData?.safety_zones?.warning || 1.0}m</span>
              </div>
              <div className="zone caution">
                <span>Caution:</span>
                <span>{lidarData?.safety_zones?.caution || 2.0}m</span>
              </div>
            </div>
          </div>

          <div className="legend-section">
            <h4>Legend</h4>
            <div className="legend">
              <div className="legend-item">
                <div className="color-box" style={{backgroundColor: '#00ff00'}}></div>
                <span>Vehicle</span>
              </div>
              <div className="legend-item">
                <div className="color-box" style={{backgroundColor: '#ff4444'}}></div>
                <span>Obstacles</span>
              </div>
              <div className="legend-item">
                <div className="color-box" style={{backgroundColor: '#ffff00'}}></div>
                <span>LiDAR Points</span>
              </div>
              <div className="legend-item">
                <div className="color-box" style={{backgroundColor: 'rgba(255,0,0,0.3)'}}></div>
                <span>Danger Zone</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LidarVisualization;