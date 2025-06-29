import React, { useState, useEffect, useRef, useCallback } from 'react';
import { apiService } from '../api';
import { 
  subscribeSystemStatus, 
  unsubscribeSystemStatus,
  subscribeLidarData,
  unsubscribeLidarData,
  subscribeIMUData,
  unsubscribeIMUData
} from '../services/socket';
import './BirdEyeSimulation.css';

const BirdEyeSimulation = () => {
  const [simulationData, setSimulationData] = useState({
    vehicle: { x: 0, y: 0, heading: 0, speed: 0 },
    lanes: [],
    obstacles: [],
    lidarPoints: [],
    trafficSigns: [],
    plannedPath: [],
    safetyZones: { immediate: 0.5, warning: 1.0, caution: 2.0 }
  });
  
  const [systemStatus, setSystemStatus] = useState(null);
  const [viewSettings, setViewSettings] = useState({
    zoom: 1.0,
    centerOnVehicle: true,
    showGrid: true,
    showTrajectory: true,
    showLidarPoints: true,
    showSafetyZones: true,
    showVelocityVectors: true,
    viewRange: 50 // meters
  });
  
  const [simulationStats, setSimulationStats] = useState({
    fps: 0,
    dataLatency: 0,
    lastUpdate: 0
  });

  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const lastFrameTime = useRef(0);
  const frameCount = useRef(0);

  // Real-time data subscription
  useEffect(() => {
    const handleSystemUpdate = (data) => {
      updateSimulationData(data);
      setSystemStatus(data);
    };

    const handleLidarUpdate = (data) => {
      updateLidarData(data);
    };

    const handleIMUUpdate = (data) => {
      updateVehicleState(data);
    };

    // Subscribe to real-time data
    subscribeSystemStatus(handleSystemUpdate);
    subscribeLidarData(handleLidarUpdate);
    subscribeIMUData(handleIMUUpdate);

    // Initial data fetch
    fetchInitialData();

    return () => {
      unsubscribeSystemStatus();
      unsubscribeLidarData();
      unsubscribeIMUData();
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  // Animation loop
  useEffect(() => {
    const animate = (timestamp) => {
      if (timestamp - lastFrameTime.current >= 33) { // ~30 FPS
        drawSimulation();
        updateFPS(timestamp);
        lastFrameTime.current = timestamp;
      }
      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [simulationData, viewSettings]);

  const fetchInitialData = async () => {
    try {
      const status = await apiService.getSystemStatus();
      updateSimulationData(status);
      setSystemStatus(status);
    } catch (error) {
      console.error('Failed to fetch initial simulation data:', error);
    }
  };

  const updateSimulationData = useCallback((data) => {
    setSimulationData(prev => ({
      ...prev,
      vehicle: {
        x: prev.vehicle.x,
        y: prev.vehicle.y,
        heading: data.imu_data?.heading_degrees || prev.vehicle.heading,
        speed: data.direction_data?.target_speed || prev.vehicle.speed
      },
      lanes: processLaneData(data.lane_results),
      obstacles: processObstacleData(data.obstacle_results),
      trafficSigns: processTrafficSignData(data.detection_results),
      plannedPath: generatePlannedPath(data.direction_data),
      safetyZones: data.lidar_results?.safety_zones || prev.safetyZones
    }));
  }, []);

  const updateLidarData = useCallback((data) => {
    setSimulationData(prev => ({
      ...prev,
      lidarPoints: processLidarPoints(data.points || []),
      obstacles: [...prev.obstacles, ...processLidarObstacles(data.obstacles || [])]
    }));
  }, []);

  const updateVehicleState = useCallback((data) => {
    setSimulationData(prev => ({
      ...prev,
      vehicle: {
        ...prev.vehicle,
        heading: data.heading_degrees || prev.vehicle.heading,
        speed: data.speed_kmh || prev.vehicle.speed
      }
    }));
  }, []);

  const processLaneData = (laneResults) => {
    if (!laneResults || !laneResults.lanes) return [];
    
    return laneResults.lanes.map((lane, index) => ({
      id: `lane_${index}`,
      type: lane.lane_type,
      confidence: lane.confidence,
      points: generateLanePoints(lane, index),
      curvature: lane.curvature || 0
    }));
  };

  const generateLanePoints = (lane, index) => {
    const points = [];
    const laneOffset = (index - 1) * 3.5; // 3.5m lane width
    
    for (let i = 0; i <= 50; i++) {
      const y = i * 2; // 2m intervals
      const curvature = lane.curvature || 0;
      const x = laneOffset + curvature * y * y * 0.001;
      points.push({ x, y });
    }
    
    return points;
  };

  const processObstacleData = (obstacleResults) => {
    if (!obstacleResults || !obstacleResults.obstacles) return [];
    
    return obstacleResults.obstacles.map((obs, index) => ({
      id: `obstacle_${index}`,
      x: obs.center?.x || (Math.random() - 0.5) * 20,
      y: obs.center?.y || Math.random() * 30 + 10,
      z: obs.center?.z || 0,
      size: obs.size || [2, 2, 1.5],
      confidence: obs.confidence || 0.8,
      type: 'unknown'
    }));
  };

  const processLidarPoints = (points) => {
    return points.slice(0, 500).map(point => ({ // Limit for performance
      x: point.x || 0,
      y: point.y || 0,
      distance: point.distance || 0,
      intensity: point.quality || 200
    }));
  };

  const processLidarObstacles = (obstacles) => {
    return obstacles.map((obs, index) => ({
      id: `lidar_obstacle_${index}`,
      x: obs.center_x || 0,
      y: obs.center_y || 0,
      z: 0,
      size: [obs.size || 1, obs.size || 1, 1],
      confidence: obs.confidence || 0.7,
      type: 'lidar_detected'
    }));
  };

  const processTrafficSignData = (detectionResults) => {
    if (!detectionResults || !detectionResults.traffic_signs) return [];
    
    return detectionResults.traffic_signs.map((sign, index) => ({
      id: `sign_${index}`,
      x: (Math.random() - 0.5) * 10,
      y: Math.random() * 20 + 15,
      z: 2,
      type: sign.label,
      confidence: sign.confidence
    }));
  };

  const generatePlannedPath = (directionData) => {
    if (!directionData) return [];
    
    const path = [];
    const steeringAngle = (directionData.steering_angle || 0) * Math.PI / 180;
    const speed = directionData.target_speed || 0;
    
    for (let i = 0; i <= 20; i++) {
      const t = i * 0.5;
      const x = Math.sin(steeringAngle * t * 0.1) * t;
      const y = t * 2;
      path.push({ x, y, speed: speed * (1 - t * 0.02) });
    }
    
    return path;
  };

  const drawSimulation = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    
    // Clear canvas
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);

    // Setup coordinate system
    ctx.save();
    ctx.translate(width / 2, height * 0.8); // Vehicle at bottom center
    ctx.scale(viewSettings.zoom * 10, -viewSettings.zoom * 10); // 10 pixels per meter, flip Y

    // Draw grid
    if (viewSettings.showGrid) {
      drawGrid(ctx);
    }

    // Draw safety zones
    if (viewSettings.showSafetyZones) {
      drawSafetyZones(ctx);
    }

    // Draw LiDAR points
    if (viewSettings.showLidarPoints) {
      drawLidarPoints(ctx);
    }

    // Draw lanes
    drawLanes(ctx);

    // Draw obstacles
    drawObstacles(ctx);

    // Draw traffic signs
    drawTrafficSigns(ctx);

    // Draw planned path
    if (viewSettings.showTrajectory) {
      drawPlannedPath(ctx);
    }

    // Draw velocity vectors
    if (viewSettings.showVelocityVectors) {
      drawVelocityVectors(ctx);
    }

    // Draw vehicle
    drawVehicle(ctx);

    ctx.restore();

    // Draw UI overlays
    drawUIOverlays(ctx, width, height);
  }, [simulationData, viewSettings]);

  const drawGrid = (ctx) => {
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.05;
    
    const range = viewSettings.viewRange;
    for (let i = -range; i <= range; i += 5) {
      // Vertical lines
      ctx.beginPath();
      ctx.moveTo(i, -range);
      ctx.lineTo(i, range);
      ctx.stroke();
      
      // Horizontal lines
      ctx.beginPath();
      ctx.moveTo(-range, i);
      ctx.lineTo(range, i);
      ctx.stroke();
    }
  };

  const drawSafetyZones = (ctx) => {
    const zones = [
      { radius: simulationData.safetyZones.caution, color: 'rgba(255, 255, 0, 0.1)' },
      { radius: simulationData.safetyZones.warning, color: 'rgba(255, 165, 0, 0.2)' },
      { radius: simulationData.safetyZones.immediate, color: 'rgba(255, 0, 0, 0.3)' }
    ];

    zones.forEach(zone => {
      ctx.fillStyle = zone.color;
      ctx.beginPath();
      ctx.arc(0, 0, zone.radius, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  const drawLidarPoints = (ctx) => {
    ctx.fillStyle = '#00ff88';
    simulationData.lidarPoints.forEach(point => {
      const intensity = point.intensity / 255;
      ctx.globalAlpha = intensity * 0.8;
      ctx.beginPath();
      ctx.arc(point.x, point.y, 0.1, 0, 2 * Math.PI);
      ctx.fill();
    });
    ctx.globalAlpha = 1;
  };

  const drawLanes = (ctx) => {
    simulationData.lanes.forEach(lane => {
      ctx.strokeStyle = lane.type === 'center' ? '#ffff00' : '#ffffff';
      ctx.lineWidth = 0.15;
      ctx.setLineDash(lane.type === 'center' ? [1, 0.5] : []);
      
      ctx.beginPath();
      lane.points.forEach((point, index) => {
        if (index === 0) {
          ctx.moveTo(point.x, point.y);
        } else {
          ctx.lineTo(point.x, point.y);
        }
      });
      ctx.stroke();
    });
    ctx.setLineDash([]);
  };

  const drawObstacles = (ctx) => {
    simulationData.obstacles.forEach(obstacle => {
      const alpha = obstacle.confidence;
      ctx.globalAlpha = alpha;
      
      if (obstacle.type === 'lidar_detected') {
        ctx.fillStyle = '#ff4444';
      } else {
        ctx.fillStyle = '#ff8800';
      }
      
      ctx.fillRect(
        obstacle.x - obstacle.size[0] / 2,
        obstacle.y - obstacle.size[1] / 2,
        obstacle.size[0],
        obstacle.size[1]
      );
      
      // Draw confidence text
      ctx.fillStyle = '#ffffff';
      ctx.font = '0.5px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(
        `${(obstacle.confidence * 100).toFixed(0)}%`,
        obstacle.x,
        obstacle.y
      );
    });
    ctx.globalAlpha = 1;
  };

  const drawTrafficSigns = (ctx) => {
    simulationData.trafficSigns.forEach(sign => {
      ctx.fillStyle = '#00aaff';
      ctx.fillRect(sign.x - 0.5, sign.y - 0.5, 1, 1);
      
      // Draw sign type
      ctx.fillStyle = '#ffffff';
      ctx.font = '0.3px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(sign.type.substring(0, 4), sign.x, sign.y + 1.5);
    });
  };

  const drawPlannedPath = (ctx) => {
    if (simulationData.plannedPath.length < 2) return;
    
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 0.2;
    ctx.setLineDash([0.5, 0.3]);
    
    ctx.beginPath();
    simulationData.plannedPath.forEach((point, index) => {
      if (index === 0) {
        ctx.moveTo(point.x, point.y);
      } else {
        ctx.lineTo(point.x, point.y);
      }
    });
    ctx.stroke();
    ctx.setLineDash([]);
  };

  const drawVelocityVectors = (ctx) => {
    const vehicle = simulationData.vehicle;
    const speed = vehicle.speed / 10; // Scale for visualization
    const heading = vehicle.heading * Math.PI / 180;
    
    ctx.strokeStyle = '#00ffff';
    ctx.lineWidth = 0.15;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(Math.sin(heading) * speed, Math.cos(heading) * speed);
    ctx.stroke();
    
    // Draw arrowhead
    const arrowSize = 0.3;
    const endX = Math.sin(heading) * speed;
    const endY = Math.cos(heading) * speed;
    
    ctx.beginPath();
    ctx.moveTo(endX, endY);
    ctx.lineTo(
      endX - arrowSize * Math.sin(heading + 0.5),
      endY - arrowSize * Math.cos(heading + 0.5)
    );
    ctx.moveTo(endX, endY);
    ctx.lineTo(
      endX - arrowSize * Math.sin(heading - 0.5),
      endY - arrowSize * Math.cos(heading - 0.5)
    );
    ctx.stroke();
  };

  const drawVehicle = (ctx) => {
    const vehicle = simulationData.vehicle;
    const heading = vehicle.heading * Math.PI / 180;
    
    ctx.save();
    ctx.rotate(heading);
    
    // Vehicle body
    ctx.fillStyle = '#00ff00';
    ctx.fillRect(-0.9, -2, 1.8, 4);
    
    // Vehicle direction indicator
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.moveTo(0, 2);
    ctx.lineTo(-0.5, 1);
    ctx.lineTo(0.5, 1);
    ctx.closePath();
    ctx.fill();
    
    ctx.restore();
  };

  const drawUIOverlays = (ctx, width, height) => {
    // Vehicle info
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(10, 10, 200, 120);
    
    ctx.fillStyle = '#ffffff';
    ctx.font = '14px Arial';
    ctx.fillText(`Speed: ${simulationData.vehicle.speed.toFixed(1)} km/h`, 20, 30);
    ctx.fillText(`Heading: ${simulationData.vehicle.heading.toFixed(1)}°`, 20, 50);
    ctx.fillText(`Lanes: ${simulationData.lanes.length}`, 20, 70);
    ctx.fillText(`Obstacles: ${simulationData.obstacles.length}`, 20, 90);
    ctx.fillText(`LiDAR Points: ${simulationData.lidarPoints.length}`, 20, 110);

    // Performance info
    ctx.fillRect(width - 150, 10, 140, 80);
    ctx.fillText(`Sim FPS: ${simulationStats.fps.toFixed(1)}`, width - 140, 30);
    ctx.fillText(`Zoom: ${(viewSettings.zoom * 100).toFixed(0)}%`, width - 140, 50);
    ctx.fillText(`Range: ${viewSettings.viewRange}m`, width - 140, 70);
  };

  const updateFPS = (timestamp) => {
    frameCount.current++;
    if (frameCount.current % 30 === 0) {
      const fps = 30000 / (timestamp - simulationStats.lastUpdate);
      setSimulationStats(prev => ({
        ...prev,
        fps: fps || 0,
        lastUpdate: timestamp
      }));
    }
  };

  const handleZoomChange = (delta) => {
    setViewSettings(prev => ({
      ...prev,
      zoom: Math.max(0.1, Math.min(3.0, prev.zoom + delta))
    }));
  };

  const handleRangeChange = (delta) => {
    setViewSettings(prev => ({
      ...prev,
      viewRange: Math.max(10, Math.min(100, prev.viewRange + delta))
    }));
  };

  const toggleSetting = (setting) => {
    setViewSettings(prev => ({
      ...prev,
      [setting]: !prev[setting]
    }));
  };

  return (
    <div className="bird-eye-simulation">
      <div className="simulation-header">
        <h2>3D Kuş Bakışı Sanal Simülasyon</h2>
        <div className="simulation-controls">
          <div className="control-group">
            <label>Zoom:</label>
            <button onClick={() => handleZoomChange(-0.1)}>-</button>
            <span>{(viewSettings.zoom * 100).toFixed(0)}%</span>
            <button onClick={() => handleZoomChange(0.1)}>+</button>
          </div>
          
          <div className="control-group">
            <label>Range:</label>
            <button onClick={() => handleRangeChange(-5)}>-</button>
            <span>{viewSettings.viewRange}m</span>
            <button onClick={() => handleRangeChange(5)}>+</button>
          </div>
          
          <div className="toggle-controls">
            <label>
              <input 
                type="checkbox" 
                checked={viewSettings.showGrid}
                onChange={() => toggleSetting('showGrid')}
              />
              Grid
            </label>
            <label>
              <input 
                type="checkbox" 
                checked={viewSettings.showLidarPoints}
                onChange={() => toggleSetting('showLidarPoints')}
              />
              LiDAR
            </label>
            <label>
              <input 
                type="checkbox" 
                checked={viewSettings.showTrajectory}
                onChange={() => toggleSetting('showTrajectory')}
              />
              Path
            </label>
            <label>
              <input 
                type="checkbox" 
                checked={viewSettings.showSafetyZones}
                onChange={() => toggleSetting('showSafetyZones')}
              />
              Safety
            </label>
            <label>
              <input 
                type="checkbox" 
                checked={viewSettings.showVelocityVectors}
                onChange={() => toggleSetting('showVelocityVectors')}
              />
              Vectors
            </label>
          </div>
        </div>
      </div>

      <div className="simulation-content">
        <div className="simulation-canvas-container">
          <canvas
            ref={canvasRef}
            width={1200}
            height={800}
            className="simulation-canvas"
          />
        </div>

        <div className="simulation-sidebar">
          <div className="data-panel">
            <h4>Vehicle Status</h4>
            <div className="status-grid">
              <div className="status-item">
                <span>Speed:</span>
                <span className="value">{simulationData.vehicle.speed.toFixed(1)} km/h</span>
              </div>
              <div className="status-item">
                <span>Heading:</span>
                <span className="value">{simulationData.vehicle.heading.toFixed(1)}°</span>
              </div>
              <div className="status-item">
                <span>Position:</span>
                <span className="value">
                  ({simulationData.vehicle.x.toFixed(1)}, {simulationData.vehicle.y.toFixed(1)})
                </span>
              </div>
            </div>
          </div>

          <div className="data-panel">
            <h4>Detection Summary</h4>
            <div className="status-grid">
              <div className="status-item">
                <span>Lanes:</span>
                <span className="value">{simulationData.lanes.length}</span>
              </div>
              <div className="status-item">
                <span>Obstacles:</span>
                <span className="value">{simulationData.obstacles.length}</span>
              </div>
              <div className="status-item">
                <span>Traffic Signs:</span>
                <span className="value">{simulationData.trafficSigns.length}</span>
              </div>
              <div className="status-item">
                <span>LiDAR Points:</span>
                <span className="value">{simulationData.lidarPoints.length}</span>
              </div>
            </div>
          </div>

          <div className="data-panel">
            <h4>System Performance</h4>
            <div className="status-grid">
              <div className="status-item">
                <span>Simulation FPS:</span>
                <span className="value">{simulationStats.fps.toFixed(1)}</span>
              </div>
              <div className="status-item">
                <span>Data Latency:</span>
                <span className="value">{simulationStats.dataLatency.toFixed(0)}ms</span>
              </div>
              <div className="status-item">
                <span>Camera:</span>
                <span className={`value ${systemStatus?.camera_status?.is_connected ? 'connected' : 'disconnected'}`}>
                  {systemStatus?.camera_status?.camera_type || 'None'}
                </span>
              </div>
              <div className="status-item">
                <span>LiDAR:</span>
                <span className={`value ${systemStatus?.lidar_status?.is_connected ? 'connected' : 'disconnected'}`}>
                  {systemStatus?.lidar_status?.is_connected ? 'Active' : 'Inactive'}
                </span>
              </div>
            </div>
          </div>

          <div className="data-panel">
            <h4>Safety Status</h4>
            <div className="status-grid">
              <div className="status-item">
                <span>Safety State:</span>
                <span className={`value ${systemStatus?.safety_status?.current_state?.toLowerCase() || 'unknown'}`}>
                  {systemStatus?.safety_status?.current_state || 'UNKNOWN'}
                </span>
              </div>
              <div className="status-item">
                <span>Emergency Stop:</span>
                <span className={`value ${systemStatus?.safety_status?.emergency_stop_active ? 'active' : 'inactive'}`}>
                  {systemStatus?.safety_status?.emergency_stop_active ? 'ACTIVE' : 'Inactive'}
                </span>
              </div>
              <div className="status-item">
                <span>Immediate Zone:</span>
                <span className="value">{simulationData.safetyZones.immediate}m</span>
              </div>
              <div className="status-item">
                <span>Warning Zone:</span>
                <span className="value">{simulationData.safetyZones.warning}m</span>
              </div>
            </div>
          </div>

          <div className="data-panel">
            <h4>Legend</h4>
            <div className="legend">
              <div className="legend-item">
                <div className="color-box" style={{backgroundColor: '#00ff00'}}></div>
                <span>Vehicle</span>
              </div>
              <div className="legend-item">
                <div className="color-box" style={{backgroundColor: '#ffffff'}}></div>
                <span>Lane Lines</span>
              </div>
              <div className="legend-item">
                <div className="color-box" style={{backgroundColor: '#ff4444'}}></div>
                <span>Obstacles</span>
              </div>
              <div className="legend-item">
                <div className="color-box" style={{backgroundColor: '#00ff88'}}></div>
                <span>LiDAR Points</span>
              </div>
              <div className="legend-item">
                <div className="color-box" style={{backgroundColor: '#00aaff'}}></div>
                <span>Traffic Signs</span>
              </div>
              <div className="legend-item">
                <div className="color-box" style={{backgroundColor: '#00ff00', opacity: 0.3}}></div>
                <span>Planned Path</span>
              </div>
              <div className="legend-item">
                <div className="color-box" style={{backgroundColor: '#00ffff'}}></div>
                <span>Velocity Vector</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BirdEyeSimulation;