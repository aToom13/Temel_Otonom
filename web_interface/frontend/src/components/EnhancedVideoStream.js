import React, { useEffect, useRef, useState, useCallback } from 'react';
import { subscribeCameraFrame, unsubscribeCameraFrame } from '../services/socket';
import { apiService } from '../api';

const EnhancedVideoStream = ({ 
  showOverlays = false, 
  overlayType = 'none',
  onFrameUpdate = null,
  className = '',
  style = {}
}) => {
  const [frame, setFrame] = useState(null);
  const [streamStatus, setStreamStatus] = useState('connecting');
  const [error, setError] = useState(null);
  const [cameraInfo, setCameraInfo] = useState(null);
  const [overlayData, setOverlayData] = useState(null);
  
  const imgRef = useRef();
  const canvasRef = useRef();
  const streamRef = useRef();
  const overlayRef = useRef();

  // Fetch camera and overlay data
  const fetchCameraData = useCallback(async () => {
    try {
      const status = await apiService.getSystemStatus();
      setCameraInfo(status.camera_status);
      
      if (showOverlays) {
        setOverlayData({
          lanes: status.lane_results,
          objects: status.detection_results,
          obstacles: status.obstacle_results
        });
      }
    } catch (error) {
      console.warn('Failed to fetch camera data:', error);
    }
  }, [showOverlays]);

  useEffect(() => {
    let mounted = true;
    let frameUpdateInterval;

    // WebSocket frame handler
    const handleFrame = (base64Frame) => {
      if (mounted) {
        setFrame(base64Frame);
        setStreamStatus('connected');
        setError(null);
        
        if (onFrameUpdate) {
          onFrameUpdate(base64Frame);
        }
      }
    };

    // Subscribe to WebSocket camera frames
    subscribeCameraFrame(handleFrame);

    // Fetch camera data periodically
    fetchCameraData();
    const dataInterval = setInterval(fetchCameraData, 1000);

    // Fallback to HTTP stream if WebSocket fails
    const fallbackToHttpStream = () => {
      if (!mounted) return;

      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      img.onload = () => {
        if (mounted) {
          setStreamStatus('connected');
          setError(null);
          
          // Draw to canvas for overlay support
          if (canvasRef.current && showOverlays) {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            drawOverlays(ctx, canvas.width, canvas.height);
          }
        }
      };
      
      img.onerror = () => {
        if (mounted) {
          setStreamStatus('error');
          setError('Failed to load video stream');
        }
      };

      streamRef.current = img;
      img.src = `${process.env.REACT_APP_API_URL || 'http://localhost:5000'}/video_feed?t=${Date.now()}`;
      
      // Refresh HTTP stream periodically
      frameUpdateInterval = setInterval(() => {
        if (mounted && img) {
          img.src = `${process.env.REACT_APP_API_URL || 'http://localhost:5000'}/video_feed?t=${Date.now()}`;
        }
      }, 100); // 10 FPS
    };

    // Try WebSocket for 3 seconds, then fallback to HTTP
    const fallbackTimer = setTimeout(() => {
      if (mounted && !frame) {
        console.log('WebSocket camera frames not available, falling back to HTTP stream');
        fallbackToHttpStream();
      }
    }, 3000);

    return () => {
      mounted = false;
      clearTimeout(fallbackTimer);
      clearInterval(dataInterval);
      if (frameUpdateInterval) {
        clearInterval(frameUpdateInterval);
      }
      unsubscribeCameraFrame();
      if (streamRef.current) {
        streamRef.current.src = '';
      }
    };
  }, [frame, showOverlays, onFrameUpdate, fetchCameraData]);

  // Draw overlays on canvas
  const drawOverlays = useCallback((ctx, width, height) => {
    if (!overlayData || !showOverlays) return;

    ctx.save();

    // Draw lane detection overlays
    if (overlayType === 'lanes' || overlayType === 'combined') {
      drawLaneOverlays(ctx, width, height, overlayData.lanes);
    }

    // Draw object detection overlays
    if (overlayType === 'objects' || overlayType === 'combined') {
      drawObjectOverlays(ctx, width, height, overlayData.objects);
    }

    // Draw obstacle overlays
    if (overlayType === 'obstacles' || overlayType === 'combined') {
      drawObstacleOverlays(ctx, width, height, overlayData.obstacles);
    }

    ctx.restore();
  }, [overlayData, overlayType, showOverlays]);

  const drawLaneOverlays = (ctx, width, height, laneData) => {
    if (!laneData || !laneData.lanes) return;

    ctx.strokeStyle = '#bb86fc';
    ctx.lineWidth = 3;
    ctx.setLineDash([]);

    // Draw lane lines (simplified)
    const leftLane = laneData.lanes.find(l => l.lane_type === 'left');
    const rightLane = laneData.lanes.find(l => l.lane_type === 'right');

    if (leftLane) {
      ctx.beginPath();
      ctx.moveTo(width * 0.2, height);
      ctx.quadraticCurveTo(width * 0.3, height * 0.5, width * 0.35, 0);
      ctx.stroke();
    }

    if (rightLane) {
      ctx.beginPath();
      ctx.moveTo(width * 0.8, height);
      ctx.quadraticCurveTo(width * 0.7, height * 0.5, width * 0.65, 0);
      ctx.stroke();
    }

    // Draw center line
    ctx.strokeStyle = '#4CAF50';
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 5]);
    ctx.beginPath();
    ctx.moveTo(width * 0.5, height);
    ctx.lineTo(width * 0.5, 0);
    ctx.stroke();

    // Lane departure warning
    if (laneData.lane_departure_warning) {
      ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
      ctx.fillRect(0, 0, width, height);
      
      ctx.fillStyle = '#ff0000';
      ctx.font = '20px Arial';
      ctx.fillText('LANE DEPARTURE WARNING', 20, 40);
    }
  };

  const drawObjectOverlays = (ctx, width, height, objectData) => {
    if (!objectData || !objectData.traffic_signs) return;

    objectData.traffic_signs.forEach((sign, index) => {
      const bbox = sign.bbox || [100 + index * 150, 100, 80, 60];
      const [x, y, w, h] = bbox;

      // Draw bounding box
      ctx.strokeStyle = '#4CAF50';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      // Draw label
      ctx.fillStyle = '#4CAF50';
      ctx.font = '14px Arial';
      ctx.fillText(
        `${sign.label} (${(sign.confidence * 100).toFixed(0)}%)`,
        x, y - 5
      );
    });
  };

  const drawObstacleOverlays = (ctx, width, height, obstacleData) => {
    if (!obstacleData || !obstacleData.obstacles) return;

    obstacleData.obstacles.forEach((obstacle, index) => {
      const x = width * 0.3 + index * 100;
      const y = height * 0.4;
      const size = 50;

      // Draw obstacle indicator
      ctx.strokeStyle = '#ff4444';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, size, size);

      ctx.fillStyle = 'rgba(255, 68, 68, 0.3)';
      ctx.fillRect(x, y, size, size);

      // Draw distance
      ctx.fillStyle = '#ff4444';
      ctx.font = '12px Arial';
      ctx.fillText(
        `${obstacle.confidence.toFixed(2)}`,
        x + 5, y + 25
      );
    });
  };

  // Update canvas when frame or overlay data changes
  useEffect(() => {
    if (frame && canvasRef.current && showOverlays) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      
      const img = new Image();
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        drawOverlays(ctx, img.width, img.height);
      };
      img.src = `data:image/jpeg;base64,${frame}`;
    }
  }, [frame, overlayData, drawOverlays, showOverlays]);

  const renderStreamStatus = () => {
    const statusConfig = {
      connecting: { color: '#FF9800', text: 'Connecting to camera...' },
      connected: { color: '#4CAF50', text: `${cameraInfo?.camera_type || 'Camera'} connected` },
      error: { color: '#F44336', text: error || 'Camera error' }
    };

    const config = statusConfig[streamStatus];
    
    return (
      <div style={{
        position: 'absolute',
        top: '10px',
        left: '10px',
        background: 'rgba(0,0,0,0.7)',
        padding: '5px 10px',
        borderRadius: '4px',
        color: config.color,
        fontSize: '0.8rem',
        zIndex: 10
      }}>
        <span style={{
          display: 'inline-block',
          width: '6px',
          height: '6px',
          borderRadius: '50%',
          backgroundColor: config.color,
          marginRight: '6px'
        }}></span>
        {config.text}
        {cameraInfo && (
          <div style={{ fontSize: '0.7rem', marginTop: '2px' }}>
            {cameraInfo.resolution[0]}x{cameraInfo.resolution[1]} @ {cameraInfo.fps.toFixed(1)}fps
          </div>
        )}
      </div>
    );
  };

  const renderCameraInfo = () => {
    if (!cameraInfo) return null;

    return (
      <div style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        background: 'rgba(0,0,0,0.7)',
        padding: '8px 12px',
        borderRadius: '4px',
        color: '#fff',
        fontSize: '0.8rem',
        zIndex: 10
      }}>
        <div>Type: {cameraInfo.camera_type}</div>
        <div>Depth: {cameraInfo.has_depth ? 'Yes' : 'No'}</div>
        <div>IMU: {cameraInfo.has_imu ? 'Yes' : 'No'}</div>
      </div>
    );
  };

  return (
    <div 
      className={`enhanced-video-stream ${className}`}
      style={{ 
        position: 'relative', 
        width: '100%', 
        height: '100%',
        ...style 
      }}
    >
      {renderStreamStatus()}
      {renderCameraInfo()}
      
      {showOverlays && (
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            objectFit: 'contain',
            zIndex: 2
          }}
        />
      )}
      
      {frame ? (
        <img
          ref={imgRef}
          src={`data:image/jpeg;base64,${frame}`}
          alt="Camera Stream"
          style={{ 
            width: '100%', 
            height: '100%', 
            objectFit: 'contain',
            background: '#222',
            zIndex: showOverlays ? 1 : 2
          }}
          onError={() => {
            setStreamStatus('error');
            setError('Failed to display frame');
          }}
        />
      ) : streamRef.current ? (
        <img
          src={streamRef.current.src}
          alt="Camera Stream"
          style={{ 
            width: '100%', 
            height: '100%', 
            objectFit: 'contain',
            background: '#222',
            zIndex: showOverlays ? 1 : 2
          }}
          onError={() => {
            setStreamStatus('error');
            setError('HTTP stream failed');
          }}
        />
      ) : (
        <div className="feed-placeholder" style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexDirection: 'column',
          background: '#222',
          color: '#888'
        }}>
          <div style={{ fontSize: '1.2rem', marginBottom: '10px' }}>
            {streamStatus === 'connecting' ? 'Connecting to camera...' : 'No camera data available'}
          </div>
          {error && (
            <div style={{ fontSize: '0.9rem', color: '#f44336' }}>
              {error}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default EnhancedVideoStream;