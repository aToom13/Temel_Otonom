import React, { useEffect, useRef, useState } from 'react';
import { subscribeCameraFrame, unsubscribeCameraFrame } from '../services/socket';

function VideoStream() {
  const [frame, setFrame] = useState(null);
  const [streamStatus, setStreamStatus] = useState('connecting');
  const [error, setError] = useState(null);
  const imgRef = useRef();
  const streamRef = useRef();

  useEffect(() => {
    let mounted = true;

    // Try WebSocket first for real-time frames
    const handleFrame = (base64Frame) => {
      if (mounted) {
        setFrame(base64Frame);
        setStreamStatus('connected');
        setError(null);
      }
    };

    // Subscribe to WebSocket camera frames
    subscribeCameraFrame(handleFrame);

    // Fallback to HTTP stream if WebSocket fails
    const fallbackToHttpStream = () => {
      if (!mounted) return;

      const img = new Image();
      img.onload = () => {
        if (mounted) {
          setStreamStatus('connected');
          setError(null);
        }
      };
      
      img.onerror = () => {
        if (mounted) {
          setStreamStatus('error');
          setError('Failed to load video stream');
        }
      };

      // Set up HTTP stream
      streamRef.current = img;
      img.src = `${process.env.REACT_APP_API_URL || 'http://localhost:5000'}/video_feed?t=${Date.now()}`;
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
      unsubscribeCameraFrame();
      if (streamRef.current) {
        streamRef.current.src = '';
      }
    };
  }, [frame]);

  const renderStreamStatus = () => {
    const statusConfig = {
      connecting: { color: '#FF9800', text: 'Connecting to camera...' },
      connected: { color: '#4CAF50', text: 'Camera connected' },
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
      </div>
    );
  };

  return (
    <div className="video-stream" style={{ position: 'relative', width: '100%', height: '100%' }}>
      {renderStreamStatus()}
      
      {frame ? (
        // WebSocket frame
        <img
          ref={imgRef}
          src={`data:image/jpeg;base64,${frame}`}
          alt="Camera Stream"
          style={{ 
            width: '100%', 
            height: '100%', 
            objectFit: 'contain',
            background: '#222' 
          }}
          onError={() => {
            setStreamStatus('error');
            setError('Failed to display frame');
          }}
        />
      ) : streamRef.current ? (
        // HTTP stream fallback
        <img
          src={streamRef.current.src}
          alt="Camera Stream"
          style={{ 
            width: '100%', 
            height: '100%', 
            objectFit: 'contain',
            background: '#222' 
          }}
          onError={() => {
            setStreamStatus('error');
            setError('HTTP stream failed');
          }}
        />
      ) : (
        // Placeholder
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
}

export default VideoStream;