import React, { useEffect, useRef, useState } from 'react';
import { subscribeCameraFrame, unsubscribeCameraFrame } from '../services/socket';

function VideoStream() {
  const [frame, setFrame] = useState(null);
  const imgRef = useRef();

  useEffect(() => {
    subscribeCameraFrame((base64Frame) => {
      setFrame(base64Frame);
    });
    return () => {
      unsubscribeCameraFrame();
    };
  }, []);

  return (
    <div className="video-stream">
      {frame ? (
        <img
          ref={imgRef}
          src={`data:image/jpeg;base64,${frame}`}
          alt="Camera Stream"
          style={{ width: '100%', height: 'auto', background: '#222' }}
        />
      ) : (
        <div className="feed-placeholder">Kamera verisi bekleniyor...</div>
      )}
    </div>
  );
}

export default VideoStream;
