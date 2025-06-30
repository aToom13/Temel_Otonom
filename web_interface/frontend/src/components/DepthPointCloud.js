import React, { useEffect, useRef, useState } from 'react';
import { subscribePointcloud, unsubscribePointcloud } from '../services/socket';
import './DepthPointCloud.css';

/**
 * DepthPointCloud – lightweight 2-D canvas visualiser for the point-cloud
 * streamed by the backend.  It renders up to 2 048 points at ~10 Hz and
 * requires no external dependencies beyond the HTML5 canvas API.
 *
 * The backend emits an object of shape: { points: [[x, y, z], ...] } where the
 * units are metres in the camera coordinate frame (x → right, y → down, z →
 * forward).  We project these onto a virtual image-plane with a very simple
 * perspective transform
 *   Xs =  fx * (x / z)  + cx
 *   Ys =  fy * (y / z)  + cy
 * and draw them as coloured pixels.  Distant points fade into blue while
 * nearby points are rendered red-yellow, giving a quick depth cue.
 */
const DepthPointCloud = ({ width = 640, height = 480 }) => {
  const canvasRef = useRef(null);
  const [points, setPoints] = useState([]);

  // Keep track of the most recent point-cloud
  useEffect(() => {
    const handlePointcloud = (data) => {
      if (Array.isArray(data?.points)) {
        setPoints(data.points);
      }
    };

    subscribePointcloud(handlePointcloud);
    return () => unsubscribePointcloud();
  }, []);

  // Draw whenever points update
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (points.length === 0) return;

    // Simple camera intrinsics (same as backend)
    const fx = 300; // arbitrary focal lengths for screen-space projection
    const fy = 300;
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;

    for (let i = 0; i < points.length; i++) {
      const [x, y, z] = points[i];
      if (z <= 0.01) continue; // avoid divide-by-zero & back-projection

      const xs = (x * fx) / z + cx;
      const ys = (y * fy) / z + cy;

      if (xs < 0 || xs >= canvas.width || ys < 0 || ys >= canvas.height) {
        continue;
      }

      // Depth-based colour (near ⇒ warm, far ⇒ cool)
      const norm = Math.min(z / 10, 1); // assume 0-10 m range
      const r = Math.floor(255 * (1 - norm));
      const g = Math.floor(255 * norm);
      const b = 255 - r;
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(xs, ys, 2, 2);
    }
  }, [points]);

  return (
    <div className="depth-pointcloud-wrapper" style={{ textAlign: 'center' }}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ background: '#000', border: '1px solid #333' }}
      />
      <p style={{ color: '#e0e0e0', fontSize: '0.8rem', marginTop: '4px' }}>
        Point Cloud – {points.length} points
      </p>
    </div>
  );
};

export default DepthPointCloud;
