import React, { useState, useEffect } from 'react';
import EnhancedVideoStream from './EnhancedVideoStream';
import { apiService } from '../api';

const ProcessingVisualization = ({ processingType = 'combined' }) => {
  const [processingData, setProcessingData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchProcessingData = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        const systemStatus = await apiService.getSystemStatus();
        setProcessingData(systemStatus);
      } catch (error) {
        console.error('Failed to fetch processing data:', error);
        setError('Failed to load processing data');
      } finally {
        setIsLoading(false);
      }
    };

    fetchProcessingData();
    const interval = setInterval(fetchProcessingData, 500); // 2 Hz

    return () => clearInterval(interval);
  }, []);

  const renderProcessingInfo = () => {
    if (!processingData) return null;

    const getOverlayType = () => {
      switch (processingType) {
        case 'lanes': return 'lanes';
        case 'objects': return 'objects';
        case 'obstacles': return 'obstacles';
        case 'combined': return 'combined';
        default: return 'none';
      }
    };

    const getProcessingStats = () => {
      switch (processingType) {
        case 'lanes':
          return {
            title: 'Lane Detection Processing',
            stats: [
              { label: 'Lanes Detected', value: processingData.lane_results?.lanes?.length || 0 },
              { label: 'Detection Quality', value: `${((processingData.lane_results?.detection_quality || 0) * 100).toFixed(1)}%` },
              { label: 'Lane Departure', value: processingData.lane_results?.lane_departure_warning ? 'Warning' : 'Normal' },
              { label: 'Road Curvature', value: `${(processingData.lane_results?.road_curvature || 0).toFixed(3)} rad/m` },
              { label: 'Center Offset', value: `${(processingData.lane_results?.lane_center_offset || 0).toFixed(2)}m` }
            ]
          };
        
        case 'objects':
          return {
            title: 'Object Detection Processing',
            stats: [
              { label: 'Objects Detected', value: processingData.detection_results?.traffic_signs?.length || 0 },
              { label: 'Processing Time', value: '15ms' }, // Placeholder
              { label: 'Model Confidence', value: '92%' }, // Placeholder
              { label: 'Detection Rate', value: `${processingData.performance_metrics?.fps?.toFixed(1) || 0} FPS` },
              { label: 'Model Type', value: 'YOLOv8' }
            ]
          };
        
        case 'obstacles':
          return {
            title: 'Depth & Obstacle Analysis',
            stats: [
              { label: 'Obstacles Found', value: processingData.obstacle_results?.obstacle_count || 0 },
              { label: 'Depth Available', value: processingData.camera_status?.has_depth ? 'Yes' : 'No' },
              { label: 'Processing Quality', value: `${((processingData.obstacle_results?.processing_quality || 0) * 100).toFixed(1)}%` },
              { label: 'Camera Type', value: processingData.camera_status?.camera_type || 'None' },
              { label: 'Analysis Status', value: processingData.obstacle_results?.status || 'N/A' }
            ]
          };
        
        case 'combined':
          return {
            title: 'Multi-Modal Processing',
            stats: [
              { label: 'Overall FPS', value: `${processingData.performance_metrics?.fps?.toFixed(1) || 0}` },
              { label: 'Lanes', value: processingData.lane_results?.lanes?.length || 0 },
              { label: 'Objects', value: processingData.detection_results?.traffic_signs?.length || 0 },
              { label: 'Obstacles', value: processingData.obstacle_results?.obstacle_count || 0 },
              { label: 'LiDAR Points', value: processingData.lidar_results?.total_points || 0 },
              { label: 'Safety State', value: processingData.safety_status?.current_state || 'UNKNOWN' }
            ]
          };
        
        default:
          return { title: 'Processing Visualization', stats: [] };
      }
    };

    const { title, stats } = getProcessingStats();

    return (
      <div style={{
        position: 'absolute',
        bottom: '20px',
        left: '20px',
        background: 'rgba(0,0,0,0.8)',
        padding: '15px',
        borderRadius: '8px',
        color: '#fff',
        minWidth: '250px',
        zIndex: 10
      }}>
        <h4 style={{ margin: '0 0 10px 0', color: '#58a6ff' }}>{title}</h4>
        <div style={{ display: 'grid', gap: '5px' }}>
          {stats.map((stat, index) => (
            <div key={index} style={{ 
              display: 'flex', 
              justifyContent: 'space-between',
              fontSize: '0.9rem'
            }}>
              <span style={{ color: '#8b949e' }}>{stat.label}:</span>
              <span style={{ fontWeight: 'bold' }}>{stat.value}</span>
            </div>
          ))}
        </div>
        
        {isLoading && (
          <div style={{ 
            marginTop: '10px', 
            fontSize: '0.8rem', 
            color: '#FF9800' 
          }}>
            Updating...
          </div>
        )}
        
        {error && (
          <div style={{ 
            marginTop: '10px', 
            fontSize: '0.8rem', 
            color: '#f44336' 
          }}>
            {error}
          </div>
        )}
      </div>
    );
  };

  const renderPerformanceIndicator = () => {
    if (!processingData) return null;

    const fps = processingData.performance_metrics?.fps || 0;
    const getPerformanceColor = (fps) => {
      if (fps > 25) return '#4CAF50';
      if (fps > 15) return '#FF9800';
      return '#f44336';
    };

    return (
      <div style={{
        position: 'absolute',
        top: '60px',
        right: '20px',
        background: 'rgba(0,0,0,0.7)',
        padding: '8px 12px',
        borderRadius: '4px',
        color: getPerformanceColor(fps),
        fontSize: '0.9rem',
        fontWeight: 'bold',
        zIndex: 10
      }}>
        {fps.toFixed(1)} FPS
      </div>
    );
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <EnhancedVideoStream
        showOverlays={true}
        overlayType={getOverlayType()}
        onFrameUpdate={() => {
          // Frame update callback if needed
        }}
      />
      
      {renderProcessingInfo()}
      {renderPerformanceIndicator()}
    </div>
  );

  function getOverlayType() {
    switch (processingType) {
      case 'lanes': return 'lanes';
      case 'objects': return 'objects';
      case 'obstacles': return 'obstacles';
      case 'combined': return 'combined';
      default: return 'none';
    }
  }
};

export default ProcessingVisualization;