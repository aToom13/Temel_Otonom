"""
Güvenlik izleme ve acil durum yönetimi modülü.
PRD'de belirtilen functional safety gereksinimleri için.
"""
import threading
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import psutil
import os

logger = logging.getLogger(__name__)

class SafetyState(Enum):
    """Güvenlik durumu"""
    SAFE = "SAFE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY_STOP = "EMERGENCY_STOP"

class SystemComponent(Enum):
    """Sistem bileşenleri"""
    CAMERA = "CAMERA"
    ARDUINO = "ARDUINO"
    LANE_DETECTION = "LANE_DETECTION"
    OBJECT_DETECTION = "OBJECT_DETECTION"
    DEPTH_ANALYSIS = "DEPTH_ANALYSIS"
    COMMUNICATION = "COMMUNICATION"
    PROCESSING = "PROCESSING"

@dataclass
class HealthStatus:
    """Sağlık durumu"""
    component: SystemComponent
    status: SafetyState
    message: str
    timestamp: float
    metrics: Dict[str, Any]

@dataclass
class SafetyEvent:
    """Güvenlik olayı"""
    event_type: str
    severity: SafetyState
    component: SystemComponent
    description: str
    timestamp: float
    data: Dict[str, Any]

class WatchdogTimer:
    """Watchdog timer sınıfı"""
    
    def __init__(self, timeout: float = 1.0, callback: Callable = None):
        self.timeout = timeout
        self.callback = callback
        self.last_reset = time.time()
        self.active = False
        self.timer_thread = None
    
    def start(self):
        """Watchdog'u başlat"""
        if self.active:
            return
        
        self.active = True
        self.last_reset = time.time()
        self.timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self.timer_thread.start()
        logger.info(f"Watchdog timer started with {self.timeout}s timeout")
    
    def stop(self):
        """Watchdog'u durdur"""
        self.active = False
        if self.timer_thread:
            self.timer_thread.join(timeout=1.0)
        logger.info("Watchdog timer stopped")
    
    def reset(self):
        """Watchdog'u sıfırla"""
        self.last_reset = time.time()
    
    def _timer_loop(self):
        """Watchdog timer döngüsü"""
        while self.active:
            current_time = time.time()
            if current_time - self.last_reset > self.timeout:
                logger.critical("Watchdog timeout detected!")
                if self.callback:
                    self.callback()
                break
            
            time.sleep(0.1)

class SafetyMonitor:
    """
    Ana güvenlik izleme sınıfı.
    - Continuous health monitoring
    - Fault detection and isolation
    - Graceful degradation
    - Emergency stop capabilities
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Safety state
        self.current_state = SafetyState.SAFE
        self.component_health = {}
        self.safety_events = []
        
        # Monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.health_callbacks = {}
        
        # Watchdog
        self.watchdog = WatchdogTimer(
            timeout=self.config.get('watchdog_timeout', 1.0),
            callback=self._watchdog_timeout_handler
        )
        
        # Performance monitoring
        self.performance_thresholds = self.config.get('performance_thresholds', {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'processing_latency': 0.5,  # seconds
            'frame_rate': 10.0  # minimum FPS
        })
        
        # Emergency stop
        self.emergency_stop_active = False
        self.emergency_callbacks = []
        
        logger.info("Safety Monitor initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Varsayılan konfigürasyon"""
        return {
            'watchdog_timeout': 1.0,
            'health_check_interval': 0.5,
            'max_safety_events': 1000,
            'performance_thresholds': {
                'cpu_percent': 90.0,
                'memory_percent': 85.0,
                'processing_latency': 0.5,
                'frame_rate': 10.0
            },
            'degradation_rules': {
                'camera_failure': 'use_backup_camera',
                'high_cpu': 'reduce_processing_quality',
                'high_memory': 'cleanup_memory'
            }
        }
    
    def start_monitoring(self):
        """Güvenlik izlemeyi başlat"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        # Start watchdog
        self.watchdog.start()
        
        logger.info("Safety monitoring started")
    
    def stop_monitoring(self):
        """Güvenlik izlemeyi durdur"""
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        self.watchdog.stop()
        
        logger.info("Safety monitoring stopped")
    
    def register_health_callback(self, component: SystemComponent, callback: Callable):
        """Sağlık kontrolü callback'i kaydet"""
        self.health_callbacks[component] = callback
        logger.debug(f"Health callback registered for {component.value}")
    
    def register_emergency_callback(self, callback: Callable):
        """Acil durum callback'i kaydet"""
        self.emergency_callbacks.append(callback)
        logger.debug("Emergency callback registered")
    
    def report_component_health(self, health_status: HealthStatus):
        """Bileşen sağlığını rapor et"""
        self.component_health[health_status.component] = health_status
        
        # Check if state change is needed
        self._evaluate_safety_state()
        
        # Reset watchdog for critical components
        if health_status.component in [SystemComponent.PROCESSING, SystemComponent.CAMERA]:
            self.watchdog.reset()
    
    def trigger_emergency_stop(self, reason: str = "Manual trigger"):
        """Acil durumu tetikle"""
        if self.emergency_stop_active:
            return
        
        self.emergency_stop_active = True
        self.current_state = SafetyState.EMERGENCY_STOP
        
        # Log emergency event
        event = SafetyEvent(
            event_type="EMERGENCY_STOP",
            severity=SafetyState.EMERGENCY_STOP,
            component=SystemComponent.PROCESSING,
            description=reason,
            timestamp=time.time(),
            data={"reason": reason}
        )
        self._log_safety_event(event)
        
        # Execute emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                callback(reason)
            except Exception as e:
                logger.error(f"Emergency callback failed: {e}")
        
        logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
    
    def reset_emergency_stop(self):
        """Acil durumu sıfırla"""
        if not self.emergency_stop_active:
            return
        
        self.emergency_stop_active = False
        self.current_state = SafetyState.SAFE
        
        logger.info("Emergency stop reset")
    
    def validate_control_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Kontrol komutunu doğrula ve güvenli hale getir.
        
        Args:
            command: Kontrol komutu (angle, speed, status)
            
        Returns:
            Doğrulanmış ve güvenli komut
        """
        if self.emergency_stop_active:
            return self._generate_emergency_command()
        
        validated_command = command.copy()
        
        # Angle validation
        angle = validated_command.get('angle', 0)
        validated_command['angle'] = max(-45, min(45, angle))  # Limit to ±45 degrees
        
        # Speed validation
        speed = validated_command.get('speed', 0)
        max_speed = self._get_max_safe_speed()
        validated_command['speed'] = max(0, min(max_speed, speed))
        
        # Status validation
        status = validated_command.get('status', 'Düz')
        if self.current_state == SafetyState.CRITICAL:
            validated_command['status'] = 'Dur'
            validated_command['speed'] = 0
        
        return validated_command
    
    def _monitoring_loop(self):
        """Ana izleme döngüsü"""
        interval = self.config.get('health_check_interval', 0.5)
        
        while self.monitoring_active:
            try:
                # System health checks
                self._check_system_performance()
                self._check_component_health()
                self._check_processing_pipeline()
                
                # Evaluate overall safety state
                self._evaluate_safety_state()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Safety monitoring error: {e}")
                time.sleep(interval)
    
    def _check_system_performance(self):
        """Sistem performansını kontrol et"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.performance_thresholds['cpu_percent']:
                self._handle_performance_issue('high_cpu', cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.performance_thresholds['memory_percent']:
                self._handle_performance_issue('high_memory', memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                self._handle_performance_issue('high_disk', disk.percent)
            
        except Exception as e:
            logger.error(f"Performance check failed: {e}")
    
    def _check_component_health(self):
        """Bileşen sağlığını kontrol et"""
        current_time = time.time()
        
        for component, callback in self.health_callbacks.items():
            try:
                health_status = callback()
                if health_status:
                    self.report_component_health(health_status)
            except Exception as e:
                # Component health check failed
                health_status = HealthStatus(
                    component=component,
                    status=SafetyState.CRITICAL,
                    message=f"Health check failed: {e}",
                    timestamp=current_time,
                    metrics={}
                )
                self.report_component_health(health_status)
    
    def _check_processing_pipeline(self):
        """İşleme pipeline'ını kontrol et"""
        # Check for stalled processing
        current_time = time.time()
        
        for component, health in self.component_health.items():
            if current_time - health.timestamp > 5.0:  # 5 second timeout
                logger.warning(f"Component {component.value} appears stalled")
                
                # Create stale health status
                stale_health = HealthStatus(
                    component=component,
                    status=SafetyState.WARNING,
                    message="Component appears stalled",
                    timestamp=current_time,
                    metrics={"last_update": health.timestamp}
                )
                self.component_health[component] = stale_health
    
    def _evaluate_safety_state(self):
        """Genel güvenlik durumunu değerlendir"""
        if self.emergency_stop_active:
            return
        
        # Count component states
        critical_count = 0
        warning_count = 0
        
        for health in self.component_health.values():
            if health.status == SafetyState.CRITICAL:
                critical_count += 1
            elif health.status == SafetyState.WARNING:
                warning_count += 1
        
        # Determine overall state
        previous_state = self.current_state
        
        if critical_count > 0:
            self.current_state = SafetyState.CRITICAL
        elif warning_count > 2:
            self.current_state = SafetyState.WARNING
        else:
            self.current_state = SafetyState.SAFE
        
        # Log state changes
        if self.current_state != previous_state:
            logger.info(f"Safety state changed: {previous_state.value} -> {self.current_state.value}")
            
            # Trigger degradation if needed
            if self.current_state in [SafetyState.WARNING, SafetyState.CRITICAL]:
                self._trigger_graceful_degradation()
    
    def _handle_performance_issue(self, issue_type: str, value: float):
        """Performans sorununu ele al"""
        logger.warning(f"Performance issue detected: {issue_type} = {value}")
        
        # Apply degradation rules
        degradation_action = self.config['degradation_rules'].get(issue_type)
        if degradation_action:
            self._apply_degradation_action(degradation_action, issue_type, value)
    
    def _apply_degradation_action(self, action: str, issue_type: str, value: float):
        """Degradasyon aksiyonunu uygula"""
        logger.info(f"Applying degradation action: {action} for {issue_type}")
        
        if action == 'reduce_processing_quality':
            # Reduce processing quality to save resources
            pass  # Implementation depends on specific modules
        elif action == 'cleanup_memory':
            # Trigger memory cleanup
            import gc
            gc.collect()
        elif action == 'use_backup_camera':
            # Switch to backup camera
            pass  # Implementation depends on camera module
    
    def _trigger_graceful_degradation(self):
        """Zarif degradasyonu tetikle"""
        logger.info("Triggering graceful degradation")
        
        # Reduce processing load
        # Disable non-critical features
        # Switch to safe mode
        pass  # Implementation depends on specific requirements
    
    def _watchdog_timeout_handler(self):
        """Watchdog timeout işleyicisi"""
        logger.critical("Watchdog timeout - system appears frozen!")
        self.trigger_emergency_stop("Watchdog timeout")
    
    def _generate_emergency_command(self) -> Dict[str, Any]:
        """Acil durum komutu oluştur"""
        return {
            'angle': 0,      # Straight
            'speed': 0,      # Stop
            'status': 'Dur'  # Emergency stop
        }
    
    def _get_max_safe_speed(self) -> int:
        """Maksimum güvenli hızı al"""
        if self.current_state == SafetyState.SAFE:
            return 60  # km/h
        elif self.current_state == SafetyState.WARNING:
            return 30  # km/h
        else:
            return 0   # Stop
    
    def _log_safety_event(self, event: SafetyEvent):
        """Güvenlik olayını logla"""
        self.safety_events.append(event)
        
        # Limit event history
        max_events = self.config.get('max_safety_events', 1000)
        if len(self.safety_events) > max_events:
            self.safety_events = self.safety_events[-max_events:]
        
        # Log to file
        logger.info(f"Safety Event: {event.event_type} - {event.description}")
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Güvenlik durumunu al"""
        return {
            'current_state': self.current_state.value,
            'emergency_stop_active': self.emergency_stop_active,
            'component_health': {
                comp.value: {
                    'status': health.status.value,
                    'message': health.message,
                    'timestamp': health.timestamp
                }
                for comp, health in self.component_health.items()
            },
            'recent_events': [
                {
                    'type': event.event_type,
                    'severity': event.severity.value,
                    'component': event.component.value,
                    'description': event.description,
                    'timestamp': event.timestamp
                }
                for event in self.safety_events[-10:]  # Last 10 events
            ]
        }

# Global safety monitor instance
safety_monitor = SafetyMonitor()