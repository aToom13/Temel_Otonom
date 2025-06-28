import pytest
import time
from unittest.mock import Mock, patch
from core.safety.safety_monitor import (
    SafetyMonitor, SafetyState, SystemComponent, 
    HealthStatus, SafetyEvent, WatchdogTimer
)

@pytest.fixture
def safety_monitor():
    """Create safety monitor for testing"""
    config = {
        'watchdog_timeout': 0.5,  # Short timeout for testing
        'health_check_interval': 0.1,
        'max_safety_events': 100,
        'performance_thresholds': {
            'cpu_percent': 80.0,
            'memory_percent': 75.0,
            'processing_latency': 0.3,
            'frame_rate': 15.0
        }
    }
    return SafetyMonitor(config)

@pytest.fixture
def mock_health_callback():
    """Mock health callback function"""
    def callback():
        return HealthStatus(
            component=SystemComponent.CAMERA,
            status=SafetyState.SAFE,
            message="Test component OK",
            timestamp=time.time(),
            metrics={"test_metric": 100}
        )
    return callback

def test_safety_monitor_initialization(safety_monitor):
    """Test safety monitor initialization"""
    assert safety_monitor is not None
    assert safety_monitor.current_state == SafetyState.SAFE
    assert not safety_monitor.emergency_stop_active
    assert not safety_monitor.monitoring_active

def test_watchdog_timer():
    """Test watchdog timer functionality"""
    callback_called = False
    
    def timeout_callback():
        nonlocal callback_called
        callback_called = True
    
    watchdog = WatchdogTimer(timeout=0.1, callback=timeout_callback)
    watchdog.start()
    
    # Wait for timeout
    time.sleep(0.2)
    
    assert callback_called
    watchdog.stop()

def test_watchdog_reset():
    """Test watchdog timer reset"""
    callback_called = False
    
    def timeout_callback():
        nonlocal callback_called
        callback_called = True
    
    watchdog = WatchdogTimer(timeout=0.2, callback=timeout_callback)
    watchdog.start()
    
    # Reset before timeout
    time.sleep(0.1)
    watchdog.reset()
    time.sleep(0.15)  # Should not timeout now
    
    assert not callback_called
    watchdog.stop()

def test_health_callback_registration(safety_monitor, mock_health_callback):
    """Test health callback registration"""
    safety_monitor.register_health_callback(SystemComponent.CAMERA, mock_health_callback)
    
    assert SystemComponent.CAMERA in safety_monitor.health_callbacks
    assert safety_monitor.health_callbacks[SystemComponent.CAMERA] == mock_health_callback

def test_emergency_callback_registration(safety_monitor):
    """Test emergency callback registration"""
    callback_called = False
    
    def emergency_callback(reason):
        nonlocal callback_called
        callback_called = True
    
    safety_monitor.register_emergency_callback(emergency_callback)
    safety_monitor.trigger_emergency_stop("Test emergency")
    
    assert callback_called
    assert safety_monitor.emergency_stop_active

def test_component_health_reporting(safety_monitor):
    """Test component health reporting"""
    health_status = HealthStatus(
        component=SystemComponent.CAMERA,
        status=SafetyState.SAFE,
        message="Camera OK",
        timestamp=time.time(),
        metrics={"fps": 30}
    )
    
    safety_monitor.report_component_health(health_status)
    
    assert SystemComponent.CAMERA in safety_monitor.component_health
    assert safety_monitor.component_health[SystemComponent.CAMERA] == health_status

def test_safety_state_evaluation(safety_monitor):
    """Test safety state evaluation"""
    # Report critical component
    critical_health = HealthStatus(
        component=SystemComponent.CAMERA,
        status=SafetyState.CRITICAL,
        message="Camera failed",
        timestamp=time.time(),
        metrics={}
    )
    
    safety_monitor.report_component_health(critical_health)
    
    assert safety_monitor.current_state == SafetyState.CRITICAL

def test_emergency_stop_trigger(safety_monitor):
    """Test emergency stop trigger"""
    safety_monitor.trigger_emergency_stop("Manual test")
    
    assert safety_monitor.emergency_stop_active
    assert safety_monitor.current_state == SafetyState.EMERGENCY_STOP

def test_emergency_stop_reset(safety_monitor):
    """Test emergency stop reset"""
    safety_monitor.trigger_emergency_stop("Test")
    assert safety_monitor.emergency_stop_active
    
    safety_monitor.reset_emergency_stop()
    assert not safety_monitor.emergency_stop_active
    assert safety_monitor.current_state == SafetyState.SAFE

def test_control_command_validation(safety_monitor):
    """Test control command validation"""
    # Normal command
    command = {'angle': 10, 'speed': 30, 'status': 'Düz'}
    validated = safety_monitor.validate_control_command(command)
    
    assert validated['angle'] == 10
    assert validated['speed'] == 30
    assert validated['status'] == 'Düz'
    
    # Command with excessive angle
    command = {'angle': 60, 'speed': 30, 'status': 'Düz'}
    validated = safety_monitor.validate_control_command(command)
    
    assert validated['angle'] == 45  # Should be limited to ±45 degrees

def test_emergency_command_generation(safety_monitor):
    """Test emergency command generation"""
    safety_monitor.trigger_emergency_stop("Test")
    
    command = {'angle': 10, 'speed': 30, 'status': 'Düz'}
    validated = safety_monitor.validate_control_command(command)
    
    assert validated['angle'] == 0
    assert validated['speed'] == 0
    assert validated['status'] == 'Dur'

def test_monitoring_start_stop(safety_monitor, mock_health_callback):
    """Test monitoring start and stop"""
    safety_monitor.register_health_callback(SystemComponent.CAMERA, mock_health_callback)
    
    safety_monitor.start_monitoring()
    assert safety_monitor.monitoring_active
    
    # Wait for some monitoring cycles
    time.sleep(0.3)
    
    safety_monitor.stop_monitoring()
    assert not safety_monitor.monitoring_active

def test_performance_monitoring(safety_monitor):
    """Test system performance monitoring"""
    with patch('psutil.cpu_percent', return_value=90.0):
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 85.0
            
            safety_monitor._check_system_performance()
            # Should handle high CPU and memory usage

def test_component_health_checking(safety_monitor, mock_health_callback):
    """Test component health checking"""
    safety_monitor.register_health_callback(SystemComponent.CAMERA, mock_health_callback)
    
    safety_monitor._check_component_health()
    
    # Should have updated component health
    assert SystemComponent.CAMERA in safety_monitor.component_health

def test_stalled_component_detection(safety_monitor):
    """Test stalled component detection"""
    # Add old health status
    old_health = HealthStatus(
        component=SystemComponent.CAMERA,
        status=SafetyState.SAFE,
        message="Camera OK",
        timestamp=time.time() - 10.0,  # 10 seconds ago
        metrics={}
    )
    
    safety_monitor.component_health[SystemComponent.CAMERA] = old_health
    
    safety_monitor._check_processing_pipeline()
    
    # Should detect stalled component
    updated_health = safety_monitor.component_health[SystemComponent.CAMERA]
    assert updated_health.status == SafetyState.WARNING

def test_safety_event_logging(safety_monitor):
    """Test safety event logging"""
    event = SafetyEvent(
        event_type="TEST_EVENT",
        severity=SafetyState.WARNING,
        component=SystemComponent.CAMERA,
        description="Test event",
        timestamp=time.time(),
        data={"test": "data"}
    )
    
    safety_monitor._log_safety_event(event)
    
    assert len(safety_monitor.safety_events) == 1
    assert safety_monitor.safety_events[0] == event

def test_safety_status_reporting(safety_monitor):
    """Test safety status reporting"""
    # Add some component health
    health = HealthStatus(
        component=SystemComponent.CAMERA,
        status=SafetyState.SAFE,
        message="Camera OK",
        timestamp=time.time(),
        metrics={"fps": 30}
    )
    
    safety_monitor.report_component_health(health)
    
    status = safety_monitor.get_safety_status()
    
    assert isinstance(status, dict)
    assert 'current_state' in status
    assert 'emergency_stop_active' in status
    assert 'component_health' in status
    assert 'recent_events' in status

def test_max_safe_speed_calculation(safety_monitor):
    """Test maximum safe speed calculation"""
    # Safe state
    safety_monitor.current_state = SafetyState.SAFE
    max_speed = safety_monitor._get_max_safe_speed()
    assert max_speed == 60
    
    # Warning state
    safety_monitor.current_state = SafetyState.WARNING
    max_speed = safety_monitor._get_max_safe_speed()
    assert max_speed == 30
    
    # Critical state
    safety_monitor.current_state = SafetyState.CRITICAL
    max_speed = safety_monitor._get_max_safe_speed()
    assert max_speed == 0

def test_graceful_degradation(safety_monitor):
    """Test graceful degradation trigger"""
    # Set warning state
    safety_monitor.current_state = SafetyState.WARNING
    
    # Should trigger graceful degradation
    safety_monitor._trigger_graceful_degradation()
    # Test passes if no exceptions are raised

def test_performance_issue_handling(safety_monitor):
    """Test performance issue handling"""
    # Test high CPU
    safety_monitor._handle_performance_issue('high_cpu', 95.0)
    
    # Test high memory
    safety_monitor._handle_performance_issue('high_memory', 90.0)
    
    # Should apply degradation actions without errors

def test_multiple_component_failures(safety_monitor):
    """Test handling multiple component failures"""
    # Report multiple critical components
    components = [SystemComponent.CAMERA, SystemComponent.ARDUINO, SystemComponent.PROCESSING]
    
    for component in components:
        health = HealthStatus(
            component=component,
            status=SafetyState.CRITICAL,
            message=f"{component.value} failed",
            timestamp=time.time(),
            metrics={}
        )
        safety_monitor.report_component_health(health)
    
    # Should be in critical state
    assert safety_monitor.current_state == SafetyState.CRITICAL

def test_safety_event_history_limit(safety_monitor):
    """Test safety event history limit"""
    # Add more events than the limit
    for i in range(150):  # More than max_safety_events (100)
        event = SafetyEvent(
            event_type=f"TEST_EVENT_{i}",
            severity=SafetyState.WARNING,
            component=SystemComponent.CAMERA,
            description=f"Test event {i}",
            timestamp=time.time(),
            data={}
        )
        safety_monitor._log_safety_event(event)
    
    # Should be limited to max_safety_events
    assert len(safety_monitor.safety_events) == safety_monitor.config['max_safety_events']

def test_watchdog_timeout_handling(safety_monitor):
    """Test watchdog timeout handling"""
    emergency_triggered = False
    
    def emergency_callback(reason):
        nonlocal emergency_triggered
        emergency_triggered = True
    
    safety_monitor.register_emergency_callback(emergency_callback)
    
    # Trigger watchdog timeout
    safety_monitor._watchdog_timeout_handler()
    
    assert emergency_triggered
    assert safety_monitor.emergency_stop_active

# Integration test
def test_full_safety_monitoring_cycle(safety_monitor, mock_health_callback):
    """Test complete safety monitoring cycle"""
    # Register callbacks
    safety_monitor.register_health_callback(SystemComponent.CAMERA, mock_health_callback)
    
    emergency_called = False
    def emergency_callback(reason):
        nonlocal emergency_called
        emergency_called = True
    
    safety_monitor.register_emergency_callback(emergency_callback)
    
    # Start monitoring
    safety_monitor.start_monitoring()
    
    # Let it run for a short time
    time.sleep(0.5)
    
    # Check that monitoring is working
    assert safety_monitor.monitoring_active
    assert SystemComponent.CAMERA in safety_monitor.component_health
    
    # Trigger emergency
    safety_monitor.trigger_emergency_stop("Integration test")
    assert emergency_called
    
    # Stop monitoring
    safety_monitor.stop_monitoring()
    assert not safety_monitor.monitoring_active