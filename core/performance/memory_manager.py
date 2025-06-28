"""
Gelişmiş bellek yönetimi ve optimizasyon modülü.
PRD'de belirtilen performans iyileştirmeleri için.
"""
import gc
import psutil
import threading
import time
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Bellek kullanım istatistikleri"""
    total_memory: float
    available_memory: float
    used_memory: float
    memory_percent: float
    gpu_memory: Optional[float] = None

class MemoryManager:
    """
    Gelişmiş bellek yönetimi sınıfı.
    - Memory leak detection
    - Automatic garbage collection
    - GPU memory management
    - Memory usage monitoring
    """
    
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.memory_history = deque(maxsize=100)
        self.monitoring_active = False
        self.monitor_thread = None
        self._lock = threading.Lock()
        
        # GPU memory management
        self.gpu_available = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> bool:
        """GPU kullanılabilirliğini kontrol et"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_memory_stats(self) -> MemoryStats:
        """Mevcut bellek istatistiklerini al"""
        memory = psutil.virtual_memory()
        stats = MemoryStats(
            total_memory=memory.total / (1024**3),  # GB
            available_memory=memory.available / (1024**3),
            used_memory=memory.used / (1024**3),
            memory_percent=memory.percent
        )
        
        if self.gpu_available:
            try:
                import torch
                stats.gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            except Exception as e:
                logger.warning(f"GPU memory check failed: {e}")
        
        return stats
    
    def cleanup_memory(self, force_gpu_cleanup: bool = False):
        """Bellek temizleme işlemi"""
        with self._lock:
            # Python garbage collection
            collected = gc.collect()
            logger.debug(f"Garbage collection freed {collected} objects")
            
            # GPU memory cleanup
            if self.gpu_available and force_gpu_cleanup:
                try:
                    import torch
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.debug("GPU memory cache cleared")
                except Exception as e:
                    logger.warning(f"GPU cleanup failed: {e}")
    
    def check_memory_pressure(self) -> bool:
        """Bellek baskısını kontrol et"""
        stats = self.get_memory_stats()
        return stats.memory_percent > self.max_memory_percent
    
    def start_monitoring(self, interval: float = 5.0):
        """Bellek izlemeyi başlat"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Bellek izlemeyi durdur"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Bellek izleme döngüsü"""
        while self.monitoring_active:
            try:
                stats = self.get_memory_stats()
                self.memory_history.append(stats)
                
                # Memory pressure check
                if self.check_memory_pressure():
                    logger.warning(f"High memory usage: {stats.memory_percent:.1f}%")
                    self.cleanup_memory(force_gpu_cleanup=True)
                
                # Memory leak detection
                if len(self.memory_history) >= 10:
                    self._detect_memory_leak()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(interval)
    
    def _detect_memory_leak(self):
        """Bellek sızıntısı tespiti"""
        if len(self.memory_history) < 10:
            return
        
        recent_usage = [stats.memory_percent for stats in list(self.memory_history)[-10:]]
        
        # Sürekli artış kontrolü
        increasing_trend = all(
            recent_usage[i] <= recent_usage[i+1] 
            for i in range(len(recent_usage)-1)
        )
        
        if increasing_trend and recent_usage[-1] - recent_usage[0] > 10:
            logger.warning("Potential memory leak detected - continuous memory increase")
            self.cleanup_memory(force_gpu_cleanup=True)

# Global memory manager instance
memory_manager = MemoryManager()