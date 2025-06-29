"""
Basitleştirilmiş bellek yönetimi modülü.
"""
import gc
import psutil
import threading
import time
from typing import Dict, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Bellek kullanım istatistikleri"""
    total_memory: float
    available_memory: float
    used_memory: float
    memory_percent: float

class MemoryManager:
    """Basitleştirilmiş bellek yönetimi sınıfı"""
    
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.monitoring_active = False
        self.monitor_thread = None
        self._lock = threading.Lock()
    
    def get_memory_stats(self) -> MemoryStats:
        """Mevcut bellek istatistiklerini al"""
        memory = psutil.virtual_memory()
        return MemoryStats(
            total_memory=memory.total / (1024**3),  # GB
            available_memory=memory.available / (1024**3),
            used_memory=memory.used / (1024**3),
            memory_percent=memory.percent
        )
    
    def cleanup_memory(self):
        """Bellek temizleme işlemi"""
        with self._lock:
            collected = gc.collect()
            logger.debug(f"Garbage collection freed {collected} objects")
    
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
                if self.check_memory_pressure():
                    logger.warning("High memory usage detected")
                    self.cleanup_memory()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(interval)

# Global memory manager instance
memory_manager = MemoryManager()