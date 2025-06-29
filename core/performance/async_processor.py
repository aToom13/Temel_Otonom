"""
Asenkron işleme ve thread pool optimizasyonu.
PRD'de belirtilen performans iyileştirmeleri için.
"""
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from queue import Queue, Empty
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProcessingTask:
    """İşleme görevi"""
    id: str
    data: Any
    processor: Callable
    priority: int = 0
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class AsyncProcessor:
    """
    Gelişmiş asenkron işleme sınıfı.
    - Thread pool optimization
    - Priority queue processing
    - Load balancing
    - Performance monitoring
    """
    
    def __init__(self, max_workers: int = 4, queue_size: int = 100):
        self.max_workers = max_workers
        self.queue_size = queue_size
        
        # Thread pool
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Task queues
        self.high_priority_queue = Queue(maxsize=queue_size)
        self.normal_priority_queue = Queue(maxsize=queue_size)
        self.low_priority_queue = Queue(maxsize=queue_size)
        
        # Processing statistics
        self.stats = {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'average_processing_time': 0.0,
            'queue_sizes': {'high': 0, 'normal': 0, 'low': 0}
        }
        
        # Control flags
        self.running = False
        self.worker_threads = []
        
    def start(self):
        """İşleme sistemini başlat"""
        if self.running:
            return
        
        self.running = True
        
        # Worker thread'leri başlat
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"AsyncProcessor-Worker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        logger.info(f"AsyncProcessor started with {self.max_workers} workers")
    
    def stop(self, timeout: float = 5.0):
        """İşleme sistemini durdur"""
        self.running = False
        
        # Worker thread'lerin bitmesini bekle
        for worker in self.worker_threads:
            worker.join(timeout=timeout)
        
        self.executor.shutdown(wait=True)
        logger.info("AsyncProcessor stopped")
    
    def submit_task(self, task: ProcessingTask) -> bool:
        """Görevi kuyruğa ekle"""
        try:
            if task.priority >= 2:  # High priority
                self.high_priority_queue.put_nowait(task)
                self.stats['queue_sizes']['high'] += 1
            elif task.priority == 1:  # Normal priority
                self.normal_priority_queue.put_nowait(task)
                self.stats['queue_sizes']['normal'] += 1
            else:  # Low priority
                self.low_priority_queue.put_nowait(task)
                self.stats['queue_sizes']['low'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit task {task.id}: {e}")
            return False
    
    def _get_next_task(self, timeout: float = 1.0) -> Optional[ProcessingTask]:
        """Öncelik sırasına göre sonraki görevi al"""
        # High priority first
        try:
            task = self.high_priority_queue.get(timeout=0.1)
            self.stats['queue_sizes']['high'] -= 1
            return task
        except Empty:
            pass
        
        # Normal priority second
        try:
            task = self.normal_priority_queue.get(timeout=0.1)
            self.stats['queue_sizes']['normal'] -= 1
            return task
        except Empty:
            pass
        
        # Low priority last
        try:
            task = self.low_priority_queue.get(timeout=timeout)
            self.stats['queue_sizes']['low'] -= 1
            return task
        except Empty:
            return None
    
    def _worker_loop(self):
        """Worker thread ana döngüsü"""
        while self.running:
            try:
                task = self._get_next_task()
                if task is None:
                    continue
                
                # Process task
                start_time = time.time()
                try:
                    result = task.processor(task.data)
                    processing_time = time.time() - start_time
                    
                    # Update statistics
                    self.stats['tasks_processed'] += 1
                    self._update_average_time(processing_time)
                    
                    logger.debug(f"Task {task.id} completed in {processing_time:.3f}s")
                    
                except Exception as e:
                    self.stats['tasks_failed'] += 1
                    logger.error(f"Task {task.id} failed: {e}")
                
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(0.1)
    
    def _update_average_time(self, processing_time: float):
        """Ortalama işleme süresini güncelle"""
        total_tasks = self.stats['tasks_processed']
        if total_tasks == 1:
            self.stats['average_processing_time'] = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats['average_processing_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.stats['average_processing_time']
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """İşleme istatistiklerini al"""
        return self.stats.copy()
    
    async def process_async(self, data: Any, processor: Callable, priority: int = 1) -> Any:
        """Asenkron işleme"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, processor, data)

# Global async processor instance
async_processor = AsyncProcessor()