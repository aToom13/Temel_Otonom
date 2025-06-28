import threading
import time
import logging

class ThreadSupervisor:
    def __init__(self, thread_targets, check_interval=2):
        self.thread_targets = thread_targets  # {name: target_function}
        self.threads = {}
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)

    def start_all(self):
        for name, target in self.thread_targets.items():
            self.start_thread(name, target)
        threading.Thread(target=self.monitor_threads, daemon=True).start()

    def start_thread(self, name, target):
        if name in self.threads and self.threads[name].is_alive():
            return
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self.threads[name] = t
        self.logger.info(f"Thread '{name}' started.")

    def monitor_threads(self):
        while True:
            for name, thread in list(self.threads.items()):
                if not thread.is_alive():
                    self.logger.warning(f"Thread '{name}' died. Restarting...")
                    self.start_thread(name, self.thread_targets[name])
            time.sleep(self.check_interval)
