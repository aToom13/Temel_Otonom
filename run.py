import subprocess
import sys
import os
import time

ORCHESTATOR_CMD = [sys.executable, 'orchestator.py']

print("[RUNNER] Orchestator başlatılıyor...")
orchestator_proc = subprocess.Popen(ORCHESTATOR_CMD)

try:
    orchestator_proc.wait()
except KeyboardInterrupt:
    print("\n[RUNNER] Kapatılıyor...")
    orchestator_proc.terminate()
    sys.exit(0)
