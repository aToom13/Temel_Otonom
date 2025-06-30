import subprocess
import sys
import os
import time

# Build absolute path to orchestator.py relative to this file so it works from any CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORCHESTATOR_PATH = os.path.join(SCRIPT_DIR, 'orchestator.py')
ORCHESTATOR_CMD = [sys.executable, ORCHESTATOR_PATH]

print("[RUNNER] Orchestator başlatılıyor...")
orchestator_proc = subprocess.Popen(ORCHESTATOR_CMD)

try:
    orchestator_proc.wait()
except KeyboardInterrupt:
    print("\n[RUNNER] Kapatılıyor...")
    orchestator_proc.terminate()
    sys.exit(0)