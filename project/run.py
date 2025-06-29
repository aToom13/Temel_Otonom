import subprocess
import sys
import os
import time

ORCHESTATOR_CMD = [sys.executable, os.path.join(os.path.dirname(__file__), 'orchestator.py')]

print("[RUNNER] Orchestator başlatılıyor...")
env = os.environ.copy()
# npm'in bulunduğu dizini ve npm.cmd dosyasını PATH'e ekle
npm_dir = r"C:\Program Files\nodejs"
npm_cmd = os.path.join(npm_dir, "npm.cmd")
if npm_dir not in env["PATH"]:
    env["PATH"] = env["PATH"] + os.pathsep + npm_dir
if npm_cmd not in env["PATH"]:
    env["PATH"] = env["PATH"] + os.pathsep + npm_cmd
orchestator_proc = subprocess.Popen(ORCHESTATOR_CMD, env=env)

try:
    orchestator_proc.wait()
except KeyboardInterrupt:
    print("\n[RUNNER] Kapatılıyor...")
    orchestator_proc.terminate()
    sys.exit(0)