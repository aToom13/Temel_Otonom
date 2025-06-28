import subprocess
import sys
import os
import time

# Backend ve frontend yolları
BACKEND_CMD = [sys.executable, 'web_interface/backend/app.py']
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), 'web_interface', 'frontend')
FRONTEND_CMD = ['npm', 'start']

DEVNULL = subprocess.DEVNULL

print("[ORCHESTATOR] Backend (app.py) başlatılıyor...")
backend_proc = subprocess.Popen(BACKEND_CMD)  # stdout/stderr terminalde

print(f"[ORCHESTATOR] Frontend (React) başlatılıyor: {FRONTEND_DIR}")
frontend_proc = subprocess.Popen(FRONTEND_CMD, cwd=FRONTEND_DIR, shell=True, stdout=DEVNULL, stderr=DEVNULL)

time.sleep(3)
print("\n[INFO] Proje başlatıldı!")
print("Backend (API): http://localhost:5000")
print("Frontend (React SPA): http://localhost:3000")
print("\nÇıkmak için Ctrl+C (veya iki kez durdurun).\n")

try:
    backend_proc.wait()
    frontend_proc.wait()
except KeyboardInterrupt:
    print("\n[ORCHESTATOR] Kapatılıyor...")
    backend_proc.terminate()
    frontend_proc.terminate()
    sys.exit(0)
