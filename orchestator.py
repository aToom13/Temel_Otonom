import subprocess
import sys
import os
import time
import signal
import logging
import socket

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('orchestrator')

# Process commands
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_PATH = os.path.join(SCRIPT_DIR, 'web_interface', 'app.py')
BACKEND_CMD = [sys.executable, BACKEND_PATH]
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), 'web_interface', 'frontend')
FRONTEND_CMD = ['npm', 'start']

# Global process references
backend_proc = None
frontend_proc = None

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Shutdown signal received, cleaning up...")
    cleanup_processes()
    sys.exit(0)

def cleanup_processes():
    """Clean up all spawned processes"""
    global backend_proc, frontend_proc
    
    if backend_proc:
        logger.info("Terminating backend process...")
        try:
            backend_proc.terminate()
            backend_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Backend process didn't terminate gracefully, killing...")
            backend_proc.kill()
        except Exception as e:
            logger.error(f"Error terminating backend: {e}")
    
    if frontend_proc:
        logger.info("Terminating frontend process...")
        try:
            frontend_proc.terminate()
            frontend_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Frontend process didn't terminate gracefully, killing...")
            frontend_proc.kill()
        except Exception as e:
            logger.error(f"Error terminating frontend: {e}")

def check_prerequisites():
    """Check if all prerequisites are met"""
    # Check if Node.js is available
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Node.js version: {result.stdout.strip()}")
        else:
            logger.error("Node.js not found")
            return False
    except FileNotFoundError:
        logger.error("Node.js not found in PATH")
        return False
    
    # Check if npm is available
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"npm version: {result.stdout.strip()}")
        else:
            logger.error("npm not found")
            return False
    except FileNotFoundError:
        logger.error("npm not found in PATH")
        return False
    
    # Check if frontend directory exists
    if not os.path.exists(FRONTEND_DIR):
        logger.error(f"Frontend directory not found: {FRONTEND_DIR}")
        return False
    
    # Check if package.json exists
    package_json = os.path.join(FRONTEND_DIR, 'package.json')
    if not os.path.exists(package_json):
        logger.error(f"package.json not found: {package_json}")
        return False
    
    return True

def install_frontend_dependencies():
    """Install frontend dependencies if needed"""
    node_modules = os.path.join(FRONTEND_DIR, 'node_modules')
    
    if not os.path.exists(node_modules):
        logger.info("Installing frontend dependencies...")
        try:
            result = subprocess.run(
                ['npm', 'install'], 
                cwd=FRONTEND_DIR, 
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.info("Frontend dependencies installed successfully")
                return True
            else:
                logger.error(f"Failed to install dependencies: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Frontend dependency installation timed out")
            return False
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")
            return False
    else:
        logger.info("Frontend dependencies already installed")
        return True

def is_port_in_use(port):
    """Return True if port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def start_backend():
    """Start the backend process"""
    global backend_proc
    
    # Ensure port 5000 is free
    backend_port = 5000
    if is_port_in_use(backend_port):
        logger.warning(f"Port {backend_port} already in use. Attempting to free it...")
        try:
            # Try to kill process(es) using the port (linux-specific fuser)
            subprocess.run(['fuser', '-k', f'{backend_port}/tcp'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            time.sleep(1)  # Give the OS a moment to release the port
        except Exception as e:
            logger.error(f"Failed to free port {backend_port}: {e}")
            # Continue anyway – backend will fail if still occupied

    logger.info("Starting backend (Flask app)...")
    try:
        backend_proc = subprocess.Popen(
            BACKEND_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Wait a moment to check if it started successfully
        time.sleep(2)
        
        if backend_proc.poll() is None:
            logger.info("Backend started successfully")
            return True
        else:
            logger.error("Backend failed to start")
            return False
            
    except Exception as e:
        logger.error(f"Failed to start backend: {e}")
        return False

def start_frontend():
    """Start the frontend process"""
    global frontend_proc
    
    # Ensure port 3000 is free
    frontend_port = 3000
    if is_port_in_use(frontend_port):
        logger.warning(f"Port {frontend_port} already in use. Attempting to free it...")
        try:
            subprocess.run(['fuser', '-k', f'{frontend_port}/tcp'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            time.sleep(1)
        except Exception as e:
            logger.error(f"Failed to free port {frontend_port}: {e}")

    logger.info("Starting frontend (React app)...")
    try:
        # Use shell=True on Windows for npm commands
        use_shell = os.name == 'nt'
        
        frontend_proc = subprocess.Popen(
            FRONTEND_CMD,
            cwd=FRONTEND_DIR,
            shell=use_shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        
        # Wait a moment to check if it started successfully
        time.sleep(3)
        
        if frontend_proc.poll() is None:
            logger.info("Frontend started successfully")
            return True
        else:
            logger.error("Frontend failed to start")
            return False
            
    except Exception as e:
        logger.error(f"Failed to start frontend: {e}")
        return False

def monitor_processes():
    """Monitor running processes and restart if needed"""
    global backend_proc, frontend_proc
    
    while True:
        try:
            # Check backend
            if backend_proc and backend_proc.poll() is not None:
                logger.warning("Backend process died, restarting...")
                if not start_backend():
                    logger.error("Failed to restart backend")
                    break
            
            # Check frontend
            if frontend_proc and frontend_proc.poll() is not None:
                logger.warning("Frontend process died, restarting...")
                if not start_frontend():
                    logger.error("Failed to restart frontend")
                    break
            
            time.sleep(5)  # Check every 5 seconds
            
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted")
            break
        except Exception as e:
            logger.error(f"Error in process monitoring: {e}")
            time.sleep(5)

def main():
    """Main orchestrator function"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("=== Dursun System Orchestrator Starting ===")
    
    try:
        # Check prerequisites
        if not check_prerequisites():
            logger.error("Prerequisites not met, exiting...")
            return 1
        
        # Install frontend dependencies
        if not install_frontend_dependencies():
            logger.error("Failed to install frontend dependencies, exiting...")
            return 1
        
        # Start backend
        if not start_backend():
            logger.error("Failed to start backend, exiting...")
            return 1
        
        # Wait for backend to be ready
        time.sleep(3)
        
        # Start frontend
        if not start_frontend():
            logger.error("Failed to start frontend, exiting...")
            cleanup_processes()
            return 1
        
        # Wait for frontend to be ready
        time.sleep(5)
        
        logger.info("\n" + "="*50)
        logger.info("🚀 Dursun System Started Successfully!")
        logger.info("Backend (API): http://localhost:5000")
        logger.info("Frontend (React): http://localhost:3000")
        logger.info("Press Ctrl+C to stop all services")
        logger.info("="*50 + "\n")
        
        # Monitor processes
        monitor_processes()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
        return 1
    finally:
        cleanup_processes()
        logger.info("Orchestrator shutdown complete")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())