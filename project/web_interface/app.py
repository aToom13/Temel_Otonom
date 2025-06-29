from flask import Flask, render_template
import os
import yaml
import logging
from flask_socketio import SocketIO
from flask_cors import CORS

# Logging ayarları
config_file = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
if os.path.exists(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    log_conf = config.get('logging', {})
    log_level = getattr(logging, log_conf.get('level', 'INFO').upper(), logging.INFO)
    log_file = log_conf.get('file', 'logs/dursun.log')
else:
    log_level = logging.INFO
    log_file = 'logs/dursun.log'

# Create logs directory if it doesn't exist
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s [%(levelname)s] %(threadName)s: %(message)s',
    handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import main processing module
try:
    import main as processing_main
    logger.info("Main processing module imported successfully")
except ImportError as e:
    logger.error(f"Failed to import main processing module: {e}")
    processing_main = None

from web_interface.blueprints.api import api
from web_interface.blueprints.video import video
from web_interface.blueprints.ws import ws, socketio

def create_app():
    """Create Flask application"""
    app = Flask(__name__)
    
    # Enable CORS for all routes
    CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])
    
    # Register blueprints
    app.register_blueprint(api)
    app.register_blueprint(video)
    app.register_blueprint(ws)
    
    # Initialize SocketIO
    socketio.init_app(app, cors_allowed_origins=["http://localhost:3000", "http://127.0.0.1:3000"])
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.errorhandler(404)
    def not_found(error):
        return {"error": "Page not found"}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return {"error": "Internal server error"}, 500
    
    return app

app = create_app()

if __name__ == '__main__':
    try:
        # Start processing threads if main module is available
        if processing_main:
            processing_main.start_processing_threads()
            logger.info("Processing threads started")
        else:
            logger.warning("Processing threads not started - main module not available")
        
        logger.info("Flask app starting...")
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
        
    except Exception as e:
        logger.error(f"Failed to start Flask app: {e}")
        raise