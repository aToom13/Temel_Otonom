from flask import Flask, render_template
import os
import yaml
import logging
from flask_socketio import SocketIO

# Logging ayarları
config_file = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
if os.path.exists(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    log_conf = config.get('logging', {})
    log_level = getattr(logging, log_conf.get('level', 'INFO').upper(), logging.INFO)
    log_file = log_conf.get('file', 'dursun.log')
else:
    log_level = logging.INFO
    log_file = 'dursun.log'
logging.basicConfig(
    level=log_level,
    format='%(asctime)s [%(levelname)s] %(threadName)s: %(message)s',
    handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import main as processing_main

from web_interface.blueprints.api import api
from web_interface.blueprints.video import video
from web_interface.blueprints.ws import ws, socketio

app = Flask(__name__)
app.register_blueprint(api)
app.register_blueprint(video)
app.register_blueprint(ws)
socketio.init_app(app)

@app.route('/')
def index():
    return render_template('index.html')

def create_app():
    app = Flask(__name__)
    app.register_blueprint(api)
    app.register_blueprint(video)
    app.register_blueprint(ws)
    socketio.init_app(app)
    return app

if __name__ == '__main__':
    processing_main.start_processing_threads()
    logger.info("Flask app starting...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
