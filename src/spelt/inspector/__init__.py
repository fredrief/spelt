from flask import Flask
import secrets
from pathlib import Path

def create_app(test_config=None):
    app = Flask(__name__)

    # Default configuration
    app.config.from_mapping(
        SECRET_KEY='dev',
        DEFAULT_ROOT_DIR=str(Path.home()),  # Convert Path to string for session storage
    )

    # Set a random secret key
    app.secret_key = secrets.token_hex(16)

    from .routes import bp
    app.register_blueprint(bp)

    return app
