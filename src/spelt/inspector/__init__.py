from flask import Flask
import secrets
from pathlib import Path
import os

def create_app(test_config=None):
    # Create Flask app with explicit template folder path
    app = Flask(__name__,
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                static_folder=os.path.join(os.path.dirname(__file__), 'static'))

    # Default configuration
    app.config.from_mapping(
        SECRET_KEY='dev',
        DEFAULT_ROOT_DIR=str(Path.cwd()),  # Convert Path to string for session storage
    )

    # Set a random secret key
    app.secret_key = secrets.token_hex(16)

    from .routes import bp
    app.register_blueprint(bp)

    return app
