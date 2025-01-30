from flask import Flask
from pathlib import Path

def create_app(root_dir: str | Path = None):
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Set a secret key for session support
    app.secret_key = 'dev'  # TODO: Use a proper secret key in production

    # Use provided directory or default to current
    if root_dir is None:
        root_dir = '.'

    # Store root directory in app config
    app.config['ROOT_DIR'] = Path(root_dir).resolve()

    # Register routes
    from . import routes
    app.register_blueprint(routes.bp)

    return app
