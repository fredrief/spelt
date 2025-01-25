from flask import Flask
from pathlib import Path

def create_app(root_dir: str | Path = None):
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Use provided directory or default to current
    if root_dir is None:
        root_dir = '.'

    # Store root directory in app config
    app.config['ROOT_DIR'] = Path(root_dir).resolve()

    # Register routes
    from . import routes
    app.register_blueprint(routes.bp)

    return app
