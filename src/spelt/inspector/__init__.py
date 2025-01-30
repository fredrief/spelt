from flask import Flask
import secrets

def create_app():
    app = Flask(__name__)

    # Set a random secret key
    app.secret_key = secrets.token_hex(16)

    from .routes import bp
    app.register_blueprint(bp)

    return app
