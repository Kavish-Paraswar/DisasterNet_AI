"""
Unified Disaster Intelligence Flask Application
Single entrypoint combining Classification (Repo A) and Segmentation (Repo B) pipelines.
"""
import sys
import os

# CRITICAL: Set Keras backend to torch BEFORE any keras import.
# Keras 3 defaults to TensorFlow. Repo A uses torch backend.
os.environ["KERAS_BACKEND"] = "torch"

# Ensure the app directory is in the Python path so that config/, utils/, services/, routes/
# can be imported as top-level packages.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask

def create_app():
    app = Flask(
        __name__,
        template_folder='templates',
        static_folder='static',
    )

    # ── Verify critical dependencies at startup ─────────────────────────────────
    missing = []
    for mod_name in ['cv2', 'torch', 'segmentation_models_pytorch', 'keras', 'PIL', 'flask']:
        try:
            __import__(mod_name)
        except ImportError:
            missing.append(mod_name)
    if missing:
        print(f"[app] FATAL: Missing dependencies — {', '.join(missing)}")
        sys.exit(1)
    print("[app] All critical dependencies verified.")

    # ── Register Blueprints ─────────────────────────────────────────────────────
    from routes.main_routes import main_bp
    from routes.api_routes import api_bp
    from routes.satellite_routes import satellite_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(satellite_bp)

    print("[app] Unified Disaster Intelligence server ready.")
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, threaded=True, port=5000)
