"""
Main Routes — UI only.
Handles exclusively the '/' route to render the Repo A template.
"""
from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)


@main_bp.route('/', methods=['GET'])
def index():
    """Serve the Disaster Intelligence UI (Repo A template)."""
    return render_template('index.html')
