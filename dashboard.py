import sys
import os
# Add project root to path - this fixes ALL import issues
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from aviation_accidents_dashboard.app import server

# This is a bridge file for Render deployment
# Gunicorn will find the 'server' object imported from the package