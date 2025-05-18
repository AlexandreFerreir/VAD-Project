import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Now import
from aviation_accidents_dashboard.app import server

# This is a bridge file for Render deployment
# Gunicorn will find the 'server' object imported from the package