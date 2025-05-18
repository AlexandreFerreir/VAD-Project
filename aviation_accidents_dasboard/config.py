"""Configuration file for the aviation dashboard application."""
import os

# Define paths for the application
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Current script directory
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))  # One level up

# Path constants for data files
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'crashes_data')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets')

# Ensure assets directory exists
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)

# Full paths to specific files
CRASH_DATA_PATH = os.path.join(DATA_DIR, 'plane_crash_data.csv')
COORDINATES_PATH = os.path.join(DATA_DIR, 'coordinates_cache.csv')
HEATMAP_HTML_PATH = os.path.join(ASSETS_DIR, 'accident_heatmap.html')
ROUTES_MAP_HTML_PATH = os.path.join(ASSETS_DIR, 'bundled_routes_map.html')

# Image paths
USA_CRASH_DENSITY_IMG = os.path.join(IMAGES_DIR, 'USA_crash_volume_densitiy.png')
TOP_ROUTES_IMG = os.path.join(IMAGES_DIR, 'top20_flight_routes_by_number_of_flights.png')
US_ROUTES_IMG = os.path.join(IMAGES_DIR, 'US_flights_routes_volume_crashes.png')

# Define custom styles
CONTENT_STYLE = {
    "padding": "2rem 2rem",
    "background-color": "#f8f9fa"
}

CARD_STYLE = {
    "box-shadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
    "margin-bottom": "24px",
    "border-radius": "8px",
    "background-color": "white"
}

# Color schemes for consistent visualization
COLOR_SCHEME = {
    "primary": "#3498db",
    "secondary": "#2c3e50",
    "danger": "#e74c3c",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "info": "#1abc9c",
    "dark": "#343a40",
    "light": "#f8f9fa"
}

# Label mappings for consistent naming
PHASE_LABELS = {
    "takeoff": "Takeoff",
    "landing": "Landing", 
    "approach": "Approach",
    "cruise": "Cruise",
    "climb": "Climb", 
    "descent": "Descent"
}

ACCIDENT_LABELS = {
    "fire": "Fire",
    "explosion": "Explosion",
    "collision": "Collision",
    "engine failure": "Engine Failure",
    "mechanical": "Mechanical",
    "overrun": "Runway Overrun",
    "fuel": "Fuel-related",
    "hijack": "Hijacking"
}

# External stylesheets
EXTERNAL_STYLESHEETS = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.1/css/all.min.css"
]