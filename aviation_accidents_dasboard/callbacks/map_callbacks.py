"""
Callbacks for map visualizations
"""
from dash import Input, Output
import os
from config import HEATMAP_HTML_PATH, ROUTES_MAP_HTML_PATH

def register_map_callbacks(app):
    """
    Register callbacks related to map visualizations
    
    Args:
        app: The Dash app
    """
    @app.callback(
        [Output("routes-map-iframe", "srcDoc"), Output("heatmap-iframe", "srcDoc")],
        Input("tabs", "active_tab"),
        prevent_initial_call=False  # Allow call on initialization
    )
    def load_maps(active_tab):
        """Loads both maps when tab-2 is selected"""
        
        if active_tab != "tab-2":
            return "", ""
        
        routes_map_html = ""
        heatmap_html = ""
        # Load routes map
        try:
            if os.path.exists(ROUTES_MAP_HTML_PATH):
                with open(ROUTES_MAP_HTML_PATH, "r", encoding="utf-8") as f:
                    routes_map_html = f.read()
            else:
                routes_map_html = "<div style='color:red; text-align:center;'><h3>Error: Routes map not found</h3></div>"
        except Exception as e:
            print(f"Error loading routes map: {str(e)}")
            routes_map_html = f"<div style='color:red; text-align:center;'><h3>Error loading routes map</h3><p>{str(e)}</p></div>"
        
        # Load heat map
        try:
            if os.path.exists(HEATMAP_HTML_PATH):
                with open(HEATMAP_HTML_PATH, "r", encoding="utf-8") as f:
                    heatmap_html = f.read()
            else:
                heatmap_html = "<div style='color:red; text-align:center;'><h3>Error: Heatmap not found</h3></div>"
        except Exception as e:
            print(f"Error loading heatmap: {str(e)}")
            heatmap_html = f"<div style='color:red; text-align:center;'><h3>Error loading heatmap</h3><p>{str(e)}</p></div>"
        
        return routes_map_html, heatmap_html