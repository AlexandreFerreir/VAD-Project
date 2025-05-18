"""
Main application file for the aviation accidents dashboard
"""
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import os

# Import configurations and constants
from config import CONTENT_STYLE, EXTERNAL_STYLESHEETS, HEATMAP_HTML_PATH, ROUTES_MAP_HTML_PATH, ASSETS_DIR

# Import data loading functionality
from data_loader import load_data

# Import layout components
from layouts.header import create_header
from layouts.overview import create_overview_layout
from layouts.crash_vis import create_crash_vis_layout
from layouts.animated_timeline import create_animated_timeline_layout
from layouts.usa_crashes import create_usa_crashes_layout
from layouts.survival_stats import create_survival_stats_layout
from layouts.airline_safety import create_airline_safety_layout
from layouts.statistics import create_statistics_layout

# Import callbacks
from callbacks.map_callbacks import register_map_callbacks
from callbacks.filter_callbacks import register_filter_callbacks

# Initialize the Dash app with a nice theme
app = dash.Dash(
    __name__, 
    suppress_callback_exceptions=True,
    external_stylesheets=[
        dbc.themes.FLATLY,  # Use a modern, clean theme
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.1/css/all.min.css"  # For icons
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
server = app.server  # Needed for deployment

# Load data for the app
df, location_dict, routes_data, encoded_images = load_data()

# Initialize maps
def initialize_maps():
    """Initialize both maps at startup if they don't exist"""
    # Import maps module
    from visualizations.maps import create_folium_heatmap, create_edge_bundled_map
    
    # Generate heat map
    if not os.path.exists(HEATMAP_HTML_PATH):
        create_folium_heatmap(df)
    
    # Generate routes map
    if not os.path.exists(ROUTES_MAP_HTML_PATH):
        create_edge_bundled_map(df, routes_data)

# Call the initialization function once before defining the layout
initialize_maps()

# Define the app layout
app.layout = html.Div([
    # Add the header
    create_header(),
    
    # Main content container
    dbc.Container([
        # Navigation tabs with custom styling
        dbc.Tabs(
            [
                dbc.Tab(label="Overview", tab_id="tab-1", 
                        label_style={"font-weight": "bold", "font-size": "16px"},
                        active_label_style={"color": "#2980b9"}),
                dbc.Tab(label="Crash Visualizations", tab_id="tab-2",  
                        label_style={"font-weight": "bold", "font-size": "16px"},
                        active_label_style={"color": "#2980b9"}),
                dbc.Tab(label="Animated Timeline", tab_id="tab-3",
                        label_style={"font-weight": "bold", "font-size": "16px"},
                        active_label_style={"color": "#2980b9"}),
                dbc.Tab(label="USA Crashes & Volume", tab_id="tab-4", 
                        label_style={"font-weight": "bold", "font-size": "16px"},
                        active_label_style={"color": "#2980b9"}),
                dbc.Tab(label="Survival Statistics", tab_id="tab-5",
                        label_style={"font-weight": "bold", "font-size": "16px"},
                        active_label_style={"color": "#2980b9"}),
                dbc.Tab(label="Airline Safety", tab_id="tab-7",  
                        label_style={"font-weight": "bold", "font-size": "16px"},
                        active_label_style={"color": "#2980b9"}),
                dbc.Tab(label="Statistics", tab_id="tab-6",  
                        label_style={"font-weight": "bold", "font-size": "16px"},
                        active_label_style={"color": "#2980b9"}),
            ],
            id="tabs",
            active_tab="tab-1",
            className="mb-4"
        ),
        
        # Content for each tab
        html.Div(id="tabs-content", style=CONTENT_STYLE)
    ], fluid=True),
    
    # Footer
    html.Footer(
        dbc.Container(
            [
                html.Hr(),
                html.P(
                    [
                        "Aviation Accident Dashboard © 2025 | ",
                        html.A("Source Code", href="#", className="text-decoration-none"),
                        " | Created by Joao Tinoco and Alexandre Ferreira"
                    ],
                    className="text-center text-muted"
                )
            ],
            fluid=True
        ),
        className="mt-4 py-3"
    )
])

# Callback to update tab content
@app.callback(
    dash.Output("tabs-content", "children"),
    dash.Input("tabs", "active_tab")
)
def render_content(tab):
    """
    Render the appropriate content based on the selected tab
    
    Args:
        tab: The ID of the selected tab
        
    Returns:
        html.Div: The layout for the selected tab
    """
    if tab == "tab-1":
        return create_overview_layout(df)
    elif tab == "tab-2":
        return create_crash_vis_layout()
    elif tab == "tab-3":
        return create_animated_timeline_layout(df)
    elif tab == "tab-4":
        return create_usa_crashes_layout(df, encoded_images)
    elif tab == "tab-5":
        return create_survival_stats_layout(df)
    elif tab == "tab-6":
        return create_statistics_layout(df)
    elif tab == "tab-7":
        return create_airline_safety_layout(df)
    
    # Default return if no tab matches
    return html.Div("This tab is not yet implemented.")

# Register callbacks
register_map_callbacks(app)
register_filter_callbacks(app, df)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8052)))