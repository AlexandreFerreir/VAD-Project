"""
Callbacks for filtering data and updating visualizations
"""
from dash import Input, Output, State
import dash_bootstrap_components as dbc
from dash import dcc, html

# Import visualization function
from visualizations.scatter_maps import create_filtered_animated_map

def register_filter_callbacks(app, df):
    """
    Register callbacks related to filtering data
    
    Args:
        app: The Dash app
        df: DataFrame with accident data
    """
    @app.callback(
        Output("animated-map-container", "children"),
        Input("update-map-button", "n_clicks"),
        State("operator-dropdown", "value"),
        State("aircraft-dropdown", "value")
    )
    def update_animated_map(n_clicks, selected_operators, selected_aircraft):
        """
        Update the animated map based on user selections
        
        Args:
            n_clicks: Number of clicks on update button
            selected_operators: List of selected operators
            selected_aircraft: List of selected aircraft types
            
        Returns:
            dcc.Graph: The updated animated map
        """
        # Initialize with default values if needed
        if not selected_operators:
            selected_operators = ["All Airlines"]
        if not selected_aircraft:
            selected_aircraft = ["All Aircraft Types"]
        
        # Create the animated map
        fig = create_filtered_animated_map(df, selected_operators, selected_aircraft)
        
        # Return as a Plotly graph
        return dcc.Graph(
            figure=fig, 
            config={'displayModeBar': True},
            style={"height": "700px"}
        )