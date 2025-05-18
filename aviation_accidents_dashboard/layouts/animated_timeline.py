"""
Layout for the animated timeline tab
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from aviation_accidents_dashboard.config import CARD_STYLE

def create_animated_timeline_layout(df):
    """
    Create the layout for the animated timeline tab
    
    Args:
        df: DataFrame with accident data
        
    Returns:
        html.Div: The layout for the animated timeline tab
    """
    top_operators = ["All Airlines"] + list(df["Operator"].value_counts().head(30).index)
    top_aircraft = ["All Aircraft Types"] + list(df["AC Type"].value_counts().head(30).index)
    
    return html.Div([
        dbc.Card([
            dbc.CardHeader(html.H4("Animated Aviation Accidents Timeline", className="text-center")),
            dbc.CardBody([
                # Filter controls
                dbc.Row([
                    # Airline dropdown
                    dbc.Col([
                        html.Label("Select Airlines:", className="fw-bold"),
                        dcc.Dropdown(
                            id="operator-dropdown",
                            options=[{"label": op, "value": op} for op in top_operators],
                            value=["All Airlines"],
                            multi=True,
                            className="mb-3"
                        )
                    ], width=5),
                    
                    # Aircraft dropdown
                    dbc.Col([
                        html.Label("Select Aircraft Types:", className="fw-bold"),
                        dcc.Dropdown(
                            id="aircraft-dropdown",
                            options=[{"label": ac, "value": ac} for ac in top_aircraft],
                            value=["All Aircraft Types"],
                            multi=True,
                            className="mb-3"
                        )
                    ], width=5),
                    
                    # Update button
                    dbc.Col([
                        html.Div([
                            dbc.Button("Update Map", id="update-map-button", color="primary", className="mt-4")
                        ], className="d-flex justify-content-center")
                    ], width=2)
                ]),
                
                # Animated map container
                html.Div(
                    dcc.Loading(
                        id="loading-animation",
                        type="circle",
                        children=html.Div(id="animated-map-container")
                    )
                )
            ])
        ], style=CARD_STYLE)
    ])