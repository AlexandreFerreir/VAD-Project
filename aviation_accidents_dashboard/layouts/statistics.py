"""
Layout for the statistics tab
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from aviation_accidents_dashboard.config import CARD_STYLE

# Import visualization functions
from aviation_accidents_dashboard.visualizations.airline_analysis import create_most_dangerous_routes_chart, create_airline_aircraft_analysis

def create_statistics_layout(df):
    """
    Create the layout for the statistics tab
    
    Args:
        df: DataFrame with accident data
        
    Returns:
        html.Div: The layout for the statistics tab
    """
    return html.Div([
        # Title row
        dbc.Row([
            dbc.Col(html.H3("Advanced Aviation Accident Analysis", className="text-center mb-4"))
        ]),
        
        # First row - Most dangerous routes chart
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("World's Most Dangerous Commercial Routes", className="text-center")),
                    dbc.CardBody([
                        dcc.Graph(figure=create_most_dangerous_routes_chart(df), config={'displayModeBar': True})
                    ])
                ], style=CARD_STYLE)
            ])
        ], className="mb-4"),
        
        # Second row - Airline and aircraft analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Analysis by Airline and Aircraft Type", className="text-center")),
                    dbc.CardBody([
                        dcc.Graph(figure=create_airline_aircraft_analysis(df), config={'displayModeBar': True})
                    ])
                ], style=CARD_STYLE)
            ])
        ])
    ])