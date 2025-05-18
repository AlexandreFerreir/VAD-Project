"""
Layout for the airline safety tab
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from config import CARD_STYLE

# Import visualization function
from visualizations.airline_analysis import create_airline_safety_analysis

def create_airline_safety_layout(df):
    """
    Create the layout for the airline safety tab
    
    Args:
        df: DataFrame with accident data
        
    Returns:
        html.Div: The layout for the airline safety tab
    """
    fig_airline_fatality, fig_fatality_by_factor, fig_airline_patterns = create_airline_safety_analysis(df)
    
    return html.Div([
        # Title row
        dbc.Row([
            dbc.Col(html.H3("Airline Safety Analysis", className="text-center mb-4"))
        ]),
        
        # First row - Fatality Rate by Airline
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Fatality Rate by Airline (Top 20 Safest)", className="text-center")),
                    dbc.CardBody([
                        dcc.Graph(figure=fig_airline_fatality, config={'displayModeBar': False})
                    ])
                ], style=CARD_STYLE)
            ])
        ], className="mb-4"),
        
        # Second row - Fatality Rate by Incident Type and Flight Phase
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Fatality Analysis by Incident Type and Flight Phase", className="text-center")),
                    dbc.CardBody([
                        dcc.Graph(figure=fig_fatality_by_factor, config={'displayModeBar': False})
                    ])
                ], style=CARD_STYLE)
            ])
        ], className="mb-4"),
        
        # Third row - Incident Types and Flight Phases by Airline
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Airline-Specific Incident Patterns", className="text-center")),
                    dbc.CardBody([
                        dcc.Graph(figure=fig_airline_patterns, config={'displayModeBar': False})
                    ])
                ], style=CARD_STYLE)
            ])
        ])
    ])