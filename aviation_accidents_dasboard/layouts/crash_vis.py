"""
Layout for the crash visualizations tab
"""
import dash_bootstrap_components as dbc
from dash import html, dcc

from config import CARD_STYLE

def create_crash_vis_layout():
    """
    Create the layout for the crash visualizations tab
    
    Returns:
        html.Div: The layout for the crash visualizations tab
    """
    return html.Div([            
        # Main title
        dbc.Row([
            dbc.Col(html.H3("Global Aviation Accident Visualizations", className="text-center mb-4"))
        ]),
        
        # Primeiro o mapa de rotas com edge bundling
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Global Aviation Flight Routes with Edge Bundling", className="text-center")),
                    dbc.CardBody([
                        html.P(
                            "This map visualizes flight routes with edge bundling algorithm to better show traffic patterns. "
                            "The bundling groups together similar routes to reduce visual clutter and highlight major air corridors. "
                            "Red markers indicate accident locations, green circles show departure points.",
                            className="text-center text-muted mb-3"
                        ),
                        # Iframe para o mapa de rotas
                        html.Div(
                            dcc.Loading(
                                id="loading-routes-map",
                                type="circle",
                                children=html.Iframe(
                                    id="routes-map-iframe",
                                    srcDoc="",  # Preenchido pelo callback
                                    width="100%",
                                    height="700px",
                                    style={"border": "none", "borderRadius": "8px"}
                                )
                            )
                        )
                    ])
                ], style=CARD_STYLE)
            ])
        ], className="mb-4"),
        
        # Depois o mapa de calor
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Global Aviation Accident Density Map", className="text-center")),
                    dbc.CardBody([
                        html.P(
                            "Heat map showing concentration of accident locations. "
                            "Color gradient indicates accident density from blue (lower) to red (higher). "
                            "This visualization helps identify global accident hotspots.",
                            className="text-center text-muted mb-3"
                        ),
                        # Iframe para o mapa de calor
                        html.Div(
                            dcc.Loading(
                                id="loading-heatmap",
                                type="circle",
                                children=html.Iframe(
                                    id="heatmap-iframe",
                                    srcDoc="",  # Preenchido pelo callback
                                    width="100%",
                                    height="700px",
                                    style={"border": "none", "borderRadius": "8px"}
                                )
                            )
                        )
                    ])
                ], style=CARD_STYLE)
            ])
        ])
    ])