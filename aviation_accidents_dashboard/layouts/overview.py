"""
Layout for the overview tab
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from config import CARD_STYLE

# Import visualization functions
from visualizations.scatter_maps import create_plotly_scatter_map
from visualizations.time_series import create_accidents_by_year

def create_stat_card(title, value, icon, color):
    """
    Create a statistics card for the dashboard
    
    Args:
        title: Card title
        value: Value to display
        icon: Icon class
        color: Color for the icon and value
        
    Returns:
        dbc.Card: The card component
    """
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.I(className=icon, style={"font-size": "2rem", "color": color}),
                            html.H4(title, className="card-title mt-2", style={"color": "#2c3e50"}),
                            html.H2(value, className="card-text", style={"color": color, "font-weight": "bold"})
                        ],
                        style={"textAlign": "center"}
                    )
                ]
            )
        ],
        style={"box-shadow": "0 4px 6px rgba(0, 0, 0, 0.1)", "border": "none", "border-radius": "10px"}
    )

def create_overview_layout(df):
    """
    Create the layout for the overview tab
    
    Args:
        df: DataFrame with accident data
        
    Returns:
        html.Div: The layout for the overview tab
    """
    return html.Div([
        # Top row - stat cards
        dbc.Row([
            dbc.Col(create_stat_card("Total Accidents", f"{len(df):,}", "fas fa-plane-crash", "#e74c3c"), width=3),
            dbc.Col(create_stat_card("Total Fatalities", f"{df['Fatalities_Count'].sum():,}", "fas fa-skull-crossbones", "#c0392b"), width=3),
            dbc.Col(create_stat_card("Date Range", f"{df['Year'].min()} - {df['Year'].max()}", "fas fa-calendar-alt", "#3498db"), width=3),
            dbc.Col(create_stat_card("Unique Airlines", f"{df['Operator'].nunique():,}", "fas fa-building", "#2980b9"), width=3),
        ], className="mb-4"),
        
        # Second row - global map
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Global Accident Distribution", className="text-center")),
                    dbc.CardBody([
                        dcc.Graph(figure=create_plotly_scatter_map(df), config={'displayModeBar': False})
                    ])
                ], style=CARD_STYLE)
            ])
        ], className="mb-4"),
        
        # Third row - yearly trends
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Historical Accident Trends", className="text-center")),
                    dbc.CardBody([
                        dcc.Graph(figure=create_accidents_by_year(df), config={'displayModeBar': False})
                    ])
                ], style=CARD_STYLE)
            ])
        ]),
    ])