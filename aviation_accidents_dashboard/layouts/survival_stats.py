"""
Layout for the survival statistics tab
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from config import CARD_STYLE

# Import visualization functions
from aviation_accidents_dashboard.visualizations.survival_analysis import create_survival_statistics, create_extended_survival_analysis

def create_survival_stats_layout(df):
    """
    Create the layout for the survival statistics tab
    
    Args:
        df: DataFrame with accident data
        
    Returns:
        html.Div: The layout for the survival statistics tab
    """
    fig_original_survival, fig_original_survival_trend, fig_original_aircraft_survival = create_survival_statistics(df)
    fig_survival_outcomes, fig_survival_by_type, fig_survival_by_phase, fig_correlation = create_extended_survival_analysis(df)
    
    return html.Div([
        # Title row
        dbc.Row([
            dbc.Col(html.H3("Aviation Accident Survival Analysis", className="text-center mb-4"))
        ]),
        
        # First row - Overall survival outcomes (original pie chart + our new one)
        dbc.Row([
            # Left column - Original survival pie chart
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Overall Survival Rate", className="text-center")),
                    dbc.CardBody([
                        dcc.Graph(figure=fig_original_survival, config={'displayModeBar': False})
                    ])
                ], style=CARD_STYLE)
            ], width=6),

            # Right column - Survival trend over time
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Survival Rate Trend Over Time", className="text-center")),
                    dbc.CardBody([
                        dcc.Graph(figure=fig_original_survival_trend, config={'displayModeBar': False})
                    ])
                ], style=CARD_STYLE)
            ], width=6)

        ], className="mb-4"),
        
        # Second row - Survival by Aircraft Type and Operator
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Survival Rates by Aircraft Type and Operator", className="text-center")),
                    dbc.CardBody([
                        dcc.Graph(figure=fig_survival_by_type, config={'displayModeBar': False})
                    ])
                ], style=CARD_STYLE)
            ])
        ], className="mb-4"),
        
        # Third row - Survival by Phase and Accident Type
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Survival Rates by Flight Phase and Accident Type", className="text-center")),
                    dbc.CardBody([
                        dcc.Graph(figure=fig_survival_by_phase, config={'displayModeBar': False})
                    ])
                ], style=CARD_STYLE)
            ])
        ], className="mb-4"),
        
        # Fourth row - Correlation analysis and survival trend
        dbc.Row([
            # Left column - Correlation analysis
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Factors Correlated with Survival Rate", className="text-center")),
                    dbc.CardBody([
                        dcc.Graph(figure=fig_correlation, config={'displayModeBar': False})
                    ])
                ], style=CARD_STYLE)
            ], width=7),
        ], className="mb-4"),
        
        # Bottom row - Original Aircraft survival rates
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Survival Rate by Aircraft Type", className="text-center")),
                    dbc.CardBody([
                        dcc.Graph(figure=fig_original_aircraft_survival, config={'displayModeBar': False})
                    ])
                ], style=CARD_STYLE)
            ])
        ])
    ])