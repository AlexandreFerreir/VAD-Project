"""
Layout for the USA crashes tab
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from aviation_accidents_dashboard.config import CARD_STYLE
from aviation_accidents_dashboard.layouts.overview import create_stat_card
from aviation_accidents_dashboard.visualizations.us_analysis import create_us_crashes_analysis

def create_usa_crashes_layout(df, encoded_images):
    """
    Create the layout for the USA crashes tab
    
    Args:
        df: DataFrame with accident data
        encoded_images: Dictionary of base64 encoded images
        
    Returns:
        html.Div: The layout for the USA crashes tab
    """
    # Calculate total US crashes and fatalities
    us_df = df[df['Location'].str.contains('United States|USA|U.S.A.|US|U.S.', case=False, na=False)]
    total_us_crashes = len(us_df)
    total_us_fatalities = us_df['Fatalities_Count'].sum()
    
    # Get US analysis figures
    _, fig_timeline, fig_operators = create_us_crashes_analysis(df)
    
    return html.Div([
        # Title row
        dbc.Row([
            dbc.Col(html.H3("US Aviation Accidents Analysis", className="text-center mb-4"))
        ]),

        dbc.Row([
            dbc.Col(create_stat_card("Total US Crashes", f"{total_us_crashes:,}", "fas fa-flag-usa", "#3498db"), width=3),
            dbc.Col(create_stat_card("US Fatalities", f"{total_us_fatalities:,}", "fas fa-skull-crossbones", "#e74c3c"), width=3),
            dbc.Col(create_stat_card("% of Global Crashes", f"{(total_us_crashes/len(df)*100):.1f}%", "fas fa-globe-americas", "#2c3e50"), width=3),
            dbc.Col(create_stat_card("Date Range", f"1987-2008", "fas fa-calendar-alt", "#3498db"), width=3),
        ], className="mb-4"),
        
        # Risk map row - using the static image
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("US Accident Risk Analysis", className="text-center")),
                    dbc.CardBody([
                        # Use base64 encoded image if available
                        html.Img(
                            src=f"data:image/png;base64,{encoded_images['usa_crash_density']}" if encoded_images['usa_crash_density'] else "",
                            style={"width": "100%", "height": "auto"},
                            className="img-fluid"
                        ) if encoded_images['usa_crash_density'] else html.Div(
                            "Image could not be loaded. Please check that '../images/USA_crash_volume_densitiy.png' exists.",
                            style={"text-align": "center", "color": "red", "padding": "20px"}
                        )
                    ])
                ], style=CARD_STYLE)
            ])
        ], className="mb-4"),
        
        dbc.Row([
            # Left column - Top 20 flight routes
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Top 20 Flight Routes by Number of Flights", className="text-center")),
                    dbc.CardBody([
                        # Use base64 encoded image if available
                        html.Img(
                            src=f"data:image/png;base64,{encoded_images['top_routes']}" if encoded_images['top_routes'] else "",
                            style={"width": "100%", "height": "auto"},
                            className="img-fluid"
                        ) if encoded_images['top_routes'] else html.Div(
                            "Image could not be loaded. Please check that '../images/top20_flight_routes_by_number_of_flights.png' exists.",
                            style={"text-align": "center", "color": "red", "padding": "20px"}
                        )
                    ])
                ], style=CARD_STYLE)
            ], width=6),
            
            # Right column - US flight routes volume crashes
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("US Flight Routes Volume vs. Crashes", className="text-center")),
                    dbc.CardBody([
                        # Use base64 encoded image if available
                        html.Img(
                            src=f"data:image/png;base64,{encoded_images['us_routes']}" if encoded_images['us_routes'] else "",
                            style={"width": "100%", "height": "auto"},
                            className="img-fluid"
                        ) if encoded_images['us_routes'] else html.Div(
                            "Image could not be loaded. Please check that '../images/US_flights_routes_volume_crashes.png' exists.",
                            style={"text-align": "center", "color": "red", "padding": "20px"}
                        )
                    ])
                ], style=CARD_STYLE)
            ], width=6)
        ], className="mb-4"),

        # Volume and airline distribution row
        dbc.Row([
            # Left column - Volume trend
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("US Accident Volume by Decade", className="text-center")),
                    dbc.CardBody([
                        dcc.Graph(figure=fig_timeline, config={'displayModeBar': False})
                    ])
                ], style=CARD_STYLE)
            ], width=6),
            
            # Right column - Operator distribution
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Top Airlines by US Accidents", className="text-center")),
                    dbc.CardBody([
                        dcc.Graph(figure=fig_operators, config={'displayModeBar': False})
                    ])
                ], style=CARD_STYLE)
            ], width=6)
        ])
    ])