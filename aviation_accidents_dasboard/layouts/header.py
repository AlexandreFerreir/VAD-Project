"""
Header component for the aviation dashboard
"""
import dash_bootstrap_components as dbc
from dash import html

def create_header():
    """
    Create the header component for the dashboard
    
    Returns:
        dbc.Navbar: The header navbar component
    """
    return dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(html.I(className="fas fa-plane-crash", style={"font-size": "2rem", "margin-right": "10px"})),
                            dbc.Col(dbc.NavbarBrand("Global Aviation Accident Analysis", className="ms-2")),
                        ],
                        align="center",
                    ),
                    href="/",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink("Dashboard", href="#")),
                            dbc.NavItem(dbc.NavLink("About", href="#")),
                            dbc.NavItem(dbc.NavLink("Data Source", href="#")),
                        ],
                        className="ms-auto",
                        navbar=True,
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ],
            fluid=True,
        ),
        color="dark",
        dark=True,
        className="mb-4",
    )