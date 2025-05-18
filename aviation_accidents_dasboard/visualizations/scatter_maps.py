"""
Functions for creating scatter maps with Plotly
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_plotly_scatter_map(df):
    """
    Create a Plotly scatter map for global visualization
    
    Args:
        df: DataFrame with accident data
        
    Returns:
        Figure: Plotly figure object
    """
    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        color="Fatalities_Count",
        size="Fatalities_Count",
        hover_name="Location",
        projection="natural earth",
        title="Global Aviation Accidents",
        color_continuous_scale="Viridis",
        size_max=15,
        hover_data={
            "Year": True,
            "Operator": True,
            "AC Type": True,
            "Fatalities_Count": True
        }
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        height=600,
        paper_bgcolor="white",
        font=dict(
            family="Roboto, sans-serif",
            size=12
        ),
        title=dict(
            font=dict(
                family="Roboto, sans-serif",
                size=20,
                color="#2c3e50"
            ),
            x=0.5
        )
    )
    
    return fig

def create_filtered_animated_map(df, selected_operators=None, selected_aircraft=None):
    """
    Create a filtered animated map based on user selections
    
    Args:
        df: DataFrame with accident data
        selected_operators: List of selected operators
        selected_aircraft: List of selected aircraft types
        
    Returns:
        Figure: Plotly figure object
    """
    # Apply filters
    filtered_df = df.copy()
    
    if selected_operators and "All Airlines" not in selected_operators:
        filtered_df = filtered_df[filtered_df["Operator"].isin(selected_operators)]
    
    if selected_aircraft and "All Aircraft Types" not in selected_aircraft:
        filtered_df = filtered_df[filtered_df["AC Type"].isin(selected_aircraft)]
    
    # Check if filtered dataframe is empty
    if filtered_df.empty:
        # Create an empty figure with an error message
        fig = go.Figure()
        fig.update_layout(
            title="No Results Found",
            annotations=[
                dict(
                    text="No accidents match your filter criteria.<br>Please try different selections.",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
            ],
            height=600
        )
        return fig
    
    # Create hover information
    filtered_df["hover_info"] = filtered_df.apply(
        lambda row: f"<b>Date:</b> {row['Date'].strftime('%Y-%m-%d')}<br>" +
                    f"<b>Location:</b> {row['Location']}<br>" +
                    f"<b>Operator:</b> {row['Operator']}<br>" +
                    f"<b>Aircraft:</b> {row['AC Type']}<br>" +
                    f"<b>Fatalities:</b> {row['Fatalities_Count']}<br>" +
                    f"<b>Aboard:</b> {row['Aboard'] if pd.notna(row['Aboard']) else 'Unknown'}<br>" +
                    f"<b>Route:</b> {row['Route'] if pd.notna(row['Route']) else 'Unknown'}",
        axis=1
    )
    
    # Create a color dictionary for operators
    operators = filtered_df["Operator"].unique()
    colors = px.colors.qualitative.Bold * (1 + len(operators) // len(px.colors.qualitative.Bold))
    color_map = dict(zip(operators, colors[:len(operators)]))
    
    # Create base figure
    fig = go.Figure()
    
    # Create frames for each year
    frames = []
    for year in sorted(filtered_df["Year"].unique()):
        year_data = filtered_df[filtered_df["Year"] == year]
        
        # Calculate size reference to keep consistent marker sizes across years
        size_ref = 2.0 * max(filtered_df["Fatalities_Count"])/(30.**2) if len(filtered_df) > 0 else 1
        
        frame = go.Frame(
            data=[
                go.Scattergeo(
                    lat=year_data["lat"],
                    lon=year_data["lon"],
                    mode="markers",
                    marker=dict(
                        size=year_data["Fatalities_Count"],
                        sizemode="area",
                        sizeref=size_ref,
                        sizemin=4,
                        color=[color_map[op] for op in year_data["Operator"]],  # Color by operator
                    ),
                    text=year_data["hover_info"],
                    hoverinfo="text",
                    name=f"Year {year}"
                )
            ],
            name=str(year)
        )
        frames.append(frame)
        
        # Add the first year data as initial view
        if year == sorted(filtered_df["Year"].unique())[0]:
            fig.add_trace(
                go.Scattergeo(
                    lat=year_data["lat"],
                    lon=year_data["lon"],
                    mode="markers",
                    marker=dict(
                        size=year_data["Fatalities_Count"],
                        sizemode="area",
                        sizeref=size_ref,
                        sizemin=4,
                        color=[color_map[op] for op in year_data["Operator"]],
                    ),
                    text=year_data["hover_info"],
                    hoverinfo="text",
                    name=f"Year {sorted(filtered_df['Year'].unique())[0]}"
                )
            )
    
    # Add frames to figure
    fig.frames = frames
    
    # Add animation control buttons
    fig.update_layout(
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                    "label": "▶ Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    "label": "⏸ Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )
    
    # Add year slider
    sliders = [{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 16},
            "prefix": "Year: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {
                "args": [
                    [str(year)],
                    {"frame": {"duration": 300, "redraw": True},
                     "mode": "immediate"}
                ],
                "label": str(year),
                "method": "animate"
            } for year in sorted(filtered_df["Year"].unique())
        ]
    }]
    
    fig.update_layout(
        sliders=sliders,
        height=600,
        margin=dict(l=20, r=20, t=80, b=80)
    )
    
    # Configure map appearance
    fig.update_geos(
        projection_type="natural earth",
        showcoastlines=True, 
        coastlinecolor="Black",
        showland=True, 
        landcolor="lightgrey",
        showocean=True, 
        oceancolor="lightblue",
        showlakes=True, 
        lakecolor="lightblue",
        showcountries=True, 
        countrycolor="darkgrey"
    )
    
    # Add title with filter information
    operator_info = "All Airlines" if not selected_operators or "All Airlines" in selected_operators else f"{len(selected_operators)} selected airlines"
    aircraft_info = "All Aircraft Types" if not selected_aircraft or "All Aircraft Types" in selected_aircraft else f"{len(selected_aircraft)} selected aircraft types"
    
    fig.update_layout(
        title=dict(
            text=f"Global Aviation Accidents by Airline<br><sup>Showing: {operator_info} | {aircraft_info} | {len(filtered_df)} incidents</sup>",
            font=dict(size=20, color="#2c3e50", family="Roboto, sans-serif"),
            y=0.95
        )
    )
    
    # Add footer with data source
    fig.add_annotation(
        text="Source: Historical Aviation Accident Database",
        xref="paper", yref="paper",
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(size=10, color="gray"),
        align="center"
    )
    
    # Add legend for operators
    # Create a custom legend
    for op, color in color_map.items():
        fig.add_trace(
            go.Scattergeo(
                lat=[None],
                lon=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                name=op,
                showlegend=True
            )
        )
    
    return fig