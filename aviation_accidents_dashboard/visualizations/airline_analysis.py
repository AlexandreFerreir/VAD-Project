"""
Airline analysis visualization functions
"""
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import numpy as np

def create_most_dangerous_routes_chart(df):
    """
    Create chart of most dangerous routes
    
    Args:
        df: DataFrame with accident data
        
    Returns:
        Figure: Plotly figure object
    """
    # Words to exclude from routes
    excluded_terms = ["demonstration", "training", "test flight", "sightseeing", "test"]

    # Count accidents by route
    route_crash_counts = {}
    route_fatalities = {}

    # Analyze all routes in the accident data
    for i, row in df.iterrows():
        if pd.notna(row['Route']) and row['Route'] != '?':
            route = str(row['Route']).strip().lower()
            
            # Check if the route contains terms to be excluded
            if any(term in route for term in excluded_terms):
                continue
                
            # Use the original version (not lowercase) for display
            original_route = str(row['Route']).strip()
            
            # Increment counter for this route
            if original_route in route_crash_counts:
                route_crash_counts[original_route] += 1
            else:
                route_crash_counts[original_route] = 1
            
            # Track fatalities
            if original_route not in route_fatalities:
                route_fatalities[original_route] = 0
            
            if pd.notna(row['Fatalities_Count']):
                route_fatalities[original_route] += row['Fatalities_Count']

    # Get the 15 most dangerous routes
    most_dangerous_routes = sorted(route_crash_counts.items(), key=lambda x: x[1], reverse=True)[:15]

    if not most_dangerous_routes:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No valid routes with multiple accidents were found.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    else:
        # Prepare data for the chart
        routes = [r[0] for r in most_dangerous_routes]
        crash_counts = [r[1] for r in most_dangerous_routes]
        fatalities = [route_fatalities.get(r, 0) for r in routes]
        
        # Simplify names of very long routes
        display_routes = []
        for route in routes:
            if len(route) > 25:
                parts = route.split('-')
                if len(parts) > 2:
                    route = f"{parts[0]}-...-{parts[-1]}"
            display_routes.append(route)
        
        # Reverse the order so the most dangerous route is at the top
        display_routes.reverse()
        crash_counts.reverse()
        fatalities.reverse()
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add horizontal bars
        fig.add_trace(go.Bar(
            y=display_routes,
            x=crash_counts,
            orientation='h',
            marker_color='darkred',
            text=crash_counts,
            textposition='outside',
            name='Accidents'
        ))
        
        # Add fatality information as annotations
        for i, (route, count, fatal) in enumerate(zip(display_routes, crash_counts, fatalities)):
            if count > 2:  # Only add to bars with enough space
                fig.add_annotation(
                    x=count/2,
                    y=i,
                    text=f"{int(fatal)} deaths",
                    font=dict(color='white', size=12, family='Arial, bold'),
                    showarrow=False
                )
        
        # Configure layout
        fig.update_layout(
            title={
                'text': "World's Most Dangerous Commercial Routes<br><sup>(excluding demonstration, training, and test flights)</sup>",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 18}
            },
            xaxis_title="Number of Accidents",
            xaxis=dict(
                gridcolor='lightgrey',
                gridwidth=0.5,
            ),
            height=600,
            margin=dict(l=20, r=20, t=100, b=40),
            paper_bgcolor='white',
            plot_bgcolor='#f8f9fa',
            annotations=[
                dict(
                    text="Source: Historical global aviation accidents dataset",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.1,
                    showarrow=False,
                    font=dict(size=10)
                )
            ]
        )
        
        return fig

def create_airline_aircraft_analysis(df):
    """
    Create airline and aircraft analysis charts
    
    Args:
        df: DataFrame with accident data
        
    Returns:
        Figure: Plotly figure object
    """
    # Prepare data for analysis
    analysis_df = df.copy()
    
    # Add decade field
    analysis_df['Decade'] = (analysis_df['Year'] // 10) * 10
    
    # Create figure with 2x2 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Aircraft Types with Most Accidents', 
            'Airlines with Most Accidents',
            'Accidents by Decade',
            'Accidents by Airline and Aircraft Type (Top 10)'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # ----- CHART 1: Aircraft types with most accidents -----
    # Count accidents by aircraft type
    aircraft_counts = analysis_df['AC Type'].value_counts().reset_index()
    aircraft_counts.columns = ['Aircraft', 'Count']
    
    # Show only top 15 types
    top_15_aircraft = aircraft_counts.head(15).sort_values('Count')
    
    fig.add_trace(
        go.Bar(
            x=top_15_aircraft['Count'],
            y=top_15_aircraft['Aircraft'],
            orientation='h',
            marker_color='#E74C3C',
            name='Aircraft',
            text=top_15_aircraft['Count'],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # ----- CHART 2: Airlines with most accidents -----
    # Count accidents by airline
    operator_counts = analysis_df['Operator'].value_counts().reset_index()
    operator_counts.columns = ['Operator', 'Count']
    
    # Show only top 15 airlines
    top_15_operators = operator_counts.head(15).sort_values('Count')
    
    fig.add_trace(
        go.Bar(
            x=top_15_operators['Count'],
            y=top_15_operators['Operator'],
            orientation='h',
            marker_color='#3498DB',
            name='Airlines',
            text=top_15_operators['Count'],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    # ----- CHART 3: Temporal evolution of accidents by decade -----
    # Count accidents by decade
    decade_counts = analysis_df['Decade'].value_counts().sort_index().reset_index()
    decade_counts.columns = ['Decade', 'Count']
    
    fig.add_trace(
        go.Bar(
            x=decade_counts['Decade'].astype(str),
            y=decade_counts['Count'],
            marker_color='#E74C3C',
            name='Decades',
            text=decade_counts['Count'],
            textposition='outside'
        ),
        row=2, col=1
    )
    
    # ----- CHART 4: Heat map of aircraft by airline -----
    # Limit to top 10 airlines and top 10 aircraft types for readability
    top_10_operators = analysis_df['Operator'].value_counts().nlargest(10).index
    top_10_aircraft = analysis_df['AC Type'].value_counts().nlargest(10).index
    
    # Filter for top 10
    heatmap_df = analysis_df[
        analysis_df['Operator'].isin(top_10_operators) & 
        analysis_df['AC Type'].isin(top_10_aircraft)
    ]
    
    # Create contingency table
    heatmap_data = pd.crosstab(heatmap_df['AC Type'], heatmap_df['Operator']).values
    
    fig.add_trace(
        go.Heatmap(
            z=heatmap_data,
            x=top_10_operators,
            y=top_10_aircraft,
            colorscale='Reds',
            name='Heatmap'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        title={
            'text': "Aviation Accident Analysis",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='#f8f9fa',
    )
    
    # Update axes
    fig.update_xaxes(title_text="Number of Accidents", row=1, col=1, gridcolor='lightgrey')
    fig.update_xaxes(title_text="Number of Accidents", row=1, col=2, gridcolor='lightgrey')
    fig.update_xaxes(title_text="Decade", row=2, col=1, gridcolor='lightgrey')
    fig.update_xaxes(title_text="Airlines", row=2, col=2)
    
    fig.update_yaxes(title_text="Aircraft Type", row=1, col=1)
    fig.update_yaxes(title_text="Airline", row=1, col=2)
    fig.update_yaxes(title_text="Number of Accidents", row=2, col=1, gridcolor='lightgrey')
    fig.update_yaxes(title_text="Aircraft Type", row=2, col=2)
    
    # Add statistics to the bottom
    total_accidents = len(analysis_df)
    total_fatalities = analysis_df['Fatalities_Count'].sum()
    unique_aircraft = analysis_df['AC Type'].nunique()
    unique_operators = analysis_df['Operator'].nunique()
    
    stats_text = f"Total accidents: {total_accidents:,} | " \
                f"Total fatalities: {int(total_fatalities):,} | " \
                f"Aircraft types: {unique_aircraft} | " \
                f"Airlines: {unique_operators}"
                
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.5, y=-0.05,
        showarrow=False,
        font=dict(size=14),
        bgcolor="lightgrey",
        bordercolor="darkgrey",
        borderwidth=1,
        borderpad=6,
        opacity=0.8
    )
    
    return fig

def create_airline_safety_analysis(df):
    """
    Create airline safety analysis charts
    
    Args:
        df: DataFrame with accident data
        
    Returns:
        tuple: Tuple of Plotly figure objects
    """
    # Prepare data for airline safety analysis
    airline_df = df.copy()
    
    # Extract numeric values for aboard and fatalities
    if 'Fatalities_Count' not in airline_df.columns:
        airline_df['Fatalities_Count'] = airline_df['Fatalities'].str.extract(r'(\d+)').astype(float)
    airline_df['Aboard_Count'] = airline_df['Aboard'].str.extract(r'(\d+)').astype(float)
    
    # Calculate fatality rate per accident
    airline_df['Fatality_Rate'] = np.where(
        airline_df['Aboard_Count'] > 0,
        airline_df['Fatalities_Count'] / airline_df['Aboard_Count'],
        0
    )
    
    # Clean and prepare date-related fields
    airline_df['Year'] = airline_df['Date'].dt.year
    
    # Clean operator column and filter out non-commercial operators
    airline_df['Operator'] = airline_df['Operator'].fillna('Unknown').replace('?', 'Unknown')
    
    # Filter out military, test flights, etc.
    exclude_terms = ['military', 'air force', 'navy', 'test', 'private', 'training']
    for term in exclude_terms:
        airline_df = airline_df[~airline_df['Operator'].str.lower().str.contains(term, na=False)]
    
    # Filter to valid records
    valid_airlines = airline_df[
        (airline_df['Year'].notna()) & 
        (airline_df['Operator'] != 'Unknown')
    ].copy()
    
    # Extract flight phases from summary
    flight_phases = ['takeoff', 'landing', 'approach', 'cruise', 'climb', 'descent']
    for phase in flight_phases:
        valid_airlines[f'Phase_{phase}'] = valid_airlines['Summary'].str.contains(phase, case=False, na=False)
    
    # Extract incident types from summary
    incident_types = ['fire', 'explosion', 'collision', 'engine failure', 'mechanical', 
                      'weather', 'fog', 'ice', 'storm', 'hijack', 'overrun']
    
    for incident in incident_types:
        valid_airlines[f'Incident_{incident}'] = valid_airlines['Summary'].str.contains(
            incident, case=False, na=False)
    
    # FIGURE 1: Fatality Rate by Airline (top 20)
    # Calculate average fatality rate for each airline with at least 3 incidents
    operator_counts = valid_airlines['Operator'].value_counts()
    airlines_with_multiple_incidents = operator_counts[operator_counts >= 3].index
    
    fatality_rates = valid_airlines[valid_airlines['Operator'].isin(airlines_with_multiple_incidents)]
    fatality_rates = fatality_rates.groupby('Operator').agg({
        'Fatality_Rate': 'mean',
        'Fatalities_Count': 'sum',
        'Aboard_Count': 'sum',
        'Operator': 'count'
    }).rename(columns={'Operator': 'Incidents'}).reset_index()
    
    # Calculate overall fatality rate
    fatality_rates['Overall_Fatality_Rate'] = fatality_rates['Fatalities_Count'] / fatality_rates['Aboard_Count']
    fatality_rates = fatality_rates.sort_values('Overall_Fatality_Rate')
    
    # Get top 20
    top_20_fatality_rate = fatality_rates.head(20)
    
    # Create color scale based on fatality rate (green to red)
    colors = [
        f'rgb({int(255*x)},{int(255*(1-x))},0)' 
        for x in top_20_fatality_rate['Overall_Fatality_Rate']
    ]
    
    fig_airline_fatality = go.Figure()
    
    fig_airline_fatality.add_trace(
        go.Bar(
            x=top_20_fatality_rate['Overall_Fatality_Rate'],
            y=top_20_fatality_rate['Operator'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(width=1, color='rgba(58, 71, 80, 1.0)')
            ),
            text=[f"{incidents} incidents, {int(fatalities)} fatalities" 
                  for incidents, fatalities in zip(top_20_fatality_rate['Incidents'], 
                                                    top_20_fatality_rate['Fatalities_Count'])],
            textposition='auto',
            hovertemplate='%{y}: %{x:.1%} fatality rate<br>%{text}<extra></extra>'
        )
    )
    
    fig_airline_fatality.update_layout(
        title=dict(
            text="Fatality Rate by Airline (Top 20 Safest)",
            font=dict(size=18, color='#2c3e50', family='Roboto, sans-serif'),
            x=0.5
        ),
        xaxis_title="Fatality Rate",
        yaxis_title="Airline",
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    fig_airline_fatality.update_xaxes(tickformat='.0%')
    
    # FIGURE 2: Fatality Rate by Incident Type and Flight Phase
    fig_fatality_by_factor = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Fatality Rate by Incident Type", "Fatality Rate by Flight Phase"),
        horizontal_spacing=0.1
    )
    
    # Incident type subplot
    incident_stats = []
    
    for incident in incident_types:
        type_data = valid_airlines[valid_airlines[f'Incident_{incident}'] == True]
        
        if len(type_data) >= 10:  # Only include types with sufficient data
            fatality_rate = type_data['Fatalities_Count'].sum() / type_data['Aboard_Count'].sum()
            incident_count = len(type_data)
            
            incident_stats.append({
                'Incident_Type': incident.replace('_', ' ').capitalize(),
                'Fatality_Rate': fatality_rate,
                'Count': incident_count
            })
    
    if incident_stats:
        incident_df = pd.DataFrame(incident_stats)
        incident_df = incident_df.sort_values('Fatality_Rate', ascending=False)
        
        # Create color scale based on fatality rate
        incident_colors = [
            f'rgb({int(255*x)},{int(255*(1-x))},0)' 
            for x in incident_df['Fatality_Rate']
        ]
        
        fig_fatality_by_factor.add_trace(
            go.Bar(
                y=incident_df['Incident_Type'],
                x=incident_df['Fatality_Rate'],
                orientation='h',
                marker=dict(
                    color=incident_colors,
                    line=dict(width=1, color='darkgray')
                ),
                text=[f"{count} incidents" for count in incident_df['Count']],
                textposition='auto',
                hovertemplate='%{y}: %{x:.1%} fatality rate<br>%{text}<extra></extra>',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add reference line at the overall fatality rate
        overall_rate = valid_airlines['Fatalities_Count'].sum() / valid_airlines['Aboard_Count'].sum()
        
        fig_fatality_by_factor.add_shape(
            type="line",
            x0=overall_rate, y0=-0.5, 
            x1=overall_rate, y1=len(incident_df)-0.5,
            line=dict(color="black", width=1, dash="dot"),
            row=1, col=1
        )
    
    # Flight phase subplot
    phase_stats = []
    
    for phase in flight_phases:
        phase_data = valid_airlines[valid_airlines[f'Phase_{phase}'] == True]
        
        if len(phase_data) >= 10:  # Only include phases with sufficient data
            fatality_rate = phase_data['Fatalities_Count'].sum() / phase_data['Aboard_Count'].sum()
            phase_count = len(phase_data)
            
            phase_stats.append({
                'Flight_Phase': phase.capitalize(),
                'Fatality_Rate': fatality_rate,
                'Count': phase_count
            })
    
    if phase_stats:
        phase_df = pd.DataFrame(phase_stats)
        phase_df = phase_df.sort_values('Fatality_Rate', ascending=False)
        
        # Create color scale based on fatality rate
        phase_colors = [
            f'rgb({int(255*x)},{int(255*(1-x))},0)' 
            for x in phase_df['Fatality_Rate']
        ]
        
        fig_fatality_by_factor.add_trace(
            go.Bar(
                y=phase_df['Flight_Phase'],
                x=phase_df['Fatality_Rate'],
                orientation='h',
                marker=dict(
                    color=phase_colors,
                    line=dict(width=1, color='darkgray')
                ),
                text=[f"{count} incidents" for count in phase_df['Count']],
                textposition='auto',
                hovertemplate='%{y}: %{x:.1%} fatality rate<br>%{text}<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add reference line at the overall fatality rate
        fig_fatality_by_factor.add_shape(
            type="line",
            x0=overall_rate, y0=-0.5, 
            x1=overall_rate, y1=len(phase_df)-0.5,
            line=dict(color="black", width=1, dash="dot"),
            row=1, col=2
        )
    
    fig_fatality_by_factor.update_layout(
        title=dict(
            text="Fatality Rate Analysis by Incident Type and Flight Phase",
            font=dict(size=18, color='#2c3e50', family='Roboto, sans-serif'),
            x=0.5
        ),
        height=500,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # Update axes
    fig_fatality_by_factor.update_xaxes(
        title_text="Fatality Rate", 
        tickformat='.0%', 
        range=[0, 1],
        row=1, col=1
    )
    fig_fatality_by_factor.update_yaxes(title_text="Incident Type", row=1, col=1)
    
    fig_fatality_by_factor.update_xaxes(
        title_text="Fatality Rate", 
        tickformat='.0%', 
        range=[0, 1],
        row=1, col=2
    )
    fig_fatality_by_factor.update_yaxes(title_text="Flight Phase", row=1, col=2)
    
    # FIGURE 3: Incident Types and Flight Phases by Airline
    # Focus on top 5 airlines
    top_5_airlines = valid_airlines['Operator'].value_counts().head(5).index.tolist()
    
    # Get the top 5 incident types
    incident_counts = {incident: valid_airlines[f'Incident_{incident}'].sum() 
                       for incident in incident_types}
    top_incidents = sorted(incident_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_incident_types = [i[0] for i in top_incidents]
    
    fig_airline_patterns = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Incident Types by Airline (Top 5 Airlines)", "Flight Phase Incidents by Airline (Top 5 Airlines)"),
        horizontal_spacing=0.1
    )
    
    # Incident types heatmap
    heatmap_data = []
    
    for airline in top_5_airlines:
        airline_data = valid_airlines[valid_airlines['Operator'] == airline]
        airline_total = len(airline_data)
        
        row_data = {'Airline': airline}
        
        for incident in top_incident_types:
            incident_count = airline_data[f'Incident_{incident}'].sum()
            incident_pct = incident_count / airline_total if airline_total > 0 else 0
            row_data[incident.capitalize()] = incident_pct
        
        heatmap_data.append(row_data)
    
    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data)
        
        # Create heatmap
        z_data = []
        for _, row in heatmap_df.iterrows():
            z_data.append([row[incident.capitalize()] for incident in top_incident_types])
        
        fig_airline_patterns.add_trace(
            go.Heatmap(
                z=z_data,
                x=[incident.capitalize() for incident in top_incident_types],
                y=heatmap_df['Airline'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title='% of Incidents'),
                hovertemplate='Airline: %{y}<br>Incident: %{x}<br>Percentage: %{z:.1%}<extra></extra>',
            ),
            row=1, col=1
        )
    
    # Flight phase heatmap
    phase_heatmap_data = []
    
    for airline in top_5_airlines:
        airline_data = valid_airlines[valid_airlines['Operator'] == airline]
        airline_total = len(airline_data)
        
        row_data = {'Airline': airline}
        
        for phase in flight_phases:
            phase_count = airline_data[f'Phase_{phase}'].sum()
            phase_pct = phase_count / airline_total if airline_total > 0 else 0
            row_data[phase.capitalize()] = phase_pct
        
        phase_heatmap_data.append(row_data)
    
    if phase_heatmap_data:
        phase_heatmap_df = pd.DataFrame(phase_heatmap_data)
        
        # Create heatmap
        z_data = []
        for _, row in phase_heatmap_df.iterrows():
            z_data.append([row[phase.capitalize()] for phase in flight_phases])
        
        fig_airline_patterns.add_trace(
            go.Heatmap(
                z=z_data,
                x=[phase.capitalize() for phase in flight_phases],
                y=phase_heatmap_df['Airline'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title='% of Incidents'),
                hovertemplate='Airline: %{y}<br>Phase: %{x}<br>Percentage: %{z:.1%}<extra></extra>',
            ),
            row=1, col=2
        )
    
    fig_airline_patterns.update_layout(
        title=dict(
            text="Airline-Specific Incident Patterns",
            font=dict(size=18, color='#2c3e50', family='Roboto, sans-serif'),
            x=0.5
        ),
        height=500,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # Update axes
    fig_airline_patterns.update_xaxes(title_text="Incident Type", row=1, col=1)
    fig_airline_patterns.update_yaxes(title_text="Airline", row=1, col=1)
    
    fig_airline_patterns.update_xaxes(title_text="Flight Phase", row=1, col=2)
    fig_airline_patterns.update_yaxes(title_text="Airline", row=1, col=2)
    
    return fig_airline_fatality, fig_fatality_by_factor, fig_airline_patterns