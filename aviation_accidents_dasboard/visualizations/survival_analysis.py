"""
Survival analysis visualization functions
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_survival_statistics(df):
    """
    Create survival statistics visualizations
    
    Args:
        df: DataFrame with accident data
        
    Returns:
        tuple: Tuple of Plotly figure objects
    """
    # Calculate survival statistics
    df_copy = df.copy()
    df_copy['Total_Aboard'] = df_copy['Aboard'].str.extract(r'(\d+)').astype(float)
    df_with_aboard = df_copy.dropna(subset=['Total_Aboard', 'Fatalities_Count'])
    df_with_aboard['Survivors'] = df_with_aboard['Total_Aboard'] - df_with_aboard['Fatalities_Count']
    df_with_aboard['Survivors'] = df_with_aboard['Survivors'].clip(lower=0)  # Ensure no negative survivors
    df_with_aboard['Survival_Rate'] = df_with_aboard['Survivors'] / df_with_aboard['Total_Aboard']
    
    # Overall survival rate
    total_aboard = df_with_aboard['Total_Aboard'].sum()
    total_fatalities = df_with_aboard['Fatalities_Count'].sum()
    total_survivors = df_with_aboard['Survivors'].sum()
    
    # Survival rate pie chart
    fig_survival = go.Figure(data=[go.Pie(
        labels=['Fatalities', 'Survivors'],
        values=[total_fatalities, total_survivors],
        hole=.4,
        marker_colors=['#e74c3c', '#2ecc71']
    )])
    
    fig_survival.update_layout(
        title=dict(
            text='Overall Survival Rate in Aviation Accidents',
            font=dict(size=18, color='#2c3e50', family='Roboto, sans-serif'),
            x=0.5
        ),
        annotations=[dict(
            text=f"{int((total_survivors/total_aboard)*100)}% Survival",
            x=0.5, y=0.5,
            font_size=20,
            showarrow=False
        )],
        height=400
    )
    
    # Survival rate over time - by decade
    df_with_aboard['Decade'] = (df_with_aboard['Year'] // 10) * 10
    decade_survival = df_with_aboard.groupby('Decade').agg({
        'Total_Aboard': 'sum',
        'Fatalities_Count': 'sum',
        'Survivors': 'sum'
    }).reset_index()
    
    decade_survival['Survival_Rate'] = decade_survival['Survivors'] / decade_survival['Total_Aboard']
    
    fig_survival_trend = px.line(
        decade_survival,
        x='Decade',
        y='Survival_Rate',
        markers=True,
        title='Survival Rate by Decade',
        labels={'Survival_Rate': 'Survival Rate', 'Decade': 'Decade'},
        color_discrete_sequence=['#2ecc71']
    )
    
    fig_survival_trend.update_traces(
        line=dict(width=3),
        marker=dict(size=10)
    )
    
    fig_survival_trend.update_layout(
        height=400,
        title_font=dict(size=18, color='#2c3e50', family='Roboto, sans-serif'),
        title_x=0.5,
        yaxis=dict(
            title='Survival Rate',
            tickformat='.0%',
            range=[0, 1]
        ),
        hovermode='x unified'
    )
    
    # Survival by aircraft type
    aircraft_survival = df_with_aboard.groupby('AC Type').agg({
        'Total_Aboard': 'sum',
        'Fatalities_Count': 'sum',
        'Survivors': 'sum'
    }).reset_index()
    
    aircraft_survival['Survival_Rate'] = aircraft_survival['Survivors'] / aircraft_survival['Total_Aboard']
    aircraft_survival['Total_Incidents'] = df_with_aboard.groupby('AC Type').size().values
    
    # Filter to only include aircraft types with at least 5 incidents for statistical significance
    significant_aircraft = aircraft_survival[aircraft_survival['Total_Incidents'] >= 5].sort_values('Survival_Rate', ascending=False).head(10)
    
    fig_aircraft_survival = px.bar(
        significant_aircraft,
        y='AC Type',
        x='Survival_Rate',
        orientation='h',
        title='Top 10 Aircraft Types by Survival Rate (min. 5 incidents)',
        color='Survival_Rate',
        color_continuous_scale='RdYlGn',
        text='Total_Incidents',
        hover_data=['Total_Aboard', 'Survivors', 'Fatalities_Count']
    )
    
    fig_aircraft_survival.update_traces(
        texttemplate='%{text} incidents',
        textposition='outside'
    )
    
    fig_aircraft_survival.update_layout(
        height=500,
        title_font=dict(size=18, color='#2c3e50', family='Roboto, sans-serif'),
        title_x=0.5,
        yaxis=dict(title=''),
        xaxis=dict(
            title='Survival Rate',
            tickformat='.0%',
            range=[0, 1]
        )
    )
    
    return fig_survival, fig_survival_trend, fig_aircraft_survival

def create_extended_survival_analysis(df):
    """
    Creates comprehensive survival analysis visualizations for the dashboard
    
    Args:
        df: DataFrame with accident data
        
    Returns:
        tuple: Multiple figures for different aspects of survival analysis
    """
    # Prepare data for survival analysis
    survival_df = df.copy()
    
    # Extract numeric values for aboard and fatalities if not already done
    if 'Fatalities_Count' not in survival_df.columns:
        survival_df['Fatalities_Count'] = survival_df['Fatalities'].str.extract(r'(\d+)').astype(float)
    survival_df['Aboard_Count'] = survival_df['Aboard'].str.extract(r'(\d+)').astype(float)
    
    # Calculate survivors and survival rate
    survival_df['Survivors'] = survival_df['Aboard_Count'] - survival_df['Fatalities_Count']
    survival_df['Survival_Rate'] = np.where(
        survival_df['Aboard_Count'] > 0,
        survival_df['Survivors'] / survival_df['Aboard_Count'],
        0
    )
    
    survival_df['Year'] = survival_df['Date'].dt.year
    survival_df['Month'] = survival_df['Date'].dt.month

    # Clean data
    valid_df = survival_df[
        (survival_df['Aboard_Count'] > 0) & 
        (~survival_df['Survival_Rate'].isna()) &
        (survival_df['Year'].notna())
    ]
    
    # Create survival categories
    valid_df['Survival_Category'] = pd.cut(
        valid_df['Survival_Rate'],
        bins=[-0.001, 0.1, 0.5, 0.9, 1.001],
        labels=['Very Low (0-10%)', 'Low (10-50%)', 'High (50-90%)', 'Very High (90-100%)']
    )
    
    # Extract phase of flight from summary if available
    flight_phases = ['takeoff', 'landing', 'approach', 'cruise', 'climb', 'descent']
    for phase in flight_phases:
        valid_df[f'Phase_{phase}'] = valid_df['Summary'].str.contains(phase, case=False, na=False)
    
    # Extract accident types from summary
    accident_types = ['fire', 'explosion', 'collision', 'engine failure', 'mechanical', 'overrun', 'fuel', 'hijack']
    for accident in accident_types:
        valid_df[f'Accident_{accident}'] = valid_df['Summary'].str.contains(accident, case=False, na=False)
        
    # Extract weather conditions from summary
    weather_conditions = ['fog', 'rain', 'snow', 'storm', 'thunderstorm', 'wind', 'icing', 'clear', 'visibility']
    for condition in weather_conditions:
        valid_df[f'Weather_{condition}'] = valid_df['Summary'].str.contains(condition, case=False, na=False)
    
    # Extract water vs land information
    valid_df['Water_Crash'] = valid_df['Location'].str.contains(
        'ocean|sea|atlantic|pacific|water|lake|river|bay|strait|coast|channel|gulf', 
        case=False, 
        na=False
    )
    
    # FIGURE 1: Overall Survival Outcomes (pie chart)
    total_aboard = valid_df['Aboard_Count'].sum()
    total_fatalities = valid_df['Fatalities_Count'].sum()
    total_survivors = valid_df['Survivors'].sum()
    
    fig_survival_outcomes = go.Figure(data=[go.Pie(
        labels=['Fatalities', 'Survivors'],
        values=[total_fatalities, total_survivors],
        hole=.4,
        marker_colors=['#e74c3c', '#2ecc71']
    )])
    
    fig_survival_outcomes.update_layout(
        title=dict(
            text='Overall Survival Outcomes in Aviation Accidents',
            font=dict(size=18, color='#2c3e50', family='Roboto, sans-serif'),
            x=0.5
        ),
        annotations=[dict(
            text=f"{int((total_survivors/total_aboard)*100)}% Survival",
            x=0.5, y=0.5,
            font_size=20,
            showarrow=False
        )],
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # FIGURE 2: Survival Rate by Aircraft Type and Operator (Top 15)
    fig_survival_by_type = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Survival Rate by Aircraft Type (Top 15)", "Survival Rate by Operator (Top 15)"),
        horizontal_spacing=0.1
    )
    
    # Aircraft Type subplot
    top_aircraft_types = valid_df['AC Type'].value_counts().head(15).index
    aircraft_survival = valid_df[valid_df['AC Type'].isin(top_aircraft_types)]
    
    # Aqui é onde o código estava cortado. Adicionando o resto:
    aircraft_stats = aircraft_survival.groupby('AC Type').agg({
        'Survival_Rate': 'mean',
        'AC Type': 'count'
    }).rename(columns={'AC Type': 'Count'}).reset_index()
    
    aircraft_stats = aircraft_stats.sort_values('Survival_Rate')
    
    # Create color gradient based on survival rate
    aircraft_colors = [
        f'rgb({int(255*(1-x))},{int(255*x)},{0})' 
        for x in aircraft_stats['Survival_Rate']
    ]
    
    fig_survival_by_type.add_trace(
        go.Bar(
            x=aircraft_stats['Survival_Rate'],
            y=aircraft_stats['AC Type'],
            orientation='h',
            marker=dict(
                color=aircraft_colors,
                line=dict(width=1, color='darkgray')
            ),
            text=[f"{count} incidents" for count in aircraft_stats['Count']],
            textposition='auto',
            hovertemplate='Survival rate: %{x:.1%}<br>%{text}<extra>%{y}</extra>'
        ),
        row=1, col=1
    )
    
    # Add reference line at 0.5 survival rate
    fig_survival_by_type.add_shape(
        type="line",
        x0=0.5, y0=-0.5, x1=0.5, y1=14.5,
        line=dict(color="black", width=1, dash="dot"),
        row=1, col=1
    )
    
    # Operator subplot
    top_operators = valid_df['Operator'].value_counts().head(15).index
    operator_survival = valid_df[valid_df['Operator'].isin(top_operators)]
    
    operator_stats = operator_survival.groupby('Operator').agg({
        'Survival_Rate': 'mean',
        'Operator': 'count'
    }).rename(columns={'Operator': 'Count'}).reset_index()
    
    operator_stats = operator_stats.sort_values('Survival_Rate')
    
    # Create color gradient based on survival rate
    operator_colors = [
        f'rgb({int(255*(1-x))},{int(255*x)},{0})' 
        for x in operator_stats['Survival_Rate']
    ]
    
    fig_survival_by_type.add_trace(
        go.Bar(
            x=operator_stats['Survival_Rate'],
            y=operator_stats['Operator'],
            orientation='h',
            marker=dict(
                color=operator_colors,
                line=dict(width=1, color='darkgray')
            ),
            text=[f"{count} incidents" for count in operator_stats['Count']],
            textposition='auto',
            hovertemplate='Survival rate: %{x:.1%}<br>%{text}<extra>%{y}</extra>'
        ),
        row=1, col=2
    )
    
    # Add reference line at 0.5 survival rate
    fig_survival_by_type.add_shape(
        type="line",
        x0=0.5, y0=-0.5, x1=0.5, y1=14.5,
        line=dict(color="black", width=1, dash="dot"),
        row=1, col=2
    )
    
    fig_survival_by_type.update_layout(
        title=dict(
            text='Survival Rate Analysis by Aircraft Type and Operator',
            font=dict(size=18, color='#2c3e50', family='Roboto, sans-serif'),
            x=0.5
        ),
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False
    )
    
    # Update axes for both subplots
    fig_survival_by_type.update_xaxes(
        title_text="Survival Rate", 
        tickformat='.0%', 
        range=[0, 1],
        row=1, col=1
    )
    fig_survival_by_type.update_yaxes(title_text="Aircraft Type", row=1, col=1)
    
    fig_survival_by_type.update_xaxes(
        title_text="Survival Rate", 
        tickformat='.0%', 
        range=[0, 1],
        row=1, col=2
    )
    fig_survival_by_type.update_yaxes(title_text="Operator", row=1, col=2)
    
    # FIGURE 3: Survival Rate by Phase of Flight and Accident Type
    fig_survival_by_phase = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Survival Rate by Flight Phase", "Survival Rate by Accident Type"),
        horizontal_spacing=0.1
    )
    
    # Flight phase subplot
    # Calculate average survival rate for each flight phase
    phase_data = []
    
    for phase in flight_phases:
        phase_incidents = valid_df[valid_df[f'Phase_{phase}'] == True]
        if len(phase_incidents) > 0:
            avg_survival = phase_incidents['Survival_Rate'].mean()
            count = len(phase_incidents)
            phase_data.append({
                'Phase': phase.capitalize(),
                'Survival_Rate': avg_survival,
                'Count': count
            })
    
    phase_df = pd.DataFrame(phase_data)
    if not phase_df.empty:
        phase_df = phase_df.sort_values('Survival_Rate')
        
        # Create color gradient
        phase_colors = [
            f'rgb({int(255*(1-x))},{int(255*x)},{0})' 
            for x in phase_df['Survival_Rate']
        ]
        
        fig_survival_by_phase.add_trace(
            go.Bar(
                x=phase_df['Survival_Rate'],
                y=phase_df['Phase'],
                orientation='h',
                marker=dict(
                    color=phase_colors,
                    line=dict(width=1, color='darkgray')
                ),
                text=[f"{count} incidents" for count in phase_df['Count']],
                textposition='auto',
                hovertemplate='Survival rate: %{x:.1%}<br>%{text}<extra>%{y}</extra>'
            ),
            row=1, col=1
        )
        
        # Add reference line
        fig_survival_by_phase.add_shape(
            type="line",
            x0=0.5, y0=-0.5, x1=0.5, y1=len(phase_df)-0.5,
            line=dict(color="black", width=1, dash="dot"),
            row=1, col=1
        )
    
    # Accident type subplot
    # Calculate average survival rate for each accident type
    accident_data = []
    
    for accident in accident_types:
        accident_incidents = valid_df[valid_df[f'Accident_{accident}'] == True]
        if len(accident_incidents) > 0:
            avg_survival = accident_incidents['Survival_Rate'].mean()
            count = len(accident_incidents)
            accident_data.append({
                'Type': accident.replace('_', ' ').capitalize(),
                'Survival_Rate': avg_survival,
                'Count': count
            })
    
    accident_df = pd.DataFrame(accident_data)
    if not accident_df.empty:
        accident_df = accident_df.sort_values('Survival_Rate')
        
        # Create color gradient
        accident_colors = [
            f'rgb({int(255*(1-x))},{int(255*x)},{0})' 
            for x in accident_df['Survival_Rate']
        ]
        
        fig_survival_by_phase.add_trace(
            go.Bar(
                x=accident_df['Survival_Rate'],
                y=accident_df['Type'],
                orientation='h',
                marker=dict(
                    color=accident_colors,
                    line=dict(width=1, color='darkgray')
                ),
                text=[f"{count} incidents" for count in accident_df['Count']],
                textposition='auto',
                hovertemplate='Survival rate: %{x:.1%}<br>%{text}<extra>%{y}</extra>'
            ),
            row=1, col=2
        )
        
        # Add reference line
        fig_survival_by_phase.add_shape(
            type="line",
            x0=0.5, y0=-0.5, x1=0.5, y1=len(accident_df)-0.5,
            line=dict(color="black", width=1, dash="dot"),
            row=1, col=2
        )
    
    fig_survival_by_phase.update_layout(
        title=dict(
            text='Survival Rate Analysis by Flight Phase and Accident Type',
            font=dict(size=18, color='#2c3e50', family='Roboto, sans-serif'),
            x=0.5
        ),
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False
    )
    
    # Update axes
    fig_survival_by_phase.update_xaxes(
        title_text="Survival Rate", 
        tickformat='.0%', 
        range=[0, 1],
        row=1, col=1
    )
    fig_survival_by_phase.update_yaxes(title_text="Flight Phase", row=1, col=1)
    
    fig_survival_by_phase.update_xaxes(
        title_text="Survival Rate", 
        tickformat='.0%', 
        range=[0, 1],
        row=1, col=2
    )
    fig_survival_by_phase.update_yaxes(title_text="Accident Type", row=1, col=2)
    
    # FIGURE 4: Correlation Analysis
    # Create a correlation matrix to identify which factors correlate with survival
    # First, prepare numerical data
    numeric_cols = ['Survival_Rate', 'Aboard_Count', 'Year', 'Month']
    
    # Add binary columns for flight phases, weather conditions, and accident types
    binary_cols = []
    for phase in flight_phases:
        binary_cols.append(f'Phase_{phase}')
    for condition in weather_conditions:
        binary_cols.append(f'Weather_{condition}')
    for accident in accident_types:
        binary_cols.append(f'Accident_{accident}')
    
    binary_cols.append('Water_Crash')
    
    # Create correlation matrix
    corr_df = valid_df[numeric_cols + binary_cols].copy()
    correlation = corr_df.corr()
    
    # Create heatmap of correlations with survival rate
    survival_corr = correlation['Survival_Rate'].sort_values(ascending=False).drop('Survival_Rate')
    
    fig_correlation = go.Figure()
    
    # Add horizontal bars for correlations
    fig_correlation.add_trace(go.Bar(
        y=survival_corr.index,
        x=survival_corr.values,
        orientation='h',
        marker=dict(
            color=survival_corr.values,
            colorscale='RdBu',
            cmin=-0.3,
            cmax=0.3,
            line=dict(width=1, color='darkgray')
        ),
        hovertemplate='Correlation: %{x:.3f}<extra>%{y}</extra>'
    ))
    
    # Add reference line at 0
    fig_correlation.add_shape(
        type="line",
        x0=0, y0=-0.5, x1=0, y1=len(survival_corr)-0.5,
        line=dict(color="black", width=1, dash="dot")
    )
    
    # Update layout
    fig_correlation.update_layout(
        title=dict(
            text="Correlation of Various Factors with Survival Rate",
            font=dict(size=18, color='#2c3e50', family='Roboto, sans-serif'),
            x=0.5
        ),
        xaxis_title="Correlation Coefficient",
        yaxis_title="Factor",
        height=700,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig_survival_outcomes, fig_survival_by_type, fig_survival_by_phase, fig_correlation