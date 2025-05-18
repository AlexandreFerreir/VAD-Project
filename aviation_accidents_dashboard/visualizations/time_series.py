"""
Time series visualization functions
"""
import plotly.graph_objects as go
import pandas as pd

def create_accidents_by_year(df):
    """
    Create time series chart of accidents by year
    
    Args:
        df: DataFrame with accident data
        
    Returns:
        Figure: Plotly figure object
    """
    yearly_counts = df.groupby('Year').size().reset_index(name='Accidents')
    yearly_fatalities = df.groupby('Year')['Fatalities_Count'].sum().reset_index(name='Fatalities')
    
    # Merge counts and fatalities
    yearly_data = pd.merge(yearly_counts, yearly_fatalities, on='Year')
    
    # Create two y-axes plot
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(
        go.Bar(
            x=yearly_data['Year'],
            y=yearly_data['Accidents'],
            name='Accidents',
            marker_color='#3498db'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=yearly_data['Year'],
            y=yearly_data['Fatalities'],
            name='Fatalities',
            marker_color='#e74c3c',
            mode='lines+markers',
            yaxis='y2'
        )
    )
    
    # Set up layout with two y-axes
    fig.update_layout(
        title={
            'text': 'Accidents and Fatalities by Year',
            'font': {'size': 20, 'color': '#2c3e50', 'family': 'Roboto, sans-serif'},
            'x': 0.5
        },
        xaxis_title='Year',
        yaxis=dict(
            title='Number of Accidents',
            title_font=dict(color='#3498db'),
            tickfont=dict(color='#3498db')
        ),
        yaxis2=dict(
            title='Number of Fatalities',
            title_font=dict(color='#e74c3c'),
            tickfont=dict(color='#e74c3c'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=60, r=60, t=80, b=60),
        height=450,
        paper_bgcolor='white',
        plot_bgcolor='#f8f9fa',
        hovermode='x unified'
    )
    
    return fig