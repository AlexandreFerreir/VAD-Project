"""
US crash analysis visualization functions
"""
import plotly.express as px
import plotly.graph_objects as go

def create_us_crashes_analysis(df):
    """
    Create US crashes analysis visualizations
    
    Args:
        df: DataFrame with accident data
        
    Returns:
        tuple: Tuple of Plotly figure objects
    """
    # Filter for US crashes
    us_df = df[df['Location'].str.contains('United States|USA|U.S.A.|US|U.S.', case=False, na=False)]
    
    # Volume by decade chart
    us_df['Decade'] = (us_df['Year'] // 10) * 10
    decade_counts = us_df.groupby('Decade').size().reset_index(name='Accidents')
    
    # Fixed: explicitly name the sum column as 'Fatalities'
    decade_fatalities = us_df.groupby('Decade')['Fatalities_Count'].sum().reset_index(name='Fatalities')
    
    fig_timeline = go.Figure()
    
    fig_timeline.add_trace(go.Bar(
        x=decade_counts['Decade'],
        y=decade_counts['Accidents'],
        name='Accidents',
        marker_color='#3498db'
    ))
    
    fig_timeline.add_trace(go.Scatter(
        x=decade_fatalities['Decade'],
        y=decade_fatalities['Fatalities'],  # Using the correct column name
        name='Fatalities',
        marker_color='#e74c3c',
        mode='lines+markers',
        yaxis='y2'
    ))
    
    fig_timeline.update_layout(
        title=dict(
            text='US Aviation Accidents by Decade',
            font=dict(size=18, color='#2c3e50', family='Roboto, sans-serif'),
            x=0.5
        ),
        xaxis_title='Decade',
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
        height=400
    )
    
    # Distribution by operator - Top 10 operators
    top_us_operators = us_df['Operator'].value_counts().head(10).reset_index()
    top_us_operators.columns = ['Operator', 'Accidents']
    
    fig_operators = px.bar(
        top_us_operators,
        y='Operator',
        x='Accidents',
        orientation='h',
        title='Top 10 Airlines by Accidents in the US',
        color='Accidents',
        color_continuous_scale='Blues',
        text='Accidents'
    )
    
    fig_operators.update_traces(texttemplate='%{text}', textposition='outside')
    
    fig_operators.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        title_font=dict(size=18, color='#2c3e50', family='Roboto, sans-serif'),
        title_x=0.5,
        yaxis=dict(title=''),
        xaxis=dict(title='Number of Accidents'),
    )
    
    return None, fig_timeline, fig_operators  # Return None for the risk map, we'll use the image directly