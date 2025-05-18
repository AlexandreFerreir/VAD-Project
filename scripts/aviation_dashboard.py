# Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-05-17 15:09:06
# Current User's Login: JohnyPeters

import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import dash_bootstrap_components as dbc
import os
import base64
from datashader.bundling import hammer_bundle
import numpy as np

# Initialize the Dash app with a nice theme
app = dash.Dash(
    __name__, 
    suppress_callback_exceptions=True,
    external_stylesheets=[
        dbc.themes.FLATLY,  # Use a modern, clean theme
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.1/css/all.min.css"  # For icons
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
server = app.server  # Needed for deployment

# Define custom styles
CONTENT_STYLE = {
    "padding": "2rem 2rem",
    "background-color": "#f8f9fa"
}

CARD_STYLE = {
    "box-shadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
    "margin-bottom": "24px",
    "border-radius": "8px",
    "background-color": "white"
}

# Create assets directory if it doesn't exist
if not os.path.exists('assets'):
    os.makedirs('assets')

# Load your data
df = pd.read_csv("data/crashes_data/plane_crash_data.csv").replace('?', pd.NA)
coords_df = pd.read_csv("data/crashes_data/coordinates_cache.csv")

# Dictionary to map location -> coordinates
location_dict = dict(zip(coords_df["Location"], coords_df["Coordinates"]))

# Convert Coordinates from string "(lat, lon)" to tuple of floats
df["Coordinates"] = (
    df["Location"]
    .map(location_dict)
    .str.strip("()")
    .str.split(",", expand=True)
    .astype(float)
    .apply(tuple, axis=1)
)

# Extract date and year
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Coordinates", "Date"])
df[["lat", "lon"]] = pd.DataFrame(df["Coordinates"].tolist(), index=df.index)
df["Year"] = df["Date"].dt.year

# Extract fatality counts
df["Fatalities_Count"] = (
    df["Fatalities"]
    .str.extract(r"(\d+)")
    .astype(float)
)
df["Fatalities_Count"] = df["Fatalities_Count"].fillna(0).astype(int)

# Clean operator and aircraft type fields
df["Operator"] = df["Operator"].fillna("Unknown").replace(pd.NA, "Unknown")
df["AC Type"] = df["AC Type"].fillna("Unknown").replace(pd.NA, "Unknown")

try:
    with open("images/USA_crash_volume_densitiy.png", "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
except Exception as e:
    print(f"Error loading image: {str(e)}")
    encoded_image = None

# Load and encode the additional US visualization images
try:
    with open("images/top20_flight_routes_by_number_of_flights.png", "rb") as image_file:
        encoded_routes_image = base64.b64encode(image_file.read()).decode('utf-8')
except Exception as e:
    print(f"Error loading routes image: {str(e)}")
    encoded_routes_image = None

try:
    with open("images/US_flights_routes_volume_crashes.png", "rb") as image_file:
        encoded_us_routes_image = base64.b64encode(image_file.read()).decode('utf-8')
except Exception as e:
    print(f"Error loading US routes image: {str(e)}")
    encoded_us_routes_image = None

# Process route data for map
def parse_route_with_stops(route):
    """Extract all points from a route, including stopovers."""
    if pd.isna(route) or route == "?" or route in ["Demonstration", "Test flight", "Air show"]:
        return None
    
    # Remove quotes and clean
    route = str(route).replace('"', '').strip()
    
    # Check for slashes for alternative routes
    if '/' in route:
        # Take only the first mentioned route
        route = route.split('/')[0].strip()
    
    # Separate route points by hyphen
    waypoints = [point.strip() for point in route.split('-')]
    
    # Filter empty points
    waypoints = [wp for wp in waypoints if wp]
    
    # If there are at least two points, it's a valid route
    if len(waypoints) >= 2:
        return waypoints
    
    return None

# Generate route data for the map
def generate_routes_data():
    routes_data = []
    
    # Process each valid route
    for _, row in df.iterrows():
        if pd.notna(row['Route']):
            waypoints = parse_route_with_stops(row['Route'])
            
            if waypoints and len(waypoints) >= 2:
                # Look up coordinates for each point in the route
                waypoint_coords = []
                valid_route = True
                
                for point in waypoints:
                    if point in location_dict and location_dict[point] is not None:
                        # Extract coordinates with proper error handling
                        try:
                            coord_str = location_dict[point]
                            # Handle different possible formats of coordinates
                            if isinstance(coord_str, str):
                                # String format "(lat, lon)"
                                coord_str = coord_str.replace('(', '').replace(')', '')
                                if ',' in coord_str:
                                    try:
                                        lat, lon = map(float, coord_str.split(','))
                                        waypoint_coords.append([lat, lon])
                                    except:
                                        valid_route = False
                                        break
                                else:
                                    valid_route = False
                                    break
                            elif isinstance(coord_str, tuple):
                                # Tuple format (lat, lon)
                                waypoint_coords.append(list(coord_str))
                            elif isinstance(coord_str, list):
                                # List format [lat, lon]
                                waypoint_coords.append(coord_str)
                            else:
                                # Unknown format, skip this route
                                valid_route = False
                                break
                        except Exception as e:
                            print(f"Error processing coordinates for {point}: {str(e)}")
                            valid_route = False
                            break
                    else:
                        valid_route = False
                        break
                
                if valid_route and len(waypoint_coords) >= 2:
                    routes_data.append({
                        'route_text': row['Route'],
                        'waypoints': waypoint_coords
                    })
    
    return routes_data

# Get routes data for maps
routes_data = generate_routes_data()


# Function to create Folium map with edge bundled routes
def create_edge_bundled_map():    # Import required for edge bundling
    
    # Function to apply edge bundling to routes
    def apply_edge_bundling(routes_data):
        """
        Converts flight routes into a format suitable for edge bundling and applies the algorithm.
        Returns the bundled routes DataFrame.
        """
        
        # Create a dictionary of unique waypoints
        unique_locations = {}
        location_id_map = {}
        counter = 0

        # Extract all unique waypoints from routes
        for route_data in routes_data:
            if route_data['waypoints']:
                for lat, lon in route_data['waypoints']:
                    # Use a composite key as unique identifier
                    location_key = f"{lat:.6f}_{lon:.6f}"
                    if location_key not in unique_locations:
                        unique_locations[location_key] = (lat, lon)
                        location_id_map[location_key] = counter
                        counter += 1

        print(f"Identified {len(unique_locations)} unique waypoints")
        
        # Create nodes DataFrame (required format for hammer_bundle)
        nodes_data = []
        for loc_key, (lat, lon) in unique_locations.items():
            nodes_data.append([loc_key, lon, lat])  # Note: x=longitude, y=latitude for maps

        ds_nodes = pd.DataFrame(nodes_data, columns=['name', 'x', 'y'])

        # Create edges DataFrame (required format for hammer_bundle)
        edges_data = []
        for route_data in routes_data:
            if route_data['waypoints'] and len(route_data['waypoints']) >= 2:
                for i in range(len(route_data['waypoints']) - 1):
                    lat1, lon1 = route_data['waypoints'][i]
                    lat2, lon2 = route_data['waypoints'][i + 1]
                    
                    source_key = f"{lat1:.6f}_{lon1:.6f}"
                    target_key = f"{lat2:.6f}_{lon2:.6f}"
                    
                    source_id = location_id_map[source_key]
                    target_id = location_id_map[target_key]
                    
                    edges_data.append([source_id, target_id])

        ds_edges = pd.DataFrame(edges_data, columns=['source', 'target'])
        
        # For large datasets, adjust parameters
        if len(edges_data) > 10000:
            bundled_routes = hammer_bundle(ds_nodes, ds_edges, iterations=10)
        else:
            bundled_routes = hammer_bundle(ds_nodes, ds_edges)
    
        return bundled_routes, unique_locations

    # Function to add bundled routes to the map
    def add_bundled_routes_to_map(bundled_routes, accident_map):
        """
        Takes bundled routes data and adds them to the map
        Returns the feature group containing all routes
        """
        route_layer = folium.FeatureGroup(name="Flight Routes")

        # Convert bundled routes to segments
        bundled_np = bundled_routes.to_numpy()
        splits = (np.isnan(bundled_np[:,0])).nonzero()[0]

        start = 0
        segments = []
        for stop in splits:
            seg = bundled_np[start:stop, :]
            if len(seg) > 0:
                segments.append(seg)
            start = stop + 1


        # Add each bundled segment to the map
        routes_added = 0
        for i, seg in enumerate(segments):
            if len(seg) > 1:
                try:
                    # Convert coordinates (x,y) -> (lat,lon)
                    route_points = [(y, x) for x, y in seg]
                    
                    # Add polyline to map
                    folium.PolyLine(
                        locations=route_points,
                        color='blue',
                        weight=2,
                        opacity=0.7,
                        popup=f"Bundled Route"
                    ).add_to(route_layer)
                    
                    routes_added += 1
                except Exception as e:
                    print(f"Error processing bundled segment #{i}: {str(e)}")
                    continue

        return route_layer, routes_added

    accident_map = folium.Map(location=[30, 0], zoom_start=2, tiles='CartoDB positron')

    # Add marker cluster for accidents
    marker_cluster = MarkerCluster(name="Accidents").add_to(accident_map)

    # Counter for processed accidents
    accidents_added = 0

    for i, row in df.iterrows():
        if pd.notna(row['lat']) and pd.notna(row['lon']):
            try:
                lat, lon = row['lat'], row['lon']
                
                # Prepare popup text
                date_str = row['Date'] if pd.notna(row['Date']) else "Unknown date"
                location_str = row['Location'] if pd.notna(row['Location']) else "Unknown location"
                operator_str = row['Operator'] if pd.notna(row['Operator']) else "Unknown operator"
                fatalities_str = row['Fatalities'] if pd.notna(row['Fatalities']) else "Unknown fatalities"
                summary_str = row['Summary'] if pd.notna(row['Summary']) else "No summary available"
                
                popup_text = f"""
                <b>Date:</b> {date_str}<br>
                <b>Location:</b> {location_str}<br>
                <b>Operator:</b> {operator_str}<br>
                <b>Fatalities:</b> {fatalities_str}<br>
                <b>Summary:</b> {str(summary_str)[:200]}{"..." if len(str(summary_str)) > 200 else ""}
                """
                
                # Add marker to cluster - ALL in RED for accident locations
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(color='red', icon='plane', prefix='fa')
                ).add_to(marker_cluster)
                
                accidents_added += 1
            except Exception as e:
                print(f"Error processing accident #{i}: {str(e)}")
                continue

    # Apply edge bundling to routes
    bundled_routes, unique_waypoints = apply_edge_bundling(routes_data)

    # Add bundled routes to the map
    route_layer, routes_added = add_bundled_routes_to_map(bundled_routes, accident_map)

    # Add origin markers
    origin_layer = folium.FeatureGroup(name="Origin Points")
    origins_added = 0

    for route_data in routes_data:
        if route_data['waypoints'] and len(route_data['waypoints']) >= 2:
            try:
                # Add marker for origin point
                origin = route_data['waypoints'][0]
                route_text = route_data['route_text']
                origin_name = route_text.split('-')[0].strip() if '-' in route_text else "Origin"
                
                folium.Circle(
                    location=origin,
                    radius=10000,  # radius in meters (10km)
                    color='green',
                    fill=True,
                    fill_color='green',
                    fill_opacity=0.7,
                    popup=f"Origin: {origin_name}"
                ).add_to(origin_layer)
                
                origins_added += 1
            except Exception as e:
                print(f"Error adding origin marker: {str(e)}")
                continue

    # Add layers to map
    route_layer.add_to(accident_map)
    origin_layer.add_to(accident_map)

    # Add layer control
    folium.LayerControl().add_to(accident_map)

    # Add custom legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
                padding: 10px; border: 2px solid grey; border-radius: 5px">
        <h4>Legend</h4>
        <p><i class="fa fa-plane fa-1x" style="color:red"></i> Accident Location</p>
        <p><span style="background-color: green; height: 10px; width: 10px; border-radius: 50%; display: inline-block"></span> Origin/Stopover</p>
        <p><span style="background-color: blue; height: 2px; width: 50px; display: inline-block"></span> Bundled Flight Route</p>
    </div>
    '''

    # Add title to the map
    title_html = '''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index: 1000; 
                background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px; 
                text-align: center; font-family: Arial; font-size: 20px; font-weight: bold;">
        Global Aircraft Accident Map with Route Bundling
    </div>
    '''
    accident_map.get_root().html.add_child(folium.Element(title_html))
    accident_map.get_root().html.add_child(folium.Element(legend_html))

    # Save to HTML file for Dash to serve
    accident_map.save("assets/bundled_routes_map.html")
    
    return accidents_added, routes_added, origins_added

            

# Function to create Folium heat map with improved gradient and legend
def create_folium_heatmap():
    # Creation of a separate map just for the heatmap
    heatmap_only = folium.Map(location=[30, 0], zoom_start=2, tiles='CartoDB positron')

    # Simplified data for the heatmap - only using coordinates without weights
    heat_data = []
    for i, row in df.iterrows():
        if pd.notna(row['lat']) and pd.notna(row['lon']):
            try:
                lat, lon = row['lat'], row['lon']
                # Just add the coordinates without weight
                heat_data.append([lat, lon])
            except:
                continue

    # Define gradient for better visualization
    gradient = {
        '0.4': 'blue',
        '0.65': 'lime', 
        '0.8': 'yellow',
        '1': 'red'
    }

    # Add heatmap with optimized parameters
    HeatMap(
        heat_data,
        radius=12,           # Smaller radius to reduce bleeding effect
        blur=10,             # Less blur for better definition
        min_opacity=0.2,     # Lower minimum opacity to make low-density areas less prominent
        max_zoom=13,         # Control maximum zoom level for heat effect
        gradient=gradient    # Custom color gradient
    ).add_to(heatmap_only)

    # Add title to the map
    title_html = '''
    <h3 align="center" style="font-size:16px"><b>Heat Map - Aviation Accident Concentration</b></h3>
    '''
    heatmap_only.get_root().html.add_child(folium.Element(title_html))

    # Add legend to the heatmap
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 120px; height: 130px; 
                border: 2px solid grey; z-index: 9999; font-size: 14px;
                background-color: white; padding: 10px;
                border-radius: 5px;">
        <div style="text-align: center; margin-bottom: 5px;"><b>Density</b></div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 15px; height: 15px; background-color: blue; margin-right: 5px;"></div>
            <div>40%</div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 15px; height: 15px; background-color: lime; margin-right: 5px;"></div>
            <div>65%</div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 15px; height: 15px; background-color: yellow; margin-right: 5px;"></div>
            <div>80%</div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 15px; height: 15px; background-color: red; margin-right: 5px;"></div>
            <div>100%</div>
        </div>
    </div>
    '''
    heatmap_only.get_root().html.add_child(folium.Element(legend_html))

    # Save to HTML file for Dash to serve
    heatmap_only.save("assets/accident_heatmap.html")
    return len(heat_data)

# Add these callbacks to generate maps when needed
# 1. Function to initialize maps - this is the only one that should remain
def initialize_maps():
    """Initialize both maps at startup if they don't exist"""
    # Check and create assets directory if it doesn't exist
    if not os.path.exists('assets'):
        os.makedirs('assets')
    
    # Generate heat map
    if not os.path.exists("assets/accident_heatmap.html"):
        create_folium_heatmap()
    
    # Generate routes map
    if not os.path.exists("assets/bundled_routes_map.html"):
        create_edge_bundled_map()

# Call the initialization function once before defining the layout
initialize_maps()

@app.callback(
    [Output("routes-map-iframe", "srcDoc"), Output("heatmap-iframe", "srcDoc")],
    Input("tabs", "active_tab"),
    prevent_initial_call=False  # Allow call on initialization
)
def load_maps(active_tab):
    """Loads both maps when tab-2 is selected"""
    
    if active_tab != "tab-2":
        return "", ""
    
    routes_map_html = ""
    heatmap_html = ""
      # Load routes map
    try:
        if os.path.exists("assets/bundled_routes_map.html"):
            with open("assets/bundled_routes_map.html", "r", encoding="utf-8") as f:
                routes_map_html = f.read()
        else:
            routes_map_html = "<div style='color:red; text-align:center;'><h3>Error: Routes map not found</h3></div>"
    except Exception as e:
        print(f"Error loading routes map: {str(e)}")
        routes_map_html = f"<div style='color:red; text-align:center;'><h3>Error loading routes map</h3><p>{str(e)}</p></div>"
    
    # Load heat map
    try:
        if os.path.exists("assets/accident_heatmap.html"):
            with open("assets/accident_heatmap.html", "r", encoding="utf-8") as f:
                heatmap_html = f.read()
        else:
            heatmap_html = "<div style='color:red; text-align:center;'><h3>Error: Heatmap not found</h3></div>"
    except Exception as e:
        print(f"Error loading heatmap: {str(e)}")
        heatmap_html = f"<div style='color:red; text-align:center;'><h3>Error loading heatmap</h3><p>{str(e)}</p></div>"
    
    return routes_map_html, heatmap_html

# Create plotly scatter map for global visualization
def create_plotly_scatter_map():
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


# Create time series chart of accidents by year
def create_accidents_by_year():
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

# Function to create most dangerous routes chart
def create_most_dangerous_routes_chart():
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

# Function to create airline and aircraft analysis charts
def create_airline_aircraft_analysis():
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

# Function to create filtered animated map
def create_filtered_animated_map(selected_operators=None, selected_aircraft=None):
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

# Function to create US crashes analysis
def create_us_crashes_analysis():
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

# Function to create survival statistics visualizations
def create_survival_statistics():
    # Calculate survival statistics
    df['Total_Aboard'] = df['Aboard'].str.extract(r'(\d+)').astype(float)
    df_with_aboard = df.dropna(subset=['Total_Aboard', 'Fatalities_Count'])
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


# Function to create survival analysis charts for the "Survival Statistics" tab
def create_extended_survival_analysis():
    """
    Creates comprehensive survival analysis visualizations for the dashboard
    Returns multiple figures for different aspects of survival analysis
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


# Função para criar gráficos de análise de segurança das companhias aéreas para a nova aba "Airline Safety"
def create_airline_safety_analysis():
    """
    Creates airline safety analysis visualizations for the dashboard
    Returns multiple figures for different aspects of airline safety analysis
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

# Create app header
header = dbc.Navbar(
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

# Create stat cards for the overview page
def create_stat_card(title, value, icon, color):
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

# Define the app layout
app.layout = html.Div([
    # Add the header
    header,
    
    # Main content container
    dbc.Container([
        # Navigation tabs with custom styling
        dbc.Tabs(
            [
                dbc.Tab(label="Overview", tab_id="tab-1", 
                        label_style={"font-weight": "bold", "font-size": "16px"},
                        active_label_style={"color": "#2980b9"}),
                dbc.Tab(label="Crash Visualizations", tab_id="tab-2",  
                        label_style={"font-weight": "bold", "font-size": "16px"},
                        active_label_style={"color": "#2980b9"}),
                dbc.Tab(label="Animated Timeline", tab_id="tab-3",  # New separate tab for the animated timeline
                        label_style={"font-weight": "bold", "font-size": "16px"},
                        active_label_style={"color": "#2980b9"}),
                dbc.Tab(label="USA Crashes & Volume", tab_id="tab-4", 
                        label_style={"font-weight": "bold", "font-size": "16px"},
                        active_label_style={"color": "#2980b9"}),
                dbc.Tab(label="Survival Statistics", tab_id="tab-5",
                        label_style={"font-weight": "bold", "font-size": "16px"},
                        active_label_style={"color": "#2980b9"}),
                dbc.Tab(label="Airline Safety", tab_id="tab-7",  
                        label_style={"font-weight": "bold", "font-size": "16px"},
                        active_label_style={"color": "#2980b9"}),
                dbc.Tab(label="Statistics", tab_id="tab-6",  
                        label_style={"font-weight": "bold", "font-size": "16px"},
                        active_label_style={"color": "#2980b9"}),
            ],
            id="tabs",
            active_tab="tab-1",
            className="mb-4"
        ),
        
        # Content for each tab
        html.Div(id="tabs-content", style=CONTENT_STYLE)
    ], fluid=True),
    
    # Footer
    html.Footer(
        dbc.Container(
            [
                html.Hr(),
                html.P(
                    [
                        "Aviation Accident Dashboard © 2025 | ",
                        html.A("Source Code", href="#", className="text-decoration-none"),
                        " | Created by Joao Tinoco and Alexandre Ferreira"
                    ],
                    className="text-center text-muted"
                )
            ],
            fluid=True
        ),
        className="mt-4 py-3"
    )
])

# Callback to update tab content
@app.callback(
    Output("tabs-content", "children"),
    Input("tabs", "active_tab")
)
def render_content(tab):
    if tab == "tab-1":
        # Overview tab with stat cards and visualizations
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
                            dcc.Graph(figure=create_plotly_scatter_map(), config={'displayModeBar': False})
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
                            dcc.Graph(figure=create_accidents_by_year(), config={'displayModeBar': False})
                        ])
                    ], style=CARD_STYLE)
                ])
            ]),
        ])
    
    elif tab == "tab-2":
        # Crash Visualizations tab - showing both visualizations in sequence
        return html.Div([            # Main title
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
                                f"Heat map showing concentration of {len(df):,} accident locations. "
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
    
    elif tab == "tab-3":
        # Animated Timeline tab
        top_operators = ["All Airlines"] + list(df["Operator"].value_counts().head(30).index)
        top_aircraft = ["All Aircraft Types"] + list(df["AC Type"].value_counts().head(30).index)
        
        return html.Div([
            dbc.Card([
                dbc.CardHeader(html.H4("Animated Aviation Accidents Timeline", className="text-center")),
                dbc.CardBody([
                    # Filter controls
                    dbc.Row([
                        # Airline dropdown
                        dbc.Col([
                            html.Label("Select Airlines:", className="fw-bold"),
                            dcc.Dropdown(
                                id="operator-dropdown",
                                options=[{"label": op, "value": op} for op in top_operators],
                                value=["All Airlines"],
                                multi=True,
                                className="mb-3"
                            )
                        ], width=5),
                        
                        # Aircraft dropdown
                        dbc.Col([
                            html.Label("Select Aircraft Types:", className="fw-bold"),
                            dcc.Dropdown(
                                id="aircraft-dropdown",
                                options=[{"label": ac, "value": ac} for ac in top_aircraft],
                                value=["All Aircraft Types"],
                                multi=True,
                                className="mb-3"
                            )
                        ], width=5),
                        
                        # Update button
                        dbc.Col([
                            html.Div([
                                dbc.Button("Update Map", id="update-map-button", color="primary", className="mt-4")
                            ], className="d-flex justify-content-center")
                        ], width=2)
                    ]),
                    
                    # Animated map container
                    html.Div(
                        dcc.Loading(
                            id="loading-animation",
                            type="circle",
                            children=html.Div(id="animated-map-container")
                        )
                    )
                ])
            ], style=CARD_STYLE)
        ])
    
    elif tab == "tab-4":
        # USA Crashes & Volume tab
        _, fig_timeline, fig_operators = create_us_crashes_analysis()
        
        # Calculate total US crashes and fatalities
        us_df = df[df['Location'].str.contains('United States|USA|U.S.A.|US|U.S.', case=False, na=False)]
        total_us_crashes = len(us_df)
        total_us_fatalities = us_df['Fatalities_Count'].sum()
        
        return html.Div([
            # Title row
            dbc.Row([
                dbc.Col(html.H3("US Aviation Accidents Analysis", className="text-center mb-4"))
            ]),

            dbc.Row([
                dbc.Col(create_stat_card("Total US Crashes", f"{total_us_crashes:,}", "fas fa-flag-usa", "#3498db"), width=3),
                dbc.Col(create_stat_card("US Fatalities", f"{total_us_fatalities:,}", "fas fa-skull-crossbones", "#e74c3c"), width=3),
                dbc.Col(create_stat_card("% of Global Crashes", f"{(total_us_crashes/len(df)*100):.1f}%", "fas fa-globe-americas", "#2c3e50"), width=3),
                dbc.Col(create_stat_card("Date Range", f"1987 - 2008", "fas fa-calendar-alt", "#3498db"), width=3),
            ], className="mb-4"),
            
            # Risk map row - using the static image
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("US Accident Risk Analysis", className="text-center")),
                        dbc.CardBody([
                            # Use base64 encoded image if available
                            html.Img(
                                src=f"data:image/png;base64,{encoded_image}" if encoded_image else "",
                                style={"width": "100%", "height": "auto"},
                                className="img-fluid"
                            ) if encoded_image else html.Div(
                                "Image could not be loaded. Please check that 'images/USA_crash_volume_densitiy.png' exists.",
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
                                src=f"data:image/png;base64,{encoded_routes_image}" if encoded_routes_image else "",
                                style={"width": "100%", "height": "auto"},
                                className="img-fluid"
                            ) if encoded_routes_image else html.Div(
                                "Image could not be loaded. Please check that 'images/top20_flight_routes_by_number_of_flights.png' exists.",
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
                                src=f"data:image/png;base64,{encoded_us_routes_image}" if encoded_us_routes_image else "",
                                style={"width": "100%", "height": "auto"},
                                className="img-fluid"
                            ) if encoded_us_routes_image else html.Div(
                                "Image could not be loaded. Please check that 'images/US_flights_routes_volume_crashes.png' exists.",
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
    
    elif tab == "tab-5":
        # Enhanced Survival Statistics tab with new visualizations
        fig_original_survival, fig_original_survival_trend, fig_original_aircraft_survival = create_survival_statistics()
        fig_survival_outcomes, fig_survival_by_type, fig_survival_by_phase, fig_correlation = create_extended_survival_analysis()
        
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
    
    elif tab == "tab-6":
        # New Statistics tab with airline, aircraft, and route analysis
        return html.Div([
            # Title row
            dbc.Row([
                dbc.Col(html.H3("Advanced Aviation Accident Analysis", className="text-center mb-4"))
            ]),
            
            # First row - Most dangerous routes chart
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("World's Most Dangerous Commercial Routes", className="text-center")),
                        dbc.CardBody([
                            dcc.Graph(figure=create_most_dangerous_routes_chart(), config={'displayModeBar': True})
                        ])
                    ], style=CARD_STYLE)
                ])
            ], className="mb-4"),
            
            # Second row - Airline and aircraft analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Analysis by Airline and Aircraft Type", className="text-center")),
                        dbc.CardBody([
                            dcc.Graph(figure=create_airline_aircraft_analysis(), config={'displayModeBar': True})
                        ])
                    ], style=CARD_STYLE)
                ])
            ])
        ])
    elif tab == "tab-7":
        # New Airline Safety tab
        fig_airline_fatality, fig_fatality_by_factor, fig_airline_patterns = create_airline_safety_analysis()
        
        return html.Div([
            # Title row
            dbc.Row([
                dbc.Col(html.H3("Airline Safety Analysis", className="text-center mb-4"))
            ]),
            
            # First row - Fatality Rate by Airline
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Fatality Rate by Airline (Top 20 Safest)", className="text-center")),
                        dbc.CardBody([
                            dcc.Graph(figure=fig_airline_fatality, config={'displayModeBar': False})
                        ])
                    ], style=CARD_STYLE)
                ])
            ], className="mb-4"),
            
            # Second row - Fatality Rate by Incident Type and Flight Phase
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Fatality Analysis by Incident Type and Flight Phase", className="text-center")),
                        dbc.CardBody([
                            dcc.Graph(figure=fig_fatality_by_factor, config={'displayModeBar': False})
                        ])
                    ], style=CARD_STYLE)
                ])
            ], className="mb-4"),
            
            # Third row - Incident Types and Flight Phases by Airline
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Airline-Specific Incident Patterns", className="text-center")),
                        dbc.CardBody([
                            dcc.Graph(figure=fig_airline_patterns, config={'displayModeBar': False})
                        ])
                    ], style=CARD_STYLE)
                ])
            ])
        ])

# Callback to update animated map based on selections
@app.callback(
    Output("animated-map-container", "children"),
    Input("update-map-button", "n_clicks"),
    State("operator-dropdown", "value"),
    State("aircraft-dropdown", "value")
)
def update_animated_map(n_clicks, selected_operators, selected_aircraft):
    # Initialize with default values if needed
    if not selected_operators:
        selected_operators = ["All Airlines"]
    if not selected_aircraft:
        selected_aircraft = ["All Aircraft Types"]
    
    # Create the animated map
    fig = create_filtered_animated_map(selected_operators, selected_aircraft)
    
    # Return as a Plotly graph
    return dcc.Graph(
        figure=fig, 
        config={'displayModeBar': True},
        style={"height": "700px"}
    )

# # Run the app
# if __name__ == "__main__":
#     app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8052)))

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=8052)