"""
Functions for creating maps with Folium
"""

import folium
from folium.plugins import MarkerCluster, HeatMap
import pandas as pd
import numpy as np
from datashader.bundling import hammer_bundle
import os
from aviation_accidents_dashboard.config import HEATMAP_HTML_PATH, ROUTES_MAP_HTML_PATH, ASSETS_DIR

def create_folium_heatmap(df):
    """
    Create a heat map of accident locations
    
    Args:
        df: DataFrame with accident data
        
    Returns:
        int: Number of points added to the heatmap
    """
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

    heatmap_only.save(HEATMAP_HTML_PATH)
    return len(heat_data)

def create_edge_bundled_map(df, routes_data):
    """
    Create a map with edge bundled flight routes
    
    Args:
        df: DataFrame with accident data
        routes_data: List of route data dictionaries
        
    Returns:
        tuple: Tuple containing counts of accidents, routes, and origins added
    """
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

    # Start creating the map
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
    accident_map.save(ROUTES_MAP_HTML_PATH)
    
    return accidents_added, routes_added, origins_added