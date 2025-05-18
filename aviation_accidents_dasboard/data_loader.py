"""
Data loading and preparation for aviation dashboard
"""
import pandas as pd
import numpy as np
import os
import base64
from config import CRASH_DATA_PATH, COORDINATES_PATH, ASSETS_DIR, USA_CRASH_DENSITY_IMG, TOP_ROUTES_IMG, US_ROUTES_IMG

def load_data():
    """
    Load and prepare data for the aviation dashboard
    
    Returns:
        tuple: Tuple containing dataframes and processed data
    """
    # Load your data
    df = pd.read_csv(CRASH_DATA_PATH).replace('?', pd.NA)
    coords_df = pd.read_csv(COORDINATES_PATH)

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
    
    # Load additional images
    encoded_images = load_images()
    
    # Process route data
    routes_data = generate_routes_data(df, location_dict)
    
    return df, location_dict, routes_data, encoded_images

def load_images():
    """
    Load and encode images for visualizations
    
    Returns:
        dict: Dictionary of encoded images
    """
    images = {
        'usa_crash_density': None,
        'top_routes': None,
        'us_routes': None
    }
    
    # USA crash volume density image
    try:
        with open(USA_CRASH_DENSITY_IMG, "rb") as image_file:
            images['usa_crash_density'] = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error loading USA crash density image: {str(e)}")
    
    # Top 20 flight routes image
    try:
        with open(TOP_ROUTES_IMG, "rb") as image_file:
            images['top_routes'] = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error loading top routes image: {str(e)}")
    
    # US flights routes volume image
    try:
        with open(US_ROUTES_IMG, "rb") as image_file:
            images['us_routes'] = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error loading US routes image: {str(e)}")
    
    return images

def parse_route_with_stops(route):
    """
    Extract all points from a route, including stopovers.
    
    Args:
        route: Route string to parse
        
    Returns:
        list: List of waypoints or None if invalid
    """
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

def generate_routes_data(df, location_dict):
    """
    Generate route data for the map
    
    Args:
        df: DataFrame with accident data
        location_dict: Dictionary mapping locations to coordinates
        
    Returns:
        list: List of route data dictionaries
    """
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