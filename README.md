# VAD-Project

# Aviation Accidents Dashboard

## Overview

This interactive dashboard visualizes historical aviation accident data from around the world. The application provides various visualizations including interactive maps, time series analyses, airline comparisons, and survival statistics. It's built with Python, Dash, and Plotly, offering an intuitive interface for exploring aviation safety trends.

## Features

- **Global Accident Maps**: Interactive maps showing accident locations with edge bundling for flight routes and heat maps for accident density
- **Time-Based Analysis**: Animated timeline of accidents by year with filtering options
- **US-Specific Analysis**: Dedicated visualizations for US aviation accidents and flight volume
- **Survival Statistics**: Detailed analysis of survival rates by aircraft type, airline, and accident conditions
- **Airline Safety Comparison**: Airline safety rankings and incident patterns
- **Statistical Breakdowns**: Most dangerous routes, aircraft types, and airlines

## Project Structure


VAD-Project/ 
├── aviation_accidents_dashboard/ (Main application code) 
│ ├── app.py # Main application entry point 
│ ├── config.py # Configuration and constants 
│ ├── data_loader.py # Data loading and preparation 
│ ├── layouts/ # UI layout components
│       ├── init.py 
│       ├── header.py # Dashboard header 
│       ├── overview.py # Overview tab layout 
│       ├── crash_vis.py # Crash visualizations tab layout 
│       ├── animated_timeline.py # Animated timeline tab layout 
│       ├── usa_crashes.py # USA crashes tab layout 
│       ├── survival_stats.py # Survival statistics tab layout 
│       ├── airline_safety.py # Airline safety tab layout 
│       └── statistics.py # Statistics tab layout 
│ ├── callbacks/ # Dashboard interactivity
│       ├── init.py 
│       ├── map_callbacks.py # Callbacks for maps
│       └── filter_callbacks.py # Callbacks for filters 
│ └── visualizations/ # Visualization functions 
│       ├── init.py 
│       ├── maps.py # Folium maps creation 
│       ├── time_series.py # Time series charts 
│       ├── scatter_maps.py # Plotly scatter maps 
│       ├── airline_analysis.py # Airline analysis charts 
│       ├── survival_analysis.py # Survival analysis charts 
│       └── us_analysis.py # US-specific analysis 
├── data/ # Data files 
│     └── crashes_data/ 
│           ├── plane_crash_data.csv # Main crash data 
│           └── coordinates_cache.csv # Geocoded locations 
├── images/ # Static images for analyses 
│     ├── USA_crash_volume_densitiy.png 
│     ├── top20_flight_routes_by_number_of_flights.png 
│     └── US_flights_routes_volume_crashes.png 
└── assets/ # Generated assets like HTML maps 
      ├── accident_heatmap.html 
      └── bundled_routes_map.html



## Detailed Module Descriptions

### Main Files

- **app.py**: The entry point of the application. It initializes the Dash app, loads data, registers callbacks, and defines the main layout structure with tabs.

- **config.py**: Contains configuration settings, path definitions, styling constants, and color schemes used throughout the application.

- **data_loader.py**: Handles loading and preprocessing of the crash data and coordinates, as well as loading images and generating route data.

### Layouts

- **header.py**: Defines the header component with navigation links.

- **overview.py**: Creates the overview dashboard with summary statistics and global accident visualization.

- **crash_vis.py**: Implements the crash visualizations tab with edge bundled routes and heatmaps.

- **animated_timeline.py**: Creates an animated timeline of accidents with filtering capabilities.

- **usa_crashes.py**: Builds the USA-specific analyses including accident density maps and route visualizations.

- **survival_stats.py**: Implements survival statistics analyses with multi-dimensional survival rate breakdowns.

- **airline_safety.py**: Creates airline safety comparisons and patterns of incidents by airline.

- **statistics.py**: Implements statistical breakdowns of routes, airlines, and aircraft types.

### Callbacks

- **map_callbacks.py**: Contains callback functions for updating and interacting with maps.

- **filter_callbacks.py**: Contains callbacks for filtering data and updating visualizations based on user selections.

### Visualizations

- **maps.py**: Functions for creating Folium maps with edge bundling and heatmaps.

- **time_series.py**: Functions for creating time-based visualizations of accidents and fatalities.

- **scatter_maps.py**: Functions for creating interactive scatter maps with Plotly, including the animated timeline.

- **airline_analysis.py**: Functions for airline-specific analyses, including dangerous routes and airline comparisons.

- **survival_analysis.py**: Functions related to survival rate analysis across various dimensions.

- **us_analysis.py**: US-specific analysis functions including route volume and regional breakdowns.