import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import networkx as nx
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

# Load the datasets
print("Loading railway datasets...")

# Load stations
with open('archive/stations.json', 'r') as f:
    stations_data = json.load(f)

# Load trains  
with open('archive/trains.json', 'r') as f:
    trains_data = json.load(f)

# Load schedules
with open('archive/schedules.json', 'r') as f:
    schedules_data = json.load(f)

print(f"Loaded {len(stations_data['features'])} stations")
print(f"Loaded {len(trains_data['features'])} trains") 
print(f"Loaded {len(schedules_data)} schedule entries")

# Convert to DataFrames for easier analysis
print("\nConverting to DataFrames...")

# Stations DataFrame
stations_list = []
for feature in stations_data['features']:
    # Skip stations with missing geometry
    if feature['geometry'] is None or feature['geometry']['coordinates'] is None:
        continue
    
    station = {
        'code': feature['properties']['code'],
        'name': feature['properties']['name'],
        'state': feature['properties']['state'],
        'zone': feature['properties']['zone'],
        'address': feature['properties']['address'],
        'longitude': feature['geometry']['coordinates'][0],
        'latitude': feature['geometry']['coordinates'][1]
    }
    stations_list.append(station)

stations_df = pd.DataFrame(stations_list)

# Trains DataFrame
trains_list = []
for feature in trains_data['features']:
    train = feature['properties'].copy()
    # Add route coordinates
    train['route_coordinates'] = feature['geometry']['coordinates']
    train['num_stops'] = len(feature['geometry']['coordinates'])
    trains_list.append(train)

trains_df = pd.DataFrame(trains_list)

# Schedules DataFrame
schedules_df = pd.DataFrame(schedules_data)

print("Data loading complete!")
print(f"\nStations DataFrame shape: {stations_df.shape}")
print(f"Trains DataFrame shape: {trains_df.shape}")
print(f"Schedules DataFrame shape: {schedules_df.shape}")

# Basic statistics
print("\n" + "="*50)
print("BASIC DATASET STATISTICS")
print("="*50)

print(f"\nSTATIONS:")
print(f"- Total stations: {len(stations_df)}")
print(f"- States covered: {stations_df['state'].nunique()}")
print(f"- Railway zones: {stations_df['zone'].nunique()}")
print(f"- Top 5 states by station count:")
print(stations_df['state'].value_counts().head())

print(f"\nTRAINS:")
print(f"- Total trains: {len(trains_df)}")
print(f"- Average route distance: {trains_df['distance'].mean():.1f} km")
print(f"- Average journey duration: {trains_df['duration_h'].mean():.1f} hours")
print(f"- Train types:")
print(trains_df['type'].value_counts().head())

print(f"\nSCHEDULES:")
print(f"- Total schedule entries: {len(schedules_df)}")
print(f"- Unique trains in schedules: {schedules_df['train_number'].nunique()}")
print(f"- Unique stations in schedules: {schedules_df['station_code'].nunique()}")