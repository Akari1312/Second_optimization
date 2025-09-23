import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import networkx as nx
from geopy.distance import geodesic
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Load the datasets
print("Loading railway datasets for optimization analysis...")

# Load stations
with open('archive/stations.json', 'r') as f:
    stations_data = json.load(f)

# Load trains
with open('archive/trains.json', 'r') as f:
    trains_data = json.load(f)

# Load schedules
with open('archive/schedules.json', 'r') as f:
    schedules_data = json.load(f)

# Convert to DataFrames
stations_list = []
for feature in stations_data['features']:
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

trains_list = []
for feature in trains_data['features']:
    train = feature['properties'].copy()
    train['route_coordinates'] = feature['geometry']['coordinates']
    train['num_stops'] = len(feature['geometry']['coordinates'])
    trains_list.append(train)

trains_df = pd.DataFrame(trains_list)
schedules_df = pd.DataFrame(schedules_data)

print("="*60)
print("RAILWAY NETWORK OPTIMIZATION ANALYSIS")
print("="*60)

# 1. ROUTE EFFICIENCY ANALYSIS
print("\n1. ROUTE EFFICIENCY ANALYSIS")
print("-" * 40)

# Calculate actual vs direct distance for efficiency
def calculate_route_efficiency(trains_df, stations_df):
    efficiency_data = []

    for _, train in trains_df.iterrows():
        if train['route_coordinates'] and len(train['route_coordinates']) >= 2:
            # Get start and end coordinates
            start_coord = train['route_coordinates'][0]
            end_coord = train['route_coordinates'][-1]

            # Calculate direct distance
            direct_distance = geodesic(
                (start_coord[1], start_coord[0]),
                (end_coord[1], end_coord[0])
            ).kilometers

            # Calculate route efficiency (direct distance / actual distance)
            if train['distance'] > 0:
                efficiency = direct_distance / train['distance']

                efficiency_data.append({
                    'train_number': train['number'],
                    'train_name': train['name'],
                    'type': train['type'],
                    'actual_distance': train['distance'],
                    'direct_distance': direct_distance,
                    'efficiency': efficiency,
                    'detour_km': train['distance'] - direct_distance,
                    'num_stops': train['num_stops']
                })

    return pd.DataFrame(efficiency_data)

route_efficiency_df = calculate_route_efficiency(trains_df, stations_df)

print(f"Average route efficiency: {route_efficiency_df['efficiency'].mean():.3f}")
print(f"Best route efficiency: {route_efficiency_df['efficiency'].max():.3f}")
print(f"Worst route efficiency: {route_efficiency_df['efficiency'].min():.3f}")

# Most inefficient routes (optimization candidates)
inefficient_routes = route_efficiency_df.nsmallest(10, 'efficiency')
print(f"\nTop 10 Most Inefficient Routes (Optimization Candidates):")
for _, route in inefficient_routes.iterrows():
    print(f"- {route['train_name']} ({route['train_number']}): "
          f"{route['efficiency']:.3f} efficiency, "
          f"{route['detour_km']:.1f}km detour")

# 2. SCHEDULING GAP ANALYSIS
print("\n\n2. SCHEDULING GAP ANALYSIS")
print("-" * 40)

# Convert schedule times to datetime for analysis
schedules_df['departure_time'] = pd.to_datetime(schedules_df['departure'], format='%H:%M', errors='coerce')
schedules_df['arrival_time'] = pd.to_datetime(schedules_df['arrival'], format='%H:%M', errors='coerce')

# Analyze frequency of trains between major station pairs
def analyze_scheduling_gaps():
    # Group by route (origin-destination pairs)
    route_frequency = defaultdict(list)

    for train_num in schedules_df['train_number'].unique():
        train_schedule = schedules_df[schedules_df['train_number'] == train_num].sort_values('day')
        if len(train_schedule) >= 2:
            origin = train_schedule.iloc[0]['station_code']
            destination = train_schedule.iloc[-1]['station_code']
            route_key = f"{origin}->{destination}"
            route_frequency[route_key].append(train_num)

    # Find routes with low frequency (gap opportunities)
    route_analysis = []
    for route, trains in route_frequency.items():
        if len(trains) >= 1:  # Only consider routes with at least 1 train
            route_analysis.append({
                'route': route,
                'train_count': len(trains),
                'trains': trains
            })

    route_df = pd.DataFrame(route_analysis)
    return route_df.sort_values('train_count')

route_frequency_df = analyze_scheduling_gaps()

print(f"Total unique routes: {len(route_frequency_df)}")
print(f"Routes with only 1 train: {len(route_frequency_df[route_frequency_df['train_count'] == 1])}")
print(f"Routes with 2-3 trains: {len(route_frequency_df[route_frequency_df['train_count'].between(2, 3)])}")

print(f"\nRoutes with potential for additional services (only 1 train):")
single_train_routes = route_frequency_df[route_frequency_df['train_count'] == 1].head(10)
for _, route in single_train_routes.iterrows():
    print(f"- {route['route']} (Train: {route['trains'][0]})")

# 3. CAPACITY UTILIZATION ANALYSIS
print("\n\n3. CAPACITY UTILIZATION ANALYSIS")
print("-" * 40)

# Analyze train types and their distribution
train_type_analysis = trains_df.groupby('type').agg({
    'distance': ['count', 'mean', 'sum'],
    'duration_h': 'mean',
    'num_stops': 'mean'
}).round(2)

train_type_analysis.columns = ['count', 'avg_distance', 'total_distance', 'avg_duration', 'avg_stops']
print("Train Type Distribution:")
print(train_type_analysis)

# Station utilization analysis
station_usage = schedules_df['station_code'].value_counts()
print(f"\nStation Utilization:")
print(f"Most utilized station: {station_usage.index[0]} ({station_usage.iloc[0]} train visits)")
print(f"Average station visits: {station_usage.mean():.1f}")
print(f"Stations with < 10 visits: {len(station_usage[station_usage < 10])}")

underutilized_stations = station_usage[station_usage < 5]
print(f"\nUnderutilized Stations (< 5 train visits):")
for station, count in underutilized_stations.head(10).items():
    station_name = stations_df[stations_df['code'] == station]['name'].iloc[0] if len(stations_df[stations_df['code'] == station]) > 0 else "Unknown"
    print(f"- {station} ({station_name}): {count} visits")

# 4. OPTIMIZATION RECOMMENDATIONS
print("\n\n4. OPTIMIZATION RECOMMENDATIONS")
print("-" * 40)

print("ROUTE OPTIMIZATION:")
print("- Consider rerouting the 10 most inefficient trains to reduce travel time")
print(f"- Potential distance savings: {inefficient_routes['detour_km'].sum():.1f} km")
print(f"- Estimated time savings: {(inefficient_routes['detour_km'].sum() / 50):.1f} hours")

print("\nSCHEDULE OPTIMIZATION:")
print(f"- Add services to {len(route_frequency_df[route_frequency_df['train_count'] == 1])} single-train routes")
print("- Consider increasing frequency on popular routes")

print("\nCAPACITY OPTIMIZATION:")
print(f"- Focus on {len(underutilized_stations)} underutilized stations")
print("- Consider connecting underutilized stations to main network")
print(f"- Passenger trains (Pass) dominate with {train_type_analysis.loc['Pass', 'count']} services")

# 5. NETWORK CONNECTIVITY ANALYSIS
print("\n\n5. NETWORK CONNECTIVITY ANALYSIS")
print("-" * 40)

# Create network graph
G = nx.Graph()

# Add stations as nodes
for _, station in stations_df.iterrows():
    G.add_node(station['code'],
               name=station['name'],
               state=station['state'],
               zone=station['zone'])

# Add train routes as edges
for _, train in trains_df.iterrows():
    route_coords = train['route_coordinates']
    if route_coords and len(route_coords) >= 2:
        # Find stations closest to route coordinates
        for i in range(len(route_coords) - 1):
            # For simplicity, we'll connect consecutive route points
            # In a real scenario, you'd match coordinates to actual stations
            pass

# Analyze connectivity by state
state_connectivity = stations_df.groupby('state').agg({
    'code': 'count',
    'zone': 'nunique'
}).rename(columns={'code': 'station_count', 'zone': 'zones_covered'})

print("State Connectivity Analysis:")
print(state_connectivity.sort_values('station_count', ascending=False).head(10))

print("\n" + "="*60)
print("ANALYSIS COMPLETE - OPTIMIZATION OPPORTUNITIES IDENTIFIED")
print("="*60)