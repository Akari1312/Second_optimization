import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("Loading railway datasets for quick optimization analysis...")

# Load datasets
with open('archive/stations.json', 'r') as f:
    stations_data = json.load(f)

with open('archive/trains.json', 'r') as f:
    trains_data = json.load(f)

with open('archive/schedules.json', 'r') as f:
    schedules_data = json.load(f)

print("="*60)
print("RAILWAY OPTIMIZATION ANALYSIS - QUICK INSIGHTS")
print("="*60)

# Basic stats
print(f"Dataset size: {len(stations_data['features'])} stations, {len(trains_data['features'])} trains, {len(schedules_data)} schedules")

# 1. TRAIN TYPE OPTIMIZATION ANALYSIS
print("\n1. TRAIN TYPE & EFFICIENCY ANALYSIS")
print("-" * 40)

trains_analysis = []
for feature in trains_data['features'][:1000]:  # Process first 1000 for speed
    props = feature['properties']
    trains_analysis.append({
        'number': props['number'],
        'name': props['name'],
        'type': props['type'],
        'distance': props['distance'],
        'duration_h': props['duration_h'],
        'from_station': props['from_station_name'],
        'to_station': props['to_station_name']
    })

trains_df = pd.DataFrame(trains_analysis)

# Train type analysis
train_types = trains_df.groupby('type').agg({
    'distance': ['count', 'mean', 'sum'],
    'duration_h': 'mean'
}).round(2)

print("Train Type Distribution (Sample of 1000 trains):")
for train_type in trains_df['type'].value_counts().head(5).index:
    type_data = trains_df[trains_df['type'] == train_type]
    avg_distance = type_data['distance'].mean()
    avg_duration = type_data['duration_h'].mean()
    count = len(type_data)
    speed = avg_distance / avg_duration if avg_duration > 0 else 0

    print(f"- {train_type}: {count} trains, {avg_distance:.1f}km avg, {speed:.1f} km/h avg speed")

# 2. ROUTE DISTANCE EFFICIENCY
print("\n2. ROUTE EFFICIENCY INSIGHTS")
print("-" * 40)

# Speed analysis for optimization
trains_df['speed_kmh'] = trains_df['distance'] / trains_df['duration_h']
trains_df['speed_kmh'] = trains_df['speed_kmh'].replace([np.inf, -np.inf], np.nan)

print(f"Average train speed: {trains_df['speed_kmh'].mean():.1f} km/h")
print(f"Fastest trains (top 5):")

fastest_trains = trains_df.nlargest(5, 'speed_kmh')
for _, train in fastest_trains.iterrows():
    print(f"- {train['name']} ({train['type']}): {train['speed_kmh']:.1f} km/h")

print(f"\nSlowest trains (optimization candidates):")
slowest_trains = trains_df.nsmallest(5, 'speed_kmh')
for _, train in slowest_trains.iterrows():
    print(f"- {train['name']} ({train['type']}): {train['speed_kmh']:.1f} km/h")

# 3. SCHEDULING ANALYSIS
print("\n3. SCHEDULING OPTIMIZATION OPPORTUNITIES")
print("-" * 40)

# Analyze schedule distribution
schedules_df = pd.DataFrame(schedules_data[:10000])  # Process first 10k for speed

# Train frequency analysis
train_frequency = schedules_df['train_number'].value_counts()
print(f"Schedule entries analyzed: {len(schedules_df)}")
print(f"Unique trains in sample: {schedules_df['train_number'].nunique()}")
print(f"Average stops per train: {train_frequency.mean():.1f}")

# Station utilization
station_usage = schedules_df['station_code'].value_counts()
print(f"\nStation Usage Distribution:")
print(f"Most visited stations (top 5):")
for station, count in station_usage.head(5).items():
    print(f"- {station}: {count} train visits")

print(f"\nLeast visited stations (optimization targets):")
underutilized = station_usage[station_usage <= 2]
print(f"Stations with <=2 visits: {len(underutilized)}")

# 4. GEOGRAPHIC DISTRIBUTION
print("\n4. GEOGRAPHIC DISTRIBUTION ANALYSIS")
print("-" * 40)

# State-wise station distribution
stations_analysis = []
for feature in stations_data['features']:
    if feature['properties']['state']:
        stations_analysis.append({
            'code': feature['properties']['code'],
            'name': feature['properties']['name'],
            'state': feature['properties']['state'],
            'zone': feature['properties']['zone']
        })

stations_df = pd.DataFrame(stations_analysis)

state_stats = stations_df.groupby('state').agg({
    'code': 'count',
    'zone': 'nunique'
}).rename(columns={'code': 'station_count', 'zone': 'zones'})

print("State-wise Network Coverage:")
for state in state_stats.sort_values('station_count', ascending=False).head(5).index:
    count = state_stats.loc[state, 'station_count']
    zones = state_stats.loc[state, 'zones']
    print(f"- {state}: {count} stations across {zones} zones")

# 5. OPTIMIZATION RECOMMENDATIONS
print("\n5. KEY OPTIMIZATION RECOMMENDATIONS")
print("-" * 40)

print("IMMEDIATE OPTIMIZATION OPPORTUNITIES:")

# Speed optimization
slow_threshold = trains_df['speed_kmh'].quantile(0.2)  # Bottom 20%
slow_trains_count = len(trains_df[trains_df['speed_kmh'] < slow_threshold])
print(f"1. Speed Optimization: {slow_trains_count} trains below {slow_threshold:.1f} km/h need route/schedule review")

# Underutilized stations
print(f"2. Network Expansion: {len(underutilized)} underutilized stations could benefit from additional services")

# Train type optimization
passenger_trains = len(trains_df[trains_df['type'] == 'Pass'])
express_trains = len(trains_df[trains_df['type'] == 'Exp'])
print(f"3. Service Mix: {passenger_trains} passenger vs {express_trains} express trains - consider upgrading slow routes")

# Distance distribution
long_routes = trains_df[trains_df['distance'] > trains_df['distance'].quantile(0.8)]
print(f"4. Long Route Efficiency: {len(long_routes)} routes >800km could benefit from intermediate express services")

print("\nQUANTIFIABLE BENEFITS:")
avg_slow_speed = trains_df[trains_df['speed_kmh'] < slow_threshold]['speed_kmh'].mean()
target_speed = trains_df['speed_kmh'].median()
if pd.notna(avg_slow_speed) and pd.notna(target_speed):
    speed_improvement = ((target_speed - avg_slow_speed) / avg_slow_speed) * 100
    print(f"- Potential {speed_improvement:.1f}% speed improvement for slow trains")

print(f"- Network coverage: {len(stations_df)} stations across {stations_df['state'].nunique()} states")
print(f"- Capacity utilization varies {station_usage.min()}-{station_usage.max()} visits per station")

print("\n" + "="*60)
print("QUICK ANALYSIS COMPLETE")
print("="*60)