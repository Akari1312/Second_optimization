"""
Complete Railway Network Builder - ALL 8,990 Stations
=====================================================

This creates a comprehensive graph that includes ALL stations.
"""

import json
import pandas as pd
import networkx as nx
from collections import defaultdict
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

def load_all_stations():
    """Load ALL 8,990 stations"""
    print("Loading ALL railway stations...")

    with open('archive/stations.json', 'r') as f:
        stations_data = json.load(f)

    stations = {}
    stations_with_coords = 0

    for feature in stations_data['features']:
        props = feature['properties']
        code = props['code']

        station_info = {
            'name': props['name'],
            'state': props['state'],
            'zone': props['zone'] or 'Unknown',
            'has_coordinates': False,
            'coordinates': None,
            'connection_type': 'isolated'
        }

        if feature['geometry'] and feature['geometry']['coordinates']:
            coords = feature['geometry']['coordinates']
            station_info['has_coordinates'] = True
            station_info['coordinates'] = (coords[0], coords[1])
            stations_with_coords += 1

        stations[code] = station_info

    print(f"Loaded {len(stations):,} total stations")
    print(f"{stations_with_coords:,} stations have coordinates")

    return stations

def create_complete_graph(stations):
    """Create graph with ALL stations"""
    print("Creating complete network graph...")

    G = nx.Graph()

    # Add ALL stations as nodes
    for code, info in stations.items():
        G.add_node(code, **info)

    print(f"Added {G.number_of_nodes():,} station nodes")

    # Add major train connections
    print("Adding major train route connections...")

    with open('archive/trains.json', 'r') as f:
        trains_data = json.load(f)

    major_connections = 0
    for feature in trains_data['features']:
        props = feature['properties']
        from_station = props['from_station_code']
        to_station = props['to_station_code']

        if from_station in stations and to_station in stations:
            distance = props.get('distance', 0) or 0

            G.add_edge(from_station, to_station,
                      connection_type='major_route',
                      distance=distance,
                      train_type=props['type'])
            major_connections += 1

            # Update connection types
            stations[from_station]['connection_type'] = 'major_hub'
            stations[to_station]['connection_type'] = 'major_hub'

    print(f"Added {major_connections:,} major route connections")

    # Add schedule-based connections (sample for performance)
    print("Adding schedule-based connections...")

    with open('archive/schedules.json', 'r') as f:
        schedules_data = json.load(f)

    # Group schedules by train number (sample for performance)
    train_schedules = defaultdict(list)

    for schedule in schedules_data[:100000]:  # Process first 100k for performance
        train_num = schedule['train_number']
        station_code = schedule['station_code']

        if station_code in stations:
            train_schedules[train_num].append({
                'station': station_code,
                'day': schedule['day']
            })

    schedule_connections = 0
    for train_num, stops in train_schedules.items():
        if len(stops) >= 2:
            # Filter out stops with None day values and sort
            valid_stops = [stop for stop in stops if stop['day'] is not None]
            if len(valid_stops) >= 2:
                valid_stops.sort(key=lambda x: x['day'])
                stops = valid_stops

            for i in range(len(stops) - 1):
                station1 = stops[i]['station']
                station2 = stops[i + 1]['station']

                if not G.has_edge(station1, station2):
                    G.add_edge(station1, station2,
                              connection_type='scheduled_route',
                              source='schedules')
                    schedule_connections += 1

                    # Update connection types
                    if stations[station1]['connection_type'] == 'isolated':
                        stations[station1]['connection_type'] = 'scheduled_stop'
                    if stations[station2]['connection_type'] == 'isolated':
                        stations[station2]['connection_type'] = 'scheduled_stop'

    print(f"Added {schedule_connections:,} schedule-based connections")

    # Add proximity connections for remaining isolated stations
    print("Adding proximity connections for isolated stations...")

    isolated_stations = [code for code, info in stations.items()
                        if info['connection_type'] == 'isolated' and info['has_coordinates']]

    print(f"Found {len(isolated_stations):,} isolated stations with coordinates")

    proximity_connections = 0
    max_proximity_connections = 5000  # Limit for performance

    # Group by zone for efficient processing
    isolated_by_zone = defaultdict(list)
    for code in isolated_stations:
        zone = stations[code]['zone']
        isolated_by_zone[zone].append(code)

    for zone, zone_stations in isolated_by_zone.items():
        if proximity_connections >= max_proximity_connections:
            break

        print(f"  Processing {len(zone_stations)} isolated stations in zone {zone}")

        for i, station1 in enumerate(zone_stations[:100]):  # Limit per zone
            if proximity_connections >= max_proximity_connections:
                break

            coord1 = stations[station1]['coordinates']

            for station2 in zone_stations[i+1:i+10]:  # Check only nearby stations
                if proximity_connections >= max_proximity_connections:
                    break

                coord2 = stations[station2]['coordinates']
                distance = geodesic((coord1[1], coord1[0]), (coord2[1], coord2[0])).kilometers

                if distance <= 25:  # Within 25km
                    G.add_edge(station1, station2,
                              connection_type='proximity',
                              distance=distance)
                    proximity_connections += 1

                    stations[station1]['connection_type'] = 'local_connected'
                    stations[station2]['connection_type'] = 'local_connected'

    print(f"Added {proximity_connections:,} proximity connections")

    # Update node attributes
    for code, info in stations.items():
        G.nodes[code].update(info)

    return G, stations

def analyze_complete_network(G, stations):
    """Analyze the complete network"""
    print("\n" + "="*60)
    print("COMPLETE NETWORK ANALYSIS")
    print("="*60)

    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()

    print(f"NETWORK STATISTICS:")
    print(f"  Total stations: {total_nodes:,}")
    print(f"  Total connections: {total_edges:,}")
    print(f"  Network density: {nx.density(G):.8f}")
    print(f"  Average connections per station: {total_edges * 2 / total_nodes:.1f}")

    # Connectivity
    is_connected = nx.is_connected(G)
    print(f"  Network is connected: {is_connected}")

    if not is_connected:
        components = list(nx.connected_components(G))
        largest_component = max(components, key=len)
        print(f"  Connected components: {len(components):,}")
        print(f"  Largest component: {len(largest_component):,} stations")

    # Connection types
    connection_types = defaultdict(int)
    for code, info in stations.items():
        connection_types[info['connection_type']] += 1

    print(f"\nSTATION TYPES:")
    for conn_type, count in sorted(connection_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_nodes) * 100
        print(f"  {conn_type}: {count:,} ({percentage:.1f}%)")

    # Edge types
    edge_types = defaultdict(int)
    for edge in G.edges(data=True):
        edge_type = edge[2].get('connection_type', 'unknown')
        edge_types[edge_type] += 1

    print(f"\nCONNECTION TYPES:")
    for edge_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_edges) * 100
        print(f"  {edge_type}: {count:,} ({percentage:.1f}%)")

    # Top hubs
    degrees = dict(G.degree())
    top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]

    print(f"\nTOP 10 HUB STATIONS:")
    for station_code, degree in top_hubs:
        station_name = stations[station_code]['name']
        zone = stations[station_code]['zone']
        conn_type = stations[station_code]['connection_type']
        print(f"  {station_code} ({station_name}, {zone}): {degree} connections [{conn_type}]")

    # Zone analysis
    zone_stats = defaultdict(lambda: {'stations': 0, 'connections': 0})
    for code, info in stations.items():
        zone = info['zone']
        zone_stats[zone]['stations'] += 1
        zone_stats[zone]['connections'] += G.degree(code)

    print(f"\nTOP 10 ZONES BY STATION COUNT:")
    sorted_zones = sorted(zone_stats.items(), key=lambda x: x[1]['stations'], reverse=True)
    for zone, stats in sorted_zones[:10]:
        avg_connections = stats['connections'] / stats['stations'] if stats['stations'] > 0 else 0
        print(f"  {zone}: {stats['stations']:,} stations, {avg_connections:.1f} avg connections")

    return {
        'total_stations': total_nodes,
        'total_connections': total_edges,
        'connection_types': dict(connection_types),
        'edge_types': dict(edge_types),
        'top_hubs': top_hubs[:5],
        'is_connected': is_connected
    }

def export_network_summary(G, stations, stats):
    """Export network summary"""
    print(f"\nExporting network summary...")

    summary = {
        'metadata': {
            'total_stations': stats['total_stations'],
            'total_connections': stats['total_connections'],
            'is_connected': stats['is_connected'],
            'export_timestamp': pd.Timestamp.now().isoformat()
        },
        'station_types': stats['connection_types'],
        'connection_types': stats['edge_types'],
        'top_hubs': [{'code': code, 'name': stations[code]['name'], 'connections': degree}
                     for code, degree in stats['top_hubs']],
        'zones': {}
    }

    # Zone summary
    zone_counts = defaultdict(int)
    for code, info in stations.items():
        zone_counts[info['zone']] += 1

    summary['zones'] = dict(zone_counts)

    with open('complete_network_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Network summary exported to 'complete_network_summary.json'")

def main():
    """Main function"""
    print("="*70)
    print("BUILDING COMPLETE INDIAN RAILWAY NETWORK")
    print("Including ALL 8,990 stations from the dataset")
    print("="*70)

    # Load all stations
    stations = load_all_stations()

    # Create complete graph
    G, stations = create_complete_graph(stations)

    # Analyze network
    stats = analyze_complete_network(G, stations)

    # Export summary
    export_network_summary(G, stations, stats)

    print(f"\n" + "="*70)
    print("COMPLETE NETWORK READY!")
    print(f"SUCCESS: All {stats['total_stations']:,} stations included")
    print(f"Total connections: {stats['total_connections']:,}")
    print(f"Coverage: {len(set(s['zone'] for s in stations.values()))} railway zones")
    print("="*70)

    return G, stations, stats

if __name__ == "__main__":
    complete_graph, all_stations, network_stats = main()