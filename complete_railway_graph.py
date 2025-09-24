"""
Complete Railway Network Graph - ALL Stations
==============================================

This creates a comprehensive graph that includes ALL 8,990 stations by:
1. Adding all stations as nodes
2. Connecting via major train routes
3. Adding connections from schedules data
4. Creating proximity-based local connections
5. Building a multi-layer network representation
"""

import json
import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

class CompleteRailwayNetwork:
    """Complete railway network with ALL stations"""

    def __init__(self):
        self.complete_graph = nx.Graph()
        self.major_routes_graph = nx.Graph()
        self.local_connections_graph = nx.Graph()
        self.stations = {}
        self.connection_stats = {}

    def load_all_stations(self):
        """Load ALL stations from the dataset"""
        print("Loading ALL railway stations...")

        with open('archive/stations.json', 'r') as f:
            stations_data = json.load(f)

        stations_loaded = 0
        stations_with_coords = 0
        stations_by_zone = defaultdict(int)

        for feature in stations_data['features']:
            props = feature['properties']
            code = props['code']

            # Create station entry
            station_info = {
                'name': props['name'],
                'state': props['state'],
                'zone': props['zone'] or 'Unknown',
                'address': props.get('address', ''),
                'has_coordinates': False,
                'coordinates': None,
                'connection_type': 'isolated'  # Will be updated based on connections
            }

            # Add coordinates if available
            if feature['geometry'] and feature['geometry']['coordinates']:
                coords = feature['geometry']['coordinates']
                station_info['has_coordinates'] = True
                station_info['coordinates'] = (coords[0], coords[1])  # (lon, lat)
                stations_with_coords += 1

            self.stations[code] = station_info
            stations_loaded += 1
            stations_by_zone[station_info['zone']] += 1

            # Add to complete graph
            self.complete_graph.add_node(code, **station_info)

        print(f"✓ Loaded {stations_loaded:,} total stations")
        print(f"✓ {stations_with_coords:,} stations have coordinates")
        print(f"✓ Coverage: {len(stations_by_zone)} railway zones")

        return stations_loaded

    def add_major_train_connections(self):
        """Add connections from major train routes"""
        print("Adding major train route connections...")

        with open('archive/trains.json', 'r') as f:
            trains_data = json.load(f)

        major_connections = 0
        trains_processed = 0

        for feature in trains_data['features']:
            props = feature['properties']
            from_station = props['from_station_code']
            to_station = props['to_station_code']

            # Only add if both stations exist in our complete network
            if from_station in self.stations and to_station in self.stations:
                distance = props.get('distance', 0) or 0
                train_type = props['type']

                # Add to complete graph
                if not self.complete_graph.has_edge(from_station, to_station):
                    self.complete_graph.add_edge(
                        from_station, to_station,
                        connection_type='major_route',
                        distance=distance,
                        train_types=[train_type],
                        train_count=1,
                        trains=[props['number']]
                    )
                    major_connections += 1
                else:
                    # Update existing edge
                    edge_data = self.complete_graph[from_station][to_station]
                    edge_data['train_types'].append(train_type)
                    edge_data['train_count'] += 1
                    edge_data['trains'].append(props['number'])

                # Also add to major routes subgraph
                self.major_routes_graph.add_node(from_station, **self.stations[from_station])
                self.major_routes_graph.add_node(to_station, **self.stations[to_station])
                self.major_routes_graph.add_edge(from_station, to_station,
                                               distance=distance, train_type=train_type)

                # Update connection types
                self.stations[from_station]['connection_type'] = 'major_hub'
                self.stations[to_station]['connection_type'] = 'major_hub'

            trains_processed += 1

        print(f"✓ Added {major_connections:,} major route connections")
        print(f"✓ Processed {trains_processed:,} trains")

        return major_connections

    def add_schedule_connections(self):
        """Add connections based on train schedules"""
        print("Adding connections from schedule data...")

        with open('archive/schedules.json', 'r') as f:
            schedules_data = json.load(f)

        # Group schedules by train number
        train_schedules = defaultdict(list)
        schedule_stations = set()

        # Process schedules in chunks for memory efficiency
        chunk_size = 50000
        for i in range(0, len(schedules_data), chunk_size):
            chunk = schedules_data[i:i + chunk_size]

            for schedule in chunk:
                train_num = schedule['train_number']
                station_code = schedule['station_code']

                if station_code in self.stations:
                    train_schedules[train_num].append({
                        'station': station_code,
                        'day': schedule['day'],
                        'arrival': schedule.get('arrival'),
                        'departure': schedule.get('departure')
                    })
                    schedule_stations.add(station_code)

        print(f"✓ Found {len(schedule_stations):,} stations in schedules")

        # Create connections within each train's route
        schedule_connections = 0
        trains_with_schedules = 0

        for train_num, stops in train_schedules.items():
            if len(stops) >= 2:
                # Sort stops by day (approximate route order)
                stops.sort(key=lambda x: x['day'])

                # Connect consecutive stations in the route
                for i in range(len(stops) - 1):
                    station1 = stops[i]['station']
                    station2 = stops[i + 1]['station']

                    if (station1 in self.stations and station2 in self.stations and
                        not self.complete_graph.has_edge(station1, station2)):

                        # Calculate approximate distance if coordinates available
                        distance = 0
                        if (self.stations[station1]['has_coordinates'] and
                            self.stations[station2]['has_coordinates']):
                            coord1 = self.stations[station1]['coordinates']
                            coord2 = self.stations[station2]['coordinates']
                            distance = geodesic((coord1[1], coord1[0]), (coord2[1], coord2[0])).kilometers

                        self.complete_graph.add_edge(
                            station1, station2,
                            connection_type='scheduled_route',
                            distance=distance,
                            train_number=train_num,
                            source='schedules'
                        )
                        schedule_connections += 1

                        # Update connection types
                        if self.stations[station1]['connection_type'] == 'isolated':
                            self.stations[station1]['connection_type'] = 'scheduled_stop'
                        if self.stations[station2]['connection_type'] == 'isolated':
                            self.stations[station2]['connection_type'] = 'scheduled_stop'

                trains_with_schedules += 1

        print(f"✓ Added {schedule_connections:,} schedule-based connections")
        print(f"✓ Processed {trains_with_schedules:,} trains with schedules")

        return schedule_connections

    def add_proximity_connections(self, max_distance_km=50, max_connections_per_zone=1000):
        """Add proximity-based connections for nearby stations"""
        print("Adding proximity-based local connections...")

        stations_with_coords = {code: info for code, info in self.stations.items()
                              if info['has_coordinates']}

        print(f"✓ Processing {len(stations_with_coords):,} stations with coordinates")

        proximity_connections = 0
        zone_connection_counts = defaultdict(int)

        # Group stations by zone for efficiency
        stations_by_zone = defaultdict(list)
        for code, info in stations_with_coords.items():
            stations_by_zone[info['zone']].append(code)

        # Add proximity connections within each zone
        for zone, station_codes in stations_by_zone.items():
            if zone_connection_counts[zone] >= max_connections_per_zone:
                continue

            print(f"  Processing zone {zone}: {len(station_codes)} stations")

            for i, station1 in enumerate(station_codes):
                if zone_connection_counts[zone] >= max_connections_per_zone:
                    break

                # Only check unconnected or lightly connected stations
                if self.complete_graph.degree(station1) > 5:
                    continue

                coord1 = self.stations[station1]['coordinates']

                for station2 in station_codes[i+1:]:
                    if zone_connection_counts[zone] >= max_connections_per_zone:
                        break

                    if self.complete_graph.has_edge(station1, station2):
                        continue

                    coord2 = self.stations[station2]['coordinates']
                    distance = geodesic((coord1[1], coord1[0]), (coord2[1], coord2[0])).kilometers

                    if distance <= max_distance_km:
                        self.complete_graph.add_edge(
                            station1, station2,
                            connection_type='proximity',
                            distance=distance,
                            source='proximity'
                        )
                        proximity_connections += 1
                        zone_connection_counts[zone] += 1

                        # Update connection types
                        if self.stations[station1]['connection_type'] == 'isolated':
                            self.stations[station1]['connection_type'] = 'local_connected'
                        if self.stations[station2]['connection_type'] == 'isolated':
                            self.stations[station2]['connection_type'] = 'local_connected'

        print(f"✓ Added {proximity_connections:,} proximity connections")
        return proximity_connections

    def analyze_complete_network(self):
        """Analyze the complete network"""
        print("\n" + "="*60)
        print("COMPLETE NETWORK ANALYSIS")
        print("="*60)

        # Basic statistics
        total_nodes = self.complete_graph.number_of_nodes()
        total_edges = self.complete_graph.number_of_edges()
        density = nx.density(self.complete_graph)

        print(f"📊 NETWORK STATISTICS:")
        print(f"   Total stations: {total_nodes:,}")
        print(f"   Total connections: {total_edges:,}")
        print(f"   Network density: {density:.6f}")
        print(f"   Average connections per station: {total_edges * 2 / total_nodes:.1f}")

        # Connectivity analysis
        is_connected = nx.is_connected(self.complete_graph)
        print(f"   Network is connected: {is_connected}")

        if not is_connected:
            components = list(nx.connected_components(self.complete_graph))
            largest_component = max(components, key=len)
            print(f"   Connected components: {len(components):,}")
            print(f"   Largest component size: {len(largest_component):,}")

        # Connection type analysis
        connection_types = defaultdict(int)
        for code, info in self.stations.items():
            connection_types[info['connection_type']] += 1

        print(f"\n🔗 STATION TYPES:")
        for conn_type, count in sorted(connection_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_nodes) * 100
            print(f"   {conn_type}: {count:,} ({percentage:.1f}%)")

        # Edge type analysis
        edge_types = defaultdict(int)
        for edge in self.complete_graph.edges(data=True):
            edge_type = edge[2].get('connection_type', 'unknown')
            edge_types[edge_type] += 1

        print(f"\n🛤️  CONNECTION TYPES:")
        for edge_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_edges) * 100
            print(f"   {edge_type}: {count:,} ({percentage:.1f}%)")

        # Zone analysis
        zone_stats = defaultdict(lambda: {'stations': 0, 'connections': 0})
        for code, info in self.stations.items():
            zone = info['zone']
            zone_stats[zone]['stations'] += 1
            zone_stats[zone]['connections'] += self.complete_graph.degree(code)

        print(f"\n🗺️  TOP 10 ZONES BY STATION COUNT:")
        sorted_zones = sorted(zone_stats.items(), key=lambda x: x[1]['stations'], reverse=True)
        for zone, stats in sorted_zones[:10]:
            avg_connections = stats['connections'] / stats['stations'] if stats['stations'] > 0 else 0
            print(f"   {zone}: {stats['stations']:,} stations, {avg_connections:.1f} avg connections")

        # Hub analysis
        degrees = dict(self.complete_graph.degree())
        top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]

        print(f"\n🏢 TOP 10 HUB STATIONS:")
        for station_code, degree in top_hubs:
            station_name = self.stations[station_code]['name']
            zone = self.stations[station_code]['zone']
            conn_type = self.stations[station_code]['connection_type']
            print(f"   {station_code} ({station_name}, {zone}): {degree} connections [{conn_type}]")

        self.connection_stats = {
            'total_stations': total_nodes,
            'total_connections': total_edges,
            'density': density,
            'is_connected': is_connected,
            'connection_types': dict(connection_types),
            'edge_types': dict(edge_types),
            'top_hubs': top_hubs[:5]
        }

        return self.connection_stats

    def export_complete_network(self, filename='complete_railway_network.json'):
        """Export the complete network to JSON"""
        print(f"\nExporting complete network to {filename}...")

        export_data = {
            'metadata': {
                'total_stations': self.complete_graph.number_of_nodes(),
                'total_connections': self.complete_graph.number_of_edges(),
                'export_timestamp': pd.Timestamp.now().isoformat(),
                'connection_stats': self.connection_stats
            },
            'stations': [],
            'connections': []
        }

        # Export stations
        for node, data in self.complete_graph.nodes(data=True):
            station_data = data.copy()
            station_data['code'] = node
            station_data['degree'] = self.complete_graph.degree(node)
            export_data['stations'].append(station_data)

        # Export connections
        for edge in self.complete_graph.edges(data=True):
            connection_data = edge[2].copy()
            connection_data['from_station'] = edge[0]
            connection_data['to_station'] = edge[1]
            export_data['connections'].append(connection_data)

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"✓ Complete network exported to {filename}")
        file_size = len(json.dumps(export_data)) / (1024 * 1024)
        print(f"✓ File size: {file_size:.1f} MB")

    def get_network_summary(self):
        """Get a summary of the complete network"""
        stats = self.connection_stats

        summary = f"""
COMPLETE INDIAN RAILWAY NETWORK SUMMARY
=======================================

📊 SCALE:
   • {stats['total_stations']:,} total railway stations
   • {stats['total_connections']:,} direct connections
   • {len([s for s in self.stations.values() if s['has_coordinates']]):,} stations with GPS coordinates
   • {len(set(s['zone'] for s in self.stations.values())):,} railway zones covered

🔗 CONNECTIVITY:
   • Network density: {stats['density']:.6f}
   • Connected: {stats['is_connected']}
   • Average connections: {stats['total_connections'] * 2 / stats['total_stations']:.1f} per station

🏢 STATION TYPES:
"""

        for conn_type, count in stats['connection_types'].items():
            percentage = (count / stats['total_stations']) * 100
            summary += f"   • {conn_type}: {count:,} ({percentage:.1f}%)\n"

        summary += f"""
🛤️  CONNECTION TYPES:
"""

        for edge_type, count in stats['edge_types'].items():
            percentage = (count / stats['total_connections']) * 100
            summary += f"   • {edge_type}: {count:,} ({percentage:.1f}%)\n"

        summary += f"""
🎯 TOP HUBS:
"""
        for station_code, degree in stats['top_hubs']:
            name = self.stations[station_code]['name']
            summary += f"   • {station_code} ({name}): {degree} connections\n"

        return summary

def main():
    """Main function to build the complete network"""
    print("="*70)
    print("BUILDING COMPLETE INDIAN RAILWAY NETWORK")
    print("="*70)

    # Initialize network
    network = CompleteRailwayNetwork()

    # Step 1: Load all stations
    network.load_all_stations()

    # Step 2: Add major train connections
    network.add_major_train_connections()

    # Step 3: Add schedule-based connections
    network.add_schedule_connections()

    # Step 4: Add proximity connections (with limits for performance)
    network.add_proximity_connections(max_distance_km=30, max_connections_per_zone=500)

    # Step 5: Analyze the complete network
    stats = network.analyze_complete_network()

    # Step 6: Export the network
    network.export_complete_network()

    # Step 7: Print summary
    print(network.get_network_summary())

    print("\n" + "="*70)
    print("COMPLETE NETWORK CONSTRUCTION FINISHED")
    print("="*70)

    return network

if __name__ == "__main__":
    complete_network = main()