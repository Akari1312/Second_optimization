"""
Railway Network Graph API
=========================

A simple API interface for the railway network graph with key functionality:
- Load and construct the graph
- Find shortest paths
- Get station information
- Analyze routes
- Export graph data
"""

import json
import networkx as nx
from collections import defaultdict
import pickle

class RailwayGraphAPI:
    """API interface for railway network graph operations"""

    def __init__(self):
        self.graph = None
        self.stations = {}
        self.is_loaded = False

    def load_graph(self, use_cache=True):
        """Load or construct the railway graph"""
        cache_file = 'railway_graph.pkl'

        if use_cache:
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.graph = cached_data['graph']
                    self.stations = cached_data['stations']
                    self.is_loaded = True
                    print(f"Loaded cached graph: {self.graph.number_of_nodes()} stations, {self.graph.number_of_edges()} routes")
                    return True
            except FileNotFoundError:
                print("No cache found, constructing graph...")

        # Construct graph from data
        self._construct_graph()

        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump({'graph': self.graph, 'stations': self.stations}, f)

        print(f"Graph constructed and cached: {self.graph.number_of_nodes()} stations, {self.graph.number_of_edges()} routes")
        return True

    def _construct_graph(self):
        """Internal method to construct graph from JSON data"""
        # Load stations
        with open('archive/stations.json', 'r') as f:
            stations_data = json.load(f)

        stations = {}
        for feature in stations_data['features']:
            if feature['geometry'] and feature['geometry']['coordinates']:
                code = feature['properties']['code']
                stations[code] = {
                    'name': feature['properties']['name'],
                    'zone': feature['properties']['zone'] or 'Unknown',
                    'state': feature['properties']['state'],
                    'coordinates': feature['geometry']['coordinates']
                }

        # Load trains
        with open('archive/trains.json', 'r') as f:
            trains_data = json.load(f)

        valid_routes = []
        station_usage = defaultdict(int)

        for feature in trains_data['features']:
            props = feature['properties']
            from_station = props['from_station_code']
            to_station = props['to_station_code']

            if from_station in stations and to_station in stations:
                route = {
                    'train_number': props['number'],
                    'from_station': from_station,
                    'to_station': to_station,
                    'distance': props.get('distance', 0) or 0,
                    'type': props['type']
                }
                valid_routes.append(route)
                station_usage[from_station] += 1
                station_usage[to_station] += 1

        # Filter to used stations only
        self.stations = {code: info for code, info in stations.items() if code in station_usage}

        # Create graph
        self.graph = nx.Graph()

        # Add nodes
        for code, info in self.stations.items():
            self.graph.add_node(code, **info)

        # Add edges
        for route in valid_routes:
            from_station = route['from_station']
            to_station = route['to_station']

            if self.graph.has_edge(from_station, to_station):
                self.graph[from_station][to_station]['train_count'] += 1
                self.graph[from_station][to_station]['trains'].append(route['train_number'])
            else:
                self.graph.add_edge(from_station, to_station,
                                  weight=route['distance'] if route['distance'] > 0 else 100,
                                  train_count=1,
                                  trains=[route['train_number']],
                                  distance=route['distance'])

        self.is_loaded = True

    def find_shortest_path(self, from_station, to_station, weight='weight'):
        """Find shortest path between two stations"""
        if not self.is_loaded:
            self.load_graph()

        try:
            path = nx.shortest_path(self.graph, from_station, to_station, weight=weight)
            path_length = nx.shortest_path_length(self.graph, from_station, to_station, weight=weight)

            result = {
                'success': True,
                'path': path,
                'distance': path_length,
                'stations_count': len(path),
                'route_details': []
            }

            # Add detailed station information
            for station_code in path:
                if station_code in self.stations:
                    station_info = self.stations[station_code].copy()
                    station_info['code'] = station_code
                    result['route_details'].append(station_info)

            return result

        except nx.NetworkXNoPath:
            return {
                'success': False,
                'error': f'No path found between {from_station} and {to_station}',
                'path': [],
                'distance': 0,
                'stations_count': 0
            }

    def get_station_info(self, station_code):
        """Get detailed information about a station"""
        if not self.is_loaded:
            self.load_graph()

        if station_code not in self.stations:
            return {'success': False, 'error': 'Station not found'}

        station_info = self.stations[station_code].copy()
        station_info['code'] = station_code

        if station_code in self.graph:
            # Add connectivity information
            connections = list(self.graph.neighbors(station_code))
            station_info['connections'] = len(connections)
            station_info['connected_stations'] = connections[:10]  # First 10 connections

            # Add train information
            total_trains = 0
            for neighbor in connections:
                total_trains += self.graph[station_code][neighbor].get('train_count', 0)
            station_info['total_trains'] = total_trains

        return {'success': True, 'station': station_info}

    def get_network_stats(self):
        """Get overall network statistics"""
        if not self.is_loaded:
            self.load_graph()

        stats = {
            'total_stations': self.graph.number_of_nodes(),
            'total_routes': self.graph.number_of_edges(),
            'network_density': nx.density(self.graph),
            'is_connected': nx.is_connected(self.graph)
        }

        # Degree statistics
        degrees = dict(self.graph.degree())
        stats['avg_connections'] = sum(degrees.values()) / len(degrees)
        stats['max_connections'] = max(degrees.values())

        # Top hubs
        top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        stats['top_hubs'] = []
        for station_code, degree in top_hubs:
            stats['top_hubs'].append({
                'code': station_code,
                'name': self.stations[station_code]['name'],
                'connections': degree
            })

        # Zone distribution
        zone_counts = defaultdict(int)
        for code in self.graph.nodes():
            zone = self.stations[code]['zone']
            zone_counts[zone] += 1

        stats['zones'] = dict(zone_counts)

        return stats

    def find_stations_by_zone(self, zone):
        """Find all stations in a specific railway zone"""
        if not self.is_loaded:
            self.load_graph()

        stations_in_zone = []
        for code, info in self.stations.items():
            if info['zone'] == zone:
                station_data = info.copy()
                station_data['code'] = code
                if code in self.graph:
                    station_data['connections'] = self.graph.degree(code)
                stations_in_zone.append(station_data)

        return sorted(stations_in_zone, key=lambda x: x.get('connections', 0), reverse=True)

    def find_alternative_routes(self, from_station, to_station, max_routes=3):
        """Find multiple alternative routes between two stations"""
        if not self.is_loaded:
            self.load_graph()

        try:
            # Find multiple paths using different approaches
            routes = []

            # Shortest path by distance
            shortest = self.find_shortest_path(from_station, to_station, weight='weight')
            if shortest['success']:
                routes.append({
                    'type': 'shortest_distance',
                    'path': shortest['path'],
                    'distance': shortest['distance'],
                    'stations': shortest['stations_count']
                })

            # Shortest path by hops (unweighted)
            try:
                path_hops = nx.shortest_path(self.graph, from_station, to_station)
                if path_hops != shortest['path']:  # Different from distance-based path
                    routes.append({
                        'type': 'fewest_stations',
                        'path': path_hops,
                        'distance': nx.path_weight(self.graph, path_hops, weight='weight'),
                        'stations': len(path_hops)
                    })
            except:
                pass

            return {'success': True, 'routes': routes}

        except:
            return {'success': False, 'error': 'Could not find alternative routes'}

    def export_subgraph(self, station_codes, include_neighbors=True):
        """Export a subgraph containing specified stations"""
        if not self.is_loaded:
            self.load_graph()

        nodes_to_include = set(station_codes)

        if include_neighbors:
            # Add immediate neighbors of specified stations
            for station in station_codes:
                if station in self.graph:
                    nodes_to_include.update(self.graph.neighbors(station))

        # Create subgraph
        subgraph = self.graph.subgraph(nodes_to_include)

        # Export data
        export_data = {
            'nodes': [],
            'edges': []
        }

        for node in subgraph.nodes():
            node_data = self.stations[node].copy()
            node_data['code'] = node
            node_data['degree'] = subgraph.degree(node)
            export_data['nodes'].append(node_data)

        for edge in subgraph.edges():
            edge_data = subgraph[edge[0]][edge[1]].copy()
            edge_data['from'] = edge[0]
            edge_data['to'] = edge[1]
            export_data['edges'].append(edge_data)

        return export_data

# Example usage and testing
def demo_api():
    """Demonstrate the API functionality"""
    print("="*50)
    print("RAILWAY GRAPH API DEMO")
    print("="*50)

    # Initialize API
    api = RailwayGraphAPI()

    # Load graph
    print("Loading graph...")
    api.load_graph()

    # Get network statistics
    print("\nNetwork Statistics:")
    stats = api.get_network_stats()
    print(f"- Total stations: {stats['total_stations']}")
    print(f"- Total routes: {stats['total_routes']}")
    print(f"- Average connections: {stats['avg_connections']:.1f}")
    print(f"- Network is connected: {stats['is_connected']}")

    print(f"\nTop Hub Stations:")
    for hub in stats['top_hubs']:
        print(f"- {hub['code']} ({hub['name']}): {hub['connections']} connections")

    # Test pathfinding
    print(f"\nPathfinding Test:")
    if 'NDLS' in api.stations and 'MAS' in api.stations:
        path_result = api.find_shortest_path('NDLS', 'MAS')
        if path_result['success']:
            print(f"Route from NDLS to MAS:")
            print(f"Path: {' -> '.join(path_result['path'])}")
            print(f"Stations: {path_result['stations_count']}")
            print(f"Distance: {path_result['distance']:.0f}")

    # Test station info
    print(f"\nStation Information Test:")
    station_info = api.get_station_info('NDLS')
    if station_info['success']:
        station = station_info['station']
        print(f"Station: {station['name']} ({station['code']})")
        print(f"Zone: {station['zone']}")
        print(f"State: {station['state']}")
        print(f"Connections: {station.get('connections', 0)}")

    print(f"\n" + "="*50)
    print("API DEMO COMPLETE")
    print("="*50)

    return api

if __name__ == "__main__":
    api = demo_api()