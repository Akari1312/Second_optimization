"""
Complete Railway Network API
===========================

API for accessing the complete Indian Railway network with ALL 8,990 stations
"""

import json
import networkx as nx
from collections import defaultdict
import pickle

class CompleteRailwayAPI:
    """API for the complete railway network with all stations"""

    def __init__(self):
        self.graph = None
        self.stations = {}
        self.is_loaded = False
        self.network_stats = {}

    def load_complete_network(self, use_cache=True):
        """Load the complete network (will build if not cached)"""
        cache_file = 'complete_railway_network.pkl'

        if use_cache:
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.graph = cached_data['graph']
                    self.stations = cached_data['stations']
                    self.network_stats = cached_data.get('stats', {})
                    self.is_loaded = True
                    print(f"Loaded complete network: {self.graph.number_of_nodes():,} stations, {self.graph.number_of_edges():,} connections")
                    return True
            except FileNotFoundError:
                print("Complete network cache not found. Building network...")

        # Build the complete network
        self._build_complete_network()

        # Cache the result
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'stations': self.stations,
                'stats': self.network_stats
            }, f)

        print(f"Complete network built and cached: {self.graph.number_of_nodes():,} stations")
        return True

    def _build_complete_network(self):
        """Build the complete network from scratch"""
        # Import and run the complete network builder
        import subprocess
        import sys

        # Run the complete network builder
        result = subprocess.run([sys.executable, 'complete_network_builder.py'],
                              capture_output=True, text=True)

        if result.returncode == 0:
            # Load the built network (assuming it was saved)
            # For now, rebuild inline
            from complete_network_builder import load_all_stations, create_complete_graph, analyze_complete_network

            self.stations = load_all_stations()
            self.graph, self.stations = create_complete_graph(self.stations)
            self.network_stats = analyze_complete_network(self.graph, self.stations)
            self.is_loaded = True
        else:
            raise Exception(f"Failed to build complete network: {result.stderr}")

    def get_network_overview(self):
        """Get complete network overview"""
        if not self.is_loaded:
            self.load_complete_network()

        return {
            'total_stations': self.graph.number_of_nodes(),
            'total_connections': self.graph.number_of_edges(),
            'network_density': nx.density(self.graph),
            'is_connected': nx.is_connected(self.graph),
            'coverage': {
                'states': len(set(s['state'] for s in self.stations.values() if s['state'])),
                'zones': len(set(s['zone'] for s in self.stations.values())),
                'coordinates': len([s for s in self.stations.values() if s['has_coordinates']])
            },
            'station_types': self.network_stats.get('connection_types', {}),
            'connection_types': self.network_stats.get('edge_types', {})
        }

    def find_station(self, query):
        """Find stations by code, name, or partial match"""
        if not self.is_loaded:
            self.load_complete_network()

        matches = []
        query_lower = query.lower()

        for code, info in self.stations.items():
            # Exact code match
            if code.lower() == query_lower:
                station_data = info.copy()
                station_data['code'] = code
                station_data['match_type'] = 'exact_code'
                station_data['connections'] = self.graph.degree(code)
                return [station_data]

            # Name contains query
            if info['name'] and query_lower in info['name'].lower():
                station_data = info.copy()
                station_data['code'] = code
                station_data['match_type'] = 'name_match'
                station_data['connections'] = self.graph.degree(code)
                matches.append(station_data)

        # Sort by connection count (most connected first)
        matches.sort(key=lambda x: x['connections'], reverse=True)
        return matches[:20]  # Return top 20 matches

    def get_station_details(self, station_code):
        """Get detailed information about a station"""
        if not self.is_loaded:
            self.load_complete_network()

        if station_code not in self.stations:
            return {'success': False, 'error': 'Station not found'}

        station_info = self.stations[station_code].copy()
        station_info['code'] = station_code

        # Add network information
        degree = self.graph.degree(station_code)
        station_info['connections'] = degree
        station_info['neighbors'] = list(self.graph.neighbors(station_code))[:20]  # First 20 neighbors

        # Get connection types
        connection_types = defaultdict(int)
        for neighbor in self.graph.neighbors(station_code):
            edge_data = self.graph[station_code][neighbor]
            conn_type = edge_data.get('connection_type', 'unknown')
            connection_types[conn_type] += 1

        station_info['connection_breakdown'] = dict(connection_types)

        # Check if in largest connected component
        if nx.is_connected(self.graph):
            station_info['in_main_network'] = True
        else:
            components = list(nx.connected_components(self.graph))
            largest_component = max(components, key=len)
            station_info['in_main_network'] = station_code in largest_component

        return {'success': True, 'station': station_info}

    def find_path_complete(self, from_station, to_station, max_hops=None):
        """Find path between any two stations in the complete network"""
        if not self.is_loaded:
            self.load_complete_network()

        if from_station not in self.stations:
            return {'success': False, 'error': f'From station {from_station} not found'}

        if to_station not in self.stations:
            return {'success': False, 'error': f'To station {to_station} not found'}

        try:
            # Try different pathfinding approaches
            paths = []

            # 1. Shortest path by number of hops
            try:
                path_hops = nx.shortest_path(self.graph, from_station, to_station)
                paths.append({
                    'type': 'shortest_hops',
                    'path': path_hops,
                    'hops': len(path_hops) - 1,
                    'stations': len(path_hops)
                })
            except nx.NetworkXNoPath:
                pass

            # 2. If weight/distance available, shortest by distance
            try:
                path_distance = nx.shortest_path(self.graph, from_station, to_station, weight='distance')
                if path_distance != path_hops:  # Different from hop-based path
                    total_distance = nx.shortest_path_length(self.graph, from_station, to_station, weight='distance')
                    paths.append({
                        'type': 'shortest_distance',
                        'path': path_distance,
                        'distance': total_distance,
                        'stations': len(path_distance)
                    })
            except (nx.NetworkXNoPath, KeyError):
                pass

            if not paths:
                return {'success': False, 'error': 'No path found between stations'}

            # Add detailed route information
            for path_info in paths:
                path_info['route_details'] = []
                for i, station_code in enumerate(path_info['path']):
                    station_data = self.stations[station_code].copy()
                    station_data['code'] = station_code
                    station_data['sequence'] = i + 1

                    # Add connection info to next station
                    if i < len(path_info['path']) - 1:
                        next_station = path_info['path'][i + 1]
                        edge_data = self.graph[station_code][next_station]
                        station_data['next_connection'] = {
                            'to': next_station,
                            'type': edge_data.get('connection_type', 'unknown'),
                            'distance': edge_data.get('distance', 0)
                        }

                    path_info['route_details'].append(station_data)

            return {'success': True, 'paths': paths}

        except Exception as e:
            return {'success': False, 'error': f'Error finding path: {str(e)}'}

    def get_stations_by_zone(self, zone):
        """Get all stations in a specific railway zone"""
        if not self.is_loaded:
            self.load_complete_network()

        zone_stations = []
        for code, info in self.stations.items():
            if info['zone'] == zone:
                station_data = info.copy()
                station_data['code'] = code
                station_data['connections'] = self.graph.degree(code)
                zone_stations.append(station_data)

        # Sort by connections (most connected first)
        zone_stations.sort(key=lambda x: x['connections'], reverse=True)
        return zone_stations

    def get_stations_by_state(self, state):
        """Get all stations in a specific state"""
        if not self.is_loaded:
            self.load_complete_network()

        state_stations = []
        for code, info in self.stations.items():
            if info['state'] == state:
                station_data = info.copy()
                station_data['code'] = code
                station_data['connections'] = self.graph.degree(code)
                state_stations.append(station_data)

        state_stations.sort(key=lambda x: x['connections'], reverse=True)
        return state_stations

    def get_isolated_stations(self):
        """Get all isolated stations (no connections)"""
        if not self.is_loaded:
            self.load_complete_network()

        isolated = []
        for code, info in self.stations.items():
            if self.graph.degree(code) == 0:
                station_data = info.copy()
                station_data['code'] = code
                isolated.append(station_data)

        return isolated

    def get_network_components(self):
        """Analyze network connectivity components"""
        if not self.is_loaded:
            self.load_complete_network()

        if nx.is_connected(self.graph):
            return {
                'is_connected': True,
                'components': 1,
                'largest_component_size': self.graph.number_of_nodes()
            }

        components = list(nx.connected_components(self.graph))
        component_sizes = [len(comp) for comp in components]

        return {
            'is_connected': False,
            'total_components': len(components),
            'largest_component_size': max(component_sizes),
            'component_size_distribution': {
                'mean': sum(component_sizes) / len(component_sizes),
                'largest': max(component_sizes),
                'smallest': min(component_sizes),
                'single_station_components': sum(1 for size in component_sizes if size == 1)
            }
        }

    def export_subnetwork(self, station_codes, include_neighbors=True):
        """Export a subnetwork for specific stations"""
        if not self.is_loaded:
            self.load_complete_network()

        nodes_to_include = set(station_codes)

        if include_neighbors:
            for station in station_codes:
                if station in self.graph:
                    nodes_to_include.update(self.graph.neighbors(station))

        subgraph = self.graph.subgraph(nodes_to_include)

        export_data = {
            'metadata': {
                'nodes': subgraph.number_of_nodes(),
                'edges': subgraph.number_of_edges(),
                'requested_stations': len(station_codes),
                'include_neighbors': include_neighbors
            },
            'stations': [],
            'connections': []
        }

        # Export nodes
        for node in subgraph.nodes():
            station_data = self.stations[node].copy()
            station_data['code'] = node
            station_data['degree'] = subgraph.degree(node)
            export_data['stations'].append(station_data)

        # Export edges
        for edge in subgraph.edges(data=True):
            connection = edge[2].copy()
            connection['from'] = edge[0]
            connection['to'] = edge[1]
            export_data['connections'].append(connection)

        return export_data

def demo_complete_api():
    """Demonstrate the complete railway API"""
    print("="*60)
    print("COMPLETE RAILWAY NETWORK API DEMO")
    print("="*60)

    api = CompleteRailwayAPI()

    # Load complete network
    print("Loading complete network...")
    api.load_complete_network()

    # Get overview
    overview = api.get_network_overview()
    print(f"\nNETWORK OVERVIEW:")
    print(f"  Total stations: {overview['total_stations']:,}")
    print(f"  Total connections: {overview['total_connections']:,}")
    print(f"  States covered: {overview['coverage']['states']}")
    print(f"  Railway zones: {overview['coverage']['zones']}")
    print(f"  Connected: {overview['is_connected']}")

    # Find station example
    print(f"\nSTATION SEARCH TEST:")
    matches = api.find_station("Delhi")
    print(f"Stations matching 'Delhi': {len(matches)}")
    for match in matches[:3]:
        print(f"  {match['code']} - {match['name']} ({match['connections']} connections)")

    # Path finding example
    print(f"\nPATH FINDING TEST:")
    if 'NDLS' in api.stations and 'MAS' in api.stations:
        path_result = api.find_path_complete('NDLS', 'MAS')
        if path_result['success']:
            for path in path_result['paths']:
                print(f"  {path['type']}: {path['stations']} stations")
                print(f"    Route: {' -> '.join(path['path'][:5])}{'...' if len(path['path']) > 5 else ''}")

    # Zone analysis
    print(f"\nZONE ANALYSIS:")
    nr_stations = api.get_stations_by_zone('NR')
    print(f"  Northern Railway (NR): {len(nr_stations)} stations")
    print(f"    Top hub: {nr_stations[0]['code']} - {nr_stations[0]['name']} ({nr_stations[0]['connections']} connections)")

    # Component analysis
    components = api.get_network_components()
    print(f"\nCONNECTIVITY:")
    print(f"  Connected: {components['is_connected']}")
    if not components['is_connected']:
        print(f"  Components: {components['total_components']:,}")
        print(f"  Largest component: {components['largest_component_size']:,} stations")

    print(f"\n" + "="*60)
    print("COMPLETE API DEMO FINISHED")
    print("="*60)

    return api

if __name__ == "__main__":
    complete_api = demo_complete_api()