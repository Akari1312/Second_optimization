"""
Simple Railway Network Graph Construction
========================================

A focused script to construct and visualize the railway network graph
with minimal dependencies and clear output.
"""

import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_railway_data():
    """Load and prepare railway data"""
    print("Loading railway data...")

    # Load stations
    with open('archive/stations.json', 'r') as f:
        stations_data = json.load(f)

    stations = {}
    coordinates = {}

    for feature in stations_data['features']:
        if feature['geometry'] and feature['geometry']['coordinates']:
            code = feature['properties']['code']
            stations[code] = {
                'name': feature['properties']['name'],
                'state': feature['properties']['state'],
                'zone': feature['properties']['zone'],
                'coordinates': feature['geometry']['coordinates']
            }
            coordinates[code] = feature['geometry']['coordinates']

    # Load trains
    with open('archive/trains.json', 'r') as f:
        trains_data = json.load(f)

    routes = []
    for feature in trains_data['features']:
        props = feature['properties']
        route = {
            'train_number': props['number'],
            'train_name': props['name'],
            'from_station': props['from_station_code'],
            'to_station': props['to_station_code'],
            'distance': props.get('distance', 0) or 0,
            'type': props['type']
        }
        routes.append(route)

    print(f"Loaded {len(stations)} stations and {len(routes)} train routes")
    return stations, routes, coordinates

def construct_graph(stations, routes):
    """Construct NetworkX graph from railway data"""
    print("Constructing network graph...")

    # Create graph
    G = nx.Graph()

    # Add stations as nodes
    for code, info in stations.items():
        G.add_node(code,
                  name=info['name'],
                  state=info['state'],
                  zone=info['zone'],
                  pos=tuple(info['coordinates']))

    # Add routes as edges
    route_counter = defaultdict(list)

    for route in routes:
        from_station = route['from_station']
        to_station = route['to_station']

        if from_station in stations and to_station in stations:
            route_key = tuple(sorted([from_station, to_station]))
            route_counter[route_key].append(route)

            # Add or update edge
            if G.has_edge(from_station, to_station):
                G[from_station][to_station]['train_count'] += 1
                G[from_station][to_station]['trains'].append(route['train_number'])
            else:
                G.add_edge(from_station, to_station,
                          weight=route['distance'] if route['distance'] > 0 else 1,
                          distance=route['distance'],
                          train_count=1,
                          trains=[route['train_number']])

    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def analyze_graph(G):
    """Analyze basic graph properties"""
    print("\n" + "="*50)
    print("GRAPH ANALYSIS")
    print("="*50)

    # Basic properties
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)

    print(f"Nodes (Stations): {num_nodes}")
    print(f"Edges (Routes): {num_edges}")
    print(f"Network Density: {density:.4f}")

    # Connectivity
    is_connected = nx.is_connected(G)
    print(f"Connected: {is_connected}")

    if is_connected:
        diameter = nx.diameter(G)
        avg_path_length = nx.average_shortest_path_length(G)
        print(f"Network Diameter: {diameter}")
        print(f"Average Path Length: {avg_path_length:.2f}")

    # Top connected stations
    degrees = dict(G.degree())
    top_stations = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]

    print(f"\nTop 10 Most Connected Stations:")
    for station_code, degree in top_stations:
        station_name = G.nodes[station_code]['name']
        print(f"- {station_code} ({station_name}): {degree} connections")

    # Zone analysis
    zones = defaultdict(int)
    for node in G.nodes():
        zone = G.nodes[node]['zone'] or 'Unknown'
        zones[zone] += 1

    print(f"\nStations by Railway Zone:")
    for zone, count in sorted(zones.items(), key=lambda x: x[1], reverse=True):
        print(f"- {zone}: {count} stations")

    return {
        'nodes': num_nodes,
        'edges': num_edges,
        'density': density,
        'connected': is_connected,
        'top_stations': top_stations[:5]
    }

def visualize_graph(G, sample_size=500):
    """Create a visual representation of the graph"""
    print(f"\nCreating graph visualization (sample of {sample_size} stations)...")

    # Sample nodes if graph is too large
    if len(G.nodes()) > sample_size:
        # Select nodes with highest degree
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:sample_size]
        nodes_to_show = [node for node, _ in top_nodes]
        subgraph = G.subgraph(nodes_to_show)
    else:
        subgraph = G

    # Create layout using geographic coordinates
    pos = {}
    for node in subgraph.nodes():
        pos[node] = subgraph.nodes[node]['pos']

    # Set up the plot
    plt.figure(figsize=(16, 12))

    # Color nodes by zone (handle None values)
    zones = list(set(subgraph.nodes[node]['zone'] or 'Unknown' for node in subgraph.nodes()))
    zone_colors = plt.cm.Set3(np.linspace(0, 1, len(zones)))
    zone_color_map = dict(zip(zones, zone_colors))

    node_colors = [zone_color_map[subgraph.nodes[node]['zone'] or 'Unknown'] for node in subgraph.nodes()]

    # Size nodes by degree
    node_sizes = [subgraph.degree(node) * 30 + 50 for node in subgraph.nodes()]

    # Draw the network
    nx.draw_networkx_nodes(subgraph, pos,
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.8)

    # Draw edges with thickness based on train count
    edge_widths = []
    for edge in subgraph.edges():
        train_count = subgraph[edge[0]][edge[1]].get('train_count', 1)
        edge_widths.append(min(train_count * 0.3, 4))

    nx.draw_networkx_edges(subgraph, pos,
                          width=edge_widths,
                          alpha=0.5,
                          edge_color='gray')

    # Add labels for major stations (degree > 10)
    major_stations = {node: node for node in subgraph.nodes() if subgraph.degree(node) > 10}
    nx.draw_networkx_labels(subgraph, pos, major_stations, font_size=8, font_weight='bold')

    plt.title(f"Indian Railway Network Graph\n{len(subgraph.nodes())} stations, {len(subgraph.edges())} routes",
             fontsize=16, fontweight='bold')
    plt.axis('off')

    # Create legend
    legend_elements = []
    for zone in sorted(zones)[:10]:  # Show only first 10 zones
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=zone_color_map[zone],
                                        markersize=10, label=zone))

    plt.legend(handles=legend_elements, loc='upper left', title='Railway Zones', bbox_to_anchor=(0, 1))

    plt.tight_layout()
    plt.savefig('railway_network_simple.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Graph visualization saved as 'railway_network_simple.png'")

def find_shortest_path(G, start_station, end_station):
    """Find shortest path between two stations"""
    try:
        path = nx.shortest_path(G, start_station, end_station, weight='weight')
        path_length = nx.shortest_path_length(G, start_station, end_station, weight='weight')

        # Get station names
        path_with_names = []
        for station in path:
            name = G.nodes[station]['name']
            path_with_names.append(f"{station} ({name})")

        return {
            'path': path,
            'path_with_names': path_with_names,
            'length': path_length,
            'stations_count': len(path)
        }
    except nx.NetworkXNoPath:
        return None

def demo_pathfinding(G):
    """Demonstrate pathfinding capabilities"""
    print("\n" + "="*50)
    print("PATHFINDING DEMONSTRATION")
    print("="*50)

    # Some major station codes for demonstration
    major_stations = ['NDLS', 'CST', 'HWH', 'MAS', 'SBC', 'PUNE', 'BZA']

    # Find available stations from our major list
    available_stations = [station for station in major_stations if station in G.nodes()]

    if len(available_stations) >= 2:
        start = available_stations[0]
        end = available_stations[1]

        print(f"Finding shortest path from {start} to {end}...")

        path_result = find_shortest_path(G, start, end)

        if path_result:
            print(f"Path found!")
            print(f"Route: {' -> '.join(path_result['path'])}")
            print(f"Stations: {path_result['stations_count']}")
            print(f"Total distance/weight: {path_result['length']:.2f}")

            print(f"\nDetailed route:")
            for i, station_info in enumerate(path_result['path_with_names']):
                print(f"{i+1}. {station_info}")
        else:
            print("No path found between these stations")
    else:
        print("Insufficient major stations available for pathfinding demo")

def main():
    """Main execution function"""
    print("="*60)
    print("RAILWAY NETWORK GRAPH CONSTRUCTION")
    print("="*60)

    # Load data
    stations, routes, coordinates = load_railway_data()

    # Construct graph
    G = construct_graph(stations, routes)

    # Analyze graph
    analysis_results = analyze_graph(G)

    # Visualize graph
    visualize_graph(G, sample_size=400)

    # Demonstrate pathfinding
    demo_pathfinding(G)

    print(f"\n" + "="*60)
    print("GRAPH CONSTRUCTION COMPLETE")
    print("="*60)

    print(f"\nYou now have:")
    print(f"- A NetworkX graph object 'G' with {G.number_of_nodes()} stations")
    print(f"- Graph visualization saved as 'railway_network_simple.png'")
    print(f"- Network analysis and pathfinding capabilities")

    return G, stations, routes

if __name__ == "__main__":
    graph, stations_data, routes_data = main()