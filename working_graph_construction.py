"""
Working Railway Network Graph Construction
==========================================

This creates a proper graph by matching train routes with actual stations
"""

import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def load_and_match_data():
    """Load data and properly match stations with routes"""
    print("Loading and matching railway data...")

    # Load all stations
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

    # Load trains and filter to those with valid station codes
    with open('archive/trains.json', 'r') as f:
        trains_data = json.load(f)

    valid_routes = []
    station_usage = defaultdict(int)

    for feature in trains_data['features']:
        props = feature['properties']
        from_station = props['from_station_code']
        to_station = props['to_station_code']

        # Only include routes where both stations exist in our station data
        if from_station in stations and to_station in stations:
            route = {
                'train_number': props['number'],
                'train_name': props['name'],
                'from_station': from_station,
                'to_station': to_station,
                'distance': props.get('distance', 0) or 0,
                'type': props['type']
            }
            valid_routes.append(route)
            station_usage[from_station] += 1
            station_usage[to_station] += 1

    print(f"Found {len(valid_routes)} valid routes connecting {len(station_usage)} stations")

    # Filter stations to only those used in routes
    used_stations = {code: info for code, info in stations.items() if code in station_usage}

    return used_stations, valid_routes

def create_network_graph(stations, routes):
    """Create NetworkX graph from matched data"""
    print("Creating network graph...")

    G = nx.Graph()

    # Add stations as nodes
    for code, info in stations.items():
        G.add_node(code,
                  name=info['name'],
                  zone=info['zone'],
                  state=info['state'],
                  coordinates=info['coordinates'])

    # Add routes as edges
    route_counter = defaultdict(list)

    for route in routes:
        from_station = route['from_station']
        to_station = route['to_station']

        # Create unique edge key
        edge_key = tuple(sorted([from_station, to_station]))
        route_counter[edge_key].append(route)

        # Add or update edge
        if G.has_edge(from_station, to_station):
            G[from_station][to_station]['train_count'] += 1
            G[from_station][to_station]['trains'].append(route['train_number'])
            # Keep the shortest distance as edge weight
            if route['distance'] > 0:
                current_distance = G[from_station][to_station]['weight']
                G[from_station][to_station]['weight'] = min(current_distance, route['distance'])
        else:
            G.add_edge(from_station, to_station,
                      weight=route['distance'] if route['distance'] > 0 else 100,
                      train_count=1,
                      trains=[route['train_number']],
                      distance=route['distance'])

    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def analyze_network(G, stations):
    """Analyze the network properties"""
    print("\n" + "="*50)
    print("NETWORK ANALYSIS")
    print("="*50)

    # Basic properties
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    print(f"Stations: {num_nodes}")
    print(f"Direct Routes: {num_edges}")
    print(f"Network Density: {nx.density(G):.6f}")
    print(f"Connected: {nx.is_connected(G)}")

    if nx.is_connected(G) and num_nodes < 1000:  # Only for manageable sizes
        diameter = nx.diameter(G)
        avg_path_length = nx.average_shortest_path_length(G)
        print(f"Network Diameter: {diameter}")
        print(f"Average Path Length: {avg_path_length:.2f}")

    # Degree analysis
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / len(degrees)
    max_degree = max(degrees.values())

    print(f"\nConnectivity Analysis:")
    print(f"Average connections per station: {avg_degree:.1f}")
    print(f"Maximum connections: {max_degree}")

    # Top connected stations
    top_stations = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 Hub Stations:")
    for station_code, degree in top_stations:
        station_name = stations[station_code]['name']
        zone = stations[station_code]['zone']
        print(f"- {station_code} ({station_name}, {zone}): {degree} direct connections")

    # Zone distribution
    zone_counts = defaultdict(int)
    for code in G.nodes():
        zone = stations[code]['zone']
        zone_counts[zone] += 1

    print(f"\nStations by Railway Zone:")
    for zone, count in sorted(zone_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"- {zone}: {count} stations")

    return {
        'nodes': num_nodes,
        'edges': num_edges,
        'avg_degree': avg_degree,
        'top_hubs': top_stations[:5]
    }

def visualize_network(G, stations, max_nodes=300):
    """Create network visualization"""
    print(f"\nCreating network visualization...")

    # Sample the graph if it's too large
    if G.number_of_nodes() > max_nodes:
        # Select top connected nodes
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        nodes_to_show = [node for node, _ in top_nodes]
        subgraph = G.subgraph(nodes_to_show)
        print(f"Showing top {len(subgraph.nodes())} connected stations")
    else:
        subgraph = G

    # Create figure
    plt.figure(figsize=(16, 12))

    # Geographic layout
    pos = {}
    for node in subgraph.nodes():
        coords = stations[node]['coordinates']
        pos[node] = (coords[0], coords[1])  # (longitude, latitude)

    # Color nodes by zone
    zones = list(set(stations[node]['zone'] for node in subgraph.nodes()))
    colors = plt.cm.tab20(range(len(zones)))
    zone_color_map = dict(zip(zones, colors))

    node_colors = [zone_color_map[stations[node]['zone']] for node in subgraph.nodes()]

    # Size nodes by degree
    degrees = dict(subgraph.degree())
    node_sizes = [max(degrees[node] * 20, 30) for node in subgraph.nodes()]

    # Draw network
    nx.draw_networkx_nodes(subgraph, pos,
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.7)

    # Draw edges with thickness based on train count
    edge_widths = []
    for edge in subgraph.edges():
        train_count = subgraph[edge[0]][edge[1]].get('train_count', 1)
        edge_widths.append(min(train_count * 0.5, 3))

    nx.draw_networkx_edges(subgraph, pos,
                          width=edge_widths,
                          alpha=0.4,
                          edge_color='gray')

    # Label major hubs (degree > 5)
    major_hubs = {node: node for node in subgraph.nodes() if subgraph.degree(node) > 5}
    nx.draw_networkx_labels(subgraph, pos, major_hubs, font_size=8, font_weight='bold')

    plt.title(f"Indian Railway Network Graph\n{len(subgraph.nodes())} stations, {len(subgraph.edges())} direct routes",
             fontsize=16, fontweight='bold')
    plt.axis('off')

    # Create legend
    legend_elements = []
    for zone in sorted(zones)[:12]:  # Show max 12 zones
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=zone_color_map[zone],
                                        markersize=8, label=zone))

    plt.legend(handles=legend_elements, loc='upper left', title='Railway Zones',
              bbox_to_anchor=(0, 1), ncol=2)

    plt.tight_layout()
    plt.savefig('railway_network_final.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    print("Network visualization saved as 'railway_network_final.png'")

def demonstrate_pathfinding(G, stations):
    """Demonstrate shortest path finding"""
    print("\n" + "="*50)
    print("PATHFINDING DEMONSTRATION")
    print("="*50)

    # Find some major stations for pathfinding demo
    major_stations = ['NDLS', 'CSTM', 'HWH', 'MAS', 'SBC', 'ADI', 'PUNE', 'LTT']
    available_major = [station for station in major_stations if station in G.nodes()]

    if len(available_major) >= 2:
        start_station = available_major[0]
        end_station = available_major[1]

        try:
            # Find shortest path
            path = nx.shortest_path(G, start_station, end_station)
            path_length = nx.shortest_path_length(G, start_station, end_station)

            print(f"Shortest path from {start_station} to {end_station}:")
            print(f"Route: {' -> '.join(path)}")
            print(f"Stations: {len(path)}")
            print(f"Total weight: {path_length:.0f}")

            # Show detailed route
            print(f"\nDetailed Route:")
            for i, station_code in enumerate(path):
                station_name = stations[station_code]['name']
                zone = stations[station_code]['zone']
                print(f"{i+1:2d}. {station_code} - {station_name} ({zone})")

        except nx.NetworkXNoPath:
            print(f"No path found between {start_station} and {end_station}")
    else:
        print("Insufficient major stations available for demonstration")

def main():
    """Main execution function"""
    print("="*60)
    print("RAILWAY NETWORK GRAPH CONSTRUCTION")
    print("="*60)

    # Load and match data
    stations, routes = load_and_match_data()

    # Create graph
    G = create_network_graph(stations, routes)

    # Analyze network
    analysis = analyze_network(G, stations)

    # Visualize network
    visualize_network(G, stations, max_nodes=250)

    # Demonstrate pathfinding
    demonstrate_pathfinding(G, stations)

    print(f"\n" + "="*60)
    print("GRAPH CONSTRUCTION COMPLETE")
    print("="*60)

    print(f"\nSUMMARY:")
    print(f"✓ Constructed graph with {analysis['nodes']} stations")
    print(f"✓ {analysis['edges']} direct train routes")
    print(f"✓ Average {analysis['avg_degree']:.1f} connections per station")
    print(f"✓ Network visualization saved as 'railway_network_final.png'")
    print(f"✓ Pathfinding capabilities demonstrated")

    return G, stations, routes

if __name__ == "__main__":
    graph, station_data, route_data = main()