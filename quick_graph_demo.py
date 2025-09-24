"""
Quick Railway Network Graph Demo
===============================

A fast demonstration of graph construction with minimal processing
"""

import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def quick_graph_demo():
    """Quick demonstration of graph construction"""
    print("="*50)
    print("QUICK RAILWAY GRAPH DEMO")
    print("="*50)

    # Load minimal data for demo
    print("Loading sample data...")

    # Load stations (first 100 for demo)
    with open('archive/stations.json', 'r') as f:
        stations_data = json.load(f)

    stations = {}
    for i, feature in enumerate(stations_data['features'][:200]):  # Sample first 200
        if feature['geometry'] and feature['geometry']['coordinates']:
            code = feature['properties']['code']
            stations[code] = {
                'name': feature['properties']['name'],
                'zone': feature['properties']['zone'] or 'Unknown',
                'coordinates': feature['geometry']['coordinates']
            }

    # Load trains (first 100 for demo)
    with open('archive/trains.json', 'r') as f:
        trains_data = json.load(f)

    routes = []
    for i, feature in enumerate(trains_data['features'][:100]):  # Sample first 100
        props = feature['properties']
        routes.append({
            'from': props['from_station_code'],
            'to': props['to_station_code'],
            'train': props['number'],
            'distance': props.get('distance', 0) or 0
        })

    print(f"Sample: {len(stations)} stations, {len(routes)} routes")

    # Create graph
    G = nx.Graph()

    # Add nodes
    for code, info in stations.items():
        G.add_node(code, **info)

    # Add edges
    edge_count = 0
    for route in routes:
        if route['from'] in stations and route['to'] in stations:
            if not G.has_edge(route['from'], route['to']):
                G.add_edge(route['from'], route['to'],
                          weight=route['distance'] if route['distance'] > 0 else 1)
                edge_count += 1

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Basic analysis
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / len(degrees)
    max_degree = max(degrees.values())

    print(f"Average degree: {avg_degree:.1f}")
    print(f"Maximum degree: {max_degree}")

    # Find top connected stations
    top_stations = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop 5 connected stations:")
    for code, degree in top_stations:
        name = stations[code]['name']
        print(f"- {code} ({name}): {degree} connections")

    # Quick visualization
    print(f"\nCreating visualization...")

    plt.figure(figsize=(12, 8))

    # Use geographic layout
    pos = {}
    for node in G.nodes():
        coords = stations[node]['coordinates']
        pos[node] = (coords[0], coords[1])  # longitude, latitude

    # Color by zone
    zones = list(set(stations[node]['zone'] for node in G.nodes()))
    colors = plt.cm.Set3(range(len(zones)))
    zone_color_map = dict(zip(zones, colors))

    node_colors = [zone_color_map[stations[node]['zone']] for node in G.nodes()]
    node_sizes = [degrees[node] * 100 + 50 for node in G.nodes()]

    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)

    # Label top stations
    labels = {code: code for code, _ in top_stations}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')

    plt.title(f"Railway Network Sample\n{G.number_of_nodes()} stations, {G.number_of_edges()} routes")
    plt.axis('off')

    # Create simple legend
    legend_elements = []
    for zone in zones[:5]:  # Show first 5 zones
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=zone_color_map[zone],
                                        markersize=8, label=zone))
    plt.legend(handles=legend_elements, loc='upper right', title='Zones')

    plt.tight_layout()
    plt.savefig('quick_railway_graph.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Visualization saved as 'quick_railway_graph.png'")

    # Demonstrate pathfinding
    if len(G.nodes()) >= 2:
        nodes_list = list(G.nodes())
        start = nodes_list[0]
        end = nodes_list[-1]

        try:
            path = nx.shortest_path(G, start, end)
            print(f"\nShortest path from {start} to {end}:")
            print(f"Path: {' -> '.join(path)}")
            print(f"Length: {len(path)} stations")
        except:
            print(f"No path found between {start} and {end}")

    print(f"\n" + "="*50)
    print("DEMO COMPLETE")
    print("="*50)

    return G

if __name__ == "__main__":
    graph = quick_graph_demo()