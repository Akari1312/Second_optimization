"""
Railway Network Graph Construction and Visualization
===================================================

This module constructs a graph representation of the Indian Railway network
and provides various visualization options including:
1. Interactive network maps
2. Static network graphs
3. Route analysis and pathfinding
4. Network statistics and analysis

Author: Railway Network Visualization System
"""

import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from geopy.distance import geodesic
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class RailwayNetworkGraph:
    """Main class for constructing and analyzing railway network graphs"""

    def __init__(self):
        self.stations_df = None
        self.trains_df = None
        self.schedules_df = None
        self.network_graph = nx.Graph()
        self.directed_graph = nx.DiGraph()
        self.station_coordinates = {}
        self.route_data = {}

    def load_data(self):
        """Load railway data from JSON files"""
        print("Loading railway network data...")

        # Load stations
        with open('archive/stations.json', 'r') as f:
            stations_data = json.load(f)

        stations_list = []
        for feature in stations_data['features']:
            if feature['geometry'] and feature['geometry']['coordinates']:
                props = feature['properties']
                coords = feature['geometry']['coordinates']

                station = {
                    'code': props['code'],
                    'name': props['name'],
                    'state': props['state'],
                    'zone': props['zone'],
                    'longitude': coords[0],
                    'latitude': coords[1]
                }
                stations_list.append(station)

                # Store coordinates for graph positioning
                self.station_coordinates[props['code']] = (coords[0], coords[1])

        self.stations_df = pd.DataFrame(stations_list)

        # Load trains
        with open('archive/trains.json', 'r') as f:
            trains_data = json.load(f)

        trains_list = []
        for feature in trains_data['features']:
            props = feature['properties']
            coords = feature['geometry']['coordinates'] if feature['geometry'] else []

            train = {
                'number': props['number'],
                'name': props['name'],
                'type': props['type'],
                'from_station_code': props['from_station_code'],
                'to_station_code': props['to_station_code'],
                'from_station_name': props['from_station_name'],
                'to_station_name': props['to_station_name'],
                'distance': props.get('distance', 0) or 0,
                'duration_h': props.get('duration_h', 0) or 0,
                'zone': props['zone'],
                'route_coordinates': coords
            }
            trains_list.append(train)

        self.trains_df = pd.DataFrame(trains_list)

        # Load schedules (sample for performance)
        with open('archive/schedules.json', 'r') as f:
            schedules_data = json.load(f)

        self.schedules_df = pd.DataFrame(schedules_data[:20000])  # Sample for performance

        print(f"Loaded: {len(self.stations_df)} stations, {len(self.trains_df)} trains, {len(self.schedules_df)} schedules")

    def construct_network_graph(self):
        """Construct NetworkX graph from railway data"""
        print("Constructing network graph...")

        # Add stations as nodes
        for _, station in self.stations_df.iterrows():
            self.network_graph.add_node(
                station['code'],
                name=station['name'],
                state=station['state'],
                zone=station['zone'],
                longitude=station['longitude'],
                latitude=station['latitude'],
                pos=(station['longitude'], station['latitude'])
            )

            # Also add to directed graph
            self.directed_graph.add_node(
                station['code'],
                name=station['name'],
                state=station['state'],
                zone=station['zone'],
                longitude=station['longitude'],
                latitude=station['latitude']
            )

        # Add train routes as edges
        edge_weights = defaultdict(list)

        for _, train in self.trains_df.iterrows():
            from_station = train['from_station_code']
            to_station = train['to_station_code']

            if from_station in self.station_coordinates and to_station in self.station_coordinates:
                # Calculate weight (can be distance, time, or combination)
                weight = train['distance'] if train['distance'] > 0 else 1

                # Store route information
                route_key = f"{from_station}-{to_station}"
                edge_weights[route_key].append({
                    'train_number': train['number'],
                    'train_name': train['name'],
                    'train_type': train['type'],
                    'distance': train['distance'],
                    'duration': train['duration_h'],
                    'weight': weight
                })

                # Add edge to undirected graph (for general connectivity)
                if self.network_graph.has_edge(from_station, to_station):
                    # Update existing edge with additional train
                    self.network_graph[from_station][to_station]['trains'].append(train['number'])
                    self.network_graph[from_station][to_station]['weight'] = min(
                        self.network_graph[from_station][to_station]['weight'], weight
                    )
                else:
                    self.network_graph.add_edge(
                        from_station,
                        to_station,
                        weight=weight,
                        distance=train['distance'],
                        trains=[train['number']],
                        train_count=1
                    )

                # Add directed edge
                self.directed_graph.add_edge(
                    from_station,
                    to_station,
                    weight=weight,
                    distance=train['distance'],
                    train_number=train['number'],
                    train_name=train['name'],
                    train_type=train['type']
                )

        # Update edge attributes with train counts
        for edge in self.network_graph.edges():
            from_station, to_station = edge
            self.network_graph[from_station][to_station]['train_count'] = len(
                self.network_graph[from_station][to_station]['trains']
            )

        print(f"Graph constructed: {self.network_graph.number_of_nodes()} nodes, {self.network_graph.number_of_edges()} edges")

    def analyze_network_properties(self):
        """Analyze basic network properties"""
        print("\n" + "="*50)
        print("NETWORK ANALYSIS")
        print("="*50)

        # Basic metrics
        num_nodes = self.network_graph.number_of_nodes()
        num_edges = self.network_graph.number_of_edges()

        print(f"Nodes (Stations): {num_nodes}")
        print(f"Edges (Routes): {num_edges}")
        print(f"Density: {nx.density(self.network_graph):.4f}")

        # Connectivity
        if nx.is_connected(self.network_graph):
            print("Network is connected")
            diameter = nx.diameter(self.network_graph)
            avg_path_length = nx.average_shortest_path_length(self.network_graph)
            print(f"Diameter: {diameter}")
            print(f"Average shortest path length: {avg_path_length:.2f}")
        else:
            print("Network is NOT connected")
            components = list(nx.connected_components(self.network_graph))
            print(f"Number of connected components: {len(components)}")
            largest_component_size = max(len(comp) for comp in components)
            print(f"Largest component size: {largest_component_size}")

        # Degree analysis
        degrees = dict(self.network_graph.degree())
        avg_degree = np.mean(list(degrees.values()))
        max_degree = max(degrees.values())

        print(f"\nDegree Analysis:")
        print(f"Average degree: {avg_degree:.2f}")
        print(f"Maximum degree: {max_degree}")

        # Find hub stations (high degree)
        hub_stations = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nTop 10 Hub Stations:")
        for station_code, degree in hub_stations:
            station_name = self.stations_df[self.stations_df['code'] == station_code]['name'].iloc[0]
            print(f"- {station_code} ({station_name}): {degree} connections")

        # Centrality measures
        print(f"\nCentrality Analysis:")

        # Betweenness centrality (most important for routing)
        if self.network_graph.number_of_nodes() < 1000:  # Only for smaller networks
            betweenness = nx.betweenness_centrality(self.network_graph)
            top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"Top 5 stations by betweenness centrality:")
            for station_code, centrality in top_betweenness:
                station_name = self.stations_df[self.stations_df['code'] == station_code]['name'].iloc[0]
                print(f"- {station_code} ({station_name}): {centrality:.4f}")

        return {
            'nodes': num_nodes,
            'edges': num_edges,
            'density': nx.density(self.network_graph),
            'is_connected': nx.is_connected(self.network_graph),
            'avg_degree': avg_degree,
            'hub_stations': hub_stations[:5]
        }

    def visualize_network_matplotlib(self, sample_size=500, figsize=(15, 10)):
        """Create static network visualization using matplotlib"""
        print(f"Creating network visualization (sample of {sample_size} stations)...")

        # Sample nodes for visualization if network is too large
        if len(self.network_graph.nodes()) > sample_size:
            # Select nodes with highest degree (most connected stations)
            degrees = dict(self.network_graph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:sample_size]
            nodes_to_include = [node for node, degree in top_nodes]
            subgraph = self.network_graph.subgraph(nodes_to_include)
        else:
            subgraph = self.network_graph

        # Create figure
        plt.figure(figsize=figsize)

        # Get positions from coordinates
        pos = {}
        for node in subgraph.nodes():
            if node in self.station_coordinates:
                pos[node] = self.station_coordinates[node]

        # Draw network
        # Draw nodes colored by zone
        zones = set()
        for node in subgraph.nodes():
            zone = subgraph.nodes[node].get('zone', 'Unknown')
            zones.add(zone)

        zone_colors = plt.cm.Set3(np.linspace(0, 1, len(zones)))
        zone_color_map = dict(zip(zones, zone_colors))

        node_colors = []
        for node in subgraph.nodes():
            zone = subgraph.nodes[node].get('zone', 'Unknown')
            node_colors.append(zone_color_map[zone])

        # Node sizes based on degree
        node_sizes = [subgraph.degree(node) * 20 + 50 for node in subgraph.nodes()]

        # Draw the network
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors,
                              node_size=node_sizes, alpha=0.7)

        # Draw edges with varying thickness based on train count
        edge_weights = []
        for edge in subgraph.edges():
            train_count = subgraph[edge[0]][edge[1]].get('train_count', 1)
            edge_weights.append(min(train_count / 5, 3))  # Normalize thickness

        nx.draw_networkx_edges(subgraph, pos, width=edge_weights, alpha=0.5, edge_color='gray')

        # Add labels for major stations only
        major_stations = [node for node in subgraph.nodes() if subgraph.degree(node) > 5]
        labels = {node: node for node in major_stations}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, font_weight='bold')

        plt.title(f"Indian Railway Network Graph\n({len(subgraph.nodes())} stations, {len(subgraph.edges())} routes)",
                 fontsize=16, fontweight='bold')
        plt.axis('off')

        # Add legend for zones
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=zone_color_map[zone],
                                    markersize=10, label=zone) for zone in sorted(zones)[:10]]
        plt.legend(handles=legend_elements, loc='upper right', title='Railway Zones')

        plt.tight_layout()
        plt.savefig('railway_network_graph.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_interactive_map(self, sample_size=1000):
        """Create interactive map using Folium"""
        print(f"Creating interactive map (sample of {sample_size} stations)...")

        # Sample stations for performance
        stations_sample = self.stations_df.head(sample_size)

        # Create base map centered on India
        center_lat = stations_sample['latitude'].mean()
        center_lon = stations_sample['longitude'].mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=6,
                      tiles='OpenStreetMap')

        # Add stations as markers
        zone_colors = {
            'NR': 'red', 'SR': 'blue', 'ER': 'green', 'WR': 'orange',
            'CR': 'purple', 'NER': 'pink', 'NFR': 'gray', 'NCR': 'brown',
            'SCR': 'lightred', 'SER': 'lightblue', 'WCR': 'lightgreen',
            'ECR': 'cadetblue', 'ECoR': 'darkred', 'NWR': 'darkblue',
            'SWR': 'darkgreen', 'SECR': 'darkpurple', 'NF': 'black'
        }

        for _, station in stations_sample.iterrows():
            color = zone_colors.get(station['zone'], 'blue')

            folium.CircleMarker(
                location=[station['latitude'], station['longitude']],
                radius=3,
                popup=f"{station['name']} ({station['code']})<br>Zone: {station['zone']}<br>State: {station['state']}",
                color=color,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)

        # Add some major train routes
        major_trains = self.trains_df[self.trains_df['distance'] > 1000].head(50)

        for _, train in major_trains.iterrows():
            from_station = self.stations_df[self.stations_df['code'] == train['from_station_code']]
            to_station = self.stations_df[self.stations_df['code'] == train['to_station_code']]

            if len(from_station) > 0 and len(to_station) > 0:
                from_coords = [from_station.iloc[0]['latitude'], from_station.iloc[0]['longitude']]
                to_coords = [to_station.iloc[0]['latitude'], to_station.iloc[0]['longitude']]

                folium.PolyLine(
                    locations=[from_coords, to_coords],
                    color='red',
                    weight=2,
                    opacity=0.6,
                    popup=f"{train['name']} ({train['number']})<br>Distance: {train['distance']}km"
                ).add_to(m)

        # Add legend
        legend_html = '''
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 150px; height: 150px;
                    background-color: white; border:2px solid grey; z-index:9999;
                    font-size:14px; padding: 10px">
        <h4>Railway Zones</h4>
        '''
        for zone, color in list(zone_colors.items())[:8]:
            legend_html += f'<p><span style="color:{color};">●</span> {zone}</p>'
        legend_html += '</div>'

        m.get_root().html.add_child(folium.Element(legend_html))

        # Save map
        m.save('railway_network_interactive_map.html')
        print("Interactive map saved as 'railway_network_interactive_map.html'")

        return m

    def create_plotly_network(self, sample_size=800):
        """Create interactive network visualization using Plotly"""
        print(f"Creating Plotly network visualization (sample of {sample_size} stations)...")

        # Sample nodes for performance
        if len(self.network_graph.nodes()) > sample_size:
            degrees = dict(self.network_graph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:sample_size]
            nodes_to_include = [node for node, degree in top_nodes]
            subgraph = self.network_graph.subgraph(nodes_to_include)
        else:
            subgraph = self.network_graph

        # Prepare node data
        node_trace = go.Scatter(
            x=[], y=[], text=[], mode='markers+text',
            hoverinfo='text', textposition="middle center",
            marker=dict(size=[], color=[], colorscale='Viridis',
                       showscale=True, colorbar=dict(title="Degree"))
        )

        # Prepare edge data
        edge_trace = go.Scatter(x=[], y=[], mode='lines',
                              line=dict(width=0.5, color='#888'), hoverinfo='none')

        # Add edges
        for edge in subgraph.edges():
            from_node, to_node = edge
            if from_node in self.station_coordinates and to_node in self.station_coordinates:
                x0, y0 = self.station_coordinates[from_node]
                x1, y1 = self.station_coordinates[to_node]
                edge_trace['x'] += tuple([x0, x1, None])
                edge_trace['y'] += tuple([y0, y1, None])

        # Add nodes
        for node in subgraph.nodes():
            if node in self.station_coordinates:
                x, y = self.station_coordinates[node]
                node_trace['x'] += tuple([x])
                node_trace['y'] += tuple([y])

                # Node info
                degree = subgraph.degree(node)
                station_name = subgraph.nodes[node].get('name', node)
                state = subgraph.nodes[node].get('state', 'Unknown')
                zone = subgraph.nodes[node].get('zone', 'Unknown')

                node_info = f"{station_name}<br>Code: {node}<br>State: {state}<br>Zone: {zone}<br>Connections: {degree}"
                node_trace['text'] += tuple([node_info])
                node_trace['marker']['size'] += tuple([max(degree/2, 5)])
                node_trace['marker']['color'] += tuple([degree])

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                        title='Indian Railway Network - Interactive Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Node size and color represent degree (number of connections)",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor="left", yanchor="bottom",
                            font=dict(size=12))],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        width=1000, height=700
                        ))

        fig.write_html('railway_network_plotly.html')
        fig.show()
        print("Interactive Plotly network saved as 'railway_network_plotly.html'")

        return fig

    def find_shortest_path(self, from_station, to_station, weight='weight'):
        """Find shortest path between two stations"""
        try:
            path = nx.shortest_path(self.network_graph, from_station, to_station, weight=weight)
            path_length = nx.shortest_path_length(self.network_graph, from_station, to_station, weight=weight)

            # Get station names for the path
            path_names = []
            for station_code in path:
                station_name = self.stations_df[self.stations_df['code'] == station_code]['name']
                if len(station_name) > 0:
                    path_names.append(f"{station_code} ({station_name.iloc[0]})")
                else:
                    path_names.append(station_code)

            return {
                'path_codes': path,
                'path_names': path_names,
                'path_length': path_length,
                'num_stations': len(path)
            }
        except nx.NetworkXNoPath:
            return None

    def get_network_statistics(self):
        """Get comprehensive network statistics"""
        stats = {
            'basic_stats': {
                'total_stations': len(self.stations_df),
                'total_trains': len(self.trains_df),
                'total_routes': self.network_graph.number_of_edges(),
                'network_density': nx.density(self.network_graph)
            },
            'geographic_distribution': self.stations_df['state'].value_counts().to_dict(),
            'zone_distribution': self.stations_df['zone'].value_counts().to_dict(),
            'train_type_distribution': self.trains_df['type'].value_counts().to_dict()
        }

        return stats

def main():
    """Main function to demonstrate graph construction and visualization"""
    print("="*60)
    print("RAILWAY NETWORK GRAPH CONSTRUCTION")
    print("="*60)

    # Initialize and load data
    railway_graph = RailwayNetworkGraph()
    railway_graph.load_data()

    # Construct network graph
    railway_graph.construct_network_graph()

    # Analyze network properties
    network_analysis = railway_graph.analyze_network_properties()

    # Create visualizations
    print(f"\nCreating visualizations...")

    # 1. Static matplotlib visualization
    railway_graph.visualize_network_matplotlib(sample_size=300)

    # 2. Interactive Folium map
    railway_graph.create_interactive_map(sample_size=500)

    # 3. Interactive Plotly network
    railway_graph.create_plotly_network(sample_size=400)

    # 4. Demonstrate pathfinding
    print(f"\nDemonstrating pathfinding...")

    # Find path between major stations
    sample_path = railway_graph.find_shortest_path('NDLS', 'CST')  # New Delhi to Mumbai CST
    if sample_path:
        print(f"Shortest path from NDLS to CST:")
        print(f"Route: {' -> '.join(sample_path['path_names'])}")
        print(f"Total stations: {sample_path['num_stations']}")
        print(f"Path length (weighted): {sample_path['path_length']:.2f}")

    # Get network statistics
    stats = railway_graph.get_network_statistics()
    print(f"\nNetwork Statistics:")
    print(f"- Total stations: {stats['basic_stats']['total_stations']}")
    print(f"- Total routes: {stats['basic_stats']['total_routes']}")
    print(f"- Network density: {stats['basic_stats']['network_density']:.4f}")

    print(f"\nVisualization files created:")
    print(f"- railway_network_graph.png (static)")
    print(f"- railway_network_interactive_map.html (interactive map)")
    print(f"- railway_network_plotly.html (interactive network)")

    print(f"\n" + "="*60)
    print("GRAPH CONSTRUCTION COMPLETE")
    print("="*60)

    return railway_graph

if __name__ == "__main__":
    graph = main()