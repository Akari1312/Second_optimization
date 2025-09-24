"""
GRANDEST FINAL RAILWAY MILP OPTIMIZER
=====================================
Comprehensive Mixed Integer Linear Programming optimization for railway scheduling
using real data from trains.json with priority-based hierarchical optimization.

Author: Claude Code
Date: 2025-09-24
Version: 1.0 - GRANDEST FINAL
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pulp as lp
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class GrandestFinalRailwayMILPOptimizer:
    """
    Ultra-comprehensive MILP optimizer for railway scheduling with priority hierarchy.
    This is the GRANDEST FINAL implementation using real trains.json data.
    """

    def __init__(self):
        self.train_data = []
        self.priority_weights = self._define_priority_hierarchy()
        self.stations = set()
        self.platform_capacity = 4  # platforms per station
        self.time_slots = 24 * 60 // 15  # 15-minute slots in a day
        self.optimization_results = {}

    def _define_priority_hierarchy(self):
        """
        Define the exact priority hierarchy as specified by the user.
        Higher weights = Higher priority in optimization.
        """
        return {
            # HIGH PRIORITY TRAINS
            'Raj': 100,      # Rajdhani Express - Flagship premium
            'Shtb': 95,      # Shatabdi Express - High-speed daytime
            'JShtb': 90,     # Jan Shatabdi - Cheaper Shatabdi version
            'Drnt': 85,      # Duronto Express - Non-stop long distance
            'GR': 80,        # Garib Rath - Affordable AC trains

            # MEDIUM PRIORITY TRAINS
            'SF': 60,        # Superfast Express - Speed > 55 km/h
            'Exp': 50,       # Express - Backbone of long-distance travel
            'Mail': 45,      # Mail trains
            'Hyd': 40,       # Hyderabad trains
            'Del': 40,       # Delhi trains
            'SKr': 35,       # Shaktipunj trains
            'Klkt': 35,      # Kolkata trains

            # LOW PRIORITY TRAINS
            'Pass': 20,      # Passenger trains - Stop everywhere
            'MEMU': 15,      # Suburban/regional trains
            'DEMU': 15,      # Diesel Electric Multiple Unit
            'Toy': 10,       # Toy trains - Heritage routes
            '': 5            # Empty type - lowest priority
        }

    def load_and_process_data(self, json_file_path):
        """
        Load and process the massive trains.json file efficiently.
        Extract key features for optimization.
        """
        print("Loading GRANDEST FINAL train data...")

        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            features = data.get('features', [])
            print(f"Processing {len(features)} train records...")

            for feature in features:
                if feature.get('type') == 'Feature':
                    props = feature.get('properties', {})

                    # Extract essential train information
                    train_info = {
                        'train_number': props.get('number', 'UNKNOWN'),
                        'train_name': props.get('name', 'UNNAMED'),
                        'train_type': props.get('type', ''),
                        'from_station': props.get('from_station_code', ''),
                        'to_station': props.get('to_station_code', ''),
                        'departure_time': props.get('departure', '00:00:00'),
                        'arrival_time': props.get('arrival', '00:00:00'),
                        'duration_hours': props.get('duration_h', 0) or 0,
                        'duration_minutes': props.get('duration_m', 0) or 0,
                        'distance': props.get('distance', 0) or 0,
                        'zone': props.get('zone', 'UNKNOWN'),
                        'priority_weight': self.priority_weights.get(props.get('type', ''), 5),
                        'has_ac': int((props.get('first_ac', 0) or 0) or (props.get('second_ac', 0) or 0) or (props.get('third_ac', 0) or 0)),
                        'has_sleeper': props.get('sleeper', 0) or 0,
                        'coordinates': feature.get('geometry', {}).get('coordinates', [])
                    }

                    # Calculate speed (km/h) - handle None values
                    duration_h = train_info['duration_hours'] or 0
                    duration_m = train_info['duration_minutes'] or 0
                    distance = train_info['distance'] or 0

                    total_minutes = duration_h * 60 + duration_m
                    if total_minutes > 0 and distance > 0:
                        train_info['speed_kmph'] = (distance / total_minutes) * 60
                    else:
                        train_info['speed_kmph'] = 0

                    self.train_data.append(train_info)
                    self.stations.add(train_info['from_station'])
                    self.stations.add(train_info['to_station'])

            print(f"Successfully processed {len(self.train_data)} trains")
            print(f"Found {len(self.stations)} unique stations")

            return True

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def analyze_priority_distribution(self):
        """
        Analyze the distribution of trains by priority levels.
        """
        priority_counts = defaultdict(int)
        priority_categories = {
            'HIGH_PRIORITY': ['Raj', 'Shtb', 'JShtb', 'Drnt', 'GR'],
            'MEDIUM_PRIORITY': ['SF', 'Exp', 'Mail', 'Hyd', 'Del', 'SKr', 'Klkt'],
            'LOW_PRIORITY': ['Pass', 'MEMU', 'DEMU', 'Toy', '']
        }

        category_counts = defaultdict(int)

        for train in self.train_data:
            train_type = train['train_type']
            priority_counts[train_type] += 1

            for category, types in priority_categories.items():
                if train_type in types:
                    category_counts[category] += 1
                    break

        print("\n  PRIORITY DISTRIBUTION ANALYSIS:")
        print("=" * 50)

        for category, count in category_counts.items():
            percentage = (count / len(self.train_data)) * 100
            print(f"{category:15}: {count:5} trains ({percentage:5.1f}%)")

        print(f"\nTop Train Types by Count:")
        sorted_priorities = sorted(priority_counts.items(), key=lambda x: x[1], reverse=True)
        for train_type, count in sorted_priorities[:10]:
            priority = self.priority_weights.get(train_type, 5)
            print(f"  {train_type:8} ({priority:3}): {count:4} trains")

        return priority_counts, category_counts

    def create_sample_optimization_scenario(self, num_trains=100):
        """
        Create a representative sample scenario for MILP optimization.
        Focus on high-conflict routes and time periods.
        """
        print(f"\n  Creating optimization scenario with {num_trains} trains...")

        # Sort trains by priority (highest first) and select diverse sample
        sorted_trains = sorted(self.train_data, key=lambda x: x['priority_weight'], reverse=True)

        # Stratified sampling to ensure representation from all priority levels
        high_priority = [t for t in sorted_trains if t['priority_weight'] >= 80]
        medium_priority = [t for t in sorted_trains if 30 <= t['priority_weight'] < 80]
        low_priority = [t for t in sorted_trains if t['priority_weight'] < 30]

        # Sample proportionally
        sample_trains = []
        sample_trains.extend(high_priority[:num_trains//3])
        sample_trains.extend(medium_priority[:num_trains//3])
        sample_trains.extend(low_priority[:num_trains//3])

        # Fill remaining slots with any available trains
        while len(sample_trains) < num_trains and len(sample_trains) < len(sorted_trains):
            for train in sorted_trains:
                if train not in sample_trains:
                    sample_trains.append(train)
                    if len(sample_trains) >= num_trains:
                        break

        return sample_trains[:num_trains]

    def time_to_slot(self, time_str):
        """
        Convert HH:MM:SS time string to 15-minute slot index.
        """
        try:
            time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
            minutes_from_midnight = time_obj.hour * 60 + time_obj.minute
            return minutes_from_midnight // 15
        except:
            return 0

    def solve_milp_optimization(self, sample_trains):
        """
        Solve the GRANDEST FINAL MILP optimization problem with comprehensive:
        1. SCHEDULE OPTIMIZATION: Optimal departure times, platform assignments
        2. CAPACITY OPTIMIZATION: Network graph, demand patterns, infrastructure limits
        3. PRIORITY-BASED OPTIMIZATION: Hierarchical train priority system
        Objective: Maximize passenger capacity + minimize delays + maximize throughput
        """
        print("\n  Solving GRANDEST FINAL MILP optimization...")
        print("  COMPREHENSIVE OPTIMIZATION INCLUDES:")
        print("     Schedule Optimization: Departure times, platform assignments")
        print("      Capacity Optimization: Network limits, demand patterns")
        print("     Priority Optimization: Hierarchical train priority system")
        print("=" * 60)

        # Create the optimization problem
        problem = lp.LpProblem("GRANDEST_FINAL_Railway_Scheduling", lp.LpMaximize)

        # Extract unique stations and create network graph
        sample_stations = list(set([t['from_station'] for t in sample_trains] +
                                 [t['to_station'] for t in sample_trains]))

        # Create route network for capacity analysis
        routes = {}
        for i, train in enumerate(sample_trains):
            route_key = (train['from_station'], train['to_station'])
            if route_key not in routes:
                routes[route_key] = []
            routes[route_key].append(i)

        print(f"  COMPREHENSIVE OPTIMIZATION PARAMETERS:")
        print(f"     Trains: {len(sample_trains)}")
        print(f"     Stations: {len(sample_stations)}")
        print(f"     Routes: {len(routes)}")
        print(f"     Time slots: {self.time_slots} (15-min intervals)")
        print(f"     Platform capacity: {self.platform_capacity} per station")

        # ENHANCED Decision Variables for Comprehensive Optimization
        print("\n  Creating COMPREHENSIVE decision variables...")

        # x[train, station, slot, platform] = 1 if train uses platform at station during slot
        x = {}
        for i, train in enumerate(sample_trains):
            for station in [train['from_station'], train['to_station']]:
                if station in sample_stations:
                    for slot in range(self.time_slots):
                        for platform in range(self.platform_capacity):
                            var_name = f"x_{i}_{station}_{slot}_{platform}"
                            x[(i, station, slot, platform)] = lp.LpVariable(var_name, cat='Binary')

        # y[train] = 1 if train is scheduled (not cancelled)
        y = {}
        for i, train in enumerate(sample_trains):
            y[i] = lp.LpVariable(f"y_{i}", cat='Binary')

        # delay[train] = delay in time slots for train
        delay = {}
        for i, train in enumerate(sample_trains):
            delay[i] = lp.LpVariable(f"delay_{i}", lowBound=0, cat='Integer')

        # CAPACITY OPTIMIZATION VARIABLES
        # route_capacity[route] = total capacity utilization on route
        route_capacity = {}
        for route in routes:
            route_capacity[route] = lp.LpVariable(f"route_cap_{route[0]}_{route[1]}",
                                                lowBound=0, cat='Continuous')

        # platform_utilization[station, slot] = platform utilization at station during slot
        platform_util = {}
        for station in sample_stations:
            for slot in range(self.time_slots):
                platform_util[(station, slot)] = lp.LpVariable(f"plat_util_{station}_{slot}",
                                                              lowBound=0, upBound=1, cat='Continuous')

        print("  COMPREHENSIVE decision variables created")

        # MULTI-OBJECTIVE Function: Schedule + Capacity + Priority Optimization
        print("\n  Setting up COMPREHENSIVE objective function...")

        # 1. PRIORITY OPTIMIZATION: Maximize weighted priority throughput
        priority_term = lp.lpSum([sample_trains[i]['priority_weight'] * y[i]
                                 for i in range(len(sample_trains))])

        # 2. SCHEDULE OPTIMIZATION: Minimize delays
        delay_penalty_term = lp.lpSum([8 * delay[i] for i in range(len(sample_trains))])

        # 3. CAPACITY OPTIMIZATION: Maximize passenger capacity and network utilization
        capacity_term = lp.lpSum([sample_trains[i]['distance'] * y[i] *
                                (1 + sample_trains[i]['has_ac'] + sample_trains[i]['has_sleeper'])
                                for i in range(len(sample_trains))])

        # 4. THROUGHPUT OPTIMIZATION: Maximize platform utilization efficiency
        throughput_term = lp.lpSum([platform_util[(station, slot)] * 100
                                  for station in sample_stations
                                  for slot in range(self.time_slots)])

        # Combined multi-objective function
        problem += (priority_term * 1.0 +           # Priority optimization
                   capacity_term * 0.01 +          # Capacity optimization
                   throughput_term * 0.1 -         # Throughput optimization
                   delay_penalty_term * 1.0)       # Schedule optimization

        print("  COMPREHENSIVE objective: Priority + Capacity + Throughput - Delays")

        # COMPREHENSIVE Constraints
        print("\n   Adding COMPREHENSIVE constraints...")
        constraint_count = 0

        # 1. CAPACITY OPTIMIZATION: Platform capacity constraints
        for station in sample_stations:
            for slot in range(self.time_slots):
                trains_at_station_slot = lp.lpSum([x.get((i, station, slot, platform), 0)
                                                 for i in range(len(sample_trains))
                                                 for platform in range(self.platform_capacity)])

                # Platform utilization calculation
                problem += (platform_util[(station, slot)] ==
                          trains_at_station_slot / self.platform_capacity)
                constraint_count += 1

                # Hard capacity limit
                problem += trains_at_station_slot <= self.platform_capacity
                constraint_count += 1

        # 2. SCHEDULE OPTIMIZATION: Train scheduling constraints
        for i, train in enumerate(sample_trains):
            # If train is scheduled, it must use exactly one platform at departure
            departure_slot = self.time_to_slot(train['departure_time'])
            departure_station = train['from_station']

            if departure_station in sample_stations and departure_slot < self.time_slots:
                # Simplified scheduling: train uses one platform at scheduled time + delay
                scheduled_departure = lp.lpSum([x.get((i, departure_station, min(departure_slot + d, self.time_slots-1), platform), 0)
                                               for d in range(17)
                                               for platform in range(self.platform_capacity)
                                               if departure_slot + d < self.time_slots])
                problem += (scheduled_departure == y[i])
                constraint_count += 1

            # Arrival constraints
            arrival_slot = self.time_to_slot(train['arrival_time'])
            arrival_station = train['to_station']

            if arrival_station in sample_stations and arrival_slot < self.time_slots:
                scheduled_arrival = lp.lpSum([x.get((i, arrival_station, min(arrival_slot + d, self.time_slots-1), platform), 0)
                                             for d in range(17)
                                             for platform in range(self.platform_capacity)
                                             if arrival_slot + d < self.time_slots])
                problem += (scheduled_arrival == y[i])
                constraint_count += 1

        # 3. CAPACITY OPTIMIZATION: Route capacity constraints
        for route, train_indices in routes.items():
            # Calculate total capacity on this route
            route_trains = lp.lpSum([y[i] * (sample_trains[i]['distance'] +
                                           sample_trains[i]['has_ac'] * 50 +
                                           sample_trains[i]['has_sleeper'] * 30)
                                   for i in train_indices])
            problem += (route_capacity[route] == route_trains)

            # Route capacity should not exceed infrastructure limits
            max_route_capacity = len(train_indices) * 1000  # Estimated capacity
            problem += (route_capacity[route] <= max_route_capacity)
            constraint_count += 1

        # 4. PRIORITY-BASED OPTIMIZATION: Priority constraints
        # High-priority trains should have preference in scheduling
        for i, train in enumerate(sample_trains):
            if train['priority_weight'] >= 80:  # High priority trains
                # High priority trains get scheduling preference (lower delay tolerance)
                problem += delay[i] <= 8  # Max 2 hours delay for high priority
            elif train['priority_weight'] >= 50:  # Medium priority
                problem += delay[i] <= 12  # Max 3 hours delay for medium priority
            # Low priority trains can have up to 4 hours delay (already set globally)

        # 5. SCHEDULE OPTIMIZATION: General delay bounds
        for i in range(len(sample_trains)):
            problem += delay[i] <= 16  # Maximum 4 hours delay
            constraint_count += 1

        # 6. THROUGHPUT OPTIMIZATION: Minimum utilization requirements
        for station in sample_stations:
            # Ensure reasonable utilization during peak hours (slots 30-50 = 7:30-12:30)
            peak_utilization = lp.lpSum([platform_util[(station, slot)]
                                       for slot in range(30, min(50, self.time_slots))])
            problem += peak_utilization >= 0.3 * min(20, self.time_slots - 30)  # 30% min utilization
            constraint_count += 1

        print(f"  Added {constraint_count} COMPREHENSIVE constraints")
        print("     Capacity constraints: Platform + Route limits")
        print("     Schedule constraints: Departure + Arrival timing")
        print("     Priority constraints: High-priority preferences")
        print("     Throughput constraints: Utilization requirements")

        # Solve the problem
        print("\n  Starting MILP solver...")
        print("   This may take a few minutes for optimal solution...")

        # Use CBC solver (open source)
        solver = lp.PULP_CBC_CMD(msg=1, timeLimit=300)  # 5-minute time limit
        problem.solve(solver)

        # Process results
        print(f"\n  OPTIMIZATION RESULTS:")
        print("=" * 40)

        status = lp.LpStatus[problem.status]
        print(f"Status: {status}")

        if problem.status == lp.LpStatusOptimal:
            objective_value = lp.value(problem.objective)
            print(f"Objective Value: {objective_value:.2f}")

            # Analyze solution
            scheduled_trains = []
            total_delay = 0
            priority_distribution = defaultdict(int)

            for i, train in enumerate(sample_trains):
                if y[i].varValue == 1:
                    train_delay = delay[i].varValue
                    scheduled_trains.append({
                        'train': train,
                        'delay_slots': train_delay,
                        'delay_minutes': train_delay * 15
                    })
                    total_delay += train_delay
                    priority_distribution[train['train_type']] += 1

            print(f"\n  SOLUTION SUMMARY:")
            print(f"     Scheduled trains: {len(scheduled_trains)}/{len(sample_trains)} ({len(scheduled_trains)/len(sample_trains)*100:.1f}%)")
            print(f"     Total delay: {total_delay} slots ({total_delay * 15} minutes)")
            print(f"     Average delay per train: {total_delay/max(len(scheduled_trains), 1):.1f} slots")

            print(f"\n  SCHEDULED TRAINS BY PRIORITY:")
            high_priority_scheduled = sum(1 for st in scheduled_trains if st['train']['priority_weight'] >= 80)
            medium_priority_scheduled = sum(1 for st in scheduled_trains if 30 <= st['train']['priority_weight'] < 80)
            low_priority_scheduled = sum(1 for st in scheduled_trains if st['train']['priority_weight'] < 30)

            print(f"     High Priority ( 80):   {high_priority_scheduled:3}")
            print(f"     Medium Priority (30-79): {medium_priority_scheduled:3}")
            print(f"     Low Priority (<30):     {low_priority_scheduled:3}")

            # Store results for analysis
            self.optimization_results = {
                'status': status,
                'objective_value': objective_value,
                'scheduled_trains': scheduled_trains,
                'total_delay_minutes': total_delay * 15,
                'scheduling_efficiency': len(scheduled_trains) / len(sample_trains),
                'priority_distribution': dict(priority_distribution),
                'sample_size': len(sample_trains)
            }

            return True
        else:
            print(f"  Optimization failed with status: {status}")
            return False

    def generate_comprehensive_analysis(self):
        """
        Generate comprehensive analysis and visualizations of optimization results.
        """
        if not self.optimization_results:
            print("  No optimization results to analyze!")
            return

        print("\n  GENERATING COMPREHENSIVE ANALYSIS...")
        print("=" * 50)

        # Create analysis DataFrame
        scheduled_data = []
        for result in self.optimization_results['scheduled_trains']:
            train = result['train']
            scheduled_data.append({
                'Train_Number': train['train_number'],
                'Train_Name': train['train_name'],
                'Type': train['train_type'],
                'Priority_Weight': train['priority_weight'],
                'From': train['from_station'],
                'To': train['to_station'],
                'Original_Departure': train['departure_time'],
                'Delay_Minutes': result['delay_minutes'],
                'Distance_KM': train['distance'],
                'Speed_KMPH': train['speed_kmph'],
                'Has_AC': train['has_ac'],
                'Zone': train['zone']
            })

        df_scheduled = pd.DataFrame(scheduled_data)

        # Priority-based analysis
        print("\n  PRIORITY-BASED PERFORMANCE ANALYSIS:")
        print("-" * 45)

        priority_categories = {
            'HIGH': df_scheduled[df_scheduled['Priority_Weight'] >= 80],
            'MEDIUM': df_scheduled[(df_scheduled['Priority_Weight'] >= 30) & (df_scheduled['Priority_Weight'] < 80)],
            'LOW': df_scheduled[df_scheduled['Priority_Weight'] < 30]
        }

        for category, data in priority_categories.items():
            if len(data) > 0:
                avg_delay = data['Delay_Minutes'].mean()
                max_delay = data['Delay_Minutes'].max()
                count = len(data)
                print(f"   {category:6} Priority: {count:3} trains, Avg Delay: {avg_delay:5.1f}m, Max Delay: {max_delay:3.0f}m")

        # Speed vs Priority Analysis
        print(f"\n  SPEED vs PRIORITY CORRELATION:")
        print("-" * 35)
        speed_priority_corr = df_scheduled[['Speed_KMPH', 'Priority_Weight']].corr().iloc[0, 1]
        print(f"   Correlation coefficient: {speed_priority_corr:.3f}")

        # Zone-wise distribution
        print(f"\n   ZONE-WISE DISTRIBUTION:")
        print("-" * 25)
        zone_counts = df_scheduled['Zone'].value_counts()
        for zone, count in zone_counts.head().items():
            print(f"   {zone:4}: {count:3} trains")

        # Delay distribution by train type
        print(f"\n  DELAY DISTRIBUTION BY TRAIN TYPE:")
        print("-" * 35)
        delay_by_type = df_scheduled.groupby('Type')['Delay_Minutes'].agg(['count', 'mean', 'max']).round(1)
        delay_by_type = delay_by_type.sort_values('mean')

        for train_type, row in delay_by_type.head(10).iterrows():
            priority = self.priority_weights.get(train_type, 5)
            print(f"   {train_type:8} (P:{priority:3}): {row['count']:3} trains, {row['mean']:5.1f}m avg, {row['max']:3.0f}m max")

        return df_scheduled

    def create_optimization_visualizations(self, df_scheduled):
        """
        Create comprehensive visualizations of the optimization results.
        """
        print("\n  CREATING OPTIMIZATION VISUALIZATIONS...")

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('GRANDEST FINAL Railway MILP Optimization Results', fontsize=20, fontweight='bold')

        # 1. Priority Distribution Pie Chart
        ax1 = plt.subplot(3, 3, 1)
        priority_cats = []
        priority_counts = []

        for index, row in df_scheduled.iterrows():
            if row['Priority_Weight'] >= 80:
                priority_cats.append('HIGH')
            elif row['Priority_Weight'] >= 30:
                priority_cats.append('MEDIUM')
            else:
                priority_cats.append('LOW')

        priority_dist = pd.Series(priority_cats).value_counts()
        colors = ['#ff4444', '#ffaa44', '#44ff44']
        plt.pie(priority_dist.values, labels=priority_dist.index, autopct='%1.1f%%', colors=colors)
        plt.title('Scheduled Trains by Priority Level', fontweight='bold')

        # 2. Delay Distribution Histogram
        ax2 = plt.subplot(3, 3, 2)
        plt.hist(df_scheduled['Delay_Minutes'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Delay (Minutes)')
        plt.ylabel('Number of Trains')
        plt.title('Delay Distribution', fontweight='bold')
        plt.axvline(df_scheduled['Delay_Minutes'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df_scheduled["Delay_Minutes"].mean():.1f}m')
        plt.legend()

        # 3. Priority Weight vs Delay Scatter
        ax3 = plt.subplot(3, 3, 3)
        scatter = plt.scatter(df_scheduled['Priority_Weight'], df_scheduled['Delay_Minutes'],
                            alpha=0.6, s=50, c=df_scheduled['Distance_KM'], cmap='viridis')
        plt.xlabel('Priority Weight')
        plt.ylabel('Delay (Minutes)')
        plt.title('Priority vs Delay Analysis', fontweight='bold')
        plt.colorbar(scatter, label='Distance (KM)')

        # 4. Train Type Distribution
        ax4 = plt.subplot(3, 3, 4)
        type_counts = df_scheduled['Type'].value_counts().head(8)
        plt.bar(range(len(type_counts)), type_counts.values, color='lightcoral')
        plt.xticks(range(len(type_counts)), type_counts.index, rotation=45)
        plt.ylabel('Number of Trains')
        plt.title('Top Train Types Scheduled', fontweight='bold')

        # 5. Zone-wise Performance
        ax5 = plt.subplot(3, 3, 5)
        zone_delays = df_scheduled.groupby('Zone')['Delay_Minutes'].mean().sort_values(ascending=False).head(8)
        plt.bar(range(len(zone_delays)), zone_delays.values, color='lightgreen')
        plt.xticks(range(len(zone_delays)), zone_delays.index, rotation=45)
        plt.ylabel('Average Delay (Minutes)')
        plt.title('Average Delay by Zone', fontweight='bold')

        # 6. Speed Distribution by Priority
        ax6 = plt.subplot(3, 3, 6)
        high_speed = df_scheduled[df_scheduled['Priority_Weight'] >= 80]['Speed_KMPH']
        medium_speed = df_scheduled[(df_scheduled['Priority_Weight'] >= 30) & (df_scheduled['Priority_Weight'] < 80)]['Speed_KMPH']
        low_speed = df_scheduled[df_scheduled['Priority_Weight'] < 30]['Speed_KMPH']

        plt.hist([high_speed, medium_speed, low_speed], label=['High Priority', 'Medium Priority', 'Low Priority'],
                alpha=0.7, bins=15, color=['red', 'orange', 'green'])
        plt.xlabel('Speed (KM/H)')
        plt.ylabel('Number of Trains')
        plt.title('Speed Distribution by Priority', fontweight='bold')
        plt.legend()

        # 7. Distance vs Priority
        ax7 = plt.subplot(3, 3, 7)
        plt.scatter(df_scheduled['Distance_KM'], df_scheduled['Priority_Weight'],
                   alpha=0.6, s=60, c=df_scheduled['Delay_Minutes'], cmap='Reds')
        plt.xlabel('Distance (KM)')
        plt.ylabel('Priority Weight')
        plt.title('Distance vs Priority', fontweight='bold')
        plt.colorbar(label='Delay (Minutes)')

        # 8. AC vs Non-AC Distribution
        ax8 = plt.subplot(3, 3, 8)
        ac_counts = df_scheduled['Has_AC'].value_counts()
        plt.pie(ac_counts.values, labels=['Non-AC', 'AC'], autopct='%1.1f%%',
               colors=['lightblue', 'gold'])
        plt.title('AC vs Non-AC Trains', fontweight='bold')

        # 9. Optimization Summary Text
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        summary_text = f"""
          OPTIMIZATION SUMMARY  

          Total Trains Analyzed: {len(self.train_data):,}
          Trains Scheduled: {len(df_scheduled)}
          Scheduling Efficiency: {self.optimization_results['scheduling_efficiency']:.1%}

          Total System Delay: {self.optimization_results['total_delay_minutes']:.0f} minutes
          Average Delay per Train: {df_scheduled['Delay_Minutes'].mean():.1f} minutes

          High Priority Trains: {len(df_scheduled[df_scheduled['Priority_Weight'] >= 80])}
          Medium Priority Trains: {len(df_scheduled[(df_scheduled['Priority_Weight'] >= 30) & (df_scheduled['Priority_Weight'] < 80)])}
          Low Priority Trains: {len(df_scheduled[df_scheduled['Priority_Weight'] < 30])}

          Objective Value: {self.optimization_results['objective_value']:.0f}
        """

        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))

        plt.tight_layout()

        # Save the visualization
        output_file = "C:\\Users\\Noel\\Desktop\\Second Optimization\\grandestfinal_optimization_results.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Visualizations saved to: {output_file}")

        plt.show()

        return fig

    def export_detailed_results(self, df_scheduled):
        """
        Export detailed results to files for further analysis.
        """
        print("\n  EXPORTING DETAILED RESULTS...")

        # Export scheduled trains to CSV
        csv_file = "C:\\Users\\Noel\\Desktop\\Second Optimization\\grandestfinal_scheduled_trains.csv"
        df_scheduled.to_csv(csv_file, index=False)
        print(f"  Scheduled trains exported to: {csv_file}")

        # Export optimization summary to JSON
        summary_file = "C:\\Users\\Noel\\Desktop\\Second Optimization\\grandestfinal_optimization_summary.json"

        summary_data = {
            "optimization_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_trains_analyzed": len(self.train_data),
                "optimization_sample_size": self.optimization_results['sample_size'],
                "solver_status": self.optimization_results['status'],
                "objective_value": self.optimization_results['objective_value']
            },
            "performance_metrics": {
                "scheduling_efficiency": self.optimization_results['scheduling_efficiency'],
                "total_delay_minutes": self.optimization_results['total_delay_minutes'],
                "average_delay_per_train": df_scheduled['Delay_Minutes'].mean(),
                "max_delay_minutes": df_scheduled['Delay_Minutes'].max()
            },
            "priority_analysis": {
                "high_priority_scheduled": len(df_scheduled[df_scheduled['Priority_Weight'] >= 80]),
                "medium_priority_scheduled": len(df_scheduled[(df_scheduled['Priority_Weight'] >= 30) & (df_scheduled['Priority_Weight'] < 80)]),
                "low_priority_scheduled": len(df_scheduled[df_scheduled['Priority_Weight'] < 30])
            },
            "train_type_distribution": self.optimization_results['priority_distribution']
        }

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"  Optimization summary exported to: {summary_file}")

    def run_complete_optimization_pipeline(self, json_file_path, num_trains=150):
        """
        Run the complete GRANDEST FINAL optimization pipeline.
        """
        print("  STARTING GRANDEST FINAL RAILWAY MILP OPTIMIZATION PIPELINE")
        print("=" * 70)

        # Step 1: Load and process data
        if not self.load_and_process_data(json_file_path):
            return False

        # Step 2: Analyze priority distribution
        self.analyze_priority_distribution()

        # Step 3: Create optimization scenario
        sample_trains = self.create_sample_optimization_scenario(num_trains)

        # Step 4: Solve MILP optimization
        if not self.solve_milp_optimization(sample_trains):
            return False

        # Step 5: Generate comprehensive analysis
        df_scheduled = self.generate_comprehensive_analysis()

        # Step 6: Create visualizations
        self.create_optimization_visualizations(df_scheduled)

        # Step 7: Export results
        self.export_detailed_results(df_scheduled)

        print("\n  GRANDEST FINAL OPTIMIZATION PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        return True

def main():
    """
    Main execution function for the GRANDEST FINAL Railway MILP Optimizer.
    """
    print("GRANDEST FINAL RAILWAY MILP OPTIMIZER")
    print("Advanced Mixed Integer Linear Programming for Railway Scheduling")
    print("Date: 2025-09-24")
    print("Built with: PuLP, NumPy, Pandas, Matplotlib, Seaborn")
    print("=" * 70)

    # Initialize the optimizer
    optimizer = GrandestFinalRailwayMILPOptimizer()

    # File path to trains.json
    json_file_path = "C:\\Users\\Noel\\Desktop\\Second Optimization\\archive\\trains.json"

    # Run the complete optimization pipeline
    success = optimizer.run_complete_optimization_pipeline(json_file_path, num_trains=100)

    if success:
        print("\nMISSION ACCOMPLISHED!")
        print("All results have been saved and visualized.")
        print("\nFiles created:")
        print("  - grandestfinal_optimization_results.png")
        print("  - grandestfinal_scheduled_trains.csv")
        print("  - grandestfinal_optimization_summary.json")
    else:
        print("\nOPTIMIZATION FAILED!")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main()