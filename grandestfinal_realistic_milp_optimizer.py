"""
GRANDEST FINAL REALISTIC RAILWAY MILP OPTIMIZER
================================================
Simplified but comprehensive MILP optimization for railway scheduling
with realistic constraints that ensure feasible solutions.

Author: Claude Code
Date: 2025-09-24
Version: 2.0 - REALISTIC IMPLEMENTATION
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

class GrandestFinalRealisticMILPOptimizer:
    """
    Realistic MILP optimizer for railway scheduling with feasible solutions.
    """

    def __init__(self):
        self.train_data = []
        self.priority_weights = self._define_priority_hierarchy()
        self.optimization_results = {}

    def _define_priority_hierarchy(self):
        """Define the exact priority hierarchy."""
        return {
            # HIGH PRIORITY TRAINS
            'Raj': 100,      # Rajdhani Express
            'Shtb': 95,      # Shatabdi Express
            'JShtb': 90,     # Jan Shatabdi
            'Drnt': 85,      # Duronto Express
            'GR': 80,        # Garib Rath

            # MEDIUM PRIORITY TRAINS
            'SF': 60,        # Superfast Express
            'Exp': 50,       # Express
            'Mail': 45,      # Mail trains
            'Hyd': 40,       # Hyderabad trains
            'Del': 40,       # Delhi trains
            'SKr': 35,       # Shaktipunj trains
            'Klkt': 35,      # Kolkata trains

            # LOW PRIORITY TRAINS
            'Pass': 20,      # Passenger trains
            'MEMU': 15,      # Suburban/regional trains
            'DEMU': 15,      # Diesel Electric Multiple Unit
            'Toy': 10,       # Toy trains
            '': 5            # Empty type
        }

    def load_and_process_data(self, json_file_path):
        """Load and process train data."""
        print("Loading REALISTIC MILP train data...")

        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            features = data.get('features', [])
            print(f"Processing {len(features)} train records...")

            for feature in features:
                if feature.get('type') == 'Feature':
                    props = feature.get('properties', {})

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
                    }

                    # Calculate speed
                    duration_h = train_info['duration_hours'] or 0
                    duration_m = train_info['duration_minutes'] or 0
                    distance = train_info['distance'] or 0

                    total_minutes = duration_h * 60 + duration_m
                    if total_minutes > 0 and distance > 0:
                        train_info['speed_kmph'] = (distance / total_minutes) * 60
                    else:
                        train_info['speed_kmph'] = 0

                    self.train_data.append(train_info)

            print(f"Successfully processed {len(self.train_data)} trains")
            return True

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def time_to_slot(self, time_str):
        """Convert time to hour slot (0-23)."""
        try:
            time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
            return time_obj.hour
        except:
            return 0

    def create_realistic_sample(self, num_trains=50):
        """Create a realistic sample for optimization."""
        print(f"Creating REALISTIC sample with {num_trains} trains...")

        # Get diverse sample with different priorities
        sorted_trains = sorted(self.train_data, key=lambda x: x['priority_weight'], reverse=True)

        # Select representative sample
        sample = []
        high_priority = [t for t in sorted_trains if t['priority_weight'] >= 80]
        medium_priority = [t for t in sorted_trains if 40 <= t['priority_weight'] < 80]
        low_priority = [t for t in sorted_trains if t['priority_weight'] < 40]

        sample.extend(high_priority[:num_trains//4])
        sample.extend(medium_priority[:num_trains//2])
        sample.extend(low_priority[:num_trains//4])

        return sample[:num_trains]

    def solve_realistic_milp(self, sample_trains):
        """
        Solve REALISTIC MILP with feasible constraints.

        OPTIMIZATION COMPONENTS:
        1. SCHEDULE OPTIMIZATION: Minimize delays and conflicts
        2. CAPACITY OPTIMIZATION: Maximize network utilization
        3. PRIORITY OPTIMIZATION: Respect train hierarchy
        """
        print("\nSOLVING REALISTIC MILP OPTIMIZATION...")
        print("COMPREHENSIVE OPTIMIZATION INCLUDES:")
        print("  - Schedule Optimization: Minimize delays and conflicts")
        print("  - Capacity Optimization: Maximize network utilization")
        print("  - Priority Optimization: Respect train hierarchy")
        print("="*60)

        # Create optimization problem
        problem = lp.LpProblem("Realistic_Railway_MILP", lp.LpMaximize)

        # Get unique stations
        stations = list(set([t['from_station'] for t in sample_trains] +
                           [t['to_station'] for t in sample_trains]))

        time_slots = 24  # 24 hour slots
        platforms_per_station = 3  # Realistic platform count

        print(f"REALISTIC OPTIMIZATION PARAMETERS:")
        print(f"  - Trains: {len(sample_trains)}")
        print(f"  - Stations: {len(stations)}")
        print(f"  - Time slots: {time_slots} hours")
        print(f"  - Platforms per station: {platforms_per_station}")

        # DECISION VARIABLES
        print("\nCreating decision variables...")

        # y[i] = 1 if train i is scheduled
        y = {}
        for i in range(len(sample_trains)):
            y[i] = lp.LpVariable(f"schedule_{i}", cat='Binary')

        # delay[i] = hours of delay for train i
        delay = {}
        for i in range(len(sample_trains)):
            delay[i] = lp.LpVariable(f"delay_{i}", lowBound=0, upBound=8, cat='Integer')

        # platform_usage[station][hour] = number of platforms used
        platform_usage = {}
        for station in stations:
            for hour in range(time_slots):
                platform_usage[(station, hour)] = lp.LpVariable(
                    f"platform_{station}_{hour}", lowBound=0,
                    upBound=platforms_per_station, cat='Integer')

        print("Decision variables created")

        # MULTI-OBJECTIVE FUNCTION
        print("\nSetting up COMPREHENSIVE objective function...")

        # 1. PRIORITY OPTIMIZATION: Maximize weighted scheduled trains
        priority_term = lp.lpSum([sample_trains[i]['priority_weight'] * y[i]
                                 for i in range(len(sample_trains))])

        # 2. SCHEDULE OPTIMIZATION: Minimize total delays
        delay_penalty = lp.lpSum([delay[i] * 10 for i in range(len(sample_trains))])

        # 3. CAPACITY OPTIMIZATION: Maximize total distance covered
        capacity_term = lp.lpSum([sample_trains[i]['distance'] * y[i] / 1000
                                 for i in range(len(sample_trains))])

        # Combined objective
        problem += (priority_term * 1.0 +      # Priority weight
                   capacity_term * 0.5 -      # Capacity weight
                   delay_penalty * 1.0)       # Delay penalty

        print("COMPREHENSIVE objective: Priority + Capacity - Delays")

        # REALISTIC CONSTRAINTS
        print("\nAdding REALISTIC constraints...")
        constraint_count = 0

        # 1. CAPACITY CONSTRAINTS: Platform usage limits
        for station in stations:
            for hour in range(time_slots):
                # Count trains using this station at this hour
                trains_at_station = []
                for i, train in enumerate(sample_trains):
                    departure_hour = self.time_to_slot(train['departure_time'])
                    arrival_hour = self.time_to_slot(train['arrival_time'])

                    if ((train['from_station'] == station and
                         departure_hour <= hour <= departure_hour + 2) or
                        (train['to_station'] == station and
                         arrival_hour <= hour <= arrival_hour + 2)):
                        trains_at_station.append(y[i])

                if trains_at_station:
                    problem += (lp.lpSum(trains_at_station) <= platforms_per_station)
                    problem += (platform_usage[(station, hour)] == lp.lpSum(trains_at_station))
                    constraint_count += 2

        # 2. SCHEDULE CONSTRAINTS: Delay relationships
        for i, train in enumerate(sample_trains):
            # If scheduled, delay should be reasonable for priority
            if train['priority_weight'] >= 80:  # High priority
                problem += (delay[i] <= 2)  # Max 2 hours delay
            elif train['priority_weight'] >= 50:  # Medium priority
                problem += (delay[i] <= 4)  # Max 4 hours delay
            else:  # Low priority
                problem += (delay[i] <= 6)  # Max 6 hours delay
            constraint_count += 1

        # 3. PRIORITY CONSTRAINTS: High-priority preference
        high_priority_trains = sum(1 for t in sample_trains if t['priority_weight'] >= 80)
        if high_priority_trains > 0:
            high_priority_scheduled = lp.lpSum([y[i] for i, train in enumerate(sample_trains)
                                              if train['priority_weight'] >= 80])
            problem += (high_priority_scheduled >= high_priority_trains * 0.8)  # 80% scheduled
            constraint_count += 1

        print(f"Added {constraint_count} REALISTIC constraints")

        # SOLVE THE PROBLEM
        print("\nStarting MILP solver...")

        solver = lp.PULP_CBC_CMD(msg=1, timeLimit=120)  # 2-minute limit
        problem.solve(solver)

        # PROCESS RESULTS
        print(f"\nOPTIMIZATION RESULTS:")
        print("="*40)

        status = lp.LpStatus[problem.status]
        print(f"Status: {status}")

        if problem.status == lp.LpStatusOptimal:
            objective_value = lp.value(problem.objective)
            print(f"Objective Value: {objective_value:.2f}")

            # Analyze solution
            scheduled_trains = []
            total_delay = 0

            for i, train in enumerate(sample_trains):
                if y[i].varValue and y[i].varValue > 0.5:
                    train_delay = delay[i].varValue or 0
                    scheduled_trains.append({
                        'train': train,
                        'delay_hours': train_delay
                    })
                    total_delay += train_delay

            # Results summary
            print(f"\nSOLUTION SUMMARY:")
            print(f"  Scheduled trains: {len(scheduled_trains)}/{len(sample_trains)} ({len(scheduled_trains)/len(sample_trains)*100:.1f}%)")
            print(f"  Total delay: {total_delay:.1f} hours")
            print(f"  Average delay: {total_delay/max(len(scheduled_trains),1):.1f} hours per train")

            # Priority analysis
            high_scheduled = sum(1 for st in scheduled_trains if st['train']['priority_weight'] >= 80)
            medium_scheduled = sum(1 for st in scheduled_trains if 50 <= st['train']['priority_weight'] < 80)
            low_scheduled = sum(1 for st in scheduled_trains if st['train']['priority_weight'] < 50)

            print(f"\nSCHEDULED BY PRIORITY:")
            print(f"  High Priority (>=80):   {high_scheduled}")
            print(f"  Medium Priority (50-79): {medium_scheduled}")
            print(f"  Low Priority (<50):     {low_scheduled}")

            # Store results
            self.optimization_results = {
                'status': status,
                'objective_value': objective_value,
                'scheduled_trains': scheduled_trains,
                'total_delay_hours': total_delay,
                'scheduling_efficiency': len(scheduled_trains) / len(sample_trains),
                'sample_size': len(sample_trains)
            }

            return True
        else:
            print(f"Optimization failed with status: {status}")
            return False

    def create_results_analysis(self):
        """Create comprehensive results analysis."""
        if not self.optimization_results:
            print("No results to analyze!")
            return None

        print("\nGENERATING COMPREHENSIVE RESULTS ANALYSIS...")
        print("="*50)

        # Create DataFrame
        results_data = []
        for result in self.optimization_results['scheduled_trains']:
            train = result['train']
            results_data.append({
                'Train_Number': train['train_number'],
                'Train_Name': train['train_name'][:30] + "...",  # Truncate long names
                'Type': train['train_type'],
                'Priority': train['priority_weight'],
                'From': train['from_station'],
                'To': train['to_station'],
                'Distance': train['distance'],
                'Speed': train['speed_kmph'],
                'Delay_Hours': result['delay_hours'],
                'Has_AC': train['has_ac'],
                'Zone': train['zone']
            })

        df = pd.DataFrame(results_data)

        if len(df) == 0:
            print("No trains were scheduled!")
            return None

        # Analysis by priority
        print("\nPRIORITY-BASED PERFORMANCE:")
        print("-" * 35)

        for category, min_priority in [('HIGH', 80), ('MEDIUM', 50), ('LOW', 0)]:
            if category == 'HIGH':
                subset = df[df['Priority'] >= min_priority]
            elif category == 'MEDIUM':
                subset = df[(df['Priority'] >= min_priority) & (df['Priority'] < 80)]
            else:
                subset = df[df['Priority'] < 50]

            if len(subset) > 0:
                avg_delay = subset['Delay_Hours'].mean()
                print(f"  {category:6}: {len(subset):2} trains, Avg Delay: {avg_delay:.1f}h")

        # Speed analysis
        print(f"\nSPEED ANALYSIS:")
        print("-" * 20)
        print(f"  Average speed: {df['Speed'].mean():.1f} km/h")
        print(f"  Speed range: {df['Speed'].min():.0f} - {df['Speed'].max():.0f} km/h")

        # Distance analysis
        print(f"\nDISTANCE ANALYSIS:")
        print("-" * 20)
        print(f"  Total distance: {df['Distance'].sum():,.0f} km")
        print(f"  Average distance: {df['Distance'].mean():.0f} km")

        # Zone analysis
        print(f"\nTOP ZONES:")
        print("-" * 15)
        zone_counts = df['Zone'].value_counts().head()
        for zone, count in zone_counts.items():
            print(f"  {zone}: {count} trains")

        return df

    def create_visualizations(self, df):
        """Create optimization result visualizations."""
        if df is None or len(df) == 0:
            print("No data to visualize!")
            return

        print("\nCREATING RESULT VISUALIZATIONS...")

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create comprehensive visualization
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('GRANDEST FINAL Railway MILP Optimization Results', fontsize=16, fontweight='bold')

        # 1. Priority distribution
        ax1 = plt.subplot(2, 3, 1)
        priority_categories = []
        for _, row in df.iterrows():
            if row['Priority'] >= 80:
                priority_categories.append('HIGH')
            elif row['Priority'] >= 50:
                priority_categories.append('MEDIUM')
            else:
                priority_categories.append('LOW')

        pd.Series(priority_categories).value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax1)
        ax1.set_title('Trains by Priority Level')

        # 2. Delay distribution
        ax2 = plt.subplot(2, 3, 2)
        df['Delay_Hours'].hist(bins=10, alpha=0.7, ax=ax2)
        ax2.set_xlabel('Delay (Hours)')
        ax2.set_ylabel('Number of Trains')
        ax2.set_title('Delay Distribution')
        ax2.axvline(df['Delay_Hours'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["Delay_Hours"].mean():.1f}h')
        ax2.legend()

        # 3. Priority vs Delay scatter
        ax3 = plt.subplot(2, 3, 3)
        scatter = ax3.scatter(df['Priority'], df['Delay_Hours'],
                             c=df['Distance'], cmap='viridis', alpha=0.6)
        ax3.set_xlabel('Priority Weight')
        ax3.set_ylabel('Delay (Hours)')
        ax3.set_title('Priority vs Delay')
        plt.colorbar(scatter, ax=ax3, label='Distance (km)')

        # 4. Train types
        ax4 = plt.subplot(2, 3, 4)
        df['Type'].value_counts().head(8).plot(kind='bar', ax=ax4)
        ax4.set_title('Train Types Scheduled')
        ax4.set_xlabel('Train Type')
        ax4.set_ylabel('Count')
        plt.xticks(rotation=45)

        # 5. Speed vs Priority
        ax5 = plt.subplot(2, 3, 5)
        ax5.scatter(df['Speed'], df['Priority'], alpha=0.6)
        ax5.set_xlabel('Speed (km/h)')
        ax5.set_ylabel('Priority Weight')
        ax5.set_title('Speed vs Priority')

        # 6. Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        summary_text = f"""
OPTIMIZATION SUMMARY

Total Trains Analyzed: {len(self.train_data):,}
Trains Scheduled: {len(df)}
Efficiency: {self.optimization_results['scheduling_efficiency']:.1%}

Total Delay: {self.optimization_results['total_delay_hours']:.1f}h
Avg Delay per Train: {df['Delay_Hours'].mean():.1f}h

Objective Value: {self.optimization_results['objective_value']:.1f}

HIGH Priority: {len(df[df['Priority'] >= 80])}
MEDIUM Priority: {len(df[(df['Priority'] >= 50) & (df['Priority'] < 80)])}
LOW Priority: {len(df[df['Priority'] < 50])}
        """

        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))

        plt.tight_layout()

        # Save
        output_file = "C:\\Users\\Noel\\Desktop\\Second Optimization\\grandestfinal_realistic_results.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualizations saved to: {output_file}")

        plt.show()
        return fig

    def export_results(self, df):
        """Export detailed results."""
        if df is None:
            return

        print("\nEXPORTING RESULTS...")

        # Export to CSV
        csv_file = "C:\\Users\\Noel\\Desktop\\Second Optimization\\grandestfinal_realistic_trains.csv"
        df.to_csv(csv_file, index=False)
        print(f"Results exported to: {csv_file}")

        # Export summary
        summary_file = "C:\\Users\\Noel\\Desktop\\Second Optimization\\grandestfinal_realistic_summary.json"
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "total_trains_analyzed": len(self.train_data),
            "trains_scheduled": len(df),
            "scheduling_efficiency": self.optimization_results['scheduling_efficiency'],
            "total_delay_hours": self.optimization_results['total_delay_hours'],
            "objective_value": self.optimization_results['objective_value'],
            "priority_breakdown": {
                "high_priority": len(df[df['Priority'] >= 80]),
                "medium_priority": len(df[(df['Priority'] >= 50) & (df['Priority'] < 80)]),
                "low_priority": len(df[df['Priority'] < 50])
            }
        }

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Summary exported to: {summary_file}")

    def run_complete_pipeline(self, json_file_path, num_trains=50):
        """Run the complete optimization pipeline."""
        print("STARTING GRANDEST FINAL REALISTIC MILP OPTIMIZATION")
        print("="*60)

        # Load data
        if not self.load_and_process_data(json_file_path):
            return False

        # Create sample
        sample_trains = self.create_realistic_sample(num_trains)

        # Solve optimization
        if not self.solve_realistic_milp(sample_trains):
            return False

        # Analyze results
        df = self.create_results_analysis()

        # Create visualizations
        self.create_visualizations(df)

        # Export results
        self.export_results(df)

        print("\nGRANDEST FINAL OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("="*55)
        return True

def main():
    """Main execution function."""
    print("GRANDEST FINAL REALISTIC RAILWAY MILP OPTIMIZER")
    print("Advanced MILP with Feasible Solutions")
    print("Date: 2025-09-24")
    print("="*60)

    optimizer = GrandestFinalRealisticMILPOptimizer()
    json_file_path = "C:\\Users\\Noel\\Desktop\\Second Optimization\\archive\\trains.json"

    success = optimizer.run_complete_pipeline(json_file_path, num_trains=60)

    if success:
        print("\nMISSION ACCOMPLISHED!")
        print("All results saved and visualized.")
    else:
        print("\nOptimization failed!")

if __name__ == "__main__":
    main()