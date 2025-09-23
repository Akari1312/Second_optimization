import json
import pandas as pd

# Load trains data to examine structure
with open('archive/trains.json', 'r') as f:
    trains_data = json.load(f)

# Check a sample train entry
if trains_data['features']:
    print("Sample train feature structure:")
    sample_train = trains_data['features'][0]
    print("Properties keys:", list(sample_train['properties'].keys()))
    print("\nFirst few properties:")
    for key, value in list(sample_train['properties'].items())[:10]:
        print(f"  {key}: {value}")

# Check schedules structure
with open('archive/schedules.json', 'r') as f:
    schedules_data = json.load(f)

if schedules_data:
    print("\n\nSample schedule entry:")
    print("Keys:", list(schedules_data[0].keys()))
    print("First entry:")
    for key, value in schedules_data[0].items():
        print(f"  {key}: {value}")

# Check stations structure
with open('archive/stations.json', 'r') as f:
    stations_data = json.load(f)

if stations_data['features']:
    print("\n\nSample station feature structure:")
    sample_station = stations_data['features'][0]
    print("Properties keys:", list(sample_station['properties'].keys()))