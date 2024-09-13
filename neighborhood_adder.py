import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Load the GeoJSON file
neighborhoods = gpd.read_file('neighborhoods/Neighborhoods.geojson')

files = [
    'data/police/policecalls2013.csv.geocoded.csv',
    'data/police/policecalls2014.csv.geocoded.csv',
    'data/police/policecalls2015.csv.geocoded.csv',
    'data/police/policecalls2016.csv.geocoded.csv',
    'data/police/policecalls2017.csv.geocoded.csv',
    'data/police/policecalls2023.csv.geocoded.csv',
    'data/police/policecalls2024.csv.geocoded.csv'
]

for file in files:
    # Load your CSV file
    police_events = pd.read_csv(file)
    
    # Create a GeoDataFrame from your events
    geometry = [Point(xy) for xy in zip(police_events['LONGITUDE'], police_events['LATITUDE'])]
    police_events_gdf = gpd.GeoDataFrame(police_events, geometry=geometry, crs="EPSG:4326")
    
    # Perform spatial join using the corrected argument
    events_with_neighborhoods = gpd.sjoin(police_events_gdf, neighborhoods, how="left", predicate="within")
    
    # Add the neighborhood name to your original DataFrame
    police_events['neighborhood'] = events_with_neighborhoods['NAME']
    
    # Save the updated DataFrame back to CSV
    new_file = f'{file[:-13]}.neighborhood.csv'
    police_events.to_csv(new_file, index=False)
    
    print(f"CSV file {file} updated with neighborhood information to {new_file}")