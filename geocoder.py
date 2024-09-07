import pandas as pd
import requests
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re

# Compile the regex pattern for efficiency
range_pattern = re.compile(r'\[(\d+)\]-\[(\d+)\]')

def average_range_in_address(address):
    match = range_pattern.search(address)
    if match:
        num1 = int(match.group(1))
        num2 = int(match.group(2))
        avg_num = (num1 + num2) // 2  # Calculate average and use integer division
        # Replace the range with the average in the address string
        new_address = range_pattern.sub(str(avg_num), address)
        return new_address
    else:
        return address
    
def get_lat_lon_for_street(street, city, state, nominatim_url):
    params = {
        'street': street,
        'city': city,
        'state': state,
        'format': 'json'
    }
    url = f"{nominatim_url}/search?{urlencode(params)}"
    response = requests.get(url)
    data = response.json()
    
    if data:
        return float(data[0]['lat']), float(data[0]['lon'])
    else:
        return None, None

def get_lat_lon(row, nominatim_url):
    if not isinstance(row['ADDRESS'], str):
        return row.name, None, None
    
    try:
        address = average_range_in_address(row['ADDRESS'])
        city, state = row['CITY'], row['STATE']

        # if no address, return nothing
        if not address.strip():
            return row.name, None, None

        if '&' in address:
            # Handle cross streets by splitting the address
            streets = [street.strip() for street in address.split('&')]
            coords = [get_lat_lon_for_street(street, city, state, nominatim_url) for street in streets]
            # Filter out None values and calculate the average
            valid_coords = [coord for coord in coords if coord != (None, None)]
            if valid_coords:
                avg_lat = sum(coord[0] for coord in valid_coords) / len(valid_coords)
                avg_lon = sum(coord[1] for coord in valid_coords) / len(valid_coords)
                return row.name, avg_lat, avg_lon
            else:
                return row.name, None, None
        else:
            # Handle regular addresses
            lat, lon = get_lat_lon_for_street(address, city, state, nominatim_url)
            return row.name, lat, lon
        
        
    except Exception as e:
        print(f"Error processing row {row.name} address: {row['ADDRESS']}: {e}")
        return row.name, None, None

def add_lat_lon_to_csv(input_file, output_file, nominatim_url, max_workers=10):
    print(f"Reading CSV {input_file}")
    df = pd.read_csv(input_file) # , nrows=100)

    latitudes = [None] * len(df)
    longitudes = [None] * len(df)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {executor.submit(get_lat_lon, row, nominatim_url): row for _, row in df.iterrows()}
        
        with tqdm(total=len(df), desc="Processing", unit=" record") as pbar:
            for future in as_completed(future_to_row):
                row_index, lat, lon = future.result()
                latitudes[row_index] = lat
                longitudes[row_index] = lon
                pbar.update(1)

    df['LATITUDE'] = latitudes
    df['LONGITUDE'] = longitudes

    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    nominatim_url = "http://localhost:8088"  # URL of your locally running Nominatim instance
    input_files = ["./data/police/policecalls2020.csv", "./data/police/policecalls2021.csv", "./data/police/policecalls2022.csv"]
    for input_file in input_files:
        print(f"Processing {input_file}")
        output_file = f"{input_file}.geocoded.csv"  # Path where the output CSV will be saved
        add_lat_lon_to_csv(input_file, output_file, nominatim_url, max_workers=20)
