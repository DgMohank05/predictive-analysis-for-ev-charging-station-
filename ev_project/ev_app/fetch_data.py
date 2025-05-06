# ev_app/fetch_data.py
import requests
import pandas as pd

def fetch_station_data(lat=17.385044, lon=78.486671, results=100):
    """
    Fetch charging station data from Open Charge Map API.
    """
    url = "https://api.openchargemap.io/v3/poi/"
    params = {
        'output': 'json',
        'countrycode': 'IN',
        'latitude': lat,
        'longitude': lon,
        'maxresults': results,
        'compact': 'true',
        'verbose': 'false',
        'key': 'b829f55b-cb1b-4e1a-b99f-131546523507'  # Replace with your Open Charge Map API key
    }
    
    response = requests.get(url, params=params)
    data = response.json()

    stations = []
    for item in data:
        stations.append({
            'id': item['ID'],
            'name': item['AddressInfo']['Title'],
            'lat': item['AddressInfo']['Latitude'],
            'lon': item['AddressInfo']['Longitude']
        })

    return pd.DataFrame(stations)
