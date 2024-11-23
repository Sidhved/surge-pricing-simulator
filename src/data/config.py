from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class DataConfig:
    # NYC TLC API endpoint
    TAXI_API_URL = "https://data.cityofnewyork.us/resource/2upf-qytp.json"
    
    # Default date ranges
    DEFAULT_DAYS_BACK = 30
    
    # Data sampling and limits
    MAX_RECORDS = 50000
    SAMPLE_RATE = 0.1
    
    # Geographic boundaries for NYC
    NYC_BOUNDS = {
        'lat_min': 40.5774,
        'lat_max': 40.9176,
        'lon_min': -74.1687,
        'lon_max': -73.8062
    }
    
    # Price thresholds for outlier detection
    MIN_PRICE = 2.50
    MAX_PRICE = 200.0