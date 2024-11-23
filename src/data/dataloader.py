import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    DataLoader class for fetching and processing NYC taxi data
    """
    def __init__(self):
        self.base_url = "https://data.cityofnewyork.us/resource/m6nq-qud6.json"  # 2022 Yellow Taxi Data
    
    def load_data(self, 
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load data for the specified date range.
        """
        try:
            # Set default dates if not provided
            if start_date is None:
                start_date = datetime(2022, 1, 1)
            if end_date is None:
                end_date = datetime(2022, 1, 31)
            
            logger.info(f"Loading data from {start_date.date()} to {end_date.date()}")
            
            # Calculate the date range in days
            date_range = (end_date - start_date).days
            
            # Adjust sample size based on date range
            # Roughly 100,000 records per day with a minimum of 10,000
            sample_size = max(10000, date_range * 100000)
            
            try:
                df = self._get_api_data(start_date, end_date, sample_size)
                if len(df) < 100:  # If we get too little data
                    logger.warning("Insufficient data from API, using sample data instead")
                    df = self._get_sample_data(sample_size)
            except Exception as e:
                logger.warning(f"Error fetching API data: {e}")
                logger.info("Falling back to sample data")
                df = self._get_sample_data(sample_size)
            
            # Process the data
            processed_df = self._process_data(df)
            logger.info(f"Successfully processed {len(processed_df)} records")
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error in load_data: {str(e)}")
            raise

    def _get_api_data(self, start_date: datetime, end_date: datetime, sample_size: int) -> pd.DataFrame:
        """
        Fetch data from NYC Taxi API with dynamic sample size
        """
        try:
            # Format dates for query
            start_str = start_date.strftime('%Y-%m-%dT00:00:00')
            end_str = end_date.strftime('%Y-%m-%dT23:59:59')
            
            # Query with dynamic limit
            params = {
                '$limit': str(sample_size),
                '$where': f"tpep_pickup_datetime between '{start_str}' and '{end_str}'",
                '$order': 'tpep_pickup_datetime ASC'
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            df = pd.DataFrame(response.json())
            logger.info(f"Successfully fetched {len(df)} records from API")
            return df
            
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            raise

    def _get_sample_data(self, n_samples: int) -> pd.DataFrame:
        """
        Generate sample data with dynamic size
        """
        np.random.seed(42)
        
        # Generate sample timestamps
        date_range = pd.date_range(
            start=datetime(2022, 1, 1),
            end=datetime(2022, 12, 31),
            periods=n_samples
        )
        
        # More realistic fare calculation based on distance and time
        distances = np.random.exponential(scale=3, size=n_samples)  # More realistic distance distribution
        base_fares = 2.50 + (distances * 2.50)  # Base fare + distance fare
        
        # Add time-based variations
        hours = pd.Series(date_range).dt.hour
        weekdays = pd.Series(date_range).dt.dayofweek
        
        # Increase fares during rush hours and weekends
        time_multipliers = np.ones(n_samples)
        rush_hours = (hours.between(7, 9) | hours.between(16, 18))
        weekend = weekdays.isin([5, 6])
        night_hours = hours.between(22, 6)
        
        time_multipliers[rush_hours] *= 1.5
        time_multipliers[weekend] *= 1.3
        time_multipliers[night_hours] *= 1.2
        
        final_fares = base_fares * time_multipliers
        
        data = {
            'tpep_pickup_datetime': date_range,
            'passenger_count': np.random.randint(1, 5, n_samples),
            'trip_distance': distances,
            'total_amount': final_fares,
            'fare_amount': final_fares * 0.85,  # Base fare without extras
            'tip_amount': final_fares * 0.15,  # Approximate tip
            'payment_type': np.random.randint(1, 3, n_samples)
        }
        
        logger.info(f"Generated {n_samples} sample records")
        return pd.DataFrame(data)

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean the taxi data
        """
        try:
            # Make a copy
            processed = df.copy()
            
            # Convert datetime - handle both real and sample data column names
            datetime_col = 'tpep_pickup_datetime' if 'tpep_pickup_datetime' in df.columns else 'pickup_datetime'
            processed['pickup_datetime'] = pd.to_datetime(processed[datetime_col])
            
            # Extract time features
            processed['hour'] = processed['pickup_datetime'].dt.hour
            processed['day_of_week'] = processed['pickup_datetime'].dt.dayofweek
            processed['is_weekend'] = processed['day_of_week'].isin([5, 6]).astype(int)
            processed['is_rush_hour'] = (
                processed['hour'].isin([7,8,9,16,17,18]) & 
                ~processed['is_weekend']
            ).astype(int)
            
            # Convert numeric columns
            numeric_columns = ['passenger_count', 'trip_distance', 
                             'total_amount', 'fare_amount', 'tip_amount']
            
            for col in numeric_columns:
                if col in processed.columns:
                    processed[col] = pd.to_numeric(processed[col], errors='coerce')
            
            # Remove invalid records
            processed = processed[
                (processed['total_amount'] > 0) &
                (processed['trip_distance'] > 0) &
                (processed['total_amount'] <= 200)  # Remove extreme outliers
            ].copy()
            
            # Add derived features
            processed['price_per_mile'] = (
                processed['total_amount'] / processed['trip_distance']
            ).clip(upper=50)  # Cap at $50/mile
            
            # Select final columns
            final_columns = [
                'pickup_datetime', 'hour', 'day_of_week',
                'is_weekend', 'is_rush_hour', 'passenger_count',
                'trip_distance', 'total_amount', 'price_per_mile'
            ]
            
            processed = processed[final_columns].dropna()
            
            logger.info("Data processing completed successfully")
            return processed
            
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            raise