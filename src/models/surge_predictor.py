# src/models/surge_predictor.py
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SurgePredictor:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.baseline_fare = None
        
    def prepare_features(self, df):
        """
        Prepare features for the surge prediction model with enhanced feature engineering
        """
        features_df = pd.DataFrame()
        
        # Basic features
        features_df['hour'] = df['hour'].astype(float)
        features_df['day_of_week'] = df['day_of_week'].astype(float)
        features_df['is_weekend'] = df['is_weekend'].astype(float)
        features_df['is_rush_hour'] = df['is_rush_hour'].astype(float)
        features_df['passenger_count'] = df['passenger_count'].astype(float)
        features_df['trip_distance'] = df['trip_distance'].astype(float)
        
        # Enhanced time-based features
        features_df['morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(float)
        features_df['evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 18)).astype(float)
        features_df['late_night'] = ((df['hour'] >= 22) | (df['hour'] <= 4)).astype(float)
        features_df['business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(float)
        
        # Demand indicators
        features_df['long_trip'] = (df['trip_distance'] > 5).astype(float)
        features_df['high_occupancy'] = (df['passenger_count'] >= 3).astype(float)
        
        # Interaction features
        features_df['weekend_night'] = features_df['is_weekend'] * features_df['late_night']
        features_df['rush_hour_weekday'] = (1 - features_df['is_weekend']) * features_df['is_rush_hour']
        
        return features_df
        
    def train(self, df):
        """
        Train the surge prediction model with enhanced logic
        """
        try:
            X = self.prepare_features(df)
            y = df['total_amount']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Calculate baseline fare for reference
            self.baseline_fare = np.median(y)
            
            self.is_trained = True
            logger.info("Model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
        
    def predict_surge(self, features):
        """
        Predict surge multiplier with enhanced logic for more dynamic pricing
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        try:
            X = self.prepare_features(features)
            X_scaled = self.scaler.transform(X)
            
            # Get base prediction
            predicted_fare = self.model.predict(X_scaled)
            
            # Calculate initial surge multiplier
            base_surge = predicted_fare / self.baseline_fare
            
            # Apply dynamic factors based on conditions
            surge_multiplier = self._adjust_surge(base_surge[0], features)
            
            # Ensure surge stays within reasonable bounds (1.0x - 3.0x)
            final_surge = np.clip(surge_multiplier, 1.0, 3.0)
            
            return np.array([final_surge])
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def _adjust_surge(self, base_surge, features):
        """
        Apply situational adjustments to the surge multiplier
        """
        multiplier = base_surge
        
        # Time-based adjustments
        hour = features['hour'].iloc[0]
        is_weekend = features['is_weekend'].iloc[0]
        passenger_count = features['passenger_count'].iloc[0]
        trip_distance = features['trip_distance'].iloc[0]
        
        # Rush hour adjustment
        if hour in [7, 8, 9, 16, 17, 18] and not is_weekend:
            multiplier *= 1.2
        
        # Late night adjustment
        if hour >= 22 or hour <= 4:
            multiplier *= 1.15
        
        # Weekend adjustment
        if is_weekend:
            if hour >= 20 or hour <= 2:  # Weekend nights
                multiplier *= 1.25
            else:
                multiplier *= 1.1
        
        # High demand indicators
        if passenger_count >= 3:
            multiplier *= 1.1
        
        if trip_distance > 5:
            multiplier *= 1.15
        
        # Weather and special events would go here
        # For now, we'll add a small random factor for variety
        random_factor = np.random.uniform(0.95, 1.05)
        multiplier *= random_factor
        
        return multiplier