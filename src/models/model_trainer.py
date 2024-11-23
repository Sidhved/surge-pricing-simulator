from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple
import joblib
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        }
        self.best_model = None
        self.feature_importance = None
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for model training
        """
        # Basic features
        features = [
            'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
            'passenger_count', 'trip_distance'
        ]
        
        X = df[features].copy()
        y = df['total_amount']
        
        return X, y
    
    def train_and_evaluate(self, df: pd.DataFrame) -> Dict:
        """
        Train multiple models and select the best one
        """
        try:
            # Prepare data
            X, y = self.prepare_features(df)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            results = {}
            
            # Train and evaluate each model
            for name, model in self.models.items():
                logger.info(f"Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Calculate metrics
                results[name] = {
                    'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                    'r2_score': r2_score(y_test, test_pred),
                    'cv_scores': cross_val_score(model, X, y, cv=5, scoring='r2').mean()
                }
                
                logger.info(f"{name} Results: {results[name]}")
            
            # Select best model based on test RMSE
            best_model_name = min(results, key=lambda k: results[k]['test_rmse'])
            self.best_model = self.models[best_model_name]
            
            # Calculate feature importance for best model
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info(f"Best model: {best_model_name}")
            return results
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        try:
            joblib.dump(self.best_model, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk
        """
        try:
            self.best_model = joblib.load(filepath)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise