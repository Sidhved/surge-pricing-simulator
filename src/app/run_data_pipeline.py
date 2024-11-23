import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_path = str(Path(__file__).parent.parent)
sys.path.append(src_path)

from data.dataloader import DataLoader
from models.model_trainer import ModelTrainer
from datetime import datetime, timedelta

# Initialize components
data_loader = DataLoader()
model_trainer = ModelTrainer()

# Load data
start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 1, 31)

# Load and process data
df = data_loader.load_and_prepare_data(start_date, end_date)

# Train models
results = model_trainer.train_and_evaluate(df)

# Save the best model
model_trainer.save_model('models/saved_models/best_model.joblib')