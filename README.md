# Surge Pricing Simulator

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.24.0-red.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://surge-pricing-simulator.streamlit.app/)

A machine learning-powered surge pricing simulator that emulates dynamic pricing systems used by ride-sharing services. Built with Streamlit and Python, this application provides real-time price adjustments based on various factors including time of day, demand patterns, and trip characteristics.

## 🌍 Live Demo

Try the live application here: [Surge Pricing Simulator](https://surge-pricing-simulator.streamlit.app/)

### Quick Start Guide
1. Select a date range in 2022 to analyze historical patterns
2. Click "Load Data" to fetch and process taxi trip data
3. Explore the generated visualizations:
   - Hourly demand patterns
   - Daily demand variations
   - Fare distribution analysis
4. In the simulation section, adjust parameters:
   - Hour of day (0-23)
   - Day of week
   - Number of passengers (1-6)
   - Trip distance
5. Click "Simulate" to see the dynamic pricing results:
   - Surge multiplier (1.0x - 3.0x)
   - Base fare
   - Final surge price

## 🌟 Features

### Real-Time Pricing Simulation
- Dynamic surge multiplier calculation (1.0x - 3.0x range)
- Interactive parameter adjustment
- Rush hour and weekend demand modeling
- Trip distance and passenger count considerations

### Data Analysis & Visualization
- Interactive hourly demand patterns
- Daily fare distribution analysis
- Weekday vs weekend trends
- Price per mile metrics
- Real-time summary statistics

### Machine Learning Integration
- Gradient Boosting model for price prediction
- Time-based feature engineering
- Historical pattern recognition
- Automated model training

### Data Sources
- NYC TLC Yellow Taxi Trip Data (2022)
- Intelligent sample data generation
- Dynamic data loading based on date range

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.9+**: Core programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning implementation
- **Plotly**: Interactive visualizations
- **NumPy**: Numerical computations

## 📊 Project Structure
```
surge-pricing-simulator/
├── src/                      # Source code
│   ├── app/                  # Streamlit application
│   │   ├── __init__.py
│   │   └── main.py          # Main application file
│   ├── data/                 # Data handling
│   │   ├── __init__.py
│   │   └── data_loader.py   # Data loading and processing
│   └── models/              # ML models
│       ├── __init__.py
│       └── surge_predictor.py  # Surge pricing model
├── requirements.txt         # Dependencies
└── README.md               # Documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip (Python package installer)
- Git

### Installation

1. Clone the repository
```bash
git clone https://github.com/Sidhved/surge-pricing-simulator.git
cd surge-pricing-simulator
```

2. Create and activate virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Running Locally

1. Start the Streamlit app
```bash
streamlit run src/app/main.py
```

2. Open your browser and navigate to `http://localhost:8501`

## 🧪 Algorithm Details

### Surge Pricing Factors
The surge multiplier is calculated based on:
- Time of day
  - Rush hours (7-9 AM, 4-6 PM)
  - Late night hours (10 PM - 4 AM)
- Day of week
  - Weekday vs weekend
  - Business hours
- Trip characteristics
  - Distance
  - Passenger count
- Historical patterns
  - Average fares
  - Typical demand

### Model Features
- Gradient Boosting Regressor for base prediction
- Feature engineering including:
  - Time-based features
  - Rush hour indicators
  - Weekend flags
  - Trip characteristics
- Dynamic multiplier adjustments
- Bounded output (1.0x - 3.0x range)

## 📈 Future Improvements

- [ ] Weather data integration
- [ ] Special events consideration
- [ ] Geographic heat maps
- [ ] Extended date range support
- [ ] Additional visualization options
- [ ] Enhanced ML model features
- [ ] Performance optimizations
- [ ] Real-time data updates

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments

- [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) for providing the dataset
- [Streamlit](https://streamlit.io/) team for the amazing framework
- [Scikit-learn](https://scikit-learn.org/) for machine learning tools
- [Plotly](https://plotly.com/) for visualization capabilities

## 📬 Contact

Your Name - [sidhved.warik@gmail.com](mailto:sidhved.warik@gmail.com)

Project Links: 
- Live Demo: [https://surge-pricing-simulator.streamlit.app/](https://surge-pricing-simulator.streamlit.app/)
- GitHub: [https://github.com/Sidhved/surge-pricing-simulator](https://github.com/Sidhved/surge-pricing-simulator)