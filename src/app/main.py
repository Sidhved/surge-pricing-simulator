# src/app/main.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_path = str(Path(__file__).parent.parent)
sys.path.append(src_path)

from data.dataloader import DataLoader
from models.surge_predictor import SurgePredictor

class SurgePricingApp:
    def __init__(self):
        # Initialize session state variables if they don't exist
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        
        if 'df' not in st.session_state:
            st.session_state.df = None
            
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
            
        if 'data_loader' not in st.session_state:
            st.session_state.data_loader = DataLoader()
            
        if 'surge_predictor' not in st.session_state:
            st.session_state.surge_predictor = SurgePredictor()

    def load_data_callback(self, start_date, end_date):
        """Callback function for loading data"""
        try:
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.min.time())
            
            df = st.session_state.data_loader.load_data(start_datetime, end_datetime)
            
            if not df.empty:
                st.session_state.df = df
                st.session_state.data_loaded = True
                
                # Train model
                st.session_state.surge_predictor.train(df)
                st.session_state.model_trained = True
                return True
            return False
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False

    def run(self):
        st.set_page_config(
            page_title="Surge Pricing Simulator",
            page_icon="🚗",
            layout="wide"
        )
        
        st.title("Real-Time Surge Pricing Simulator")
        
        # Data loading section
        with st.expander("Data Loading Section", expanded=not st.session_state.data_loaded):
            st.header("Historical Data Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    datetime(2022, 1, 1),
                    key='start_date'
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    datetime(2022, 1, 31),
                    key='end_date'
                )
            
            if st.button("Load Data", key='load_data_button'):
                with st.spinner("Fetching data..."):
                    success = self.load_data_callback(start_date, end_date)
                    if success:
                        st.success("Data loaded successfully!")
        
        # Only show visualizations and simulation if data is loaded
        if st.session_state.data_loaded and st.session_state.df is not None:
            # Visualization section
            with st.expander("Data Visualizations", expanded=True):
                self.create_visualizations(st.session_state.df)
            
            # Simulation section
            with st.expander("Surge Price Simulation", expanded=True):
                self.show_simulation_section()

    def create_visualizations(self, df):
        st.subheader("Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly demand pattern
            hourly_demand = df.groupby('hour')['total_amount'].mean().reset_index()
            fig1 = px.line(
                hourly_demand,
                x='hour',
                y='total_amount',
                title='Average Hourly Demand',
                labels={'hour': 'Hour of Day', 'total_amount': 'Average Fare ($)'}
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Day of week pattern
            daily_demand = df.groupby('day_of_week')['total_amount'].mean().reset_index()
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_demand['day_name'] = daily_demand['day_of_week'].map(dict(enumerate(days)))
            fig2 = px.bar(
                daily_demand,
                x='day_name',
                y='total_amount',
                title='Average Daily Demand',
                labels={'day_name': 'Day of Week', 'total_amount': 'Average Fare ($)'}
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)

        # Price distribution
        fig3 = px.histogram(
            df,
            x='total_amount',
            nbins=50,
            title='Fare Distribution',
            labels={'total_amount': 'Fare Amount ($)', 'count': 'Frequency'}
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

        # Add metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Fare", f"${df['total_amount'].mean():.2f}")
        with col2:
            st.metric("Median Fare", f"${df['total_amount'].median():.2f}")
        with col3:
            st.metric("Average Trip Distance", f"{df['trip_distance'].mean():.1f} miles")
        with col4:
            st.metric("Total Trips", f"{len(df):,}")

    def show_simulation_section(self):
        st.header("Real-Time Simulation")
        
        if st.session_state.model_trained:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                hour = st.slider(
                    "Hour of Day",
                    0, 23, 12,
                    key='sim_hour'
                )
            with col2:
                day = st.selectbox(
                    "Day of Week",
                    ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"],
                    key='sim_day'
                )
            with col3:
                passenger_count = st.number_input(
                    "Average Passenger Count",
                    1, 6, 2,
                    key='sim_passengers'
                )
            
            distance = st.slider(
                "Trip Distance (miles)",
                0.0, 20.0, 2.5,
                step=0.5,
                key='sim_distance'
            )
            
            if st.button("Simulate", key='simulate_button'):
                sample_data = pd.DataFrame({
                    'hour': [hour],
                    'day_of_week': [["Monday", "Tuesday", "Wednesday",
                                    "Thursday", "Friday", "Saturday",
                                    "Sunday"].index(day)],
                    'passenger_count': [passenger_count],
                    'trip_distance': [distance],
                    'is_weekend': [1 if day in ["Saturday", "Sunday"] else 0],
                    'is_rush_hour': [1 if hour in [7,8,9,16,17,18] else 0]
                })
                
                try:
                    surge_multiplier = st.session_state.surge_predictor.predict_surge(sample_data)[0]
                    
                    # Create columns for results
                    col1, col2, col3 = st.columns(3)
                    
                    # Display results with more context
                    with col1:
                        st.metric(
                            "Surge Multiplier",
                            f"{surge_multiplier:.2f}x",
                            delta=f"{(surge_multiplier-1)*100:.1f}% adjustment"
                        )
                    
                    base_price = 10 + (2 * distance)  # Basic price calculation
                    surge_price = base_price * surge_multiplier
                    
                    with col2:
                        st.metric(
                            "Base Fare",
                            f"${base_price:.2f}",
                            delta=None
                        )
                    
                    with col3:
                        st.metric(
                            "Surge Fare",
                            f"${surge_price:.2f}",
                            delta=f"${surge_price - base_price:.2f}"
                        )
                    
                except Exception as e:
                    st.error(f"Error in simulation: {str(e)}")
        else:
            st.warning("Please load data first to train the model.")

if __name__ == "__main__":
    app = SurgePricingApp()
    app.run()