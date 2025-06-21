import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json
import requests
from collections import deque

# Page configuration
st.set_page_config(
    page_title="Mara Holdings Mining Center - Real-Time Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add CSS for smooth updates
st.markdown("""
<style>
    /* Smooth transitions for all elements */
    .stMetric {
        transition: all 0.3s ease-in-out;
    }
    
    /* Prevent layout shift */
    .js-plotly-plot {
        transition: opacity 0.2s ease-in-out;
    }
    
    /* Smooth metric updates */
    [data-testid="metric-container"] {
        transition: all 0.2s ease-in-out;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Smooth container updates */
    .stContainer {
        transition: all 0.1s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = deque(maxlen=300)  # 5 minutes of data
    st.session_state.current_index = 0
    st.session_state.is_running = True
    st.session_state.demand_response_active = False
    st.session_state.total_cost = 0
    st.session_state.total_revenue = 0

# Simulation parameters
BASE_POWER_MW = 5.8156
BASE_HASHRATE_THS = 183200
BTC_PRICE_USD = 65000
NETWORK_HASHRATE_EHS = 750

# Function to generate realistic data point
def generate_data_point(index):
    """Generate a single data point based on the index (seconds from start)"""
    timestamp = datetime.now()
    minute = (index // 60) % 30
    
    # Temperature profile
    temperature = 34.6 + (minute / 30) * 0.7
    
    # Base values
    power_mw = BASE_POWER_MW
    hashrate_ths = BASE_HASHRATE_THS
    
    # Environmental factor
    temp_factor = 1 + (temperature - 34.6) * 0.015
    random_variation = (np.random.random() - 0.5) * 0.02
    
    # Demand response logic
    demand_response = False
    if 15 <= minute < 20:
        demand_response = True
        power_mw *= 0.715
        hashrate_ths *= 0.715
    
    # Apply variations
    power_mw = power_mw * temp_factor * (1 + random_variation)
    hashrate_ths = hashrate_ths * (1 + random_variation * 0.8)
    
    # Electricity price dynamics
    if minute < 5:
        energy_price = 0.07845 + (np.random.random() - 0.5) * 0.002
    elif minute < 10:
        energy_price = 0.08230 + (np.random.random() - 0.5) * 0.003
    elif minute < 15:
        energy_price = 0.09155 + (np.random.random() - 0.5) * 0.005
    elif minute < 20:
        if minute == 15 and (index % 60) < 5:
            energy_price = 0.15678
        else:
            energy_price = 0.13423 + (np.random.random() - 0.5) * 0.01
    elif minute < 25:
        energy_price = 0.09567 + (np.random.random() - 0.5) * 0.004
    else:
        energy_price = 0.08645 + (np.random.random() - 0.5) * 0.003
    
    # Calculate metrics
    facility_hashrate_ehs = hashrate_ths / 1000000
    pool_share = facility_hashrate_ehs / NETWORK_HASHRATE_EHS
    btc_per_day = pool_share * 144 * 3.125
    revenue_per_day = btc_per_day * BTC_PRICE_USD
    hash_price = revenue_per_day / hashrate_ths
    
    electricity_cost = (power_mw * energy_price * 1000) / 3600
    mining_revenue = revenue_per_day / 86400
    efficiency = (power_mw * 1000000) / hashrate_ths
    profit_margin = ((mining_revenue - electricity_cost) / mining_revenue) * 100
    
    return {
        'timestamp': timestamp,
        'power_mw': power_mw,
        'energy_price': energy_price,
        'hash_price': hash_price,
        'electricity_cost': electricity_cost,
        'mining_revenue': mining_revenue,
        'hashrate_ths': hashrate_ths,
        'efficiency': efficiency,
        'temperature': temperature,
        'profit_margin': profit_margin,
        'demand_response': demand_response
    }

# Mock API call function (replace with actual API endpoint)
def fetch_latest_data():
    """Fetch data from static JSON file and cycle through data points"""
    try:
        # Load historical data if not already loaded
        if not hasattr(st.session_state, 'historical_data'):
            with open('power-historic-data-api.py', 'r') as f:
                content = f.read()
                # Extract JSON data (skip the comments at the top)
                json_start = content.find('{')
                json_content = content[json_start:]
                st.session_state.historical_data = json.loads(json_content)
        
        # Get data points
        data_points = st.session_state.historical_data.get('data', [])
        
        if data_points:
            # Cycle through data points using current_index
            data_index = st.session_state.current_index % len(data_points)
            data_point = data_points[data_index].copy()
            
            # Update timestamp to current time for real-time simulation
            data_point['timestamp'] = datetime.now()
            
            # Add missing fields if they don't exist in historical data
            if 'demand_response' not in data_point:
                # Calculate demand response based on power consumption
                # If power is significantly below base power, assume demand response is active
                power_threshold = BASE_POWER_MW * 0.85  # 85% of base power
                data_point['demand_response'] = data_point.get('power_mw', BASE_POWER_MW) < power_threshold
            
            # Ensure all required fields exist with defaults
            data_point.setdefault('temperature', 34.6)
            data_point.setdefault('efficiency', 31.7)
            data_point.setdefault('profit_margin', 0.0)
            data_point.setdefault('power_mw', BASE_POWER_MW)
            data_point.setdefault('energy_price', 0.08)
            data_point.setdefault('hashrate_ths', BASE_HASHRATE_THS)
            data_point.setdefault('electricity_cost', 0.13)
            data_point.setdefault('mining_revenue', 0.08)
            data_point.setdefault('hash_price', 0.039)
            
            st.session_state.current_index += 1
            return data_point
        else:
            st.error("No data points found in historical data")
            # Fallback to simulated data
            data_point = generate_data_point(st.session_state.current_index)
            st.session_state.current_index += 1
            return data_point
            
    except FileNotFoundError:
        st.error("power-historic-data-api.py file not found")
        # Fallback to simulated data
        data_point = generate_data_point(st.session_state.current_index)
        st.session_state.current_index += 1
        return data_point
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON data: {e}")
        # Fallback to simulated data
        data_point = generate_data_point(st.session_state.current_index)
        st.session_state.current_index += 1
        return data_point
    except Exception as e:
        st.error(f"Unexpected error loading data: {e}")
        # Fallback to simulated data
        data_point = generate_data_point(st.session_state.current_index)
        st.session_state.current_index += 1
        return data_point

# Header
st.markdown("# ⛏️ Real-Time Mining Analytics")
st.markdown("### by Mining Alpha")

# Control buttons
col1, col2, col3, col4 = st.columns([1, 1, 1, 6])
with col1:
    if st.button("▶️ Start" if not st.session_state.is_running else "⏸️ Pause"):
        st.session_state.is_running = not st.session_state.is_running
with col2:
    if st.button("🔄 Reset"):
        st.session_state.data_buffer.clear()
        st.session_state.current_index = 0
        st.session_state.total_cost = 0
        st.session_state.total_revenue = 0

# Create placeholder for real-time updates
placeholder = st.empty()

# Create persistent chart containers to avoid re-rendering
if 'chart_containers' not in st.session_state:
    st.session_state.chart_containers = {
        'metrics': st.empty(),
        'charts': st.empty(), 
        'profit': st.empty(),
        'summary': st.empty()
    }

# Main loop
while True:
    if st.session_state.is_running:
        # Fetch new data
        new_data = fetch_latest_data()
        st.session_state.data_buffer.append(new_data)
        
        # Update totals
        st.session_state.total_cost += new_data['electricity_cost']
        st.session_state.total_revenue += new_data['mining_revenue']
        st.session_state.demand_response_active = new_data['demand_response']
        
        # Create DataFrame from buffer
        df = pd.DataFrame(list(st.session_state.data_buffer))
        
        # Update metrics section smoothly
        with st.session_state.chart_containers['metrics'].container():
            # Top metrics row
            metric_cols = st.columns(6)
            
            with metric_cols[0]:
                st.metric(
                    "Power Consumption",
                    f"{new_data['power_mw']:.3f} MW",
                    f"{(new_data['power_mw'] - BASE_POWER_MW):.3f} MW"
                )
            
            with metric_cols[1]:
                st.metric(
                    "Energy Price",
                    f"${new_data['energy_price']*1000:.2f}/MWh",
                    f"{((new_data['energy_price'] - 0.08) * 1000):.2f}"
                )
            
            with metric_cols[2]:
                margin_color = "🟢" if new_data['profit_margin'] > 0 else "🔴"
                st.metric(
                    "Profit Margin",
                    f"{margin_color} {new_data['profit_margin']:.1f}%",
                    f"{new_data['profit_margin'] - 30:.1f}%"
                )
            
            with metric_cols[3]:
                st.metric(
                    "Hash Rate",
                    f"{new_data['hashrate_ths']/1000:.1f} PH/s",
                    f"{(new_data['hashrate_ths'] - BASE_HASHRATE_THS)/1000:.1f} PH/s"
                )
            
            with metric_cols[4]:
                st.metric(
                    "Temperature",
                    f"{new_data['temperature']:.1f}°C",
                    f"{new_data['temperature'] - 34.6:.1f}°C"
                )
            
            with metric_cols[5]:
                dr_status = "🚨 ACTIVE" if st.session_state.demand_response_active else "✅ Normal"
                st.metric(
                    "Grid Status",
                    dr_status,
                    "Curtailed" if st.session_state.demand_response_active else "Operating"
                )
        
        # Update charts section smoothly
        with st.session_state.chart_containers['charts'].container():
            # Charts
            col_left, col_right = st.columns(2)
            
            with col_left:
                # Power consumption chart with stable key
                fig_power = make_subplots(rows=1, cols=1)
                fig_power.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['power_mw'],
                    mode='lines',
                    name='Power',
                    line=dict(color='#f7931a', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(247, 147, 26, 0.2)'
                ), row=1, col=1)
                
                fig_power.update_layout(
                    title="Real-Time Power Consumption",
                    xaxis_title="Time",
                    yaxis_title="Power (MW)",
                    height=350,
                    template="plotly_white",  # Changed to white for better visibility
                    showlegend=False,
                    margin=dict(l=40, r=40, t=40, b=40),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig_power, use_container_width=True)
            
            with col_right:
                # Energy price chart with stable key
                fig_price = make_subplots(rows=1, cols=1)
                fig_price.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['energy_price'] * 1000,  # Convert to $/MWh
                    mode='lines',
                    name='Price',
                    line=dict(color='#4ecdc4', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(78, 205, 196, 0.2)'
                ), row=1, col=1)
                
                fig_price.update_layout(
                    title="Electricity Spot Price",
                    xaxis_title="Time",
                    yaxis_title="Price ($/MWh)",
                    height=350,
                    template="plotly_white",  # Changed to white for better visibility
                    showlegend=False,
                    margin=dict(l=40, r=40, t=40, b=40),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig_price, use_container_width=True)
        
        # Update profit chart section smoothly
        with st.session_state.chart_containers['profit'].container():
            # Profit/Loss chart
            st.subheader("Real-Time Profitability Analysis")
            
            # Create profit margin chart with color coding
            fig_profit = make_subplots(rows=1, cols=1)
            
            # Add profit margin line
            colors = ['#4ecdc4' if pm > 0 else '#ff6b6b' for pm in df['profit_margin']]
            fig_profit.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['profit_margin'],
                mode='lines+markers',
                name='Profit Margin',
                line=dict(width=3),
                marker=dict(size=6, color=colors),
                text=[f"{pm:.1f}%" for pm in df['profit_margin']],
                hovertemplate='%{text}<extra></extra>'
            ), row=1, col=1)
            
            # Add zero line
            fig_profit.add_hline(
                y=0, 
                line_dash="dash", 
                line_color="gray",
                annotation_text="Break Even"
            )
            
            fig_profit.update_layout(
                xaxis_title="Time",
                yaxis_title="Profit Margin (%)",
                height=300,
                template="plotly_white",  # Changed to white for better visibility
                showlegend=False,
                margin=dict(l=40, r=40, t=10, b=40),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_profit, use_container_width=True)
        
        # Update summary section smoothly
        with st.session_state.chart_containers['summary'].container():
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.info(f"""
                **Session Statistics**  
                Duration: {len(df)} seconds  
                Total Cost: ${st.session_state.total_cost:.2f}  
                Total Revenue: ${st.session_state.total_revenue:.2f}
                """)
            
            with col2:
                net_profit = st.session_state.total_revenue - st.session_state.total_cost
                profit_emoji = "💰" if net_profit > 0 else "📉"
                st.info(f"""
                **Financial Performance**  
                Net Profit: {profit_emoji} ${net_profit:.2f}  
                Avg Margin: {df['profit_margin'].mean():.1f}%  
                Hash Price: ${df['hash_price'].mean():.4f}/TH/day
                """)
            
            with col3:
                st.info(f"""
                **Operational Metrics**  
                Avg Power: {df['power_mw'].mean():.3f} MW  
                Avg Efficiency: {df['efficiency'].mean():.1f} W/TH  
                Total Hash: {df['hashrate_ths'].mean()/1000:.1f} PH/s
                """)
            
            with col4:
                demand_response_count = sum(1 for d in st.session_state.data_buffer if d['demand_response'])
                st.info(f"""
                **Grid Response**  
                DR Events: {1 if demand_response_count > 0 else 0}  
                DR Duration: {demand_response_count}s  
                Power Saved: {(BASE_POWER_MW - df[df['demand_response'] == True]['power_mw'].mean()) if demand_response_count > 0 else 0:.3f} MW
                """)
    
    # Sleep for 1 second
    time.sleep(1)
    
    # Check if user wants to stop
    if not st.session_state.is_running:
        time.sleep(0.1)  # Reduce CPU usage when paused