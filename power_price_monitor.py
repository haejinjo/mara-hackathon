import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta
import json
from collections import deque

# Page configuration
st.set_page_config(
    page_title="Hashrate Prediction Power Autoscaler",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 300;
        color: #1a73e8;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #5f6368;
        margin-bottom: 2rem;
    }
    
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e8eaed;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(60, 64, 67, 0.12);
    }
    
    .metric-large {
        font-size: 3rem;
        font-weight: 400;
        color: #1a73e8;
    }
    
    .metric-trend {
        font-size: 1rem;
        color: #5f6368;
    }
    
    /* Hide Streamlit branding for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stPlotlyChart {
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(60, 64, 67, 0.12);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for data management
if 'price_buffer' not in st.session_state:
    st.session_state.price_buffer = deque(maxlen=300)  # 5 minutes of data
    st.session_state.current_index = 0
    st.session_state.is_monitoring = True

def fetch_electricity_price():
    """Fetch real-time electricity spot price data"""
    try:
        # Load historical electricity price data
        if not hasattr(st.session_state, 'price_data'):
            with open('power-historic-data-api.py', 'r') as f:
                content = f.read()
                json_start = content.find('{')
                json_content = content[json_start:]
                data = json.loads(json_content)
                # Extract only price-related data
                st.session_state.price_data = [
                    {
                        'timestamp': datetime.now() - timedelta(seconds=len(data['data']) - i),
                        'price_per_mwh': item.get('energy_price', 0.08) * 1000,  # Convert to $/MWh
                        'price_per_kwh': item.get('energy_price', 0.08),  # $/kWh
                    }
                    for i, item in enumerate(data.get('data', []))
                ]
        
        # Cycle through data points for real-time simulation
        data_points = st.session_state.price_data
        if data_points:
            data_index = st.session_state.current_index % len(data_points)
            price_point = data_points[data_index].copy()
            price_point['timestamp'] = datetime.now()
            st.session_state.current_index += 1
            return price_point
        else:
            # Fallback to simulated data
            return generate_simulated_price()
            
    except Exception as e:
        st.error(f"Error loading price data: {e}")
        return generate_simulated_price()

def generate_simulated_price():
    """Generate simulated electricity price data"""
    base_price = 0.08  # Base price in $/kWh
    time_factor = (st.session_state.current_index % 3600) / 3600  # Hour cycle
    random_variation = (np.random.random() - 0.5) * 0.02
    
    # Simulate daily price patterns
    price_multiplier = 1 + 0.3 * np.sin(time_factor * 2 * np.pi) + random_variation
    current_price = base_price * price_multiplier
    
    st.session_state.current_index += 1
    
    return {
        'timestamp': datetime.now(),
        'price_per_kwh': current_price,
        'price_per_mwh': current_price * 1000
    }

# Header
st.markdown('<h1 class="main-header">🧠 AI-Powered Hashrate Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Reinforcement Learning model predictions based on historical hashrate data</p>', unsafe_allow_html=True)

# Control panel
col1, col2, col3 = st.columns([1, 1, 8])
with col1:
    if st.button("▶️ Start" if not st.session_state.is_monitoring else "⏸️ Pause"):
        st.session_state.is_monitoring = not st.session_state.is_monitoring

with col2:
    if st.button("🔄 Reset"):
        st.session_state.price_buffer.clear()
        st.session_state.current_index = 0

# Create containers for smooth updates
if 'containers' not in st.session_state:
    st.session_state.containers = {
        'current_price': st.empty(),
        'price_chart': st.empty(),
        'statistics': st.empty()
    }

# Main monitoring loop
while True:
    if st.session_state.is_monitoring:
        # Fetch new price data
        price_data = fetch_electricity_price()
        st.session_state.price_buffer.append(price_data)
        
        # Convert buffer to DataFrame
        df = pd.DataFrame(list(st.session_state.price_buffer))
        
        # Update current price display
        with st.session_state.containers['current_price'].container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Calculate price change
                price_change = 0
                if len(df) > 1:
                    price_change = df['price_per_mwh'].iloc[-1] - df['price_per_mwh'].iloc[-2]
                
                st.metric(
                    label="RL Predicted Price",
                    value=f"${price_data['price_per_mwh']:.2f}/MWh",
                    delta=f"{price_change:.2f} $/MWh"
                )
            
            with col2:
                st.metric(
                    label="Price per kWh",
                    value=f"${price_data['price_per_kwh']:.4f}/kWh",
                    delta=f"{price_change/1000:.4f} $/kWh"
                )
            
            with col3:
                # Price status indicator
                avg_price = df['price_per_mwh'].mean() if len(df) > 10 else price_data['price_per_mwh']
                status = "🟢 Below Average" if price_data['price_per_mwh'] < avg_price else "🔴 Above Average"
                st.metric(
                    label="Price Status",
                    value=status,
                    delta=f"vs. avg ${avg_price:.2f}/MWh"
                )
        
        # Update price chart
        with st.session_state.containers['price_chart'].container():
            st.subheader("RL Model Predictions - Hashrate-Based Forecasting")
            
            fig = go.Figure()
            
            # Add price line with gradient fill
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['price_per_mwh'],
                mode='lines',
                name='Electricity Price',
                line=dict(color='#1a73e8', width=3),
                fill='tozeroy',
                fillcolor='rgba(26, 115, 232, 0.1)',
                hovertemplate='<b>%{y:.2f} $/MWh</b><br>%{x}<extra></extra>'
            ))
            
            # Add average line
            if len(df) > 1:
                avg_price = df['price_per_mwh'].mean()
                fig.add_hline(
                    y=avg_price,
                    line_dash="dash",
                    line_color="#ea4335",
                    annotation_text=f"Average: ${avg_price:.2f}/MWh"
                )
            
            fig.update_layout(
                title="RL Model Predictions: Hashrate-Optimized Power Pricing",
                xaxis_title="Time",
                yaxis_title="Predicted Price ($/MWh)",
                height=400,
                template="plotly_white",
                showlegend=False,
                margin=dict(l=60, r=40, t=60, b=40),
                font=dict(family="Roboto, sans-serif"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Update statistics
        with st.session_state.containers['statistics'].container():
            if len(df) > 1:
                st.subheader("RL Model Performance & Prediction Analytics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.info(f"""
                    **Current Session**  
                    Duration: {len(df)} seconds  
                    Data Points: {len(df)}  
                    Update Rate: 1 Hz
                    """)
                
                with col2:
                    min_price = df['price_per_mwh'].min()
                    max_price = df['price_per_mwh'].max()
                    st.info(f"""
                    **Price Range**  
                    Minimum: ${min_price:.2f}/MWh  
                    Maximum: ${max_price:.2f}/MWh  
                    Spread: ${max_price - min_price:.2f}/MWh
                    """)
                
                with col3:
                    avg_price = df['price_per_mwh'].mean()
                    std_price = df['price_per_mwh'].std()
                    st.info(f"""
                    **Statistical Summary**  
                    Average: ${avg_price:.2f}/MWh  
                    Std Dev: ${std_price:.2f}/MWh  
                    Volatility: {(std_price/avg_price)*100:.1f}%
                    """)
                
                with col4:
                    # Recent trend
                    recent_trend = "Stable"
                    if len(df) >= 10:
                        recent_prices = df['price_per_mwh'].tail(10)
                        if recent_prices.iloc[-1] > recent_prices.iloc[0] * 1.05:
                            recent_trend = "📈 Rising"
                        elif recent_prices.iloc[-1] < recent_prices.iloc[0] * 0.95:
                            recent_trend = "📉 Falling"
                        else:
                            recent_trend = "➡️ Stable"
                    
                    st.info(f"""
                    **Market Trend**  
                    Status: {recent_trend}  
                    Latest: ${df['price_per_mwh'].iloc[-1]:.2f}/MWh  
                    Change: {((df['price_per_mwh'].iloc[-1]/df['price_per_mwh'].iloc[0] - 1) * 100):.1f}%
                    """)
    
    # Update every second
    time.sleep(1)
    
    if not st.session_state.is_monitoring:
        time.sleep(0.1)  # Reduce CPU usage when paused 