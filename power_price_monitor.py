import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    st.session_state.price_buffer = deque(maxlen=600)  # 10 minutes of data buffer (sliding window shows 5 minutes)
    st.session_state.current_index = 0
    st.session_state.is_monitoring = True

def fetch_electricity_price():
    """Fetch real-time electricity spot price and hashrate data"""
    try:
        # Load historical electricity price and hashrate data
        if not hasattr(st.session_state, 'price_data'):
            with open('power-historic-data-api.py', 'r') as f:
                content = f.read()
                json_start = content.find('{')
                json_content = content[json_start:]
                data = json.loads(json_content)
                # Extract power and hashrate data
                st.session_state.price_data = [
                    {
                        'timestamp': datetime.now() - timedelta(seconds=len(data['data']) - i),
                        'price_per_mwh': item.get('energy_price', 0.08) * 1000,  # Convert to $/MWh
                        'price_per_kwh': item.get('energy_price', 0.08),  # $/kWh
                        'power_mw': item.get('power_mw', 5.8156),  # MW
                        'hashrate_ths': item.get('hashrate_ths', 183200),  # TH/s
                        'hashrate_phs': item.get('hashrate_ths', 183200) / 1000,  # Convert to PH/s (183.2 PH/s)
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
    """Generate simulated power consumption, price and hashrate data"""
    base_price = 0.08  # Base price in $/kWh
    base_hashrate = 183200  # Base hashrate in TH/s
    base_power = 5.8156  # Base power in MW
    time_factor = (st.session_state.current_index % 3600) / 3600  # Hour cycle
    random_variation = (np.random.random() - 0.5) * 0.02
    
    # Simulate daily price patterns
    price_multiplier = 1 + 0.3 * np.sin(time_factor * 2 * np.pi) + random_variation
    current_price = base_price * price_multiplier
    
    # Simulate hashrate with inverse correlation to price (when price is high, hashrate might be lower due to curtailment)
    hashrate_multiplier = 1 - 0.1 * np.sin(time_factor * 2 * np.pi) + random_variation * 0.5
    current_hashrate = base_hashrate * hashrate_multiplier
    
    # Simulate power consumption correlated with hashrate
    power_multiplier = hashrate_multiplier * (1 + random_variation * 0.3)
    current_power = base_power * power_multiplier
    
    st.session_state.current_index += 1
    
    return {
        'timestamp': datetime.now(),
        'price_per_kwh': current_price,
        'price_per_mwh': current_price * 1000,
        'power_mw': current_power,
        'hashrate_ths': current_hashrate,
        'hashrate_phs': current_hashrate / 1000
    }

# Header
st.markdown('<h1 class="main-header">⚡ Mining Facility Power & Performance Monitor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time monitoring of power consumption and mining hashrate performance</p>', unsafe_allow_html=True)

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
        # Fetch 10 new data points at once
        for _ in range(10):
            price_data = fetch_electricity_price()
            st.session_state.price_buffer.append(price_data)
        
        # Convert buffer to DataFrame
        df = pd.DataFrame(list(st.session_state.price_buffer))
        
        # Get the latest data point for metrics display
        latest_data = df.iloc[-1] if len(df) > 0 else price_data
        
        # Update current price display
        with st.session_state.containers['current_price'].container():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Calculate price change (comparing latest to 10 points ago)
                price_change = 0
                if len(df) > 10:
                    price_change = df['price_per_mwh'].iloc[-1] - df['price_per_mwh'].iloc[-11]
                
                st.metric(
                    label="Energy Price",
                    value=f"${latest_data['price_per_mwh']:.2f}/MWh",
                    delta=f"{price_change:.2f} $/MWh"
                )
            
            with col2:
                # Calculate power change (comparing latest to 10 points ago)
                power_change = 0
                if len(df) > 10:
                    power_change = df['power_mw'].iloc[-1] - df['power_mw'].iloc[-11]
                
                st.metric(
                    label="Power Consumption",
                    value=f"{latest_data['power_mw']:.3f} MW",
                    delta=f"{power_change:.3f} MW"
                )
            
            with col3:
                # Calculate hashrate change (comparing latest to 10 points ago)
                hashrate_change = 0
                if len(df) > 10:
                    hashrate_change = df['hashrate_ths'].iloc[-1] - df['hashrate_ths'].iloc[-11]
                
                st.metric(
                    label="Hash Rate",
                    value=f"{latest_data['hashrate_ths']:,.0f} TH/s",
                    delta=f"{hashrate_change:,.0f} TH/s"
                )
            
            with col4:
                # Price status indicator
                avg_price = df['price_per_mwh'].mean() if len(df) > 10 else price_data['price_per_mwh']
                status = "🟢 Below Average" if price_data['price_per_mwh'] < avg_price else "🔴 Above Average"
                st.metric(
                    label="Price Status",
                    value=status,
                    delta=f"vs. avg ${avg_price:.2f}/MWh"
                )
        
        # Update power and hashrate chart
        with st.session_state.containers['price_chart'].container():
            st.subheader("Power Consumption & Hash Rate Monitoring")
            st.caption("📊 Sliding 5-minute window • Updates every 10 seconds with 10 data points")
            
            # Create subplot with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add power consumption line with gradient fill
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['power_mw'],
                mode='lines',
                name='Power Consumption',
                line=dict(color='#f7931a', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(247, 147, 26, 0.1)',
                hovertemplate='<b>%{y:.3f} MW</b><br>%{x|%H:%M:%S}<extra></extra>',
                connectgaps=False
            ), secondary_y=False)
            
            # Add hashrate line
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['hashrate_ths'],
                mode='lines',
                name='Hash Rate',
                line=dict(color='#4ecdc4', width=2),
                hovertemplate='<b>%{y:,.0f} TH/s</b><br>%{x|%H:%M:%S}<extra></extra>',
                yaxis='y2',
                connectgaps=False
            ), secondary_y=True)
            
            # Add average power line
            if len(df) > 1:
                avg_power = df['power_mw'].mean()
                fig.add_hline(
                    y=avg_power,
                    line_dash="dash",
                    line_color="#f7931a",
                    line_width=1,
                    opacity=0.5,
                    annotation_text=f"Avg Power: {avg_power:.3f} MW",
                    secondary_y=False
                )
            
            # Set y-axes titles
            fig.update_yaxes(title_text="Power Consumption (MW)", secondary_y=False)
            fig.update_yaxes(title_text="Hash Rate (TH/s)", secondary_y=True)
            
            # Calculate sliding window x-axis range (constant size)
            if len(df) > 0:
                latest_time = df['timestamp'].iloc[-1]
                
                # Fixed window size: 5 minutes of data
                window_size = pd.Timedelta(minutes=5)
                
                # Calculate window start time (5 minutes before latest)
                window_start = latest_time - window_size
                
                # Add small buffer to the right (30 seconds)
                buffer_time = pd.Timedelta(seconds=30)
                x_axis_end = latest_time + buffer_time
                
                # Set fixed sliding window range
                fig.update_layout(
                    xaxis=dict(
                        range=[window_start, x_axis_end],
                        showgrid=True,
                        gridcolor='rgba(128,128,128,0.2)',
                        fixedrange=False
                    )
                )
            
            fig.update_layout(
                title="Mining Facility: Real-Time Performance (5-Minute Sliding Window)",
                xaxis_title="Time",
                height=400,
                template="plotly_white",
                showlegend=True,
                legend=dict(x=0.02, y=0.98),
                margin=dict(l=60, r=60, t=60, b=40),
                font=dict(family="Roboto, sans-serif"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                # Add subtle grid lines for stock chart feel
                xaxis_showgrid=True,
                yaxis_showgrid=True,
                xaxis_gridcolor='rgba(128,128,128,0.2)',
                yaxis_gridcolor='rgba(128,128,128,0.2)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Update statistics
        with st.session_state.containers['statistics'].container():
            if len(df) > 1:
                st.subheader("RL Model Performance & Prediction Analytics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Calculate visible data points in the 5-minute window
                    if len(df) > 0:
                        latest_time = df['timestamp'].iloc[-1]
                        window_start = latest_time - pd.Timedelta(minutes=5)
                        visible_data = df[df['timestamp'] >= window_start]
                        visible_count = len(visible_data)
                    else:
                        visible_count = 0
                    
                    st.info(f"""
                    **Sliding Window**  
                    Window Size: 5 minutes  
                    Visible Points: {visible_count}  
                    Total Buffered: {len(df)}  
                    Update Rate: 10 points/10s
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
    
    # Update every 10 seconds (with 10 data points each time)
    time.sleep(10)
    
    if not st.session_state.is_monitoring:
        time.sleep(1)  # Reduce CPU usage when paused 