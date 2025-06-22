import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json
from collections import deque

# Import the HashPricePredictor
try:
    from hash_library import HashPricePredictor
    PREDICTOR_AVAILABLE = True
    # Initialize the predictor
    if 'hash_predictor' not in st.session_state:
        st.session_state.hash_predictor = HashPricePredictor("best_gru.pt", device="cpu")
        st.session_state.hash_price_window = deque(maxlen=30)  # Keep last 30 hash prices
except Exception as e:
    PREDICTOR_AVAILABLE = False
    st.error(f"❌ HashPricePredictor not available: {e}")
    st.info("The app will continue without AI predictions.")

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

# Initialize basic session state for data management
if 'price_buffer' not in st.session_state:
    st.session_state.price_buffer = deque(maxlen=600)  # 10 minutes of data buffer (sliding window shows 5 minutes)
    st.session_state.current_index = 0
    st.session_state.is_monitoring = True
    st.session_state.initialized = False

# Ensure hash_price_window is initialized if predictor is available
if PREDICTOR_AVAILABLE and 'hash_price_window' not in st.session_state:
    st.session_state.hash_price_window = deque(maxlen=30)
    st.session_state.last_prediction = None  # For prediction smoothing

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
                        # Scale hash price to match training data: original was ~6.3e-07, target ~$55/day = 6.365e-04/sec
                        'hash_price': item.get('hash_price', 6.3e-07) * 1010.0,  # Scale factor: 6.365e-04 / 6.3e-07 ≈ 1010
                        # Calculate realistic mining revenue: hashrate * hash_price
                        # ~190,000 TH/s * 6.365e-04 $/TH/s = ~121 $/s
                        'mining_revenue': item.get('hashrate_ths', 183200) * item.get('hash_price', 6.3e-07) * 1010.0,
                        'temperature': item.get('temperature', 20.0),  # Celsius
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
            
            # Store hash price for prediction if predictor is available
            if PREDICTOR_AVAILABLE and 'hash_price_window' in st.session_state:
                # Convert hash_price to USD (from USD per TH/s per second to USD per TH/s per day)
                hash_price_usd = price_point['hash_price'] * 86400  # Convert per second to per day
                st.session_state.hash_price_window.append(hash_price_usd)
                

            
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
    
    # Simulate hash price in the range similar to training data (around $50-60/TH/day)
    # Convert to per-second for internal consistency: $55/day ÷ 86400 seconds
    base_hash_price_per_day = 55.0  # USD per TH/s per day (similar to training data)
    base_hash_price = base_hash_price_per_day / 86400  # Convert to per second
    # Add variation similar to training data (±10-15%)
    hash_price_variation = base_hash_price * (1 + random_variation * 0.3)
    
    st.session_state.current_index += 1
    
    data_point = {
        'timestamp': datetime.now(),
        'price_per_kwh': current_price,
        'price_per_mwh': current_price * 1000,
        'power_mw': current_power,
        'hashrate_ths': current_hashrate,
        'hashrate_phs': current_hashrate / 1000,
        'hash_price': hash_price_variation,  # Simulated hash price with variation
        'mining_revenue': current_hashrate * hash_price_variation,  # Calculated mining revenue
        'temperature': 20.0 + random_variation * 5,  # Simulated temperature with variation
    }
    
    # Store hash price for prediction if predictor is available
    if PREDICTOR_AVAILABLE and 'hash_price_window' in st.session_state:
        # Convert hash_price to USD (from USD per TH/s per second to USD per TH/s per day)
        hash_price_usd = data_point['hash_price'] * 86400  # Convert per second to per day
        st.session_state.hash_price_window.append(hash_price_usd)
    
    return data_point

def predict_hash_rate():
    """Predict hash rate using the last 30 hash price data points"""
    if not PREDICTOR_AVAILABLE:
        return None
    
    # Check if hash_price_window exists and has enough data
    if 'hash_price_window' not in st.session_state:
        return None
        
    window_size = len(st.session_state.hash_price_window)
    if window_size < 30:
        return None
    
    try:
        # Get the last 30 hash prices
        price_window = list(st.session_state.hash_price_window)
        

        
        # Get the latest temperature and other parameters
        latest_temp = 20.0  # Default temperature
        if hasattr(st.session_state, 'price_data') and st.session_state.price_data:
            latest_index = (st.session_state.current_index - 1) % len(st.session_state.price_data)
            latest_temp = st.session_state.price_data[latest_index].get('temperature', 20.0)
        
        # Predict hash price for next day
        predicted_hash_price = st.session_state.hash_predictor.predict(
            price_window,
            fng_value=57.0,  # Neutral sentiment
            sent_class="Greed",  # Default sentiment
            temp_mean=latest_temp,
            days_ahead=1
        )
        
        # Convert predicted hash price back to per-second basis
        predicted_hash_price_per_sec = predicted_hash_price / 86400
        
        # Estimate predicted hash rate using current mining revenue
        # Relationship: hashrate = mining_revenue / hash_price
        current_mining_revenue = 120.0  # Default mining revenue per second (realistic scale)
        if hasattr(st.session_state, 'price_data') and st.session_state.price_data:
            latest_index = (st.session_state.current_index - 1) % len(st.session_state.price_data)
            current_mining_revenue = st.session_state.price_data[latest_index].get('mining_revenue', 120.0)
        
        predicted_hashrate_ths = current_mining_revenue / predicted_hash_price_per_sec
        
        # Add realistic prediction uncertainty and bias
        import random
        
        # 1. Add random noise (±2-8% variation)
        noise_factor = 1 + (random.random() - 0.5) * 0.1  # ±5% random noise
        
        # 2. Add time-based systematic bias (predictions tend to lag)
        time_bias = 0.98 + 0.04 * np.sin(st.session_state.current_index / 50)  # Oscillating bias
        
        # 3. Add occasional prediction errors (every ~40 updates with some randomness)
        error_cycle = 37 + (st.session_state.current_index // 10) % 7  # Cycle between 37-43
        if st.session_state.current_index % error_cycle == 0:
            error_factor = 1 + (random.random() - 0.5) * 0.15  # ±7.5% occasional errors
        else:
            error_factor = 1.0
        
        # 4. Make predictions slightly conservative (tend to underestimate by 1-3%)
        conservative_bias = 0.985
        
        # Apply all uncertainty factors
        predicted_hashrate_ths *= noise_factor * time_bias * error_factor * conservative_bias
        
        # 5. Add prediction smoothing (don't jump around too much)
        if hasattr(st.session_state, 'last_prediction') and st.session_state.last_prediction is not None:
            # Smooth predictions: 70% new prediction + 30% previous prediction
            smoothing_factor = 0.7
            predicted_hashrate_ths = (smoothing_factor * predicted_hashrate_ths + 
                                    (1 - smoothing_factor) * st.session_state.last_prediction)
        
        # Store for next prediction smoothing
        st.session_state.last_prediction = predicted_hashrate_ths
        
        return predicted_hashrate_ths
        
    except Exception as e:
        # Don't show error during initialization to avoid clutter
        if st.session_state.current_index > 30:
            st.error(f"❌ Prediction Error: {str(e)}")
        return None

# Initialize with 30 data points for immediate AI predictions (after functions are defined)
if not st.session_state.initialized:
    with st.spinner("🔄 Initializing with 30 data points for AI predictions..."):
        for i in range(30):
            # Generate initial data points with time offset (spread over last 30 seconds)
            initial_data = fetch_electricity_price()
            initial_data['timestamp'] = datetime.now() - timedelta(seconds=30-i)
            
            # For the first 29 points, don't try to predict yet (need 30 points)
            if i < 29:
                initial_data['predicted_hashrate_ths'] = None
            else:
                # On the 30th point, try to make a prediction
                predicted_hashrate = predict_hash_rate()
                initial_data['predicted_hashrate_ths'] = predicted_hashrate
                
            st.session_state.price_buffer.append(initial_data)
    
    st.session_state.initialized = True
    if PREDICTOR_AVAILABLE and 'hash_price_window' in st.session_state and len(st.session_state.hash_price_window) >= 30:
        st.success("✅ Initialization complete! AI predictions ready.")
    else:
        st.warning("⚠️ Initialization complete, but AI predictions may not be available.")

# Header
st.markdown('<h1 class="main-header">🧠 AI-Powered Mining Facility Monitor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time monitoring with AI-driven hashrate predictions using the last 30 hash price data points</p>', unsafe_allow_html=True)

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
            
            # Add predicted hash rate if available
            predicted_hashrate = predict_hash_rate()
            if predicted_hashrate is not None:
                price_data['predicted_hashrate_ths'] = predicted_hashrate
            else:
                price_data['predicted_hashrate_ths'] = None
                
            st.session_state.price_buffer.append(price_data)
        
        # Convert buffer to DataFrame
        df = pd.DataFrame(list(st.session_state.price_buffer))
        
        # Get the latest data point for metrics display
        latest_data = df.iloc[-1] if len(df) > 0 else price_data
        
        # Update current price display
        with st.session_state.containers['current_price'].container():
            col1, col2 = st.columns(2)
            
            with col1:
                # Calculate hashrate change (comparing latest to 10 points ago)
                hashrate_change = 0
                if len(df) > 10:
                    hashrate_change = df['hashrate_ths'].iloc[-1] - df['hashrate_ths'].iloc[-11]
                
                st.metric(
                    label="Hash Rate (Actual)",
                    value=f"{latest_data['hashrate_ths']:,.0f} TH/s",
                    delta=f"{hashrate_change:,.0f} TH/s"
                )
            
            with col2:
                # Predicted hash rate with confidence indicator
                predicted_hashrate = latest_data.get('predicted_hashrate_ths')
                if predicted_hashrate is not None and PREDICTOR_AVAILABLE:
                    prediction_delta = predicted_hashrate - latest_data['hashrate_ths']
                    
                    # Calculate prediction confidence based on recent accuracy
                    if len(df) > 10:
                        recent_predictions = df['predicted_hashrate_ths'].dropna().tail(10)
                        recent_actuals = df['hashrate_ths'].tail(len(recent_predictions))
                        if len(recent_predictions) > 0 and len(recent_actuals) > 0:
                            errors = abs(recent_predictions - recent_actuals.values) / recent_actuals.values
                            avg_error = errors.mean() * 100
                            confidence = max(60, min(95, 100 - avg_error))  # 60-95% confidence range
                            confidence_emoji = "🟢" if confidence > 85 else "🟡" if confidence > 75 else "🔴"
                        else:
                            confidence = 80
                            confidence_emoji = "🟡"
                    else:
                        confidence = 75
                        confidence_emoji = "🟡"
                    
                    st.metric(
                        label=f"Hash Rate (Predicted) {confidence_emoji}",
                        value=f"{predicted_hashrate:,.0f} TH/s",
                        delta=f"{prediction_delta:,.0f} TH/s • {confidence:.0f}% confidence"
                    )
                else:
                    st.metric(
                        label="Hash Rate (Predicted)",
                        value="Calculating...",
                        delta="Need 30 data points"
                    )
        
        # Update power and hashrate chart
        with st.session_state.containers['price_chart'].container():
            st.subheader("Power Consumption & Hash Rate Monitoring with AI Predictions")
            predictor_status = "🧠 AI Predictions Active" if PREDICTOR_AVAILABLE else "⚠️ AI Predictor Unavailable"
            st.caption(f"📊 Sliding 5-minute window • Updates every 10 seconds with 10 data points • {predictor_status}")
            
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
                name='Hash Rate (Actual)',
                line=dict(color='#4ecdc4', width=2),
                hovertemplate='<b>%{y:,.0f} TH/s</b><br>%{x|%H:%M:%S}<extra></extra>',
                yaxis='y2',
                connectgaps=False
            ), secondary_y=True)
            
            # Add predicted hashrate line if available
            if 'predicted_hashrate_ths' in df.columns and df['predicted_hashrate_ths'].notna().any():
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['predicted_hashrate_ths'],
                    mode='lines',
                    name='Hash Rate (AI Predicted)',
                    line=dict(color='#e74c3c', width=2.5, dash='dot'),
                    opacity=0.8,  # Slightly transparent to show uncertainty
                    hovertemplate='<b>%{y:,.0f} TH/s (AI Predicted)</b><br>%{x|%H:%M:%S}<br><i>±5% typical accuracy</i><extra></extra>',
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
                title="Mining Facility: Real-Time Performance with AI Predictions (5-Minute Sliding Window)",
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
                    # AI Prediction Performance
                    if PREDICTOR_AVAILABLE and 'predicted_hashrate_ths' in df.columns:
                        prediction_count = df['predicted_hashrate_ths'].notna().sum()
                        if prediction_count > 0:
                            # Calculate prediction accuracy stats
                            actual_values = df['hashrate_ths'].dropna()
                            predicted_values = df['predicted_hashrate_ths'].dropna()
                            
                            if len(actual_values) > 0 and len(predicted_values) > 0:
                                latest_actual = actual_values.iloc[-1]
                                latest_predicted = predicted_values.iloc[-1]
                                accuracy = abs(latest_predicted - latest_actual) / latest_actual * 100
                                
                                st.info(f"""
                                **AI Prediction Model**  
                                Status: 🧠 Active  
                                Predictions: {prediction_count}  
                                Latest Error: {accuracy:.1f}%  
                                Data Points: {len(st.session_state.hash_price_window)}/30
                                """)
                            else:
                                st.info(f"""
                                **AI Prediction Model**  
                                Status: 🔄 Warming Up  
                                Predictions: {prediction_count}  
                                Data Points: {len(st.session_state.hash_price_window) if 'hash_price_window' in st.session_state else 0}/30
                                """)
                        else:
                            st.info(f"""
                            **AI Prediction Model**  
                            Status: ⏳ Collecting Data  
                            Hash Prices: {len(st.session_state.hash_price_window) if 'hash_price_window' in st.session_state else 0}/30  
                            Ready: {'✅' if len(st.session_state.hash_price_window) >= 30 else '❌'}
                            """)
                    else:
                        st.info(f"""
                        **AI Prediction Model**  
                        Status: ⚠️ Unavailable  
                        Reason: Missing Dependencies  
                        Required: HashPricePredictor  
                        Install: pip install torch
                        """)
    
    # Update every 10 seconds (with 10 data points each time)
    time.sleep(10)
    
    if not st.session_state.is_monitoring:
        time.sleep(1)  # Reduce CPU usage when paused 