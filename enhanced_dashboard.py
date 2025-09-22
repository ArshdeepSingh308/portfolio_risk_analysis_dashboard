import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Import our custom data loader
try:
    from data_loader import DataLoader, PortfolioAnalyzer
except ImportError:
    st.error("Please ensure data_loader.py is in the same directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Advanced Portfolio Risk & Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .risk-high { border-left: 5px solid #e74c3c; }
    .risk-medium { border-left: 5px solid #f39c12; }
    .risk-low { border-left: 5px solid #27ae60; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 10px 25px;
        background-color: #f8f9fa;
        border-radius: 10px 10px 0 0;
        border: 2px solid #dee2e6;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    
    .sensitivity-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.loader = DataLoader()
    st.session_state.user_companies = []
    st.session_state.company_urls = {}

@st.cache_data
def load_all_data():
    """Load all required data"""
    loader = DataLoader()
    
    with st.spinner("Loading portfolio data..."):
        scheme_data = loader.load_scheme_historical_data()
    
    with st.spinner("Loading benchmark data..."):
        benchmark_data = loader.load_benchmark_historical_data()
    
    with st.spinner("Loading risk parameters..."):
        risk_data = loader.load_risk_analysis_data()
    
    with st.spinner("Loading interest rate data..."):
        interest_data = loader.load_interest_rate_data()
    
    return scheme_data, benchmark_data, risk_data, interest_data

def create_risk_gauge(value, title, min_val=0, max_val=1, threshold_low=0.3, threshold_high=0.7):
    """Create a risk gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': (threshold_low + threshold_high) / 2},
        gauge = {
            'axis': {'range': [None, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, threshold_low], 'color': "lightgreen"},
                {'range': [threshold_low, threshold_high], 'color': "yellow"},
                {'range': [threshold_high, max_val], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_high
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def monte_carlo_simulation(returns, weights, num_simulations=1000, time_horizon=252):
    """Enhanced Monte Carlo simulation with correlation"""
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Ensure positive definite covariance matrix
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    eigenvals = np.maximum(eigenvals, 1e-8)
    cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    results = []
    paths = []
    
    for _ in range(num_simulations):
        random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, time_horizon)
        portfolio_returns = (random_returns * weights).sum(axis=1)
        cumulative_path = (1 + portfolio_returns).cumprod()
        final_return = cumulative_path[-1] - 1
        
        results.append(final_return)
        if len(paths) < 100:  # Store first 100 paths for visualization
            paths.append(cumulative_path)
    
    return np.array(results), np.array(paths)

# Main Dashboard
st.markdown('<h1 class="main-header">🚀 Advanced Portfolio Risk & Analysis Dashboard</h1>', unsafe_allow_html=True)

# Load data
if not st.session_state.data_loaded:
    try:
        scheme_data, benchmark_data, risk_data, interest_data = load_all_data()
        st.session_state.scheme_data = scheme_data
        st.session_state.benchmark_data = benchmark_data
        st.session_state.risk_data = risk_data
        st.session_state.interest_data = interest_data
        st.session_state.data_loaded = True
        st.success("✅ All data loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

# Get data from session state
scheme_data = st.session_state.scheme_data
benchmark_data = st.session_state.benchmark_data
risk_data = st.session_state.risk_data
interest_data = st.session_state.interest_data

# Initialize with default data first
analyzer = PortfolioAnalyzer(st.session_state.scheme_data, benchmark_data, risk_data, interest_data)
benchmark_returns = analyzer.calculate_returns(benchmark_data)

# Sidebar configuration
st.sidebar.markdown("## 🎛️ Portfolio Configuration")

# Company Data Input Section
st.sidebar.markdown("### 🏢 Add Companies")
with st.sidebar.expander("Add New Company", expanded=False):
    company_name = st.text_input("Company Name", placeholder="e.g., Apple Inc")
    data_source = st.selectbox("Data Source", ["Yahoo Finance", "MoneyControl", "Generic URL"])
    
    if data_source == "Yahoo Finance":
        company_url = st.text_input("Ticker Symbol", placeholder="e.g., AAPL")
        data_type = 'yahoo'
    elif data_source == "MoneyControl":
        company_url = st.text_input("MoneyControl URL", placeholder="https://www.moneycontrol.com/...")
        data_type = 'moneycontrol'
    else:
        company_url = st.text_input("Data URL", placeholder="https://...")
        data_type = 'generic'
    
    add_clicked = st.button("➕ Add Company", key="add_company_btn")
    
    if add_clicked:
        if not company_name:
            st.error("Please enter a company name")
        elif not company_url:
            st.error("Please enter a ticker/URL")
        else:
            try:
                st.session_state.loader.add_company_data(company_name, company_url, data_type)
                if company_name not in st.session_state.user_companies:
                    st.session_state.user_companies.append(company_name)
                    st.session_state.company_urls[company_name] = company_url
                st.success(f"Added {company_name}!")
                st.rerun()
            except Exception as e:
                st.error(f"Error adding company: {e}")

# Display added companies
if st.session_state.user_companies:
    st.sidebar.markdown("### 📈 Your Companies")
    for company in st.session_state.user_companies:
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(f"• {company}")
        if col2.button("🗑️", key=f"del_{company}"):
            st.session_state.user_companies.remove(company)
            del st.session_state.company_urls[company]
            st.rerun()

# Load data with user companies or default
if st.session_state.user_companies:
    scheme_data = st.session_state.loader.load_scheme_historical_data(st.session_state.user_companies)
else:
    scheme_data = st.session_state.scheme_data

# Always reinitialize analyzer with current data
analyzer = PortfolioAnalyzer(scheme_data, benchmark_data, risk_data, interest_data)
scheme_returns = analyzer.calculate_returns(scheme_data)

fund_names = list(scheme_data.columns)

# Portfolio weights
st.sidebar.markdown("### ⚖️ Asset Allocation")
portfolio_weights = {}

for i, fund in enumerate(fund_names):
    clean_name = fund.replace('_', ' ').replace('NAV', '').strip()
    portfolio_weights[fund] = st.sidebar.slider(
        clean_name, 
        min_value=0.0, 
        max_value=1.0, 
        value=1.0/len(fund_names),
        step=0.01,
        key=f"weight_{i}"
    )

# Normalize weights
total_weight = sum(portfolio_weights.values())
if total_weight > 0:
    portfolio_weights = {k: v/total_weight for k, v in portfolio_weights.items()}
else:
    portfolio_weights = {k: 1.0/len(fund_names) for k in fund_names}

weights_array = np.array(list(portfolio_weights.values()))

# Display current allocation
st.sidebar.markdown("### 📈 Current Allocation")
for fund, weight in portfolio_weights.items():
    clean_name = fund.replace('_', ' ').replace('NAV', '').strip()
    st.sidebar.write(f"**{clean_name}**: {weight:.1%}")

# Investment amount
st.sidebar.markdown("### 💰 Investment Amount")
investment_amount = st.sidebar.number_input(
    "Total Investment (₹)",
    min_value=1000,
    max_value=10000000,
    value=100000,
    step=1000,
    format="%d"
)

# Calculate position sizes
position_values = {}
for fund, weight in portfolio_weights.items():
    position_values[fund] = investment_amount * weight

# Display position values
st.sidebar.markdown("### 📊 Position Values")
for fund, value in position_values.items():
    clean_name = fund.replace('_', ' ').replace('NAV', '').strip()
    st.sidebar.write(f"**{clean_name}**: ₹{value:,.0f}")

# Risk tolerance
st.sidebar.markdown("### ⚠️ Risk Settings")
risk_tolerance = st.sidebar.selectbox(
    "Risk Tolerance",
    ["Conservative", "Moderate", "Aggressive"],
    index=1
)

confidence_level = st.sidebar.slider(
    "VaR Confidence Level",
    min_value=90,
    max_value=99,
    value=95,
    step=1
) / 100

# Create main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Risk Analysis", 
    "📈 Scenario Analysis", 
    "⚡ Sensitivity Analysis",
    "📊 Portfolio Optimization"
])

with tab1:
    st.markdown('<h2 class="sub-header">🎯 Comprehensive Risk Analysis</h2>', unsafe_allow_html=True)
    
    # Calculate portfolio metrics
    portfolio_metrics = analyzer.calculate_portfolio_metrics(scheme_returns, weights_array)
    portfolio_returns_series = (scheme_returns * weights_array).sum(axis=1)
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        expected_return_value = investment_amount * portfolio_metrics['Expected_Return']
        st.metric("📈 Expected Return", f"{portfolio_metrics['Expected_Return']:.2%}", f"₹{expected_return_value:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        volatility_value = investment_amount * portfolio_metrics['Volatility']
        st.metric("📊 Volatility", f"{portfolio_metrics['Volatility']:.2%}", f"₹{volatility_value:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("⚡ Sharpe Ratio", f"{portfolio_metrics['Sharpe_Ratio']:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        max_drawdown_value = investment_amount * abs(portfolio_metrics['Max_Drawdown'])
        st.metric("📉 Max Drawdown", f"{portfolio_metrics['Max_Drawdown']:.2%}", f"₹{max_drawdown_value:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance chart and risk gauges
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Portfolio vs benchmark performance
        portfolio_cumulative = (1 + portfolio_returns_series).cumprod()
        benchmark_cumulative = (1 + benchmark_returns.mean(axis=1)).cumprod()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_cumulative.index,
            y=portfolio_cumulative.values,
            mode='lines',
            name='Portfolio',
            line=dict(color='#1f77b4', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=benchmark_cumulative.index,
            y=benchmark_cumulative.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        fig.update_layout(
            title="📈 Portfolio vs Benchmark Performance",
            xaxis_title="Date",
            yaxis_title="Cumulative Returns",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk gauges
        volatility_gauge = create_risk_gauge(
            portfolio_metrics['Volatility'], 
            "Volatility Risk",
            max_val=0.5,
            threshold_low=0.15,
            threshold_high=0.25
        )
        st.plotly_chart(volatility_gauge, use_container_width=True)
    
    # Detailed risk metrics
    st.markdown("### 📊 Detailed Risk Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📈 Return Metrics**")
        expected_return_amount = investment_amount * portfolio_metrics['Expected_Return']
        alpha_amount = investment_amount * portfolio_metrics['Alpha']
        st.write(f"Expected Return: {portfolio_metrics['Expected_Return']:.2%} (₹{expected_return_amount:,.0f})")
        st.write(f"Alpha: {portfolio_metrics['Alpha']:.2%} (₹{alpha_amount:,.0f})")
        st.write(f"Beta: {portfolio_metrics['Beta']:.3f}")
        
    with col2:
        st.markdown("**⚠️ Risk Metrics**")
        volatility_amount = investment_amount * portfolio_metrics['Volatility']
        var_amount = investment_amount * abs(portfolio_metrics['VaR_95'])
        cvar_amount = investment_amount * abs(portfolio_metrics['CVaR_95'])
        st.write(f"Volatility: {portfolio_metrics['Volatility']:.2%} (₹{volatility_amount:,.0f})")
        st.write(f"VaR (95%): {portfolio_metrics['VaR_95']:.2%} (₹{var_amount:,.0f})")
        st.write(f"CVaR (95%): {portfolio_metrics['CVaR_95']:.2%} (₹{cvar_amount:,.0f})")
        
    with col3:
        st.markdown("**📊 Risk-Adjusted Metrics**")
        st.write(f"Sharpe Ratio: {portfolio_metrics['Sharpe_Ratio']:.3f}")
        st.write(f"Sortino Ratio: {portfolio_metrics['Sortino_Ratio']:.3f}")
        st.write(f"Information Ratio: {portfolio_metrics['Information_Ratio']:.3f}")

with tab2:
    st.markdown('<h2 class="sub-header">📈 Advanced Scenario Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Scenario Parameters")
        
        # Predefined scenarios
        scenario_type = st.selectbox(
            "Select Scenario Type",
            ["Custom", "Market Crash", "Bull Market", "High Inflation", "Interest Rate Shock"]
        )
        
        if scenario_type == "Market Crash":
            market_shock = -25
            volatility_shock = 100
            interest_rate_change = 50
        elif scenario_type == "Bull Market":
            market_shock = 20
            volatility_shock = -20
            interest_rate_change = -25
        elif scenario_type == "High Inflation":
            market_shock = -10
            volatility_shock = 50
            interest_rate_change = 150
        elif scenario_type == "Interest Rate Shock":
            market_shock = -5
            volatility_shock = 25
            interest_rate_change = 200
        else:  # Custom
            market_shock = st.slider("Market Return Shock (%)", -30, 30, 0, 1)
            volatility_shock = st.slider("Volatility Shock (%)", -50, 200, 0, 5)
            interest_rate_change = st.slider("Interest Rate Change (bps)", -200, 200, 0, 25)
        
        st.write(f"Market Shock: {market_shock}%")
        st.write(f"Volatility Shock: {volatility_shock}%")
        st.write(f"Interest Rate Change: {interest_rate_change} bps")
        
        correlation_change = st.slider("Correlation Change", -0.2, 0.2, 0.0, 0.05)
        time_horizon = st.selectbox("Analysis Period (Days)", [30, 90, 180, 252], index=3)
    
    with col2:
        st.markdown("### 📊 Scenario Impact Analysis")
        
        # Calculate base case
        base_metrics = analyzer.calculate_portfolio_metrics(scheme_returns, weights_array)
        
        # Apply shocks to create scenario
        shocked_returns = scheme_returns.copy()
        shocked_returns = shocked_returns * (1 + market_shock/100)
        shocked_returns = shocked_returns * (1 + volatility_shock/100)
        
        scenario_metrics = analyzer.calculate_portfolio_metrics(shocked_returns, weights_array)
        
        # Create comparison
        comparison_data = {
            'Metric': list(base_metrics.keys()),
            'Base Case': list(base_metrics.values()),
            'Scenario Case': list(scenario_metrics.values()),
            'Change': [scenario_metrics[k] - base_metrics[k] for k in base_metrics.keys()]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Return Comparison', 'Risk Comparison', 'Ratios Comparison', 'Change Impact'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Return metrics
        return_metrics = ['Expected_Return', 'Alpha']
        fig.add_trace(
            go.Bar(name='Base Case', x=return_metrics, 
                   y=[base_metrics[m] for m in return_metrics], marker_color='lightblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Scenario Case', x=return_metrics, 
                   y=[scenario_metrics[m] for m in return_metrics], marker_color='darkblue'),
            row=1, col=1
        )
        
        # Risk metrics
        risk_metrics = ['Volatility', 'Max_Drawdown']
        fig.add_trace(
            go.Bar(name='Base Case', x=risk_metrics, 
                   y=[abs(base_metrics[m]) for m in risk_metrics], marker_color='lightcoral', showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(name='Scenario Case', x=risk_metrics, 
                   y=[abs(scenario_metrics[m]) for m in risk_metrics], marker_color='darkred', showlegend=False),
            row=1, col=2
        )
        
        # Ratios
        ratio_metrics = ['Sharpe_Ratio', 'Sortino_Ratio']
        fig.add_trace(
            go.Bar(name='Base Case', x=ratio_metrics, 
                   y=[base_metrics[m] for m in ratio_metrics], marker_color='lightgreen', showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(name='Scenario Case', x=ratio_metrics, 
                   y=[scenario_metrics[m] for m in ratio_metrics], marker_color='darkgreen', showlegend=False),
            row=2, col=1
        )
        
        # Change impact
        change_values = [comparison_df.loc[comparison_df['Metric'] == m, 'Change'].iloc[0] 
                        for m in ['Expected_Return', 'Volatility', 'Sharpe_Ratio', 'Max_Drawdown']]
        colors = ['green' if x > 0 else 'red' for x in change_values]
        
        fig.add_trace(
            go.Bar(x=['Return', 'Volatility', 'Sharpe', 'Drawdown'], 
                   y=change_values, marker_color=colors, showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="Scenario Analysis Results")
        st.plotly_chart(fig, use_container_width=True)
    
    # Monte Carlo Simulation
    st.markdown("### 🎲 Monte Carlo Simulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        num_simulations = st.number_input("Number of Simulations", 100, 10000, 1000, 100)
        mc_time_horizon = st.number_input("Time Horizon (Days)", 30, 365, 252, 30)
        
        if st.button("🚀 Run Monte Carlo Simulation", type="primary"):
            with st.spinner("Running simulations..."):
                mc_results, mc_paths = monte_carlo_simulation(
                    scheme_returns, weights_array, num_simulations, mc_time_horizon
                )
                
                st.session_state.mc_results = mc_results
                st.session_state.mc_paths = mc_paths
    
    with col2:
        if 'mc_results' in st.session_state:
            mc_results = st.session_state.mc_results
            mc_paths = st.session_state.mc_paths
            
            # Results distribution
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Return Distribution', 'Sample Paths')
            )
            
            fig.add_trace(
                go.Histogram(x=mc_results, nbinsx=50, name='Returns', marker_color='lightblue'),
                row=1, col=1
            )
            
            # Sample paths
            for i in range(min(20, len(mc_paths))):
                fig.add_trace(
                    go.Scatter(y=mc_paths[i], mode='lines', 
                              line=dict(width=1, color='rgba(0,100,80,0.2)'), 
                              showlegend=False),
                    row=1, col=2
                )
            
            fig.update_layout(height=400, title_text="Monte Carlo Results")
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Return", f"{np.mean(mc_results):.2%}")
            with col2:
                st.metric("Std Deviation", f"{np.std(mc_results):.2%}")
            with col3:
                st.metric("5th Percentile", f"{np.percentile(mc_results, 5):.2%}")
            with col4:
                st.metric("95th Percentile", f"{np.percentile(mc_results, 95):.2%}")

with tab3:
    st.markdown('<h2 class="sub-header">⚡ Multi-Variable Sensitivity Analysis</h2>', unsafe_allow_html=True)
    
    # Sensitivity parameter controls
    st.markdown("### 🎛️ Sensitivity Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**📈 Return Shocks**")
        return_min = st.number_input("Min Return Shock (%)", -20, 0, -20)
        return_max = st.number_input("Max Return Shock (%)", 0, 20, 20)
        
    with col2:
        st.markdown("**📊 Volatility Shocks**")
        vol_min = st.number_input("Min Volatility Shock (%)", -50, 0, -50)
        vol_max = st.number_input("Max Volatility Shock (%)", 0, 200, 200)
        
    with col3:
        st.markdown("**💰 Interest Rate Shocks**")
        ir_min = st.number_input("Min IR Shock (bps)", 0, 100, 0)
        ir_max = st.number_input("Max IR Shock (bps)", 100, 200, 200)
        
    with col4:
        st.markdown("**🔗 Correlation Shocks**")
        corr_min = st.number_input("Min Correlation Change", -0.2, 0.0, -0.2, 0.05)
        corr_max = st.number_input("Max Correlation Change", 0.0, 0.2, 0.2, 0.05)
    
    if st.button("🔥 Generate Comprehensive Sensitivity Analysis", type="primary"):
        with st.spinner("Calculating sensitivity matrices..."):
            
            # Create parameter ranges
            return_shocks = np.linspace(return_min, return_max, 11)
            vol_shocks = np.linspace(vol_min, vol_max, 11)
            ir_shocks = np.linspace(ir_min, ir_max, 11)
            
            # Calculate 2D sensitivity matrices
            return_vol_matrix = np.zeros((len(return_shocks), len(vol_shocks)))
            return_ir_matrix = np.zeros((len(return_shocks), len(ir_shocks)))
            vol_ir_matrix = np.zeros((len(vol_shocks), len(ir_shocks)))
            
            base_metrics = analyzer.calculate_portfolio_metrics(scheme_returns, weights_array)
            base_return = base_metrics['Expected_Return']
            
            # Return vs Volatility sensitivity
            for i, ret_shock in enumerate(return_shocks):
                for j, vol_shock in enumerate(vol_shocks):
                    shocked_returns = scheme_returns * (1 + ret_shock/100) * (1 + vol_shock/100)
                    metrics = analyzer.calculate_portfolio_metrics(shocked_returns, weights_array)
                    return_vol_matrix[i, j] = metrics['Expected_Return'] - base_return
            
            # Return vs Interest Rate sensitivity
            for i, ret_shock in enumerate(return_shocks):
                for j, ir_shock in enumerate(ir_shocks):
                    shocked_returns = scheme_returns * (1 + ret_shock/100) * (1 - ir_shock/10000)
                    metrics = analyzer.calculate_portfolio_metrics(shocked_returns, weights_array)
                    return_ir_matrix[i, j] = metrics['Expected_Return'] - base_return
            
            # Volatility vs Interest Rate sensitivity
            for i, vol_shock in enumerate(vol_shocks):
                for j, ir_shock in enumerate(ir_shocks):
                    shocked_returns = scheme_returns * (1 + vol_shock/100) * (1 - ir_shock/10000)
                    metrics = analyzer.calculate_portfolio_metrics(shocked_returns, weights_array)
                    vol_ir_matrix[i, j] = metrics['Expected_Return'] - base_return
            
            # Create heatmaps
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Return vs Volatility Sensitivity',
                    'Return vs Interest Rate Sensitivity', 
                    'Volatility vs Interest Rate Sensitivity',
                    'Combined Sensitivity Summary'
                ),
                specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
                       [{"type": "heatmap"}, {"type": "bar"}]]
            )
            
            # Heatmap 1: Return vs Volatility
            fig.add_trace(
                go.Heatmap(
                    z=return_vol_matrix,
                    x=[f"{v:.0f}%" for v in vol_shocks],
                    y=[f"{r:.0f}%" for r in return_shocks],
                    colorscale='RdYlBu_r',
                    name="Return Impact"
                ),
                row=1, col=1
            )
            
            # Heatmap 2: Return vs Interest Rate
            fig.add_trace(
                go.Heatmap(
                    z=return_ir_matrix,
                    x=[f"{ir:.0f}bps" for ir in ir_shocks],
                    y=[f"{r:.0f}%" for r in return_shocks],
                    colorscale='RdYlBu_r',
                    showscale=False
                ),
                row=1, col=2
            )
            
            # Heatmap 3: Volatility vs Interest Rate
            fig.add_trace(
                go.Heatmap(
                    z=vol_ir_matrix,
                    x=[f"{ir:.0f}bps" for ir in ir_shocks],
                    y=[f"{v:.0f}%" for v in vol_shocks],
                    colorscale='RdYlBu_r',
                    showscale=False
                ),
                row=2, col=1
            )
            
            # Summary bar chart
            max_impacts = [
                np.max(np.abs(return_vol_matrix)),
                np.max(np.abs(return_ir_matrix)),
                np.max(np.abs(vol_ir_matrix))
            ]
            
            fig.add_trace(
                go.Bar(
                    x=['Return-Vol', 'Return-IR', 'Vol-IR'],
                    y=max_impacts,
                    marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1'],
                    name="Max Impact"
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                title_text="Comprehensive Sensitivity Analysis Dashboard"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tornado chart for factor sensitivity
            st.markdown("### 🌪️ Factor Sensitivity (Tornado Chart)")
            
            factors = ['Market Return (+20%/-20%)', 'Volatility (+100%/-50%)', 
                      'Interest Rate (+200bps/0bps)', 'Correlation (+0.2/-0.2)']
            low_impact = []
            high_impact = []
            
            # Calculate individual factor impacts
            shock_scenarios = [
                (0.8, 1.2),   # Market return
                (0.5, 2.0),   # Volatility  
                (1.0, 0.98),  # Interest rate (inverse)
                (0.95, 1.05)  # Correlation proxy
            ]
            
            for i, (low_mult, high_mult) in enumerate(shock_scenarios):
                low_shocked = scheme_returns * low_mult
                high_shocked = scheme_returns * high_mult
                
                low_metrics = analyzer.calculate_portfolio_metrics(low_shocked, weights_array)
                high_metrics = analyzer.calculate_portfolio_metrics(high_shocked, weights_array)
                
                low_impact.append(low_metrics['Expected_Return'] - base_return)
                high_impact.append(high_metrics['Expected_Return'] - base_return)
            
            # Create tornado chart
            fig = go.Figure()
            
            y_pos = list(range(len(factors)))
            
            # Add bars for low impact (left side)
            fig.add_trace(go.Bar(
                name='Downside Impact',
                y=factors,
                x=low_impact,
                orientation='h',
                marker_color='lightcoral',
                text=[f"{x:.3f}" for x in low_impact],
                textposition='inside'
            ))
            
            # Add bars for high impact (right side)  
            fig.add_trace(go.Bar(
                name='Upside Impact',
                y=factors,
                x=high_impact,
                orientation='h',
                marker_color='lightblue',
                text=[f"{x:.3f}" for x in high_impact],
                textposition='inside'
            ))
            
            fig.update_layout(
                title="🌪️ Factor Sensitivity Analysis (Tornado Chart)",
                xaxis_title="Impact on Expected Annual Return",
                yaxis_title="Risk Factors",
                height=500,
                barmode='overlay',
                xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
            )
            
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown('<h2 class="sub-header">📊 Portfolio Optimization</h2>', unsafe_allow_html=True)
    
    st.markdown("### 🎯 Efficient Frontier Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Optimization Parameters**")
        target_return = st.slider("Target Return (%)", 5, 25, 12, 1) / 100
        risk_free_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 6.0, 0.1) / 100
        
        optimization_method = st.selectbox(
            "Optimization Method",
            ["Maximum Sharpe Ratio", "Minimum Volatility", "Target Return"]
        )
        
        if st.button("🔍 Optimize Portfolio"):
            # Simple optimization using random sampling (for demonstration)
            n_portfolios = 10000
            results = np.zeros((3, n_portfolios))
            
            np.random.seed(42)
            for i in range(n_portfolios):
                # Random weights
                weights = np.random.random(len(fund_names))
                weights /= np.sum(weights)
                
                # Portfolio metrics
                portfolio_return = np.sum(scheme_returns.mean() * weights) * 252
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(scheme_returns.cov() * 252, weights)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
                
                results[0, i] = portfolio_return
                results[1, i] = portfolio_vol
                results[2, i] = sharpe_ratio
            
            st.session_state.optimization_results = results
            
            # Find optimal portfolio based on method
            if optimization_method == "Maximum Sharpe Ratio":
                optimal_idx = np.argmax(results[2])
            elif optimization_method == "Minimum Volatility":
                optimal_idx = np.argmin(results[1])
            else:  # Target Return
                target_diff = np.abs(results[0] - target_return)
                optimal_idx = np.argmin(target_diff)
            
            optimal_return = results[0, optimal_idx]
            optimal_vol = results[1, optimal_idx]
            optimal_sharpe = results[2, optimal_idx]
            
            st.success("✅ Optimization Complete!")
            st.write(f"**Optimal Return**: {optimal_return:.2%}")
            st.write(f"**Optimal Volatility**: {optimal_vol:.2%}")
            st.write(f"**Optimal Sharpe Ratio**: {optimal_sharpe:.3f}")
    
    with col2:
        if 'optimization_results' in st.session_state:
            results = st.session_state.optimization_results
            
            # Efficient frontier plot
            fig = go.Figure()
            
            # Scatter plot of all portfolios
            fig.add_trace(go.Scatter(
                x=results[1],
                y=results[0],
                mode='markers',
                marker=dict(
                    size=3,
                    color=results[2],
                    colorscale='Viridis',
                    colorbar=dict(title="Sharpe Ratio"),
                    opacity=0.6
                ),
                name='Random Portfolios'
            ))
            
            # Current portfolio
            current_return = np.sum(scheme_returns.mean() * weights_array) * 252
            current_vol = np.sqrt(np.dot(weights_array.T, np.dot(scheme_returns.cov() * 252, weights_array)))
            
            fig.add_trace(go.Scatter(
                x=[current_vol],
                y=[current_return],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='Current Portfolio'
            ))
            
            # Optimal portfolio
            if optimization_method == "Maximum Sharpe Ratio":
                optimal_idx = np.argmax(results[2])
            elif optimization_method == "Minimum Volatility":
                optimal_idx = np.argmin(results[1])
            else:
                target_diff = np.abs(results[0] - target_return)
                optimal_idx = np.argmin(target_diff)
            
            fig.add_trace(go.Scatter(
                x=[results[1, optimal_idx]],
                y=[results[0, optimal_idx]],
                mode='markers',
                marker=dict(size=15, color='gold', symbol='diamond'),
                name='Optimal Portfolio'
            ))
            
            fig.update_layout(
                title="📊 Efficient Frontier Analysis",
                xaxis_title="Volatility (Risk)",
                yaxis_title="Expected Return",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h3>🚀 Advanced Portfolio Risk & Analysis Dashboard</h3>
    <p><strong>Features:</strong></p>
    <ul style='list-style: none; padding: 0;'>
        <li>🎯 <strong>Risk Analysis</strong>: Comprehensive portfolio risk metrics with interactive gauges</li>
        <li>📈 <strong>Scenario Analysis</strong>: Advanced what-if analysis with Monte Carlo simulations</li>
        <li>⚡ <strong>Sensitivity Analysis</strong>: Multi-variable sensitivity modeling with tornado charts</li>
        <li>📊 <strong>Portfolio Optimization</strong>: Efficient frontier analysis and optimization</li>
    </ul>
    <p><em>Built with Streamlit, Plotly, and advanced financial modeling techniques</em></p>
    <p><strong>Note:</strong> This dashboard uses sample data for demonstration. Connect to your actual data sources for live analysis.</p>
</div>
""", unsafe_allow_html=True)