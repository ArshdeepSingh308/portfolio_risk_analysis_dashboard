import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Portfolio Risk & Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sample data for demonstration
@st.cache_data
def load_sample_data():
    """Load sample portfolio and benchmark data"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    
    # Sample portfolio data
    portfolio_data = {
        'ICICI_Large_Cap': np.random.normal(0.0008, 0.015, len(dates)),
        'Parag_Parikh_Flexi': np.random.normal(0.001, 0.018, len(dates)),
        'HDFC_Mid_Cap': np.random.normal(0.0012, 0.022, len(dates))
    }
    
    # Convert to cumulative returns
    for fund in portfolio_data:
        portfolio_data[fund] = (1 + pd.Series(portfolio_data[fund])).cumprod()
    
    portfolio_df = pd.DataFrame(portfolio_data, index=dates)
    
    # Benchmark data
    benchmark_data = {
        'NIFTY_50': np.random.normal(0.0007, 0.012, len(dates)),
        'NIFTY_500': np.random.normal(0.0008, 0.014, len(dates))
    }
    
    for benchmark in benchmark_data:
        benchmark_data[benchmark] = (1 + pd.Series(benchmark_data[benchmark])).cumprod()
    
    benchmark_df = pd.DataFrame(benchmark_data, index=dates)
    
    # Interest rates
    interest_rates = pd.DataFrame({
        'Rate': np.random.normal(6.5, 0.5, len(dates))
    }, index=dates)
    
    return portfolio_df, benchmark_df, interest_rates

def calculate_portfolio_metrics(returns, weights=None):
    """Calculate portfolio risk metrics"""
    if weights is None:
        weights = np.array([1/len(returns.columns)] * len(returns.columns))
    
    portfolio_returns = (returns * weights).sum(axis=1)
    
    metrics = {
        'Annual Return': portfolio_returns.mean() * 252,
        'Annual Volatility': portfolio_returns.std() * np.sqrt(252),
        'Sharpe Ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)),
        'Max Drawdown': (portfolio_returns.cumsum() - portfolio_returns.cumsum().expanding().max()).min(),
        'VaR (95%)': np.percentile(portfolio_returns, 5),
        'CVaR (95%)': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean()
    }
    
    return metrics, portfolio_returns

def monte_carlo_simulation(returns, weights, num_simulations=1000, time_horizon=252):
    """Perform Monte Carlo simulation"""
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    results = []
    for _ in range(num_simulations):
        random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, time_horizon)
        portfolio_returns = (random_returns * weights).sum(axis=1)
        cumulative_return = (1 + portfolio_returns).prod() - 1
        results.append(cumulative_return)
    
    return np.array(results)

# Main Dashboard
st.markdown('<h1 class="main-header">Portfolio Risk & Analysis Dashboard</h1>', unsafe_allow_html=True)

# Load data
portfolio_df, benchmark_df, interest_rates = load_sample_data()
portfolio_returns = portfolio_df.pct_change().dropna()
benchmark_returns = benchmark_df.pct_change().dropna()

# Sidebar for global parameters
st.sidebar.header("Portfolio Configuration")
portfolio_weights = {}
fund_names = list(portfolio_df.columns)

st.sidebar.subheader("Portfolio Weights")
for i, fund in enumerate(fund_names):
    portfolio_weights[fund] = st.sidebar.slider(
        f"{fund.replace('_', ' ')}", 
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

weights_array = np.array(list(portfolio_weights.values()))

# Create tabs for different analysis tools
tab1, tab2, tab3 = st.tabs(["🎯 Risk Analysis", "📈 Scenario Analysis", "⚡ Sensitivity Analysis"])

with tab1:
    st.header("Risk Analysis Tool")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Portfolio performance chart
        portfolio_value = (portfolio_returns * weights_array).sum(axis=1).cumsum()
        benchmark_value = benchmark_returns.mean(axis=1).cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_value.index,
            y=portfolio_value.values,
            mode='lines',
            name='Portfolio',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=benchmark_value.index,
            y=benchmark_value.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio vs Benchmark Performance",
            xaxis_title="Date",
            yaxis_title="Cumulative Returns",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk metrics
        metrics, port_returns = calculate_portfolio_metrics(portfolio_returns, weights_array)
        
        st.subheader("Risk Metrics")
        for metric, value in metrics.items():
            st.metric(metric, f"{value:.4f}")
    
    # Risk ratios section
    st.subheader("Risk Ratios Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Beta Analysis**")
        beta = np.cov(port_returns, benchmark_returns.mean(axis=1))[0,1] / np.var(benchmark_returns.mean(axis=1))
        st.metric("Portfolio Beta", f"{beta:.3f}")
        
    with col2:
        st.markdown("**Alpha Analysis**")
        alpha = metrics['Annual Return'] - beta * benchmark_returns.mean(axis=1).mean() * 252
        st.metric("Portfolio Alpha", f"{alpha:.4f}")
        
    with col3:
        st.markdown("**Information Ratio**")
        tracking_error = (port_returns - benchmark_returns.mean(axis=1)).std() * np.sqrt(252)
        info_ratio = alpha / tracking_error if tracking_error != 0 else 0
        st.metric("Information Ratio", f"{info_ratio:.3f}")

with tab2:
    st.header("Scenario Analysis Tool")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Scenario Parameters")
        
        # Market scenario inputs
        market_shock = st.slider("Market Return Shock (%)", -30, 30, 0, 1)
        volatility_shock = st.slider("Volatility Shock (%)", -50, 200, 0, 5)
        interest_rate_change = st.slider("Interest Rate Change (bps)", -200, 200, 0, 25)
        
        # Correlation scenario
        correlation_change = st.slider("Correlation Change", -0.2, 0.2, 0.0, 0.05)
        
        st.subheader("Time Horizon")
        time_horizon = st.selectbox("Analysis Period", [30, 90, 180, 252], index=3)
    
    with col2:
        st.subheader("What-If Analysis Results")
        
        # Calculate scenario impacts
        base_metrics, _ = calculate_portfolio_metrics(portfolio_returns, weights_array)
        
        # Apply shocks
        shocked_returns = portfolio_returns.copy()
        shocked_returns = shocked_returns * (1 + market_shock/100)
        shocked_returns = shocked_returns * (1 + volatility_shock/100)
        
        scenario_metrics, _ = calculate_portfolio_metrics(shocked_returns, weights_array)
        
        # Create comparison chart
        metrics_comparison = pd.DataFrame({
            'Base Case': list(base_metrics.values()),
            'Scenario Case': list(scenario_metrics.values())
        }, index=list(base_metrics.keys()))
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Base Case',
            x=metrics_comparison.index,
            y=metrics_comparison['Base Case'],
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            name='Scenario Case',
            x=metrics_comparison.index,
            y=metrics_comparison['Scenario Case'],
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title="Base Case vs Scenario Analysis",
            xaxis_title="Metrics",
            yaxis_title="Values",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Monte Carlo simulation
    st.subheader("Monte Carlo Simulation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_simulations = st.number_input("Number of Simulations", 100, 10000, 1000, 100)
        
        if st.button("Run Monte Carlo Simulation"):
            mc_results = monte_carlo_simulation(portfolio_returns, weights_array, num_simulations, time_horizon)
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=mc_results,
                nbinsx=50,
                name='Simulated Returns',
                marker_color='lightgreen'
            ))
            
            fig.update_layout(
                title=f"Monte Carlo Simulation Results ({num_simulations} simulations)",
                xaxis_title="Portfolio Returns",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'mc_results' in locals():
            st.subheader("Simulation Statistics")
            st.metric("Mean Return", f"{np.mean(mc_results):.4f}")
            st.metric("Standard Deviation", f"{np.std(mc_results):.4f}")
            st.metric("5th Percentile", f"{np.percentile(mc_results, 5):.4f}")
            st.metric("95th Percentile", f"{np.percentile(mc_results, 95):.4f}")

with tab3:
    st.header("Sensitivity Analysis Tool")
    
    st.subheader("Multi-Variable Sensitivity Analysis")
    
    # Sensitivity parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Return Shocks**")
        return_min = st.number_input("Min Return Shock (%)", -20, 0, -20)
        return_max = st.number_input("Max Return Shock (%)", 0, 20, 20)
        
    with col2:
        st.markdown("**Volatility Shocks**")
        vol_min = st.number_input("Min Volatility Shock (%)", -50, 0, -50)
        vol_max = st.number_input("Max Volatility Shock (%)", 0, 200, 200)
        
    with col3:
        st.markdown("**Interest Rate Shocks**")
        ir_min = st.number_input("Min IR Shock (bps)", 0, 100, 0)
        ir_max = st.number_input("Max IR Shock (bps)", 100, 200, 200)
    
    # Generate sensitivity analysis
    if st.button("Generate Sensitivity Analysis"):
        # Create parameter ranges
        return_shocks = np.linspace(return_min, return_max, 11)
        vol_shocks = np.linspace(vol_min, vol_max, 11)
        
        # Calculate sensitivity matrix
        sensitivity_matrix = np.zeros((len(return_shocks), len(vol_shocks)))
        
        for i, ret_shock in enumerate(return_shocks):
            for j, vol_shock in enumerate(vol_shocks):
                shocked_returns = portfolio_returns * (1 + ret_shock/100) * (1 + vol_shock/100)
                metrics, _ = calculate_portfolio_metrics(shocked_returns, weights_array)
                sensitivity_matrix[i, j] = metrics['Annual Return']
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=sensitivity_matrix,
            x=[f"{v:.0f}%" for v in vol_shocks],
            y=[f"{r:.0f}%" for r in return_shocks],
            colorscale='RdYlBu',
            colorbar=dict(title="Annual Return")
        ))
        
        fig.update_layout(
            title="Portfolio Return Sensitivity Analysis",
            xaxis_title="Volatility Shock (%)",
            yaxis_title="Return Shock (%)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tornado chart for individual factor sensitivity
        st.subheader("Factor Sensitivity (Tornado Chart)")
        
        base_return = calculate_portfolio_metrics(portfolio_returns, weights_array)[0]['Annual Return']
        
        factors = ['Market Return', 'Volatility', 'Interest Rate', 'Correlation']
        low_impact = []
        high_impact = []
        
        # Calculate individual factor impacts
        for factor in factors:
            if factor == 'Market Return':
                low_shocked = portfolio_returns * 0.8
                high_shocked = portfolio_returns * 1.2
            elif factor == 'Volatility':
                low_shocked = portfolio_returns * 0.5
                high_shocked = portfolio_returns * 2.0
            elif factor == 'Interest Rate':
                low_shocked = portfolio_returns * 1.0
                high_shocked = portfolio_returns * 0.98
            else:  # Correlation
                low_shocked = portfolio_returns * 0.95
                high_shocked = portfolio_returns * 1.05
            
            low_return = calculate_portfolio_metrics(low_shocked, weights_array)[0]['Annual Return']
            high_return = calculate_portfolio_metrics(high_shocked, weights_array)[0]['Annual Return']
            
            low_impact.append(low_return - base_return)
            high_impact.append(high_return - base_return)
        
        # Create tornado chart
        fig = go.Figure()
        
        for i, factor in enumerate(factors):
            fig.add_trace(go.Bar(
                name=f'{factor} Low',
                y=[factor],
                x=[low_impact[i]],
                orientation='h',
                marker_color='lightcoral',
                showlegend=False
            ))
            fig.add_trace(go.Bar(
                name=f'{factor} High',
                y=[factor],
                x=[high_impact[i]],
                orientation='h',
                marker_color='lightblue',
                showlegend=False
            ))
        
        fig.update_layout(
            title="Factor Sensitivity Analysis (Tornado Chart)",
            xaxis_title="Impact on Annual Return",
            yaxis_title="Risk Factors",
            height=400,
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**Dashboard Features:**
- **Risk Analysis**: Comprehensive portfolio risk metrics and ratios
- **Scenario Analysis**: What-if analysis with Monte Carlo simulations
- **Sensitivity Analysis**: Multi-variable sensitivity modeling with tornado charts

*Note: This dashboard uses sample data for demonstration. Connect to your actual data sources for live analysis.*
""")