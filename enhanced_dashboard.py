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
    page_title="Portfolio Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stApp { background-color: #f8f9fa; }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 4px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 6px;
        color: #64748b;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: white;
        color: #1e293b;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
    }
    
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'loader' not in st.session_state:
    st.session_state.loader = DataLoader()
if 'user_companies' not in st.session_state:
    st.session_state.user_companies = []
if 'selected_defaults' not in st.session_state:
    st.session_state.selected_defaults = ['ICICI_Large_Cap_NAV', 'Parag_Parikh_Flexi_NAV', 'HDFC_Mid_Cap_NAV']

@st.cache_data
def load_all_data():
    loader = DataLoader()
    scheme_data = loader.load_scheme_historical_data()
    benchmark_data = loader.load_benchmark_historical_data()
    risk_data = loader.load_risk_analysis_data()
    interest_data = loader.load_interest_rate_data()
    return scheme_data, benchmark_data, risk_data, interest_data

def create_risk_gauge(value, title, min_val=0, max_val=1, threshold_low=0.3, threshold_high=0.7):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': (threshold_low + threshold_high) / 2},
        gauge={
            'axis': {'range': [None, max_val]},
            'bar': {'color': "#3b82f6"},
            'steps': [
                {'range': [min_val, threshold_low], 'color': "#10b981"},
                {'range': [threshold_low, threshold_high], 'color': "#f59e0b"},
                {'range': [threshold_high, max_val], 'color': "#ef4444"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_high
            }
        }
    ))
    fig.update_layout(height=300, paper_bgcolor='white', plot_bgcolor='white')
    return fig

def monte_carlo_simulation(returns, weights, num_simulations=1000, time_horizon=252):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
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
        if len(paths) < 100:
            paths.append(cumulative_path)
    
    return np.array(results), np.array(paths)

# Header
st.markdown('<h1 class="main-title">📊 Portfolio Risk & Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Professional portfolio analysis with comprehensive risk metrics</p>', unsafe_allow_html=True)

# Load data
if not st.session_state.data_loaded:
    try:
        scheme_data, benchmark_data, risk_data, interest_data = load_all_data()
        st.session_state.scheme_data = scheme_data
        st.session_state.benchmark_data = benchmark_data
        st.session_state.risk_data = risk_data
        st.session_state.interest_data = interest_data
        st.session_state.data_loaded = True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Get data
scheme_data = st.session_state.scheme_data
benchmark_data = st.session_state.benchmark_data
risk_data = st.session_state.risk_data
interest_data = st.session_state.interest_data

# Initialize analyzer
analyzer = PortfolioAnalyzer(st.session_state.scheme_data, benchmark_data, risk_data, interest_data)
benchmark_returns = analyzer.calculate_returns(benchmark_data)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Portfolio Configuration")
    
    # Default Assets Selection
    st.markdown("### 🏦 Default Assets")
    default_assets = {
        'ICICI_Large_Cap_NAV': 'ICICI Large Cap Fund',
        'Parag_Parikh_Flexi_NAV': 'Parag Parikh Flexi Cap',
        'HDFC_Mid_Cap_NAV': 'HDFC Mid Cap Fund'
    }
    
    selected_defaults = []
    for key, name in default_assets.items():
        if st.checkbox(name, value=key in st.session_state.selected_defaults, key=f"default_{key}"):
            selected_defaults.append(key)
    
    st.session_state.selected_defaults = selected_defaults
    
    st.divider()
    
    # Add Custom Companies
    st.markdown("### ➕ Add Custom Company")
    with st.expander("Add New Company", expanded=False):
        company_name = st.text_input("Company Name", placeholder="e.g., Tesla Inc")
        data_source = st.selectbox("Data Source", ["Yahoo Finance", "MoneyControl", "Generic URL"])
        
        if data_source == "Yahoo Finance":
            company_url = st.text_input("Ticker Symbol", placeholder="e.g., TSLA")
            data_type = 'yahoo'
        elif data_source == "MoneyControl":
            company_url = st.text_input("MoneyControl URL", placeholder="https://www.moneycontrol.com/...")
            data_type = 'moneycontrol'
        else:
            company_url = st.text_input("Data URL", placeholder="https://...")
            data_type = 'generic'
        
        if st.button("Add Company", use_container_width=True):
            if company_name and company_url:
                st.session_state.loader.add_company_data(company_name, company_url, data_type)
                if company_name not in st.session_state.user_companies:
                    st.session_state.user_companies.append(company_name)
                st.success(f"✅ Added {company_name}")
                st.rerun()
            else:
                st.error("Please fill all fields")
    
    # Show added companies
    if st.session_state.user_companies:
        st.markdown("### 📈 Your Companies")
        for company in st.session_state.user_companies:
            col1, col2 = st.columns([3, 1])
            col1.write(f"• {company}")
            if col2.button("🗑️", key=f"del_{company}"):
                st.session_state.user_companies.remove(company)
                st.rerun()
    
    st.divider()
    
    # Investment Amount
    st.markdown("### 💰 Investment Amount")
    investment_amount = st.number_input(
        "Total Investment (₹)",
        min_value=1000,
        value=100000,
        step=1000,
        format="%d"
    )
    
    st.divider()
    
    # Risk Settings
    st.markdown("### ⚠️ Risk Settings")
    risk_tolerance = st.selectbox(
        "Risk Tolerance",
        ["Conservative", "Moderate", "Aggressive"],
        index=1
    )
    
    confidence_level = st.slider(
        "VaR Confidence Level",
        min_value=90,
        max_value=99,
        value=95,
        step=1
    ) / 100

# Prepare portfolio data
portfolio_data = {}

# Add selected default assets
for asset in st.session_state.selected_defaults:
    if asset in scheme_data.columns:
        portfolio_data[asset] = scheme_data[asset]

# Add user companies
if st.session_state.user_companies:
    user_data = st.session_state.loader.load_dynamic_portfolio_data(st.session_state.user_companies)
    if not user_data.empty:
        for company in user_data.columns:
            portfolio_data[company] = user_data[company]

# Create final portfolio dataframe
if portfolio_data:
    final_portfolio_df = pd.DataFrame(portfolio_data)
else:
    final_portfolio_df = scheme_data

# Update analyzer with final portfolio
analyzer = PortfolioAnalyzer(final_portfolio_df, benchmark_data, risk_data, interest_data)
scheme_returns = analyzer.calculate_returns(final_portfolio_df)
fund_names = list(final_portfolio_df.columns)

# Portfolio weights in sidebar
with st.sidebar:
    st.markdown("### ⚖️ Portfolio Weights")
    portfolio_weights = {}
    
    for i, fund in enumerate(fund_names):
        clean_name = fund.replace('_', ' ').replace('NAV', '').strip()
        portfolio_weights[fund] = st.slider(
            clean_name,
            0.0, 1.0, 1.0/len(fund_names),
            step=0.01,
            key=f"weight_{i}"
        )
    
    # Normalize weights
    total_weight = sum(portfolio_weights.values())
    if total_weight > 0:
        portfolio_weights = {k: v/total_weight for k, v in portfolio_weights.items()}
    
    weights_array = np.array(list(portfolio_weights.values()))
    
    # Show position values
    st.markdown("### 📊 Position Values")
    for fund, weight in portfolio_weights.items():
        clean_name = fund.replace('_', ' ').replace('NAV', '').strip()
        value = investment_amount * weight
        st.write(f"**{clean_name}**: ₹{value:,.0f}")

# Main content
portfolio_metrics = analyzer.calculate_portfolio_metrics(scheme_returns, weights_array)
portfolio_returns_series = (scheme_returns * weights_array).sum(axis=1)

# Key metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    expected_return_value = investment_amount * portfolio_metrics['Expected_Return']
    st.metric(
        "📈 Expected Return",
        f"{portfolio_metrics['Expected_Return']:.1%}",
        f"₹{expected_return_value:,.0f}"
    )

with col2:
    volatility_value = investment_amount * portfolio_metrics['Volatility']
    st.metric(
        "📊 Volatility",
        f"{portfolio_metrics['Volatility']:.1%}",
        f"₹{volatility_value:,.0f}"
    )

with col3:
    st.metric(
        "⚡ Sharpe Ratio",
        f"{portfolio_metrics['Sharpe_Ratio']:.2f}"
    )

with col4:
    max_drawdown_value = investment_amount * abs(portfolio_metrics['Max_Drawdown'])
    st.metric(
        "📉 Max Drawdown",
        f"{portfolio_metrics['Max_Drawdown']:.1%}",
        f"₹{max_drawdown_value:,.0f}"
    )

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Risk Analysis", "📈 Scenario Analysis", "⚡ Sensitivity Analysis", "📊 Portfolio Optimization"])

with tab1:
    st.markdown("### 🎯 Comprehensive Risk Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Performance chart with multiple benchmarks
        portfolio_cumulative = (1 + portfolio_returns_series).cumprod()
        
        fig = go.Figure()
        
        # Add portfolio line
        fig.add_trace(go.Scatter(
            x=portfolio_cumulative.index,
            y=portfolio_cumulative.values,
            mode='lines',
            name='Portfolio',
            line=dict(color='#3b82f6', width=3)
        ))
        
        # Add benchmark lines
        if not benchmark_data.empty:
            for i, col in enumerate(benchmark_data.columns):
                benchmark_cumulative = (1 + benchmark_returns[col]).cumprod()
                colors = ['#ef4444', '#10b981', '#f59e0b']
                fig.add_trace(go.Scatter(
                    x=benchmark_cumulative.index,
                    y=benchmark_cumulative.values,
                    mode='lines',
                    name=col.replace('_', ' '),
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
        
        fig.update_layout(
            title="📈 Portfolio vs Benchmark Performance",
            xaxis_title="Date",
            yaxis_title="Cumulative Returns",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk gauge
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
    st.markdown("### 📈 Advanced Scenario Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**🎛️ Scenario Parameters**")
        
        scenario_type = st.selectbox(
            "Select Scenario Type",
            ["Custom", "Market Crash", "Bull Market", "High Inflation", "Interest Rate Shock"]
        )
        
        if scenario_type == "Market Crash":
            market_shock, volatility_shock, interest_rate_change = -25, 100, 50
        elif scenario_type == "Bull Market":
            market_shock, volatility_shock, interest_rate_change = 25, -30, -50
        elif scenario_type == "High Inflation":
            market_shock, volatility_shock, interest_rate_change = -10, 50, 150
        elif scenario_type == "Interest Rate Shock":
            market_shock, volatility_shock, interest_rate_change = -5, 25, 200
        else:
            market_shock = st.slider("Market Return Shock (%)", -30, 30, 0, 1)
            volatility_shock = st.slider("Volatility Shock (%)", -50, 200, 0, 5)
            interest_rate_change = st.slider("Interest Rate Change (bps)", -200, 200, 0, 25)
        
        st.write(f"Market Shock: {market_shock}%")
        st.write(f"Volatility Shock: {volatility_shock}%")
        st.write(f"Interest Rate Change: {interest_rate_change} bps")
        
        correlation_change = st.slider("Correlation Change", -0.2, 0.2, 0.0, 0.05)
        time_horizon = st.selectbox("Analysis Period (Days)", [30, 90, 180, 252], index=3)
    
    with col2:
        st.markdown("**📊 Scenario Impact Analysis**")
        
        # Calculate base case
        base_metrics = analyzer.calculate_portfolio_metrics(scheme_returns, weights_array)
        
        # Apply shocks
        shocked_returns = scheme_returns.copy()
        shocked_returns = shocked_returns * (1 + market_shock/100)
        shocked_returns = shocked_returns * (1 + volatility_shock/100)
        
        # Apply interest rate and correlation effects
        if interest_rate_change != 0:
            ir_impact = interest_rate_change / 10000  # Convert bps to decimal impact
            shocked_returns = shocked_returns * (1 - ir_impact)
        
        if correlation_change != 0:
            corr_impact = correlation_change * 0.1  # Correlation impact on returns
            shocked_returns = shocked_returns * (1 + corr_impact)
        
        scenario_metrics = analyzer.calculate_portfolio_metrics(shocked_returns, weights_array)
        
        # Create comparison chart
        metrics_comparison = pd.DataFrame({
            'Base Case': [base_metrics['Expected_Return'], base_metrics['Volatility'], base_metrics['Sharpe_Ratio']],
            'Scenario Case': [scenario_metrics['Expected_Return'], scenario_metrics['Volatility'], scenario_metrics['Sharpe_Ratio']]
        }, index=['Expected Return', 'Volatility', 'Sharpe Ratio'])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Base Case',
            x=metrics_comparison.index,
            y=metrics_comparison['Base Case'],
            marker_color='#3b82f6'
        ))
        fig.add_trace(go.Bar(
            name='Scenario Case',
            x=metrics_comparison.index,
            y=metrics_comparison['Scenario Case'],
            marker_color='#ef4444'
        ))
        
        fig.update_layout(
            title="Base Case vs Scenario Analysis",
            xaxis_title="Metrics",
            yaxis_title="Values",
            barmode='group',
            height=400,
            plot_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Monte Carlo Simulation
    st.markdown("### 🎲 Monte Carlo Simulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        num_simulations = st.number_input("Number of Simulations", 100, 10000, 1000, 100)
        mc_time_horizon = st.number_input("Time Horizon (Days)", 30, 1000, 252, 30)
        
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
                go.Histogram(x=mc_results, nbinsx=50, name='Returns', marker_color='#3b82f6'),
                row=1, col=1
            )
            
            # Sample paths
            for i in range(min(20, len(mc_paths))):
                fig.add_trace(
                    go.Scatter(y=mc_paths[i], mode='lines', 
                              line=dict(width=1, color='rgba(59, 130, 246, 0.2)'), 
                              showlegend=False),
                    row=1, col=2
                )
            
            fig.update_layout(height=400, title_text="Monte Carlo Results", plot_bgcolor='white')
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
    st.markdown("### ⚡ Multi-Variable Sensitivity Analysis")
    
    # Sensitivity parameter controls
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
            
            # Calculate sensitivity matrix
            sensitivity_matrix = np.zeros((len(return_shocks), len(vol_shocks)))
            base_return = portfolio_metrics['Expected_Return']
            
            for i, ret_shock in enumerate(return_shocks):
                for j, vol_shock in enumerate(vol_shocks):
                    shocked_returns = scheme_returns * (1 + ret_shock/100) * (1 + vol_shock/100)
                    metrics = analyzer.calculate_portfolio_metrics(shocked_returns, weights_array)
                    sensitivity_matrix[i, j] = metrics['Expected_Return'] - base_return
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=sensitivity_matrix,
                x=[f"{v:.0f}%" for v in vol_shocks],
                y=[f"{r:.0f}%" for r in return_shocks],
                colorscale='RdYlBu_r',
                colorbar=dict(title="Return Impact")
            ))
            
            fig.update_layout(
                title="Portfolio Return Sensitivity Analysis",
                xaxis_title="Volatility Shock (%)",
                yaxis_title="Return Shock (%)",
                height=500,
                plot_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tornado chart
            st.markdown("### 🌪️ Factor Sensitivity (Tornado Chart)")
            
            factors = ['Market Return (+20%/-20%)', 'Volatility (+100%/-50%)', 
                      'Interest Rate (+200bps/0bps)', 'Correlation (+0.2/-0.2)']
            low_impact = []
            high_impact = []
            
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
            
            fig.add_trace(go.Bar(
                name='Downside Impact',
                y=factors,
                x=low_impact,
                orientation='h',
                marker_color='#ef4444',
                text=[f"{x:.3f}" for x in low_impact],
                textposition='inside'
            ))
            
            fig.add_trace(go.Bar(
                name='Upside Impact',
                y=factors,
                x=high_impact,
                orientation='h',
                marker_color='#3b82f6',
                text=[f"{x:.3f}" for x in high_impact],
                textposition='inside'
            ))
            
            fig.update_layout(
                title="🌪️ Factor Sensitivity Analysis (Tornado Chart)",
                xaxis_title="Impact on Expected Annual Return",
                yaxis_title="Risk Factors",
                height=500,
                barmode='overlay',
                xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### 📊 Portfolio Optimization")
    
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
            # Simple optimization using random sampling
            n_portfolios = 10000
            results = np.zeros((3, n_portfolios))
            
            np.random.seed(42)
            for i in range(n_portfolios):
                weights = np.random.random(len(fund_names))
                weights /= np.sum(weights)
                
                portfolio_return = np.sum(scheme_returns.mean() * weights) * 252
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(scheme_returns.cov() * 252, weights)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
                
                results[0, i] = portfolio_return
                results[1, i] = portfolio_vol
                results[2, i] = sharpe_ratio
            
            st.session_state.optimization_results = results
            
            # Find optimal portfolio
            if optimization_method == "Maximum Sharpe Ratio":
                optimal_idx = np.argmax(results[2])
            elif optimization_method == "Minimum Volatility":
                optimal_idx = np.argmin(results[1])
            else:
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
                height=500,
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #64748b; padding: 1rem;'>
        <p><strong>🚀 Advanced Portfolio Risk & Analysis Dashboard</strong></p>
        <p><small>⚠️ For educational purposes only. Not investment advice.</small></p>
    </div>
    """,
    unsafe_allow_html=True
)