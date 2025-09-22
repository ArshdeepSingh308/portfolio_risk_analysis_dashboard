# 🚀 Portfolio Risk & Analysis Dashboard

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive financial dashboard for comprehensive portfolio risk analysis, scenario modeling, and sensitivity analysis using advanced financial theories including Markowitz optimization and Monte Carlo simulations.

## 🌟 Live Demo

**[🔗 Try the Live Dashboard](https://your-app-name.streamlit.app)**

## 📊 Key Features

### 🎯 Risk Analysis Tool
- **Portfolio Risk Metrics**: Expected return, volatility, Sharpe ratio, Sortino ratio
- **Value at Risk (VaR)**: 95% and 99% confidence levels  
- **Risk-Adjusted Ratios**: Alpha, Beta, Information ratio, Treynor ratio
- **Interactive Risk Gauges**: Visual risk assessment with color-coded indicators

### 📈 Scenario Analysis Tool
- **Predefined Scenarios**: Market crash, bull market, high inflation, interest rate shock
- **Monte Carlo Simulations**: Up to 10,000 simulations with path visualization
- **What-If Analysis**: Impact assessment of various market conditions
- **Markowitz Theory Implementation**: Modern portfolio theory applications

### ⚡ Sensitivity Analysis Tool
- **Multi-Variable Analysis**: Return, volatility, interest rate, and correlation shocks
- **Sensitivity Matrices**: Interactive heatmaps showing parameter interactions
- **Tornado Charts**: Factor sensitivity visualization
- **Comprehensive Stress Testing**: Extreme condition analysis

### 📊 Portfolio Optimization
- **Efficient Frontier Analysis**: Risk-return optimization
- **Maximum Sharpe Ratio**: Optimal risk-adjusted returns
- **Minimum Volatility**: Risk minimization strategies

## 🏦 Supported Assets

Pre-configured for Indian mutual funds:
- ICICI Prudential Large Cap Fund - Direct Plan - Growth
- Parag Parikh Flexi Cap Fund - Direct Plan - Growth  
- HDFC Mid Cap Opportunities Fund - Direct Plan - Growth

## 🚀 Quick Start

### Option 1: One-Click Launch
```bash
python run_dashboard.py
```

### Option 2: Manual Setup
```bash
pip install -r requirements.txt
streamlit run enhanced_dashboard.py
```

### Option 3: Try Online
**[🔗 Live Dashboard](https://your-app-name.streamlit.app)** - No installation required!

## 📁 Project Structure

```
portfolio_risk_analysis_dashboard/
├── enhanced_dashboard.py      # Main Streamlit application
├── data_loader.py            # Data fetching and processing
├── financial_dashboard.py    # Basic dashboard version
├── run_dashboard.py          # Easy launcher script
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── LICENSE                  # MIT License
└── README.md               # This file
```

## 🔧 Technical Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **Financial Models**: SciPy, Custom implementations
- **Data Sources**: Yahoo Finance, MoneyControl

## 📈 Usage Examples

### Risk Analysis
1. Adjust portfolio weights using sidebar sliders
2. View comprehensive risk metrics and gauges
3. Compare portfolio vs benchmark performance

### Scenario Analysis  
1. Select predefined scenarios or create custom ones
2. Run Monte Carlo simulations (up to 10,000 iterations)
3. Analyze probability distributions and outcomes

### Sensitivity Analysis
1. Set shock parameters for comprehensive testing
2. Generate interactive sensitivity heatmaps
3. View tornado charts for factor analysis

## 🔄 Deployment

### Deploy to Streamlit Cloud
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository and `enhanced_dashboard.py`
5. Click "Deploy!"

### Local Development
```bash
git clone https://github.com/yourusername/portfolio-risk-analysis-dashboard.git
cd portfolio-risk-analysis-dashboard
pip install -r requirements.txt
streamlit run enhanced_dashboard.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This dashboard is for educational and analysis purposes only. Past performance does not guarantee future results. All investments carry risk of loss. Consult with financial advisors before making investment decisions.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Financial data from [Yahoo Finance](https://finance.yahoo.com/) and [MoneyControl](https://www.moneycontrol.com/)
- Visualization powered by [Plotly](https://plotly.com/)

---

**⭐ Star this repository if you found it helpful!**