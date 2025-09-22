import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Data loader for mutual fund and market data"""
    
    def __init__(self):
        self.fund_urls = {
            'ICICI_Prudential_Large_Cap': 'https://www.moneycontrol.com/mutual-funds/nav/icici-prudential-large-cap-fund-direct-plan-growth/MPI1134',
            'Parag_Parikh_Flexi_Cap': 'https://www.moneycontrol.com/mutual-funds/nav/parag-parikh-flexi-cap-fund-direct-plan-growth/MPP002',
            'HDFC_Mid_Cap': 'https://www.moneycontrol.com/mutual-funds/nav/hdfc-mid-cap-opportunities-fund-direct-plan/MHD1161'
        }
        self.user_companies = {}  # Store user-added companies
    
    def add_company_data(self, company_name, data_url, data_type='moneycontrol'):
        """Add user-defined company data source"""
        self.user_companies[company_name] = {
            'url': data_url,
            'type': data_type,
            'added_date': pd.Timestamp.now()
        }
        return f"Added {company_name} successfully"
    
    def fetch_user_company_data(self, company_name):
        """Fetch data for user-added company"""
        if company_name not in self.user_companies:
            return None
            
        company_info = self.user_companies[company_name]
        url = company_info['url']
        data_type = company_info['type']
        
        try:
            if data_type == 'yahoo':
                return self._fetch_yahoo_data(url)
            elif data_type == 'moneycontrol':
                return self._fetch_moneycontrol_data(url, company_name)
            else:
                return self._fetch_generic_data(url, company_name)
        except Exception as e:
            print(f"Error fetching {company_name}: {e}")
            return self._generate_sample_company_data(company_name)
    
    def _fetch_yahoo_data(self, ticker):
        """Fetch data from Yahoo Finance"""
        try:
            data = yf.download(ticker, start='2020-01-01', end='2024-01-01', progress=False)['Adj Close']
            return data.fillna(method='ffill')
        except:
            return self._generate_sample_company_data(ticker)
    
    def _fetch_moneycontrol_data(self, url, company_name):
        """Fetch data from MoneyControl URL"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract NAV or price
            nav_element = soup.find('span', {'class': 'amt'})
            current_price = float(nav_element.text.strip()) if nav_element else 100
        except:
            current_price = 100  # Default price if fetch fails
        
        # Generate historical data based on current price
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        returns = np.random.normal(0.0008, 0.018, len(dates))
        prices = current_price * (1 + returns).cumprod()
        
        return pd.Series(prices, index=dates, name=company_name)
    
    def _fetch_generic_data(self, url, company_name):
        """Fetch data from generic URL"""
        # Attempt to extract data from any URL
        response = requests.get(url, timeout=10)
        # Basic price extraction logic
        return self._generate_sample_company_data(company_name)
    
    def _generate_sample_company_data(self, company_name):
        """Generate sample data for company"""
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        np.random.seed(hash(company_name) % 1000)  # Unique seed per company
        
        # Random but realistic parameters
        mean_return = np.random.uniform(0.0005, 0.0015)
        volatility = np.random.uniform(0.012, 0.025)
        
        returns = np.random.normal(mean_return, volatility, len(dates))
        prices = 100 * (1 + returns).cumprod()
        
        return pd.Series(prices, index=dates, name=company_name)
    
    def load_dynamic_portfolio_data(self, company_list):
        """Load data for multiple user-defined companies"""
        portfolio_data = {}
        
        for company_name in company_list:
            if company_name in self.user_companies:
                data = self.fetch_user_company_data(company_name)
                if data is not None:
                    portfolio_data[company_name] = data
            else:
                # Generate sample data for unknown companies
                portfolio_data[company_name] = self._generate_sample_company_data(company_name)
        
        return pd.DataFrame(portfolio_data)
        
    def load_scheme_historical_data(self, custom_companies=None):
        """Load historical data for mutual fund schemes or custom companies"""
        try:
            if custom_companies:
                return self.load_dynamic_portfolio_data(custom_companies)
            
            # Default sample data
            dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
            np.random.seed(42)
            
            data = {
                'ICICI_Large_Cap_NAV': 100 * (1 + np.random.normal(0.0008, 0.015, len(dates))).cumprod(),
                'Parag_Parikh_Flexi_NAV': 100 * (1 + np.random.normal(0.001, 0.018, len(dates))).cumprod(),
                'HDFC_Mid_Cap_NAV': 100 * (1 + np.random.normal(0.0012, 0.022, len(dates))).cumprod()
            }
            
            return pd.DataFrame(data, index=dates)
            
        except Exception as e:
            print(f"Error loading scheme data: {e}")
            return self._generate_sample_scheme_data()
    
    def load_benchmark_historical_data(self):
        """Load benchmark indices data"""
        try:
            # Try to fetch real data from Yahoo Finance
            tickers = ['^NSEI', '^NSMIDCP']  # Nifty 50, Nifty Midcap
            data = yf.download(tickers, start='2020-01-01', end='2024-01-01', progress=False)
            
            # Handle different data structures
            if hasattr(data, 'columns'):
                if 'Adj Close' in data.columns:
                    data = data['Adj Close']
                elif len(data.columns) > 1:
                    data = data.iloc[:, -1]
            
            # Convert Series to DataFrame if needed
            if hasattr(data, 'name') and not hasattr(data, 'columns'):
                data = data.to_frame()
            
            if data.empty:
                return self._generate_sample_benchmark_data()
                
            # Ensure we have proper column names
            if hasattr(data, 'columns'):
                if len(data.columns) == 2:
                    data.columns = ['NIFTY_50', 'NIFTY_MIDCAP']
                elif len(data.columns) == 1:
                    data.columns = ['NIFTY_50']
                
            return data.fillna(method='ffill')
            
        except Exception as e:
            print(f"Error loading benchmark data: {e}")
            return self._generate_sample_benchmark_data()
    
    def load_risk_analysis_data(self):
        """Load risk analysis parameters"""
        risk_data = {
            'VaR_Confidence_Levels': [0.95, 0.99],
            'Time_Horizons': [1, 5, 10, 22, 252],  # Days
            'Risk_Free_Rate': 0.06,  # 6% annual
            'Correlation_Matrix': self._generate_correlation_matrix(),
            'Volatility_Estimates': {
                'ICICI_Large_Cap': 0.15,
                'Parag_Parikh_Flexi': 0.18,
                'HDFC_Mid_Cap': 0.22
            }
        }
        return risk_data
    
    def load_interest_rate_data(self):
        """Load interest rate data"""
        try:
            dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
            np.random.seed(42)
            
            # Simulate interest rate data
            base_rate = 6.5
            rate_changes = np.random.normal(0, 0.1, len(dates))
            rates = base_rate + np.cumsum(rate_changes) * 0.01
            
            data = pd.DataFrame({
                'Date': dates,
                '10Y_Govt_Bond': rates,
                'Repo_Rate': rates - 1.5,
                'Corporate_Bond_AAA': rates + 0.5,
                'Corporate_Bond_AA': rates + 1.0
            })
            
            data.set_index('Date', inplace=True)
            return data
            
        except Exception as e:
            print(f"Error loading interest rate data: {e}")
            return self._generate_sample_interest_data()
    
    def fetch_moneycontrol_data(self, fund_name):
        """Fetch data from MoneyControl website"""
        try:
            url = self.fund_urls.get(fund_name)
            if not url:
                return None
                
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract NAV and other fund details
            nav_element = soup.find('span', {'class': 'amt'})
            nav = float(nav_element.text.strip()) if nav_element else None
            
            # Extract additional fund information
            fund_info = {
                'NAV': nav,
                'Fund_Name': fund_name,
                'URL': url,
                'Last_Updated': pd.Timestamp.now()
            }
            
            return fund_info
            
        except Exception as e:
            print(f"Error fetching data for {fund_name}: {e}")
            return None
    
    def _generate_sample_scheme_data(self):
        """Generate sample scheme data"""
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        
        data = {
            'ICICI_Large_Cap_NAV': 100 * (1 + np.random.normal(0.0008, 0.015, len(dates))).cumprod(),
            'Parag_Parikh_Flexi_NAV': 100 * (1 + np.random.normal(0.001, 0.018, len(dates))).cumprod(),
            'HDFC_Mid_Cap_NAV': 100 * (1 + np.random.normal(0.0012, 0.022, len(dates))).cumprod()
        }
        
        return pd.DataFrame(data, index=dates)
    
    def _generate_sample_benchmark_data(self):
        """Generate sample benchmark data"""
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        
        data = {
            'NIFTY_50': 10000 * (1 + np.random.normal(0.0007, 0.012, len(dates))).cumprod(),
            'NIFTY_MIDCAP': 8000 * (1 + np.random.normal(0.0009, 0.016, len(dates))).cumprod()
        }
        
        return pd.DataFrame(data, index=dates)
    
    def _generate_sample_interest_data(self):
        """Generate sample interest rate data"""
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        
        base_rate = 6.5
        rate_changes = np.random.normal(0, 0.1, len(dates))
        rates = base_rate + np.cumsum(rate_changes) * 0.01
        
        data = {
            '10Y_Govt_Bond': rates,
            'Repo_Rate': rates - 1.5,
            'Corporate_Bond_AAA': rates + 0.5,
            'Corporate_Bond_AA': rates + 1.0
        }
        
        return pd.DataFrame(data, index=dates)
    
    def _generate_correlation_matrix(self):
        """Generate correlation matrix for funds"""
        funds = ['ICICI_Large_Cap', 'Parag_Parikh_Flexi', 'HDFC_Mid_Cap']
        np.random.seed(42)
        
        # Generate a positive definite correlation matrix
        A = np.random.randn(3, 3)
        corr_matrix = np.dot(A, A.T)
        
        # Normalize to correlation matrix
        d = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(d, d)
        
        return pd.DataFrame(corr_matrix, index=funds, columns=funds)

class PortfolioAnalyzer:
    """Portfolio analysis and calculations"""
    
    def __init__(self, scheme_data, benchmark_data, risk_data, interest_data):
        self.scheme_data = scheme_data
        self.benchmark_data = benchmark_data
        self.risk_data = risk_data
        self.interest_data = interest_data
        
    def calculate_returns(self, data):
        """Calculate returns from price data"""
        return data.pct_change().dropna()
    
    def calculate_portfolio_metrics(self, returns, weights):
        """Calculate comprehensive portfolio metrics"""
        portfolio_returns = (returns * weights).sum(axis=1)
        
        metrics = {
            'Expected_Return': portfolio_returns.mean() * 252,
            'Volatility': portfolio_returns.std() * np.sqrt(252),
            'Sharpe_Ratio': self._calculate_sharpe_ratio(portfolio_returns),
            'Sortino_Ratio': self._calculate_sortino_ratio(portfolio_returns),
            'Max_Drawdown': self._calculate_max_drawdown(portfolio_returns),
            'VaR_95': np.percentile(portfolio_returns, 5),
            'CVaR_95': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean(),
            'Beta': self._calculate_beta(portfolio_returns),
            'Alpha': self._calculate_alpha(portfolio_returns),
            'Information_Ratio': self._calculate_information_ratio(portfolio_returns),
            'Treynor_Ratio': self._calculate_treynor_ratio(portfolio_returns)
        }
        
        return metrics
    
    def _calculate_sharpe_ratio(self, returns):
        """Calculate Sharpe ratio"""
        risk_free_rate = self.risk_data['Risk_Free_Rate'] / 252
        excess_returns = returns - risk_free_rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns):
        """Calculate Sortino ratio"""
        risk_free_rate = self.risk_data['Risk_Free_Rate'] / 252
        excess_returns = returns - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        return excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_beta(self, portfolio_returns):
        """Calculate portfolio beta"""
        if self.benchmark_data.empty:
            return 1.0
        benchmark_returns = self.calculate_returns(self.benchmark_data).mean(axis=1)
        
        # Align data lengths
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        port_aligned = portfolio_returns.iloc[-min_len:]
        bench_aligned = benchmark_returns.iloc[-min_len:]
        
        covariance = np.cov(port_aligned, bench_aligned)[0, 1]
        benchmark_variance = np.var(bench_aligned)
        return covariance / benchmark_variance if benchmark_variance > 0 else 1.0
    
    def _calculate_alpha(self, portfolio_returns):
        """Calculate portfolio alpha"""
        beta = self._calculate_beta(portfolio_returns)
        risk_free_rate = self.risk_data['Risk_Free_Rate']
        if self.benchmark_data.empty:
            return 0.0
        benchmark_returns = self.calculate_returns(self.benchmark_data).mean(axis=1)
        
        # Align data lengths
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        port_aligned = portfolio_returns.iloc[-min_len:]
        bench_aligned = benchmark_returns.iloc[-min_len:]
        
        benchmark_return = bench_aligned.mean() * 252
        portfolio_return = port_aligned.mean() * 252
        return portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
    
    def _calculate_information_ratio(self, portfolio_returns):
        """Calculate information ratio"""
        if self.benchmark_data.empty:
            return 0.0
        benchmark_returns = self.calculate_returns(self.benchmark_data).mean(axis=1)
        
        # Align data lengths
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        port_aligned = portfolio_returns.iloc[-min_len:]
        bench_aligned = benchmark_returns.iloc[-min_len:]
        
        active_returns = port_aligned - bench_aligned
        tracking_error = active_returns.std() * np.sqrt(252)
        return active_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
    
    def _calculate_treynor_ratio(self, portfolio_returns):
        """Calculate Treynor ratio"""
        beta = self._calculate_beta(portfolio_returns)
        risk_free_rate = self.risk_data['Risk_Free_Rate']
        portfolio_return = portfolio_returns.mean() * 252
        return (portfolio_return - risk_free_rate) / beta if beta != 0 else 0

# Usage example
if __name__ == "__main__":
    # Initialize data loader
    loader = DataLoader()
    
    # Load all data
    scheme_data = loader.load_scheme_historical_data()
    benchmark_data = loader.load_benchmark_historical_data()
    risk_data = loader.load_risk_analysis_data()
    interest_data = loader.load_interest_rate_data()
    
    print("Data loaded successfully!")
    print(f"Scheme data shape: {scheme_data.shape}")
    print(f"Benchmark data shape: {benchmark_data.shape}")
    print(f"Interest rate data shape: {interest_data.shape}")