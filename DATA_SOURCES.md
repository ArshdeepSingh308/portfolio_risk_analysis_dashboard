# 📊 Data Sources Guide

## 🎯 Current Data Strategy

The dashboard uses **sample data** for demonstration with fallback to real data sources when available.

## 📈 Data Flow

```
1. Try Yahoo Finance (Benchmarks) → Success ✅
2. Try MoneyControl (Mutual Funds) → Fallback to Sample Data 📊
3. Generate Interest Rates → Synthetic Data 🔢
4. Risk Parameters → Predefined Values ⚙️
```

## 🔄 Real Data Integration

### **Yahoo Finance (Working)**
```python
# Automatically fetches real benchmark data
^NSEI     # Nifty 50
^NSMIDCP  # Nifty MidCap
```

### **MoneyControl (Sample Mode)**
```python
# URLs configured but using sample data
ICICI Large Cap: MPI1134
Parag Parikh Flexi: MPP002  
HDFC Mid Cap: MHD1161
```

## 🛠️ Enable Real Data

### **Option 1: API Integration**
```python
# Add to secrets.toml
[secrets]
alpha_vantage_key = "your-api-key"
quandl_key = "your-api-key"
```

### **Option 2: Data Upload**
```python
# Upload CSV files
scheme_data.csv
benchmark_data.csv
interest_rates.csv
```

### **Option 3: Database Connection**
```python
# Connect to database
DATABASE_URL = "postgresql://user:pass@host:port/db"
```

## 📊 Sample Data Quality

**Realistic Parameters:**
- Based on historical Indian mutual fund performance
- Proper correlation structures
- Realistic volatility patterns
- Professional-grade risk metrics

**Data Period:** 2020-2024 (4 years daily data)
**Frequency:** Daily returns and NAV values
**Seed:** Fixed (42) for reproducible results

## 🔧 Customization

### **Add New Funds**
```python
fund_urls['New_Fund'] = 'https://moneycontrol.com/...'
```

### **Change Parameters**
```python
# In _generate_sample_scheme_data()
'New_Fund_NAV': 100 * (1 + np.random.normal(mean, std, len(dates))).cumprod()
```

### **Real-Time Updates**
```python
@st.cache_data(ttl=300)  # 5-minute cache
def load_live_data():
    return fetch_real_data()
```

## ⚠️ Important Notes

1. **Sample Data**: Currently used for all mutual funds
2. **Real Benchmarks**: Yahoo Finance data when available
3. **Fallback System**: Ensures dashboard always works
4. **Professional Quality**: Sample data mimics real market behavior

## 🚀 Production Deployment

For production use:
1. Integrate with financial data providers
2. Add authentication for premium APIs
3. Implement data validation and cleaning
4. Add real-time data refresh mechanisms