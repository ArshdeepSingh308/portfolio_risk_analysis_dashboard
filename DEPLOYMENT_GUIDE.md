# 🚀 Deployment Guide

Complete guide for deploying the Portfolio Risk Analysis Dashboard to various platforms.

## 🌐 Streamlit Community Cloud (Recommended)

### Prerequisites
- GitHub account
- Repository with your dashboard code

### Step-by-Step Deployment

#### 1. Prepare Repository
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

#### 2. Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file: `enhanced_dashboard.py`
6. Click "Deploy!"

#### 3. Configuration
Your app will be available at: `https://your-app-name.streamlit.app`

### Environment Variables (Optional)
If using external APIs, add secrets in Streamlit Cloud:
1. Go to your app settings
2. Click "Secrets"
3. Add your API keys:
```toml
[secrets]
api_key = "your-api-key"
database_url = "your-database-url"
```

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "enhanced_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run
```bash
# Build image
docker build -t portfolio-dashboard .

# Run container
docker run -p 8501:8501 portfolio-dashboard
```

### Docker Compose
```yaml
version: '3.8'
services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
    volumes:
      - ./data:/app/data
```

## ☁️ Cloud Platform Deployment

### Heroku
1. Create `Procfile`:
```
web: sh setup.sh && streamlit run enhanced_dashboard.py
```

2. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### AWS EC2
1. Launch EC2 instance (Ubuntu)
2. Install dependencies:
```bash
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements.txt
```

3. Run with PM2:
```bash
npm install -g pm2
pm2 start "streamlit run enhanced_dashboard.py" --name dashboard
```

### Google Cloud Platform
1. Create `app.yaml`:
```yaml
runtime: python39
service: default

basic_scaling:
  max_instances: 2
  idle_timeout: 10m

resources:
  cpu: 1
  memory_gb: 0.5
  disk_size_gb: 10
```

2. Deploy:
```bash
gcloud app deploy
```

## 🔧 Production Configuration

### Performance Optimization
```toml
# .streamlit/config.toml
[server]
maxUploadSize = 200
maxMessageSize = 200

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
```

### Caching Strategy
```python
# In your code
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    return expensive_data_operation()

@st.cache_resource
def init_model():
    return load_ml_model()
```

### Error Handling
```python
try:
    data = load_external_data()
except Exception as e:
    st.error(f"Data loading failed: {e}")
    data = load_fallback_data()
```

## 📊 Monitoring & Analytics

### Health Checks
```python
# Add to your app
def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

if st.sidebar.button("Health Check"):
    st.json(health_check())
```

### Usage Analytics
```python
# Track user interactions
def log_user_action(action, parameters):
    """Log user actions for analytics"""
    timestamp = datetime.now()
    # Log to your analytics service
```

## 🔒 Security Considerations

### Environment Variables
```python
import os
import streamlit as st

# Use secrets for sensitive data
api_key = st.secrets.get("api_key") or os.getenv("API_KEY")
```

### Input Validation
```python
def validate_input(value, min_val, max_val):
    """Validate user input"""
    if not isinstance(value, (int, float)):
        raise ValueError("Invalid input type")
    if not min_val <= value <= max_val:
        raise ValueError(f"Value must be between {min_val} and {max_val}")
    return value
```

## 🚀 CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python -m pytest tests/
    - name: Deploy to Streamlit
      # Streamlit Cloud auto-deploys on push to main
      run: echo "Deployed to Streamlit Cloud"
```

## 📱 Mobile Optimization

### Responsive Design
```python
# Check if mobile
def is_mobile():
    return st.session_state.get('mobile', False)

# Adjust layout for mobile
if is_mobile():
    st.markdown("Mobile-optimized layout")
    cols = st.columns(1)
else:
    cols = st.columns([2, 1])
```

## 🔄 Backup & Recovery

### Data Backup
```python
import json
from datetime import datetime

def backup_user_data():
    """Backup user configurations"""
    backup_data = {
        'timestamp': datetime.now().isoformat(),
        'portfolio_weights': st.session_state.get('weights', {}),
        'settings': st.session_state.get('settings', {})
    }
    return json.dumps(backup_data)
```

## 📈 Scaling Considerations

### Load Balancing
- Use multiple Streamlit instances behind a load balancer
- Implement session affinity if needed
- Consider using Redis for shared session state

### Database Integration
```python
# Example with SQLite
import sqlite3

@st.cache_resource
def init_database():
    conn = sqlite3.connect('portfolio_data.db')
    return conn

def save_portfolio(user_id, portfolio_data):
    conn = init_database()
    # Save to database
```

## 🛠️ Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce data size
   - Implement pagination
   - Use data sampling

2. **Slow Loading**
   - Add caching
   - Optimize data queries
   - Use progress bars

3. **Deployment Failures**
   - Check requirements.txt
   - Verify Python version
   - Review error logs

### Debug Mode
```python
# Enable debug mode
if st.sidebar.checkbox("Debug Mode"):
    st.write("Session State:", st.session_state)
    st.write("Cache Info:", st.cache_data.clear())
```

## 📞 Support

For deployment issues:
1. Check Streamlit Community Forum
2. Review platform-specific documentation
3. Check application logs
4. Test locally first

---

**🎉 Your dashboard is now ready for production deployment!**