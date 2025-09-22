"""
Portfolio Risk & Analysis Dashboard Launcher
============================================

This script launches the interactive financial dashboard with comprehensive
risk analysis, scenario analysis, and sensitivity analysis tools.

Features:
- Risk Analysis Tool: Portfolio risk metrics, VaR, CVaR, Sharpe ratio, etc.
- Scenario Analysis Tool: What-if analysis with Monte Carlo simulations
- Sensitivity Analysis Tool: Multi-variable sensitivity modeling
- Portfolio Optimization: Efficient frontier analysis

Usage:
    python run_dashboard.py

Or run directly with Streamlit:
    streamlit run enhanced_dashboard.py
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    try:
        # Check if enhanced_dashboard.py exists
        if not os.path.exists("enhanced_dashboard.py"):
            print("❌ enhanced_dashboard.py not found!")
            print("Please ensure all files are in the same directory.")
            return False
        
        print("🚀 Launching Portfolio Risk & Analysis Dashboard...")
        print("📊 Dashboard will open in your default web browser")
        print("🔗 URL: http://localhost:8501")
        print("\n" + "="*50)
        print("DASHBOARD FEATURES:")
        print("="*50)
        print("🎯 Risk Analysis Tool:")
        print("   - Portfolio risk metrics and ratios")
        print("   - VaR, CVaR, Sharpe ratio calculations")
        print("   - Interactive risk gauges")
        print("   - Performance vs benchmark comparison")
        print()
        print("📈 Scenario Analysis Tool:")
        print("   - What-if scenario modeling")
        print("   - Predefined scenarios (Market Crash, Bull Market, etc.)")
        print("   - Monte Carlo simulations")
        print("   - Custom parameter adjustments")
        print()
        print("⚡ Sensitivity Analysis Tool:")
        print("   - Multi-variable sensitivity matrices")
        print("   - Tornado charts for factor analysis")
        print("   - Comprehensive shock testing")
        print("   - Return/Volatility/Interest Rate impacts")
        print()
        print("📊 Portfolio Optimization:")
        print("   - Efficient frontier analysis")
        print("   - Maximum Sharpe ratio optimization")
        print("   - Minimum volatility optimization")
        print("   - Target return optimization")
        print("="*50)
        print("\n💡 Tip: Use the sidebar to adjust portfolio weights and risk parameters")
        print("🔄 The dashboard updates interactively as you change inputs")
        print("\nPress Ctrl+C to stop the dashboard")
        print("="*50)
        
        # Launch Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "enhanced_dashboard.py"])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🚀 Portfolio Risk & Analysis Dashboard Launcher")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Install requirements
    print("\n📦 Installing requirements...")
    if not install_requirements():
        return
    
    # Launch dashboard
    print("\n🚀 Starting dashboard...")
    launch_dashboard()

if __name__ == "__main__":
    main()