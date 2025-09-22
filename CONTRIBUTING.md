# Contributing to Portfolio Risk Analysis Dashboard

Thank you for your interest in contributing to this project! 🎉

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic knowledge of financial concepts and Python

### Development Setup
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/portfolio-risk-analysis-dashboard.git
   cd portfolio-risk-analysis-dashboard
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the dashboard:
   ```bash
   streamlit run enhanced_dashboard.py
   ```

## 🔧 Development Guidelines

### Code Style
- Follow PEP 8 Python style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular

### File Structure
- `enhanced_dashboard.py`: Main Streamlit application
- `data_loader.py`: Data fetching and processing logic
- `financial_dashboard.py`: Basic dashboard version
- `run_dashboard.py`: Launcher script

### Adding New Features

#### 1. New Risk Metrics
Add new metrics to `PortfolioAnalyzer` class in `data_loader.py`:
```python
def _calculate_new_metric(self, returns):
    """Calculate new risk metric"""
    # Implementation here
    return metric_value
```

#### 2. New Data Sources
Extend `DataLoader` class:
```python
def load_new_data_source(self):
    """Load data from new source"""
    # Implementation here
    return data
```

#### 3. New Visualizations
Add to appropriate tab in `enhanced_dashboard.py`:
```python
# Create new chart
fig = go.Figure()
# Add traces and layout
st.plotly_chart(fig, use_container_width=True)
```

## 🧪 Testing

### Manual Testing
1. Test all interactive elements (sliders, buttons, dropdowns)
2. Verify calculations with known datasets
3. Check responsiveness on different screen sizes
4. Test error handling with invalid inputs

### Adding Tests
Create test files following the pattern:
```python
# test_data_loader.py
import unittest
from data_loader import DataLoader, PortfolioAnalyzer

class TestDataLoader(unittest.TestCase):
    def test_load_sample_data(self):
        # Test implementation
        pass
```

## 📝 Documentation

### Code Documentation
- Add docstrings to all functions and classes
- Include parameter descriptions and return types
- Provide usage examples for complex functions

### README Updates
When adding features, update:
- Feature list in README.md
- Usage examples
- Screenshots if UI changes

## 🐛 Bug Reports

### Before Submitting
1. Check existing issues
2. Test with latest version
3. Reproduce the bug consistently

### Bug Report Template
```markdown
**Bug Description**
Clear description of the bug

**Steps to Reproduce**
1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior**
What should happen

**Screenshots**
If applicable

**Environment**
- OS: [e.g., Windows 10]
- Python version: [e.g., 3.9.0]
- Browser: [e.g., Chrome 91.0]
```

## 💡 Feature Requests

### Feature Request Template
```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other approaches you've considered
```

## 🔄 Pull Request Process

### Before Submitting
1. Create a feature branch: `git checkout -b feature/amazing-feature`
2. Make your changes
3. Test thoroughly
4. Update documentation
5. Commit with clear messages

### PR Template
```markdown
**Description**
Brief description of changes

**Type of Change**
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

**Testing**
- [ ] Manual testing completed
- [ ] All existing features work
- [ ] New feature works as expected

**Screenshots**
If applicable
```

## 🎯 Priority Areas

### High Priority
- Performance optimization
- Additional risk metrics
- Real-time data integration
- Mobile responsiveness

### Medium Priority
- New asset classes (bonds, commodities)
- Advanced optimization algorithms
- Export functionality
- User preferences

### Low Priority
- UI/UX improvements
- Additional chart types
- Internationalization
- Advanced customization

## 📞 Getting Help

- **Questions**: Open a GitHub issue with the "question" label
- **Discussions**: Use GitHub Discussions for general topics
- **Documentation**: Check README.md and code comments

## 🏆 Recognition

Contributors will be:
- Listed in the README.md
- Mentioned in release notes
- Given credit in code comments

## 📄 Code of Conduct

### Our Standards
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

### Unacceptable Behavior
- Harassment or discrimination
- Trolling or insulting comments
- Publishing private information
- Unprofessional conduct

Thank you for contributing! 🙏