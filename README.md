<div align="center">
  <table>
    <tr>
      <td align="center" style="padding-right: 20px;">
        <h1>AGNBoost</h1>
        <p><strong>A machine learning toolkit for astronomical data analysis using XGBoost</strong></p>
      </td>
      <td align="center">
        <img src="agnboost_logo.png" alt="AGNBoost Logo" width="120" height="120" style="border-radius: 50%; object-fit: cover;"/>
      </td>
    </tr>
  </table>
  
  [![Documentation](https://img.shields.io/badge/docs-github--pages-blue)](https://kurthamblin.github.io/agnboost/)
  [![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

AGNBoost is a specialized Python package designed for astronomers working with photometric and spectroscopic data. Built on the foundation of [XGBoostLSS](https://statmixedml.github.io/XGBoostLSS/), AGNBoost provides a streamlined workflow for classification and regression tasks in astronomy, with particular focus on Active Galactic Nuclei (AGN) identification and photometric redshift estimation. The package offers distributional modeling capabilities that go beyond simple point estimates, providing robust uncertainty quantification essential for astronomical analysis.

## ✨ Features

### 🔭 **Astronomy-Focused Data Management**
- **Smart Catalog System**: Load and manage data from FITS files, CSV, or pandas DataFrames
- **Flexible Band Configuration**: JSON-based photometric band setup with metadata support
- **Automatic Feature Engineering**: Built-in color calculations and magnitude transformations
- **Data Validation**: Ensures data quality and compatibility across datasets

### 🚀 **Advanced Machine Learning**
- **XGBoostLSS Integration**: Full distributional modeling with uncertainty quantification
- **Hyperparameter Optimization**: Intelligent tuning with custom parameter grids
- **Multi-Target Support**: Train models for multiple astronomical targets simultaneously
- **Robust Cross-Validation**: Stratified splitting for both classification and regression

### ⚡ **Research-Ready Tools**
- **Pre-trained Models**: Ready-to-use models for AGN classification and redshift estimation
- **Signal-to-Noise Filtering**: Built-in S/N cuts for photometric data quality control
- **Model Persistence**: Comprehensive saving/loading with full metadata tracking
- **Extensible Architecture**: Designed for customization and integration with existing workflows

## 🚀 Installation

### Prerequisites
- Python 3.8 or later
- Git

### Quick Install

```bash
# Clone the repository
git clone https://github.com/kurthamblin/agnboost.git
cd agnboost

# Install in development mode (recommended)
pip install -e .
```

### Using Virtual Environment (Recommended)

```bash
# Create and activate a conda environment
conda create -n agnboost python=3.11
conda activate agnboost

# Clone and install
git clone https://github.com/kurthamblin/agnboost.git
cd agnboost
pip install -e .
```

### Verify Installation

```python
import agnboost
from agnboost import Catalog, AGNBoost

print("AGNBoost installed successfully!")
```

## 📖 Documentation

Complete documentation is available on GitHub Pages:

**🌐 [https://kurthamblin.github.io/agnboost/](https://kurthamblin.github.io/agnboost/)**

The documentation includes:
- **Quick Start Guide**: Get up and running in minutes
- **Comprehensive Tutorials**: Step-by-step guides for common workflows
- **API Reference**: Complete documentation of all classes and methods
- **Examples**: Real-world use cases with JWST and other survey data


## 🤝 Contributing

We welcome contributions from the astronomy community! Please see our [Contributing Guide](https://kurthamblin.github.io/agnboost/contributing/) for details on how to get involved.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [XGBoostLSS](https://statmixedml.github.io/XGBoostLSS/) for distributional modeling
- Designed for compatibility with JWST, HST, and ground-based survey data
- Developed by astronomers, for astronomers

---

<div align="center">
  <strong>If you use AGNBoost in your research, please see our <a href="https://kurthamblin.github.io/agnboost/citation/">citation guide</a>.</strong>
</div>
```

