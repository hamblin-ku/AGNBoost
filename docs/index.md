# AGNBoost

**A powerful machine learning toolkit for astronomical data analysis using advanced XGBoost techniques.**

---

## Overview

AGNBoost is a specialized Python package designed for astronomers working with photometric and spectroscopic data. Built on the foundation of [XGBoostLSS](https://statmixedml.github.io/XGBoostLSS/), AGNBoost provides a streamlined workflow for classification and regression tasks in astronomy, with particular focus on Active Galactic Nuclei (AGN) identification and galaxy property estimation.

Whether you're analyzing JWST observations, conducting large-scale surveys, or developing custom models for your research, AGNBoost offers the tools and flexibility you need to extract meaningful insights from your astronomical data.

!!! tip "Quick Start"
    New to AGNBoost? Check out our [Quick Start Guide](getting-started.md) to get up and running in minutes!

---

## Core Features

### 🔭 **Astronomy-Focused Data Management**
- **Smart Catalog System**: Load and manage astronomical data from FITS files, CSV, or pandas DataFrames
- **Band Configuration**: Flexible JSON-based configuration for photometric bands with metadata (wavelengths, shorthand names)
- **Automatic Feature Engineering**: Built-in color calculations, magnitude transformations, and signal-to-noise filtering
- **Data Validation**: Ensures data quality and compatibility across different datasets

### 🚀 **Advanced Machine Learning Pipeline**
- **Hyperparameter Optimization**: Intelligent tuning with custom parameter grids and early stopping
- **Cross-Validation**: Robust model validation with stratified splitting for both classification and regression
- **Multi-Target Support**: Train models for multiple astronomical targets simultaneously
- **Model Persistence**: Comprehensive saving and loading with full metadata tracking

### ⚡ **XGBoostLSS Integration**
- **Distributional Modeling**: Go beyond point estimates with full probability distributions
- **Custom Objectives**: Specialized loss functions for astronomical applications
- **Efficient Training**: Optimized for large astronomical datasets with GPU acceleration support
- **Uncertainty Quantification**: Robust uncertainty estimates for scientific analysis

### 🛠 **Research-Ready Tools**
- **Flexible Data Splitting**: Intelligent train/validation/test splits with optional stratification
- **Signal-to-Noise Filtering**: Built-in S/N cuts for photometric data quality control
- **Transform Pipeline**: Easy-to-use data transformation and augmentation tools
- **Extensible Architecture**: Designed for customization and integration with existing workflows

---

## Installation

We recommend using a virtual environment to manage dependencies. AGNBoost works best with Python 3.8 or later.

### Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n agnboost python=3.11
conda activate agnboost

# Clone the repository
git clone https://github.com/yourusername/agnboost.git
cd agnboost

# Install in development mode
pip install -e .
```

### Using pip and venv

```bash
# Create a virtual environment
python -m venv agnboost-env
source agnboost-env/bin/activate  # On Windows: agnboost-env\Scripts\activate

# Clone and install
git clone https://github.com/yourusername/agnboost.git
cd agnboost
pip install -e .
```

### Verify Installation

```python
import agnboost
from agnboost import Catalog, AGNBoost

print(f"AGNBoost version: {agnboost.__version__}")
```

!!! note "Development Installation"
    The `-e` flag installs AGNBoost in "editable" mode, which means changes to the source code will be immediately available without reinstalling. This is particularly useful if you plan to contribute to the project or customize it for your research.

---

## Built on XGBoostLSS

AGNBoost leverages the power of [XGBoostLSS](https://statmixedml.github.io/XGBoostLSS/), a cutting-edge extension of XGBoost that enables **distributional modeling**. Instead of predicting single point estimates, XGBoostLSS can model entire probability distributions, providing:

- **Full Distributional Predictions**: Estimate not just the mean, but the entire shape of the target distribution
- **Robust Uncertainty Quantification**: Get principled uncertainty estimates for your astronomical measurements  
- **Flexible Loss Functions**: Choose from a wide variety of probability distributions tailored to your data
- **Advanced Regularization**: Built-in techniques to prevent overfitting and improve generalization

This makes AGNBoost particularly powerful for astronomical applications where uncertainty quantification is crucial, such as:

- Photometric redshift estimation with confidence intervals
- AGN probability classification with uncertainty bounds
- Stellar mass estimation with full posterior distributions
- Flux measurements with proper error propagation

---


## What's Next?

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    New to AGNBoost? Start with our comprehensive tutorial covering installation, basic usage, and your first analysis.

    [:octicons-arrow-right-24: Quick Start](getting-started.md)

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    Step-by-step guides covering everything from basic usage to advanced training pipelines.

    [:octicons-arrow-right-24: View Tutorials](tutorials/basic-usage.md)

-   :material-code-braces:{ .lg .middle } **API Reference**

    ---

    Complete documentation of all classes, methods, and functions in AGNBoost.

    [:octicons-arrow-right-24: API Docs](api/catalog.md)

-   :material-telescope:{ .lg .middle } **Examples**

    ---

    Real-world examples using JWST data, custom features, and advanced modeling techniques.

    [:octicons-arrow-right-24: See Examples](examples/jwst-analysis.md)

</div>

---

## Community and Support

AGNBoost is developed by astronomers, for astronomers. We welcome contributions, feedback, and collaboration from the community.

- **GitHub Repository**: [github.com/yourusername/agnboost](https://github.com/yourusername/agnboost)
- **Issues and Discussions**: Use GitHub Issues for bug reports and feature requests
- **Contributing**: See our [Contributing Guide](contributing.md) to get involved

---

*AGNBoost is open-source software released under the MIT License. If you use AGNBoost in your research, please see our [Citation Guide](citation.md).*

