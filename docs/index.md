# AGNBoost

**A powerful machine learning toolkit for astronomical data analysis using advanced XGBoost techniques.**

---

## Overview

AGNBoost is a specialized Python framework designed for astronomers working with photometric data. Built on the foundation of [XGBoostLSS](https://statmixedml.github.io/XGBoostLSS/), AGNBoost provides a streamlined workflow for disitrubtional regression with photometric data, with particular focus on Active Galactic Nuclei (AGN) identification and galaxy property estimation.

!!! tip "Quick Start"
    New to AGNBoost? Check out our [Quick Start Guide](getting-started.md) to get up and running in minutes!

---

## Core Features

### 🔭 **Astronomy-Focused Data Management**
- **Smart Catalog System**: Load and manage astronomical data from FITS files, CSV, or pandas DataFrames
- **Band Configuration**: Flexible JSON-based configuration for photometric bands with metadata (wavelengths, shorthand names)
- **Automatic Feature Engineering**: Built-in color calculations, magnitude transformations, and signal-to-noise filtering
- **Data Validation**: Ensures data quality and compatibility across different datasets

### 🚀 **Convenient Pipeline**
- **Hyperparameter Optimization**: Intelligent tuning with custom parameter grids and early stopping
- **Cross-Validation**: Robust model validation with stratified splitting for both classification and regression
- **Model Persistence**: Comprehensive saving and loading with full metadata tracking

### ⚡ **XGBoostLSS Integration**
- **Distributional Modeling**: Go beyond point estimates with full probability distributions
- **Custom Objectives**: Specialized loss functions for astronomical applications
- **Efficient Training**: Optimized for large astronomical datasets with GPU acceleration support
- **Uncertainty Quantification**: Robust uncertainty estimates for Astronomical analysis

### 🛠 **Research-Ready Tools**
- **Flexible Data Splitting**:  Train/validation/test splits with optional stratification
- **Signal-to-Noise Filtering**: Built-in S/N cuts for photometric data quality control
- **Transform Pipeline**: Easy-to-use data transformation and augmentation tools
- **Extensible Architecture**: Designed for customization and integration with existing workflows

---

## Installation

We recommend using a virtual environment to manage dependencies. AGNBoost works best with Python 3.10/3.11 or later.

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

AGNBoost leverages the power of [XGBoostLSS](https://statmixedml.github.io/XGBoostLSS/), a cutting-edge extension of XGBoost that enables **distributional modeling**. Instead of predicting single point estimates, XGBoostLSS models the entire conditioan distributions, providing:

- **Full Distributional Predictions**: Estimate not just the mean, but the entire shape of the target distribution
- **Robust Uncertainty Quantification**: Get principled uncertainty estimates for your astronomical measurements  
- **Flexible Distributions**: Choose from a wide variety of probability distributions tailored to your data
- **Advanced Regularization**: Built-in techniques to prevent overfitting and improve generalization

This makes AGNBoost particularly powerful for astronomical applications where uncertainty quantification is crucial, such as:

- Photometric redshift estimation with confidence intervals
- AGN identification with uncertainty bounds
- Stellar mass estimation with full posterior distributions

---


## What's Next?

<div class="grid cards" markdown>

-   **Getting Started**

    ---

    New to AGNBoost? Start with our comprehensive tutorial covering installation, basic usage, and your first analysis.

    [Quick Start →](getting-started.md)

-   **Tutorials**

    ---

    Step-by-step guides covering everything from basic usage to advanced training pipelines.

    [View Tutorials →](tutorials/)

-   **API Reference**

    ---

    Complete documentation of all classes, methods, and functions in AGNBoost.

    [API Docs →](api.md)

-   **Examples**

    ---

    Real-world examples using JWST data, custom features, and advanced modeling techniques.

    [See Examples →](examples/jwst-analysis.md)

</div>

---

## Community and Support

AGNBoost is developed by astronomers, for astronomers. We welcome contributions, feedback, and collaboration from the community.

- **GitHub Repository**: [github.com/yourusername/agnboost](https://github.com/yourusername/agnboost)
- **Issues and Discussions**: Use GitHub Issues for bug reports and feature requests
- **Contributing**: See our [Contributing Guide](contributing.md) to get involved

---

*AGNBoost is open-source software released under the MIT License. If you use AGNBoost in your research, please see our [Citation Guide](citation.md).*

