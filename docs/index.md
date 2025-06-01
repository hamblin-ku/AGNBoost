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

### Using Virtual Conda Environment (Recommended)

```bash
# Create and activate a conda environment
conda create -n agnboost python=3.11
conda activate agnboost

pip install git+https://github.com/hamblin-ku/AGNBoost.git
```

Otherwise, you can jsut install with pip (python 3.10 or later):

### Quick Install
To directly install the latest development version with pip, please use:
```bash
pip install git+https://github.com/hamblin-ku/AGNBoost.git
```

### Verify Installation

```python
import agnboost
from agnboost import Catalog, AGNBoost

print(f"AGNBoost version: {agnboost.__version__}")
```

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

-   **Tutorials**

    ---

    Step-by-step guides covering everything from basic usage to advanced training pipelines.

    [View Tutorials →](tutorials/)

-   **API Reference**

    ---

    Complete documentation of all classes, methods, and functions in AGNBoost.

    [API Docs →](api.md)

</div>

---

## Community and Support

We welcome contributions, feedback, and collaboration from the community!

- **GitHub Repository**: [https://github.com/hamblin-ku/AGNBoost](https://github.com/hamblin-ku/AGNBoost)
- **Issues and Discussions**: Use GitHub Issues for bug reports and feature requests
- **Contributing**: See our [Contributing Guide](contributing.md) to get involved

---

*AGNBoost is open-source software released under the MIT License. If you use AGNBoost in your research, please see our [Citation Guide](citation.md).*

