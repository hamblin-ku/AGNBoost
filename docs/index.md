# AGNBoost

**A powerful machine learning toolkit for astronomical data analysis using advanced XGBoost techniques.**

---

## Overview

AGNBoost is a specialized Python framework designed for astronomers working with photometric data. Built with [XGBoostLSS](https://statmixedml.github.io/XGBoostLSS/) as the foundation, AGNBoost provides a streamlined workflow for distributional regression with photometric data, with particular focus on Active Galactic Nuclei (AGN) identification and galaxy property estimation. Ultimately, AGNBoost is well-suited to any Astronimical regression task due to its robust uncertainty estimation and ease-of-use. 

!!! tip "Quick Start"
    New to AGNBoost? Check out our [Tutorials](tutorials/basic-usage/) to get started!!

---

## Core Features

### **Focus on Astronomical Datasets**
- **Smart Catalog System**: Load and manage astronomical data from FITS files, CSV, or pandas DataFrames
- **Automatic Feature Engineering**: Built-in color calculations, magnitude transformations, and signal-to-noise filtering. 
- **Data Validation**: Ensures data quality and compatibility across different datasets
- **Pre-trained Models**: Comes packaged with two pre-trained models, for frac$_\text{AGN}$ estimation and photometric redshift estimation from NIRCam+MIRI photometry. 

### **Convenient and Simple Pipeline**
- **Model Creation and Tuning**: Simple to create models for any target variable in data, and tune them with Optuna.
- **Model Persistence**: Automatic model and hyperparameter saving 


### **[XGBoostLSS Integration](https://github.com/StatMixedML/XGBoostLSS/tree/master)**
- **Distributional Modeling**: Go beyond point estimates with full probability distributions
- **Efficient Training**: Optimized for large astronomical datasets with GPU acceleration support
- **Uncertainty Quantification**: Provides robust uncertainty estimates, including both aleatoric uncertainty (i.e., that due to randomness in data) and epistemic uncertainty (i.e., that due to a lack of model knowledge).
---

## Installation

We recommend using a virtual environment to manage dependencies. AGNBoost works best with Python 3.10/3.11 or later.

### Using Virtual Conda Environment (Recommended)

```bash
# Create and activate a conda environment
conda create -n agnboost python=3.11
conda activate agnboost

git clone https://github.com/hamblin-ku/AGNBoost.git
cd AGNBoost
pip install -e . 
```

Otherwise, you can just clone the github repository and install the dependencies with pip:

### Quick Install
To directly install the latest development version with pip, please use:
```bash
git clone https://github.com/hamblin-ku/AGNBoost.git
cd AGNBoost
pip install -e . 
```

!!! Installation Note
    The "-e" flag installs AGNBoost in "editable" mode, which means that local changes to the source code will immediately be available without any further steps. This is particularly useful if you plan to contribute to the project or desire to customize AGNBoost for your research.

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

    [View Tutorials →](tutorials/basic-usage/)

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
- **Contributing**: See our [Contributing Guide](CONTRIBUTING.md) to get involved

---

*AGNBoost is open-source software released under the MIT License. If you use AGNBoost in your research, please see our [Citation Guide](citation.md).*

