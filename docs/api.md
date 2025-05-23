# API Reference

This page provides complete documentation for all AGNBoost classes and functions.

## Overview

AGNBoost consists of two main classes:

- **Catalog**: For data loading, management, and feature engineering
- **AGNBoost**: For machine learning model training, tuning, and prediction

---

## Catalog Class

The `Catalog` class is your entry point for working with astronomical data. It handles loading data from various formats, validates photometric bands, creates features, and manages data splits.

### Key Features
- Load FITS files, CSV files, or pandas DataFrames
- Automatic photometric band validation
- Feature engineering (colors, transformations)
- Train/validation/test data splitting
- Signal-to-noise filtering

::: agnboost.agnboost.dataset.Catalog
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3
      members_order: source
      filters: ["!^_"]

---

## AGNBoost Class

The `AGNBoost` class provides the machine learning functionality, including model training, hyperparameter tuning, and prediction with uncertainty quantification.

### Key Features
- XGBoostLSS integration for distributional modeling
- Hyperparameter optimization
- Model persistence and loading
- Multi-target support
- Comprehensive logging and validation

::: agnboost.agnboost.model.AGNBoost
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3
      members_order: source
      filters: ["!^_"]

---

## Utility Functions

Helper functions for data processing, feature engineering, and model management.

::: agnboost.utils
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3
      members_order: source
      filters: ["!^_"]