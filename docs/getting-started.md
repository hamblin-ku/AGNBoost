**docs/getting-started.md:**
```markdown
# Quick Start

This guide will get you up and running with AGNBoost in just a few minutes.

## Prerequisites

Make sure you have Python 3.8 or later installed and AGNBoost is set up in your environment.

```bash
# Verify your installation
python -c "import agnboost; print('AGNBoost is ready!')"
```

## Your First Catalog

The `Catalog` class is the starting point for working with astronomical data in AGNBoost.

### Loading Data from a File

```python
from agnboost import Catalog

# Load data from a FITS file
catalog = Catalog(path="your_data.fits")

# Or from a CSV file
catalog = Catalog(path="your_data.csv")

# View basic information about your data
catalog.print_data_summary()
```

### Loading Data from a DataFrame

If you already have your data in a pandas DataFrame:

```python
import pandas as pd
from agnboost import Catalog

# Load your data however you prefer
df = pd.read_csv("your_data.csv")

# Create a catalog from the DataFrame
catalog = Catalog(data=df)
```

## Working with Features

AGNBoost can automatically create features from your photometric data:

```python
# Generate features (colors, log magnitudes, etc.)
catalog.create_feature_dataframe()

# Check what features were created
features = catalog.get_features()
print(f"Created {features.shape[1]} features from {features.shape[0]} objects")
```

## Using Pre-trained Models

AGNBoost comes with pre-trained models for common astronomical tasks:

```python
from agnboost import AGNBoost

# Initialize AGNBoost
agnboost = AGNBoost()

# Load pre-trained models
models_loaded = agnboost.load_models()

if models_loaded:
    print("Pre-trained models loaded successfully!")
    
    # Make predictions
    predictions = agnboost.predict(catalog)
    print("Available predictions:", list(predictions.keys()))
else:
    print("No pre-trained models found. You may need to train new models.")
```

## A Complete Example

Here's a complete workflow from data loading to predictions:

```python
from agnboost import Catalog, AGNBoost

# Step 1: Load your data
catalog = Catalog(path="jwst_photometry.fits")
print(f"Loaded {len(catalog.get_data())} astronomical objects")

# Step 2: Create features
catalog.create_feature_dataframe()
print(f"Generated {catalog.get_features().shape[1]} features")

# Step 3: Load pre-trained models
agnboost = AGNBoost()
agnboost.load_models()

# Step 4: Make predictions
predictions = agnboost.predict(catalog)

# Step 5: Access your results
for model_name, preds in predictions.items():
    print(f"Predictions from {model_name} model: {len(preds)} objects")
```

## Data Requirements

For AGNBoost to work optimally, your data should include:

- **Photometric measurements**: Flux or magnitude values in supported bands
- **Error estimates**: Uncertainty values for photometric measurements  
- **Valid band names**: Column names that match the `allowed_bands.json` configuration

You can check which bands are recognized:

```python
# See which photometric bands were found in your data
valid_bands = catalog.get_valid_bands()
print("Valid photometric bands found:")
for band_name, info in valid_bands.items():
    print(f"  {band_name}: {info['shorthand']} ({info['wavelength']} μm)")
```

## Next Steps

Now that you have the basics working, you can:

- **Explore the tutorials**: Learn about advanced features and customization
- **Check the API documentation**: Deep dive into all available methods and options
- **Train custom models**: Use your own data to train specialized models

### Useful Links

- [Band Configuration Tutorial](tutorials/band-configuration.md) - Learn how to add new photometric bands
- [Training from Scratch Tutorial](tutorials/training-from-scratch.md) - Train models on your own data
- [API Reference](api.md) - Complete documentation of all classes and methods

## Troubleshooting

### Common Issues

**"No valid band columns found"**: Your data column names don't match the expected photometric band names. Check the [Band Configuration Tutorial](tutorials/band-configuration.md) to learn how to add your bands.

**"No pre-trained models found"**: The models directory is empty or models are incompatible. You may need to train new models or check that the models were properly downloaded.

**Import errors**: Make sure all dependencies are installed. See the [Installation guide](installation.md) for details.

### Getting Help

If you encounter issues:

1. Check the [API documentation](api.md) for detailed method descriptions
2. Look at the tutorial examples for similar use cases
3. Open an issue on the [GitHub repository](https://github.com/kurthamblin/agnboost/issues)
