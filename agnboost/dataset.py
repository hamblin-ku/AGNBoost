from pathlib import Path

from loguru import logger
from tqdm import tqdm
import json
from agnboost.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


# catalog.py
from xgboost import DMatrix
import os
import pandas as pd
import numpy as np
from agnboost.utils import log_message
from sklearn.model_selection import train_test_split 

import logging


# need to add a imputation method
# there also needs to be a imputation tune
# should take: inplace, cols_missing
# where cols_missing is an optional list of photometry column names that have missing data
# (otherwise consider all the phot columns)

class Catalog:
    """
    A class for loading, managing, and manipulating astronomical data.
    
    The Catalog class provides tools for loading astronomical data from various formats,
    performing feature engineering, data validation, and preparing data for machine learning
    workflows. It supports FITS files, CSV files, and pandas DataFrames.
    
    Attributes:
        data (pandas.DataFrame): The main astronomical dataset.
        features_df (pandas.DataFrame): Engineered features for machine learning.
        valid_columns (dict): Metadata for validated photometric band columns.
        train_indices (pandas.Index): Indices for training data split.
        val_indices (pandas.Index): Indices for validation data split.
        test_indices (pandas.Index): Indices for test data split.
    
    Examples:
        Basic usage with a FITS file:
        
        ```python
        from agnboost import Catalog
        
        # Load data
        catalog = Catalog(path="jwst_data.fits")
        
        # Create features
        catalog.create_feature_dataframe()
        
        # Split data
        catalog.split_data(test_size=0.2, val_size=0.2)
        ```
        
        Loading data from a pandas DataFrame:
        
        ```python
        import pandas as pd
        df = pd.read_csv("data.csv")
        catalog = Catalog(data=df)
        '''
    """


    # Class-level logger
    logger = logging.getLogger('AGNBoost.Catalog')

    def __init__(self, path=None, data=None, delimiter=',', bands_file="allowed_bands.json", summarize = True, logger=None):
        """
        Initialize a Catalog object to load and manipulate data.
        
        Args:
            path (str, optional): Path to the data file to load. Not used if data is provided.
            data (pandas.DataFrame or astropy.table.Table, optional): Pre-loaded data to use. 
                If provided, path is ignored.
            delimiter (str): Delimiter to use for CSV files. Only used if path is provided.
                Defaults to ','.
            bands_file (str): Path to the JSON file containing allowed band definitions.
                Defaults to "allowed_bands.json".
                
        Raises:
            ValueError: If neither path nor data is provided.
            FileNotFoundError: If the specified file path does not exist.
            
        Examples:
            Load from file:
            ```python
            catalog = Catalog(path="data.fits")
            ```
            
            Load from DataFrame:
            ```python
            catalog = Catalog(data=my_dataframe)
            ```
        """

        # Set up instance logger (use provided logger or class logger)
        self.logger = logger or self.__class__.logger

        self.path = path
        self.delimiter = delimiter
        self.data = None
        self.bands_file = 'agnboost/' + bands_file
        self.allowed_bands = {}
        self.valid_columns = {}  # Will store metadata for columns that pass validation
        
        # Load the allowed bands from JSON
        self._load_allowed_bands()
        
        # Load the data
        if data is not None:
            self._process_input_data(data)
        elif path is not None:
            self._load_data_from_path()
        else:
            self.logger.error("Error: Either path or data must be provided.")
            #log_message("Error: Either path or data must be provided.", "ERROR")
            return
        
        # Validate and Process the data
        if self.data is not None:
            # Ensure that there are at least 2 allowed columns
            if not self._validate_columns():
                self.logger.warning("Data does not contain at least 2 columns from the allowed bands.")
                #log_message("Warning: Data does not contain at least 2 columns from the allowed bands.", "WARNING")
            
            # Remove negative and zero flux values
            self._process_data()

            # Print summary statistics
            if summarize:
                self.print_data_summary()

        self.train_indices = None
        self.val_indices = None
        self.trainval_indices = None
        self.test_indices = None

        self.feature_functions = None

    @staticmethod         
    def create_color_dataframe(data, bands):
        """
        Calculate photometric colors from band fluxes.
        
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing color columns.
        """
        '''print(data)
        if data is None:
            data = self.data

        bands = self.valid_columns'''

        # Sort bands by wavelength for consistent color creation
        sorted_bands = sorted(bands.items(), key=lambda x: x[1]['wavelength'], reverse = True)
        band_names = [item[0] for item in sorted_bands]
        num_bands = len(band_names)

        # If there are not at least two bands, we cannot create any colors
        if len(band_names) < 2:
            #log_message("Not enough bands to create colors.", "WARNING")
            Catalog.logger.error("Not enough bands to create colors.")
            return None
        
        colors_df = pd.DataFrame(index=data.index)
        
        # Iterate through the bands and create all possible (non-reciprocal) colors
        for i in range(num_bands):
            band_i = band_names[i]
            shorthand_i = bands[band_i]['shorthand']
            for j in range(1, num_bands - i):
                band_j = band_names[i + j]
                shorthand_j = bands[band_j]['shorthand']
                try:
                    color_ij_name = f"{shorthand_i}/{shorthand_j}"
                    colors_df[color_ij_name] = np.log10( data[band_i] / data[band_j] )
                    Catalog.logger.info(f"Created color: {color_ij_name}")
                    #log_message(f"Created color: {color_ij_name}")

                except Exception as e:
                    #log_message(f"Error creating color {shorthand_i}/{shorthand_j}: {str(e)}", "WARNING")
                    Catalog.logger.error(f"Error creating color {shorthand_i}/{shorthand_j}: {str(e)}", "WARNING")

        return colors_df

    @staticmethod
    def remove_negative_fluxes(data, bands):
        bands = list(bands.keys())

        valid_data = data.copy()
        data_columns = list(data.columns)

        for col_name in bands:
            col_idx = data_columns.index(col_name)

            try:
                if (valid_data[col_name] <= 0).any():
                    Catalog.logger.warning(f"Warning: Column {col_name} contains non-positive values. "
                               f"Setting them to NaN for log calculation.")
                    valid_data.loc[ (valid_data[col_name] <= 0), col_name ] = np.nan

            except Exception as e:
                Catalog.logger.error(f"Error setting negative or zero values to nan in {col_name}: {str(e)}")
                return None

        return valid_data

    def get_features_from_phots(self, phot_df):
        # Use default feature configuration
        if self.feature_functions is None:

            phot_i_df = phot_df.apply(lambda x: np.log10(x) )#.reset_index(drop = True)
            #phot_i_df = np.log10( pd.DataFrame(phot_i_arr).T ).reset_index(drop = True)

            color_i_df = self.create_color_dataframe(data = phot_df, bands = self.valid_columns)#.reset_index(drop = True)

            color_i_df_2 = color_i_df.apply(lambda x: x**2).rename( columns = {color: color +'^2' for color in color_i_df.columns} )#.reset_index(drop = True)

            feature_i_df = pd.concat([phot_i_df, color_i_df, color_i_df_2], axis=1, join='outer')

        return feature_i_df

    def create_feature_dataframe(self, feature_funcs=None):
        """
        Create a dataframe of features for eventual use with AGNBoost class.
        
        Parameters:
        -----------
        feature_funcs : list or None
            List of functions to apply to the validated columns. If None,
            default features (log10 of bands, colors, and square of colors) are created.
            
        Returns:
        --------
        pandas.DataFrame
            The created feature dataframe (also stored as self.features_df).
        """
        if self.data is None or not self.valid_columns:
            #log_message("Cannot create features: No data or valid columns available.", "ERROR")
            self.logger.error("Cannot create features: No data or valid columns available.")
            return None
        
        feature_dfs = []
        
        # If no custom feature functions are passed
        if feature_funcs is None:
            validated_bands_dict = self.get_valid_bands()
            validated_bands_list = list(validated_bands_dict.keys()) 
            #shorthands = [validated_bands_dict[ band ]['shorthand'] for band in validated_bands_list]

            # 1. Create log10 of validated band columns
            log_bands_df = self.data[validated_bands_list].apply( np.log10 )
            feature_dfs.append(log_bands_df)
            
            # 2. Create colors from validated bands
            colors_df = self.create_color_dataframe(data = self.data[validated_bands_list], bands = self.valid_columns)
            feature_dfs.append(colors_df)

            # 3. Create squares of colors
            colors_squared_df = colors_df.apply( lambda x : x**2 )
            colors_squared_df.rename( columns = {color: color +'^2' for color in colors_df.columns}, inplace = True)
            feature_dfs.append(colors_squared_df)

        # Need to change feature funcs to a dict, where the entries are the suffixes for the feature names
        # Need to save the functions used for features for photometric error calculations
        else:
            # Custom functions
            if not isinstance(feature_funcs, (list, tuple)):
                log_message("feature_funcs must be a list or tuple of functions.", "ERROR")
                return None
            
            # Validate and apply each function
            for i, func in enumerate(feature_funcs):
                if not callable(func):
                    
                    continuelog_message(f"Item {i} in feature_funcs is not callable. Skipping.", "WARNING")
                
                try:
                    # Apply function to the relevant data subset
                    result = func(self.data, self.valid_columns)
                    
                    # Check if result is a DataFrame
                    if isinstance(result, pd.DataFrame):
                        if result.index.equals(self.data.index):
                            feature_dfs.append(result)
                            log_message(f"Applied custom feature function {func.__name__}: "
                                       f"added {result.shape[1]} features.")
                        else:
                            log_message(f"Function {func.__name__} returned DataFrame with "
                                       f"mismatched index. Skipping.", "WARNING")
                    else:
                        log_message(f"Function {func.__name__} did not return a DataFrame. "
                                   f"Got {type(result)} instead. Skipping.", "WARNING")
                except Exception as e:
                    log_message(f"Error applying function {func.__name__}: {str(e)}", "WARNING")
        
        # Combine all feature dataframes
        if not feature_dfs:
            log_message("No features were created.", "WARNING")
            return None
        
        try:
            self.features_df = pd.concat(feature_dfs, axis=1, join = 'outer')
            log_message(f"Created feature dataframe with {self.features_df.shape[1]} columns "
                       f"and {self.features_df.shape[0]} rows.")
            
            # Handle NaN values
            nan_count = self.features_df.isna().sum().sum()
            if nan_count > 0:
                log_message(f"Warning: Feature dataframe contains {nan_count} NaN values.", "WARNING")
            
            return self.features_df
            
        except Exception as e:
            log_message(f"Error creating final feature dataframe: {str(e)}", "ERROR")
            return None

    def transform(self, column_name, transform_func, new_column_name=None, inplace=False):
        """
        Apply a transformation function to a column and save the result.
        
        Parameters:
        -----------
        column_name : str
            Name of the column to transform.
        transform_func : callable
            Function to apply to the column.
        new_column_name : str or None, default=None
            Name for the transformed column. If None and not inplace, will use
            f"{column_name}_transformed".
        inplace : bool, default=False
            If True, overwrite the original column. If False, create a new column.
            
        Returns:
        --------
        pandas.Series
            The transformed column.
        """
        # Verify data exists
        if self.data is None:
            self.logger.error("No data loaded to transform.")
            return None
        
        # Verify column exists
        if column_name not in self.data.columns:
            self.logger.error(f"Column '{column_name}' not found in data.")
            return None
        
        # Verify transform_func is callable
        if not callable(transform_func):
            self.logger.error(f"Transform function is not callable: {transform_func}")
            return None
        
        # Determine output column name
        if inplace:
            output_column = column_name
            self.logger.info(f"Will transform column '{column_name}' in-place")
        else:
            if new_column_name is None:
                # Generate a name based on function name if available, otherwise use default
                func_name = getattr(transform_func, '__name__', '')
                if func_name and func_name != '<lambda>':
                    output_column = f"{column_name}_{func_name}"
                else:
                    output_column = f"{column_name}_transformed"
            else:
                output_column = new_column_name
                
            self.logger.info(f"Will transform column '{column_name}' and save as '{output_column}'")
        
        # Apply transformation
        try:
            # Get the column data
            column_data = self.data[column_name]
            
            # Apply the transformation
            self.logger.info(f"Applying transformation to column '{column_name}'")
            transformed_data = transform_func(column_data)
            
            # Validate transformed data
            if transformed_data is None:
                self.logger.error("Transformation returned None")
                return None
                
            # Check if the transformation returned a Series or array-like object
            if not isinstance(transformed_data, pd.Series):
                try:
                    # Convert to Series with original index
                    transformed_data = pd.Series(transformed_data, index=self.data.index)
                except Exception as e:
                    self.logger.error(f"Could not convert transformation result to Series: {str(e)}")
                    return None
            
            # Save the transformed column
            self.data[output_column] = transformed_data
            
            self.logger.info(f"Successfully transformed column. Result saved as '{output_column}'")
            
            # Log some stats about the transformation
            if pd.api.types.is_numeric_dtype(transformed_data):
                stats = {
                    'min': transformed_data.min(),
                    'max': transformed_data.max(),
                    'mean': transformed_data.mean(),
                    'median': transformed_data.median(),
                    'null_count': transformed_data.isna().sum()
                }
                self.logger.info(f"Transformation statistics: {stats}")
            else:
                null_count = transformed_data.isna().sum()
                unique_count = transformed_data.nunique()
                self.logger.info(f"Transformed column has {unique_count} unique values and {null_count} null values")
            
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Error applying transformation: {str(e)}")
            return None
    
    def get_targets(self, target_names):
        """
        Extract target columns from the data.
        
        Parameters:
        -----------
        target_names : str or list
            Name(s) of target column(s) to extract.
        drop_na : bool, default=False
            If True, drop rows with NA values in any of the target columns.
            
        Returns:
        --------
        pandas.DataFrame or pandas.Series
            Target column(s). Returns a Series if a single target name is provided,
            or a DataFrame if multiple targets are requested.
        """
        if self.data is None:
            self.logger.error("No data loaded to extract targets.")
            return None
        
        # Convert single target name to list
        if isinstance(target_names, str):
            target_names = [target_names]
        
        # Check if all target columns exist
        missing_targets = [name for name in target_names if name not in self.data.columns]
        if missing_targets:
            self.logger.error(f"Target column(s) not found in data: {missing_targets}")
            return None
        
        # Extract targets
        try:
            targets = self.data[target_names]
            self.logger.info(f"Extracted {len(targets)} rows for target(s): {target_names}")
            
            return targets
        
        except Exception as e:
            self.logger.error(f"Error extracting target(s): {str(e)}")
            return None


    def get_features(self):
        """
        Get the feature dataframe, creating it if it doesn't exist.
        
        Returns:
        --------
        pandas.DataFrame
            The feature dataframe.
        """
        if not hasattr(self, 'features_df') or self.features_df is None:
            log_message("Features not yet created. Creating with default parameters.")
            self.create_feature_dataframe()
        
        return self.features_df

    def get_feature_names(self):
        """
        Get the feature names dataframe, creating it if it doesn't exist.
        
        Returns:
        --------
        pandas.DataFrame
            The feature dataframe.
        """
        if not hasattr(self, 'features_df') or self.features_df is None:
            log_message("Features not yet created. Creating with default parameters.")
            self.create_feature_dataframe()
        
        return self.features_df.columns


    def get_data(self):
        return self.data

    def get_length(self):
        return len(self.data)


    def _load_allowed_bands(self):
        """
        Load allowed bands from the JSON configuration file.
        """

        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for bands file at: {os.path.join(os.getcwd(), 'allowed_bands.json')}")

        try:
            if not os.path.exists(self.bands_file):
                log_message(f"Error: Bands file '{self.bands_file}' does not exist.", "ERROR")
                return
            
            with open(self.bands_file, 'r') as f:
                bands_data = json.load(f)
            
            # Remove metadata entry if present
            if "_metadata" in bands_data:
                metadata = bands_data.pop("_metadata")
                log_message(f"Loaded bands file metadata: {metadata.get('description', 'No description')}")
            
            self.allowed_bands = bands_data
            log_message(f"Loaded {len(self.allowed_bands)} allowed bands from {self.bands_file}")
            
        except json.JSONDecodeError:
            log_message(f"Error: Invalid JSON format in bands file '{self.bands_file}'.", "ERROR")
        except Exception as e:
            log_message(f"Error loading bands file: {str(e)}", "ERROR")
    
    def _process_input_data(self, data):
        """
        Process input data that was passed directly.
        
        Parameters:
        -----------
        data : pandas.DataFrame or astropy.table.Table
            The data to process.
        """
        if isinstance(data, pd.DataFrame):
            self.data = data
            log_message("Using provided pandas DataFrame.")
        else:
            # Try to handle Astropy Table
            try:
                from astropy.table import Table
                if isinstance(data, Table):
                    self.data = data.to_pandas()
                    log_message("Converted Astropy Table to pandas DataFrame.")
                else:
                    log_message(f"Unsupported data type: {type(data)}. Expected DataFrame or Table.", "ERROR")
            except ImportError:
                log_message("Astropy not available. Cannot convert Table to DataFrame.", "ERROR")
            except Exception as e:
                log_message(f"Error processing input data: {str(e)}", "ERROR")
    
    def _load_data_from_path(self):
        """
        Load data from the specified path.
        """
        if not os.path.exists(self.path):
            log_message(f"Error: File '{self.path}' does not exist.", "ERROR")
            return
        
        file_extension = os.path.splitext(self.path)[1].lower()
        
        if file_extension == '.fits':
            self._load_fits_file()
        else:
            self._load_csv_file()
    
    def _process_data(self):
        """
        Process the saved data and remove any zero or negative values from the flux columns.
        This serves two purposes:
            (1)  It handles the case of using altenrative flags for missing values (-99,-999, etc.) and just sets these to NaN from the start
            (2)  Prevents errors later on when creating the feature DataFrames, since we will not be able to take the log of these values or divide by them.
        """
        self.data = self.remove_negative_fluxes(data = self.data, bands = self.valid_columns)


    def _validate_columns(self):
        """
        Validate that the data contains at least 2 columns from the allowed bands.
        Store metadata for valid columns.
        
        Returns:
        --------
        bool
            True if validation passes, False otherwise.
        """
        if self.data is None or not self.allowed_bands:
            return False
        
        # Find matching columns between data and allowed bands
        matching_columns = []
        for column in list(self.allowed_bands.keys()):
            if column in self.data.columns:
                matching_columns.append(column)
                # Store metadata for this valid column
                self.valid_columns[column] = {
                    'shorthand': self.allowed_bands[column]['shorthand'],
                    'wavelength': self.allowed_bands[column]['wavelength']
                }

        '''for column in self.data.columns:
            if column in self.allowed_bands:
                matching_columns.append(column)
                # Store metadata for this valid column
                self.valid_columns[column] = {
                    'shorthand': self.allowed_bands[column]['shorthand'],
                    'wavelength': self.allowed_bands[column]['wavelength']
                }'''
        
        # Log the matching columns
        if matching_columns:
            log_message(f"Found {len(matching_columns)} valid band columns:")
            for col in matching_columns:
                log_message(f"  - {col} ({self.valid_columns[col]['shorthand']}): "
                           f"{self.valid_columns[col]['wavelength']} μm")
        else:
            log_message("No valid band columns found in the dataset.", "WARNING")
        
        # Also check for common columns that might be needed
        common_cols = ['id', 'ra', 'dec', 'redshift', 'z', 'class', 'fagn']
        found_common = [col for col in common_cols if col in self.data.columns]
        if found_common:
            log_message(f"Found {len(found_common)} common non-band columns: {found_common}")
        
        # Check if there are at least 2 matching columns
        return len(matching_columns) >= 2
    
    def _load_fits_file(self):
        """
        Load a FITS file using Astropy and convert to pandas DataFrame.
        """
        try:
            from astropy.table import Table
            
            log_message(f"Loading FITS file: {self.path}")
            
            # Load the FITS file as an Astropy Table
            table = Table.read(self.path)
            
            # Convert to pandas DataFrame
            self.data = table.to_pandas()
            
            log_message(f"Successfully loaded FITS file with {len(self.data)} rows.")
        except ImportError:
            log_message("Error: astropy is required to load FITS files. Please install it using 'pip install astropy'.", "ERROR")
        except Exception as e:
            log_message(f"Error loading FITS file: {str(e)}", "ERROR")
    
    def _load_csv_file(self):
        """
        Load a CSV or similar delimited file using pandas.
        Will try multiple delimiters if the specified one fails.
        """
        # Common delimiters to try if the specified one fails
        delimiters_to_try = [self.delimiter, ',', '\t', ';', '|', ' ']
        
        # Make sure we only try each delimiter once
        delimiters_to_try = list(dict.fromkeys(delimiters_to_try))
        
        for delimiter in delimiters_to_try:
            try:
                log_message(f"Attempting to load file with delimiter: '{delimiter}'")
                self.data = pd.read_csv(self.path, delimiter=delimiter)
                
                # If we get here, the file was loaded successfully
                if delimiter != self.delimiter:
                    log_message(f"Note: File was loaded with delimiter '{delimiter}' instead of '{self.delimiter}'.")
                    self.delimiter = delimiter
                
                log_message(f"Successfully loaded data with {len(self.data)} rows.")
                return
            except pd.errors.ParserError:
                log_message(f"Failed to parse file with delimiter: '{delimiter}'", "WARNING")
            except Exception as e:
                log_message(f"Error when trying delimiter '{delimiter}': {str(e)}", "WARNING")
        
        # If we get here, all delimiters failed
        log_message("Failed to load file with any of the attempted delimiters.", "ERROR")
    
    def print_data_summary(self):
        """
        Print a formatted summary of the loaded data.
        """
        if self.data is None:
            log_message("No data loaded to summarize.", "WARNING")
            return
        
        # Get basic information
        rows, cols = self.data.shape
        dtypes = self.data.dtypes
        mem_usage = self.data.memory_usage(deep=True).sum() / 1024**2  # in MB
        
        # Print header
        print("\n" + "="*80)
        source = os.path.basename(self.path) if self.path else "User-provided data"
        print(f"DATA SUMMARY: {source}")
        print("="*80)
        
        # Print basic stats
        print(f"Dimensions: {rows} rows × {cols} columns")
        print(f"Memory usage: {mem_usage:.2f} MB")
        print("-"*80)
        
        # Print information about valid bands first
        if self.valid_columns:
            print("Valid Band Columns:")
            print("-"*80)
            print(f"{'Column Name':<30} {'Shorthand':<15} {'Wavelength (μm)':<15}")
            print("-"*80)
            
            # Sort by wavelength
            sorted_bands = sorted(self.valid_columns.items(), key=lambda x: x[1]['wavelength'])
            for col_name, info in sorted_bands:
                print(f"{col_name[:30]:<30} {info['shorthand']:<15} {info['wavelength']:<15.3f}")
            
            print("-"*80)
        
        # Column information
        print("Column Information:")
        print("-"*80)
        print(f"{'Column Name':<30} {'Type':<15} {'Non-Null':<15} {'Null %':<10}")
        print("-"*80)
        
        for col in self.data.columns:
            col_type = str(dtypes[col])
            non_null = self.data[col].count()
            null_pct = 100 - (non_null / rows * 100)
            
            # Count unique values (with a limit to avoid performance issues)
            try:
                if self.data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(self.data[col]):
                    # For string/categorical columns, just count uniques
                    unique_count = self.data[col].nunique()
                else:
                    # For numeric columns, include min/max
                    if pd.api.types.is_numeric_dtype(self.data[col]):
                        col_min = self.data[col].min()
                        col_max = self.data[col].max()

            except:
                unique_info = "Error"
            
            print(f"{col[:30]:<30} {col_type:<15} {non_null}/{rows:<15} {null_pct:.2f}%{'':<5}")
        
        print("-"*80)
        
        # Additional statistics for numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("\nNumeric Column Statistics:")
            print("-"*80)
            
            # Calculate statistics
            stats = self.data[numeric_cols].describe().T
            
            # Print statistics
            print(f"{'Column':<30} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
            print("-"*80)
            
            for col in numeric_cols:
                if col in stats.index:
                    row = stats.loc[col]
                    print(f"{col[:20]:<20} {row['mean']:<12.4g} {row['std']:<12.4g} {row['min']:<12.4g} {row['max']:<12.4g}")

        
        print("="*80 + "\n")
    
    def get_valid_bands(self):
        """
        Get information about valid band columns in the data.
        
        Returns:
        --------
        dict
            Dictionary of valid band columns with their metadata.
        """
        return self.valid_columns

    def get_valid_bands_list(self):
        """
        Get information about valid band columns in the data.
        
        Returns:
        --------
        dict
            Dictionary of valid band columns with their metadata.
        """
        return list(self.valid_columns.keys())
    
    def get_band_wavelengths(self):
        """
        Get a dictionary mapping band columns to their wavelengths.
        
        Returns:
        --------
        dict
            Dictionary mapping column names to wavelengths in microns.
        """
        return {col: info['wavelength'] for col, info in self.valid_columns.items()}
    
    def get_band_shorthands(self):
        """
        Get a dictionary mapping band columns to their shorthand names.
        
        Returns:
        --------
        dict
            Dictionary mapping column names to shorthand identifiers.
        """
        return {col: info['shorthand'] for col, info in self.valid_columns.items()}
    
    def sn_cut(self, columns=None, threshold=3.0, inplace=False, suffix='_err'):
        """
        Perform a signal-to-noise ratio cut on the specified columns.
        
        Parameters:
        -----------
        columns : dict or None
            Dictionary mapping value columns to their error columns.
            Example: {'flux': 'flux_err', 'redshift': 'redshift_err'}
            If None, automatically detect band columns and use {band: band+suffix} pairs.
        threshold : float, default=3.0
            Minimum S/N ratio to keep a row.
        inplace : bool, default=False
            If True, modifies the data in place; otherwise, returns a copy.
        suffix : str, default='_err'
            Suffix to append to band names to find error columns, if columns=None.
            
        Returns:
        --------
        pandas.DataFrame or None
            Filtered dataframe if inplace=False, otherwise None.
        """
        if self.data is None:
            log_message("No data loaded to filter.", "WARNING")
            return None
        
        result = self.data.copy() if not inplace else self.data
        
        # If no columns specified, auto-detect from valid band columns
        if columns is None:
            columns = {}
            for band_col in self.valid_columns:
                err_col = f"{band_col}{suffix}"
                if err_col in result.columns:
                    columns[band_col] = err_col
            
            if not columns:
                log_message("No valid band/error column pairs found for S/N cut.", "WARNING")
                return result if not inplace else None
            
            log_message(f"Auto-detected {len(columns)} band/error column pairs for S/N cut.")
        
        # Keep track of the total number of rows dropped
        original_rows = len(result)
        rows_remaining = original_rows
        
        for value_col, error_col in columns.items():
            # Check if both columns exist
            if value_col not in result.columns:
                log_message(f"Value column '{value_col}' not found in data. Skipping.", "WARNING")
                continue
                
            if error_col not in result.columns:
                log_message(f"Error column '{error_col}' not found in data. Skipping.", "WARNING")
                continue
            
            # Calculate S/N ratio
            # Avoid division by zero and handle NaN values
            sn_ratio = np.abs(result[value_col]) / result[error_col]
            sn_ratio = sn_ratio.replace([np.inf, -np.inf], np.nan)
            
            # Apply the cut
            previous_rows = len(result)
            result = result[sn_ratio >= threshold]
            rows_dropped = previous_rows - len(result)
            
            # Get shorthand name if it's a valid band
            if value_col in self.valid_columns:
                band_name = f"{value_col} ({self.valid_columns[value_col]['shorthand']})"
            else:
                band_name = value_col
            
            log_message(f"S/N cut on {band_name} with threshold {threshold}: "
                       f"dropped {rows_dropped} rows ({rows_dropped/previous_rows*100:.2f}%).")
            rows_remaining = len(result)
        
        total_dropped = original_rows - rows_remaining
        log_message(f"Total S/N cut result: kept {rows_remaining}/{original_rows} rows "
                   f"({rows_remaining/original_rows*100:.2f}%), "
                   f"dropped {total_dropped} rows ({total_dropped/original_rows*100:.2f}%).")
        
        if inplace:
            self.data = result
            return None
        else:
            return result
    
    def get_data(self):
        """
        Return the loaded dataframe.
        
        Returns:
        --------
        pandas.DataFrame
            The loaded data.
        """
        return self.data
    
    def get_columns(self):
        """
        Return the column names of the loaded dataframe.
        
        Returns:
        --------
        list
            List of column names.
        """
        if self.data is not None:
            return list(self.data.columns)
        return []
    
    def filter_data(self, filter_conditions):
        """
        Filter the data based on provided conditions.
        
        Parameters:
        -----------
        filter_conditions : dict
            Dictionary of column:value pairs to filter on.
            
        Returns:
        --------
        pandas.DataFrame
            Filtered dataframe.
        """
        if self.data is None:
            log_message("No data loaded to filter.", "WARNING")
            return None
        
        filtered_data = self.data.copy()
        
        for column, value in filter_conditions.items():
            if column in filtered_data.columns:
                if isinstance(value, (list, tuple)):
                    filtered_data = filtered_data[filtered_data[column].isin(value)]
                else:
                    filtered_data = filtered_data[filtered_data[column] == value]
            else:
                log_message(f"Column '{column}' not found in data.", "WARNING")
        
        log_message(f"Filtered data from {len(self.data)} to {len(filtered_data)} rows.")
        return filtered_data
    
    def get_subset(self, columns=None, rows=None):
        """
        Get a subset of the data by columns and/or rows.
        
        Parameters:
        -----------
        columns : list or None
            List of column names to include. If None, includes all columns.
        rows : slice, list, or None
            Rows to include. Can be a slice (e.g., slice(0, 100)), a list of indices,
            or None to include all rows.
            
        Returns:
        --------
        pandas.DataFrame
            Subset of the data.
        """
        if self.data is None:
            log_message("No data loaded to subset.", "WARNING")
            return None
        
        # Select columns if specified
        if columns is not None:
            missing_cols = [col for col in columns if col not in self.data.columns]
            if missing_cols:
                log_message(f"Warning: Columns not found in data: {missing_cols}", "WARNING")
                columns = [col for col in columns if col in self.data.columns]
            
            subset = self.data[columns]
        else:
            subset = self.data.copy()
        
        # Select rows if specified
        if rows is not None:
            try:
                subset = subset.iloc[rows]
                log_message(f"Selected {len(subset)} rows from data.")
            except Exception as e:
                log_message(f"Error selecting rows: {str(e)}", "ERROR")
        
        return subset
    
    def save_to_csv(self, output_path, index=False):
        """
        Save the current data to a CSV file.
        
        Parameters:
        -----------
        output_path : str
            Path to save the CSV file.
        index : bool, default=False
            Whether to include the index in the saved file.
            
        Returns:
        --------
        bool
            True if successful, False otherwise.
        """
        if self.data is None:
            log_message("No data loaded to save.", "WARNING")
            return False
        
        try:
            self.data.to_csv(output_path, index=index)
            log_message(f"Data saved successfully to {output_path}")
            return True
        except Exception as e:
            log_message(f"Error saving data to {output_path}: {str(e)}", "ERROR")
            return False

    def drop_nan(self, columns=None, inplace=False, how='any'):
        """
        Drop rows with NaN values in specified columns or validated columns.
        
        Parameters:
        -----------
        columns : list or None, default=None
            List of column names to check for NaN. If None, uses all validated columns.
        inplace : bool, default=False
            If True, performs operation in-place and returns None.
            If False, returns a copy of the data with rows dropped.
        how : {'any', 'all'}, default='any'
            'any' : Drop if any of the specified columns has NaN
            'all' : Drop only if all of the specified columns have NaN
        
        Returns:
        --------
        pandas.DataFrame or None
            The DataFrame with NaN rows dropped, or None if inplace=True.
        """
        if self.data is None:
            self.logger.error("No data loaded to perform drop_nan operation.")
            return None
        


        # If no columns specified, use validated columns
        if columns is None:
            columns = list(self.valid_columns.keys())
            if not columns:
                self.logger.warning("No validated columns found. Using all columns for NaN check.")
                columns = list(self.data.columns)
        
        # Check that specified columns exist
        missing_cols = [col for col in columns if col not in self.data.columns]
        if missing_cols:
            self.logger.warning(f"Columns not found in data: {missing_cols}")
            columns = [col for col in columns if col in self.data.columns]
            
        if not columns:
            self.logger.error("No valid columns to check for NaNs. Operation aborted.")
            return self.data.copy() if not inplace else None
        
        # Count rows before dropping
        original_rows = len(self.data)
        
        # Create a view or copy based on inplace parameter
        result = self.data.copy()

        # Set inf values to nan
        result.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Set negative value to nans
        #result = self.remove_negative_fluxes(data = result, bands = self.valid_columns)

        # Get column names for logging
        column_names = [f"{col} ({self.valid_columns[col]['shorthand']})" if col in self.valid_columns 
                       else col for col in columns]
        column_desc = ", ".join(column_names)
        
        # Drop NaN rows
        result = result.dropna(axis = 0, subset=columns, how=how)


        # Count rows after dropping
        remaining_rows = len(result)
        dropped_rows = original_rows - remaining_rows
        drop_percentage = (dropped_rows / original_rows * 100) if original_rows > 0 else 0
        
        # Log the results
        if dropped_rows > 0:
            log_message(f"Dropped {dropped_rows} rows ({drop_percentage:.2f}%) with NaN values in {how} of "
                            f"these columns: {column_desc}")
            log_message(f"Rows remaining: {remaining_rows}/{original_rows} ({remaining_rows/original_rows*100:.2f}%)")

        else:
            log_message(f"No rows with NaN values found in the specified columns.")
        
        if inplace:
            self.data = result
            return None
        else:
            return result



    def split_data(self, test_size=0.2, val_size=0.2, random_state=None, stratify_col=None, n_bins=10):
        """
        Split the data into train, validation, and test sets.
        
        Parameters:
        -----------
        test_size : float, default=0.2
            Proportion of the data to use for testing (0.0 to 1.0).
        val_size : float, default=0.2
            Proportion of the data to use for validation (0.0 to 1.0).
        random_state : int or None, default=None
            Random seed for reproducibility.
        stratify_col : str or None, default=None
            Column name to use for stratified splitting. If provided, 
            ensures proportional representation of this column's values
            in all splits. For continuous columns, binning will be applied.
        n_bins : int, default=10
            Number of bins to use when stratifying on a continuous column.
            Only used when stratify_col is a continuous variable.
            
        Returns:
        --------
        tuple
            (train_indices, val_indices, test_indices) - Indices for each split
        """
        if self.data is None:
            self.logger.error("No data loaded to split.")
            return None, None, None
        
        # Validate input parameters
        if not 0 <= test_size < 1:
            self.logger.error(f"Invalid test_size: {test_size}. Must be between 0 and 1.")
            return None, None, None
        
        if not 0 <= val_size < 1:
            self.logger.error(f"Invalid val_size: {val_size}. Must be between 0 and 1.")
            return None, None, None
        
        if test_size + val_size >= 1:
            self.logger.error(f"Sum of test_size ({test_size}) and val_size ({val_size}) "
                             f"must be less than 1.")
            return None, None, None
        
        # Calculate proportions for the first split (test vs. rest)
        first_test_size = test_size
        
        # Calculate proportions for the second split (val vs. train, out of the remaining data)
        second_test_size = val_size / (1 - first_test_size)
        
        # Prepare stratification if requested
        stratify = None
        if stratify_col is not None:
            if stratify_col in self.data.columns:
                # Check if it's a continuous column that needs binning
                if pd.api.types.is_numeric_dtype(self.data[stratify_col]) and self.data[stratify_col].nunique() > n_bins:
                    self.logger.info(f"Stratifying by continuous column '{stratify_col}' using {n_bins} bins")
                    
                    # Create bins for the continuous target
                    try:
                        # Get column data without NaN values
                        col_data = self.data[stratify_col].dropna()
                        
                        # Use pandas qcut for equal-sized bins (based on quantiles)
                        # Fall back to cut if we have too many repeated values
                        try:
                            bins = pd.qcut(col_data, n_bins, duplicates='drop')
                        except ValueError:
                            self.logger.warning(f"Using uniform binning due to repeated values in '{stratify_col}'")
                            bins = pd.cut(col_data, n_bins)
                        
                        # Create a Series mapping index to bin label
                        binned_values = pd.Series(bins, index=col_data.index)
                        
                        # Any rows with NaN values in stratify_col will not have a bin
                        # Join this back to the full index
                        full_binned = pd.Series(index=self.data.index)
                        full_binned.loc[binned_values.index] = binned_values
                        
                        stratify = full_binned
                        self.logger.info(f"Created {len(bins.categories)} bins for stratification")
                        self.logger.info(f"Bin distribution: {bins.value_counts(normalize=True).sort_index().to_dict()}")
                    except Exception as e:
                        self.logger.warning(f"Error creating bins for stratification: {str(e)}. Using random splitting.")
                        stratify = None
                else:
                    # Categorical or discrete numeric column, use directly
                    stratify = self.data[stratify_col]
                    self.logger.info(f"Using column '{stratify_col}' for stratified splitting")
            else:
                self.logger.warning(f"Stratify column '{stratify_col}' not found in data. Using random splitting.")
        
        try:
            # First split: separate test set
            train_val_indices, test_indices = train_test_split(
                self.data.index,
                test_size=first_test_size,
                random_state=random_state,
                stratify=stratify
            )
            
            # Update stratify for the second split if needed
            if stratify is not None:
                stratify = stratify.loc[train_val_indices]
            
            # Second split: separate train and validation sets
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=second_test_size,
                random_state=random_state,
                stratify=stratify
            )
            
            # Save the indices
            self.train_indices = train_indices
            self.val_indices = val_indices
            self.trainval_indices = train_indices.union(val_indices)
            self.test_indices = test_indices
            
            # Log the split sizes
            self.logger.info(f"Data split complete:")
            self.logger.info(f"  - Train set: {len(train_indices)} rows ({len(train_indices)/len(self.data)*100:.1f}%)")
            self.logger.info(f"  - Validation set: {len(val_indices)} rows ({len(val_indices)/len(self.data)*100:.1f}%)")
            self.logger.info(f"  - Test set: {len(test_indices)} rows ({len(test_indices)/len(self.data)*100:.1f}%)")
            
            return train_indices, val_indices, test_indices
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            return None, None, None


    def get_split_df(self, split_type='train', include_features=True, include_target=None, return_DMatrix = False, missing=np.nan):
        """
        Get a dataframe for the specified data split.
        
        Parameters:
        -----------
        split_type : str, default='train'
            Which data split to use. Options: 'train', 'val'/'validation', or 'test'.
        include_features : bool, default=True
            If True, include the feature columns in the result.
        include_target : str, list, or None, default=None
            Target column(s) to include. If None, no target columns are included.
        return_DMatrix: bool, default = False
            Whether to return a XGBoost DMatrix instead of a pandas Dataframe.
        missing : int, float, or None, default=np.nan
            Value to represent missing values in XGBoost.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame for the specified split with requested columns.
        """
        # Validate split_type
        valid_split_types = {'train', 'val', 'validation', 'test', 'trainval'}
        if split_type not in valid_split_types:
            self.logger.error(f"Invalid split_type: '{split_type}'. Must be one of {valid_split_types}")
            return None
        
        # Normalize split_type
        if split_type == 'validation':
            split_type = 'val'
 
        # Split data if not already split
        if not hasattr(self, f'{split_type}_indices') or getattr(self, f'{split_type}_indices') is None:
            self.logger.warning("Data has not been split. Running split_data with default parameters.")
            self.split_data()
        
        # Get indices for the specified split
        indices = getattr(self, f'{split_type}_indices')
        
        # Create result dataframe
        result_parts = []
        
        # Add features if requested
        if include_features:
            # Create feature dataframe if it doesn't exist
            if not hasattr(self, 'features_df') or self.features_df is None:
                self.logger.info("Features not yet created. Creating with default parameters.")
                self.create_feature_dataframe()
            
            # Get features for the specified split
            try:
                # Call the appropriate feature getter method
                #features_method = getattr(self, f'get_{split_type}_features')
                features = self.get_features().iloc[indices]
                
                if features is not None and not features.empty:
                    result_parts.append(features)
                else:
                    self.logger.warning(f"No features available for {split_type} split.")
                    # If we're only returning features, return None
                    if include_target is None:
                        return None
            except Exception as e:
                self.logger.error(f"Error getting features for {split_type} split: {str(e)}")
                # If we're only returning features, return None
                if include_target is None:
                    return None
        
        # Add targets if requested
        if include_target is not None:
            # Convert to list if string
            if not isinstance(include_target, str):
                self.logger.error("Target column is a list, but it must be a string")
                return None
            
            # Validate target columns
            if include_target not in self.data.columns:
                self.logger.error(f"Target column(s) not found in data: {missing_targets}")
                if split_type == 'test':
                    self.logger.warning(f"Target column(s) not found in data: {missing_targets}. Not needed for 'test' set, will set to None.")
                    include_target = None
                else:
                    self.logger.error(f"Target column(s) not found in data: {missing_targets}, but required for non-test set.")
                    return None
            
            else:
                # Get targets
                try:
                    # Extract target columns from the main dataframe
                    targets = self.data.loc[indices, include_target]
                    
                    if targets is not None and not targets.empty:
                        result_parts.append(targets)
                    else:
                        self.logger.warning(f"No targets available for {split_type} split.")
                        # If we're only returning targets, return None
                        if not include_features or not result_parts:
                            return None
                except Exception as e:
                    self.logger.error(f"Error getting targets for {split_type} split: {str(e)}")
                    # If we're only returning targets, return None
                    if not include_features or not result_parts:
                        return None

        # If both include features and include_target are False/None, just return self.data[indices]
        if not include_features and include_target is None:
            self.logger.info(f"Returning indexed {split_type} dataframe.")
            return self.data.iloc[indices]
        
        # Combine features and targets if we have both
        if not result_parts:
            self.logger.error("No data available for the specified configuration.")
            return None
        
        if len(result_parts) == 1:
            # Only one type of data (features or targets)
            result = result_parts[0]
        else:
            # Both features and targets - merge them
            # We need to handle potential index mismatches
            common_indices = result_parts[0].index
            for df in result_parts[1:]:
                common_indices = common_indices.intersection(df.index)
            
            # Check if we lost rows due to index mismatches
            if len(common_indices) < len(result_parts[0]) or len(common_indices) < len(result_parts[1]):
                self.logger.warning(f"Lost {len(result_parts[0]) - len(common_indices)} rows due to index mismatches between features and targets.")
            
            # Filter all dataframes to common indices and concatenate
            filtered_parts = [df.loc[common_indices] for df in result_parts]
            result = pd.concat(filtered_parts, axis=1)
        
        if not include_features and include_target is not None:
            self.logger.info(f"Created {split_type} dataframe with {len(result)} rows only target ({include_target}) column.")
        else:
            self.logger.info(f"Created {split_type} dataframe with {len(result)} rows and {len(result.columns)} columns.")

        # If a Dmatrix was requested
        if return_DMatrix:
            try:
                if len(result_parts) == 1:
                    dmatrix = DMatrix(result, label= None, missing= missing)
                    self.logger.info( f"Created {split_type} DMatrix with {dmatrix.num_row()} rows and {dmatrix.num_col()} features" )
                else:
                    dmatrix = DMatrix(result_parts[0], label = result_parts[1], missing= missing)
                    self.logger.info(f"Created {split_type} DMatrix with {dmatrix.num_row()} rows and {dmatrix.num_col()} features with labels")

                return dmatrix

            except Exception as e:
                self.logger.error(f"Error creating DMatrix: {str(e)}")
                return None

        return result

    def get_train_val_test_sizes(self):
        """
        Get the sizes and percentages of the train, validation, and test sets.
        
        Returns:
        --------
        dict
            Dictionary with set sizes and percentages.
            Format: {
                'total': int,
                'train': {'size': int, 'percentage': float},
                'validation': {'size': int, 'percentage': float}, 
                'test': {'size': int, 'percentage': float}
            }
        """
        if self.data is None:
            self.logger.error("No data loaded.")
            return None
        
        # Check if splits exist, create them if they don't
        if not hasattr(self, 'train_indices') or self.train_indices is None:
            self.logger.warning("Data has not been split. Running split_data with default parameters.")
            self.split_data()
        
        # Get sizes
        total = len(self.data)
        train_size = len(self.train_indices)
        val_size = len(self.val_indices)
        test_size = len(self.test_indices)
        
        # Calculate percentages
        train_pct = (train_size / total * 100) if total > 0 else 0
        val_pct = (val_size / total * 100) if total > 0 else 0
        test_pct = (test_size / total * 100) if total > 0 else 0
        
        # Create result dictionary
        result = {
            'total': total,
            'train': {
                'size': train_size,
                'percentage': train_pct
            },
            'validation': {
                'size': val_size,
                'percentage': val_pct
            },
            'test': {
                'size': test_size,
                'percentage': test_pct
            }
        }
        
        self.logger.info(f"Data split sizes: Train={train_size}, Val={val_size}, Test={test_size}, Total={total}")
        
        return result