# utils.py

#from agnboost.dataset import Catalog
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from xgboostlss import model as xgboostlss_model
from xgboost import DMatrix
import pandas as pd
import numpy as np

def log_message(message, level="INFO"):
    """
    Log a message with the specified level.
    
    Parameters:
    -----------
    message : str
        Message to log.
    level : str
        Log level (INFO, WARNING, ERROR).
    """
    print(f"[{level}] {message}")


def validate_param_grid(param_grid):
    """
    Validate that the parameter grid is correctly formatted.
    
    Expected format:
    {
        "param_name": ["type", type_specific_value],
        ...
    }
    
    Where "type" is one of:
    - "none": type_specific_value should be a list of specific values
    - "int": type_specific_value should be a dict with "low", "high", "log" keys
    - "float": type_specific_value should be a dict with "low", "high", "log" keys
    - "categorical": type_specific_value should be a list of possible values
    
    Parameters:
    -----------
    param_grid : dict
        Dictionary of parameter grid specifications
        
    Returns:
    --------
    tuple
        (is_valid, error_message) - (True, None) if valid, (False, error_msg) if invalid
    """
    # Check if param_grid is a dictionary
    if not isinstance(param_grid, dict):
        return False, f"param_grid must be a dictionary, got {type(param_grid).__name__}"
    
    # Valid types
    valid_types = ["none", "int", "float", "categorical"]
    
    # Validate each parameter
    for param_name, param_spec in param_grid.items():
        # Check if param_spec is a list
        if not isinstance(param_spec, list):
            return False, f"Parameter '{param_name}' specification must be a list, got {type(param_spec).__name__}"
        
        # Check if param_spec has 2 elements
        if len(param_spec) != 2:
            return False, f"Parameter '{param_name}' specification must have exactly 2 elements, got {len(param_spec)}"
        
        param_type, param_value = param_spec
        
        # Check if param_type is a string
        if not isinstance(param_type, str):
            return False, f"Parameter '{param_name}' type must be a string, got {type(param_type).__name__}"
        
        # Check if param_type is valid
        if param_type not in valid_types:
            return False, f"Parameter '{param_name}' has invalid type '{param_type}'. Valid types: {valid_types}"
        
        # Validate based on type
        if param_type == "none":
            # Should be a list of values
            if not isinstance(param_value, list):
                return False, f"Parameter '{param_name}' of type 'none' must have a list of values, got {type(param_value).__name__}"
            
        elif param_type in ["int", "float"]:
            # Should be a dict with "low", "high", "log" keys
            if not isinstance(param_value, dict):
                return False, f"Parameter '{param_name}' of type '{param_type}' must have a dict value, got {type(param_value).__name__}"
            
            # Check required keys
            required_keys = ["low", "high", "log"]
            missing_keys = [key for key in required_keys if key not in param_value]
            if missing_keys:
                return False, f"Parameter '{param_name}' is missing required keys: {missing_keys}"
            
            # Check value types
            if param_type == "int":
                if not (isinstance(param_value["low"], int) and isinstance(param_value["high"], int)):
                    return False, f"Parameter '{param_name}' of type 'int' must have integer 'low' and 'high' values"
            
            # Check low < high
            if param_value["low"] >= param_value["high"]:
                return False, f"Parameter '{param_name}' must have 'low' < 'high', got {param_value['low']} >= {param_value['high']}"
            
            # Check log is boolean
            if not isinstance(param_value["log"], bool):
                return False, f"Parameter '{param_name}' must have boolean 'log' value, got {type(param_value['log']).__name__}"
            
        elif param_type == "categorical":
            # Should be a list of possible values
            if not isinstance(param_value, list):
                return False, f"Parameter '{param_name}' of type 'categorical' must have a list of values, got {type(param_value).__name__}"
            
            if len(param_value) == 0:
                return False, f"Parameter '{param_name}' of type 'categorical' must have at least one value"
    
    # All checks passed
    return True, None



def validate_param_dict(params):
    """
    Validate that the passed params are a dictionary of single values
    
    Expected format:
    {
        "param_name": value,
        ...
    }

    Parameters:
    -----------
    params : dict
        Dictionary of parameter grid specifications
        
    Returns:
    --------
    tuple
        (is_valid, error_message) - (True, None) if valid, (False, error_msg) if invalid
    """

    # Verify that params is a dictionary of single values
    if not isinstance(params, dict):
        return False, f"params must be a dictionary, got {type(params).__name__}"
    
    # Check that each parameter value is a simple type, not a list or dictionary
    invalid_params = {}
    for param_name, param_value in params.items():
        # Allow only simple types: numbers, strings, and booleans
        if isinstance(param_value, (list, dict, tuple, set)):
            invalid_params[param_name] = type(param_value).__name__
    
    if invalid_params:
        error_msg = "Invalid parameter values found. XGBoost parameters must be single values, not lists or dictionaries:"
        for param_name, type_name in invalid_params.items():
            error_msg += f"\n  - '{param_name}': {type_name}"

        return False, error_msg

    # All checks passed
    return True, None


def BetaMean(alpha, beta):
    return alpha/(alpha+beta)

def BetaVar(alpha, beta):
    var = alpha*beta / ( ((alpha + beta)**2)*(alpha + beta + 1) )
    return var

def BetaStd(alpha, beta):
    var = alpha*beta / ( ((alpha + beta)**2)*(alpha + beta + 1) )
    std = np.sqrt(var)
    return std

def ZABetaMean(alpha, beta, gate):
    return (1 - gate) * BetaMean(alpha, beta)

def ZABetaVar(alpha, beta, gate):
    var = (1 - gate) * (BetaVar(alpha, beta) + (alpha**2)*gate/((alpha+beta)**2))
    return var

def ZABetaStd(alpha, beta , gate):
    std = np.sqrt(ZABetaVar(alpha, beta , gate))
    return std


def predict_mean_ZABeta(model: xgboostlss_model,                     
                data: DMatrix,
                seed: int = 123):
    pred_params = model.predict(data, pred_type="parameters", seed = seed)
    mu_array = np.array( [ZABetaMean(alpha = pred_params['concentration1'].iloc[j], 
                                   beta= pred_params['concentration0'].iloc[j],
                                    gate = pred_params['gate'].iloc[j]) 
                                    for j in range( data.num_row() )] )
    return mu_array

def predict_mean_fromparams_ZABeta(params: pd.DataFrame, seed: int = 123):
    mu_array = np.array( [ZABetaMean(alpha = params['concentration1'].iloc[j], 
                                           beta= params['concentration0'].iloc[j],
                                            gate = params['gate'].iloc[j]) 
                                        for j in range( params.shape[0] )] )
    return mu_array


def predict_std_fromparams_ZABeta(params: pd.DataFrame,seed: int = 123):
    std_array = np.array( [ZABetaStd(alpha = params['concentration1'].iloc[j], 
                                           beta= params['concentration0'].iloc[j],
                                            gate = params['gate'].iloc[j]) 
                                        for j in range( params.shape[0] )] )
    return std_array


def predict_mean( trained_model: xgboostlss_model,
                dist_name : str,
                data : DMatrix,
                seed : int = 123
                    ):

    pred_params = trained_model.predict(data = data, pred_type = "parameters", seed = seed)

    match dist_name:
        case "ZABeta":
            mu_array = predict_mean_fromparams_ZABeta( params = pred_params,  seed = seed)

    return mu_array


def predict_std( trained_model: xgboostlss_model,
                dist_name : str,
                data : DMatrix,
                seed : int = 123
                    ):

    pred_params = trained_model.predict(data = data, pred_type = "parameters", seed = seed)

    match dist_name:
        case "ZABeta":
            std_array = predict_std_fromparams_ZABeta( params = pred_params,  seed = seed)

    return std_array



