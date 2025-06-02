from pathlib import Path

import logging
from tqdm import tqdm


# agnboost.py
from agnboost.dataset import Catalog
from datetime import datetime
from xgboost import DMatrix, Booster
import functools
from math import comb
import numpy as np
import os
import glob
import optuna
import pandas as pd
import gzip
import pickle
from agnboost.utils import * #log_message, preprocess_data
import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from xgboostlss import model as xgboostlss_model
from xgboostlss.distributions.distribution_utils import DistributionClass
from xgboostlss.distributions.ZABeta import *
from xgboostlss.model import *
import matplotlib.pyplot as plt
#from xgboostlss.distributions import *
#from xgboostlss.model import XGBoostLSS


class AGNBoost:
    """
    A machine learning framework for simultaneous AGN identification and photometric redshift estimation.
    
    AGNBoost utilizes the XGBoostLSS algorithm to predict both the fraction of mid-IR 
    3-30 μm emission attributable to an AGN power law (fracAGN) and photometric redshift 
    from JWST NIRCam and MIRI photometry. The framework analyzes 121 input features 
    derived from multi-band infrared observations, including original magnitudes, color 
    indices, and their squared values to enhance AGN discrimination. The computational 
    efficiency and scalability make AGNBoost well-suited for large-scale JWST surveys 
    requiring rapid AGN identification and redshift estimation.
    
    Attributes:
        models_dir (str): Directory path for storing/loading trained models.
        models (dict): Dictionary containing trained XGBoostLSS models for each target variable.
        model_info (dict): Dictionary storing metadata and performance information for each model.
        feature_names (list): Feature (column) names used for model input.
        target_variables (dict): Dictionary mapping target variable names to their probability distributions.
        logger (logging.Logger): Logger instance for this AGNBoost object.
        ALLOWED_DISTS (list): Class-level list of allowed probability distributions ['ZABeta', 'Beta'].
    
    Examples:
        Basic usage with default settings:
        
        ```python
        from agnboost import AGNBoost
        
        # Initialize with defaults
        agnboost = AGNBoost()
        
        # Train models (assuming you have training data)
        agnboost.train(training_data)
        
        # Make predictions
        predictions = agnboost.predict(test_data)
        ```
        
        Initialize with custom target variables:
        
        ```python
        targets = {'fagn': 'ZABeta', 'redshift': 'Beta'}
        agnboost = AGNBoost(target_variables=targets)
        ```
        
        Initialize with custom features:
        
        ```python
        features = ['F115W', 'F150W', 'F200W', 'F277W', 'F770W', 'F1000W', 'F1130W']
        agnboost = AGNBoost(feature_names=features)
        ```
    """
    # Class-level logger
    logger = logging.getLogger('AGNBoost.AGNBoost')

    ALLOWED_DISTS = ['ZABeta', 'Beta']

    
    def __init__(self, feature_names=None, target_variables=None, logger=None):
        """
        Initialize the AGNBoost object for AGN identification and redshift estimation.
        
        Args:
            feature_names (list, optional): Feature (column) names to use for model input.
                If None, default features will be used based on JWST NIRCam and MIRI bands.
                Defaults to None.
            target_variables (dict, optional): Dictionary mapping target variable names to 
                their probability distributions. Keys should be target names (e.g., 'fagn', 
                'z_transformed') and values should be distribution types from ALLOWED_DISTS.
                If None, defaults to {'fagn': 'ZABeta', 'z_transformed': 'Beta'}.
            logger (logging.Logger, optional): Custom logger instance to use for logging.
                If None, uses the class-level logger. Defaults to None.
                
        Raises:
            TypeError: If target_variables is not a dictionary.
            ValueError: If target_variables contains invalid distribution values.
            
        Examples:
            Initialize with default settings:
            ```python
            agnboost = AGNBoost()
            ```
            
            Initialize with custom target variables:
            ```python
            targets = {'fagn': 'ZABeta', 'redshift': 'Beta'}
            agnboost = AGNBoost(target_variables=targets)
            ```
        """
        # Set up instance logger (use provided logger or class logger)
        self.logger = logger or self.__class__.logger
        
        # Configure model directory
        self.models_dir = os.environ.get('AGNBOOST_MODELS_DIR', 'models/')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Set attributes with provided or default values
        self.feature_names = feature_names

        # Validate target_variables if provided
        if target_variables is not None:
            # Check that target_variables is a dictionary
            if not isinstance(target_variables, dict):
                self.logger.error(f"target_variables must be a dictionary, got {type(target_variables).__name__}")
                raise TypeError(f"target_variables must be a dictionary, got {type(target_variables).__name__}")
            
            # Check that all values are in the allowed list
            invalid_values = [value for value in target_variables.values() 
                              if value not in self.__class__.ALLOWED_DISTS]
            
            if invalid_values:
                self.logger.error(f"Invalid target variable values: {invalid_values}. "
                                 f"Allowed values are: {ALLOWED_DISTS}")
                raise ValueError(f"Invalid target variable values: {invalid_values}. "
                               f"Allowed values are: {ALLOWED_DISTS}")
            
            self.logger.info(f"Validated target_variables dictionary with {len(target_variables)} entries")
        
        self.target_variables = target_variables if target_variables is not None else {'fagn' : 'ZABeta', 'z_transformed' : 'Beta'}
        
        # Initialize models dictionary
        self.models = {target_variables: None for target_variables in self.target_variables.keys()}
        #self.dists =  self.target_variables.values()
        self.model_info = {model_name: {} for model_name in self.models.keys()}

    def get_models(self):
        """
        Get the feature dataframe, creating it if it doesn't exist.
        
        Returns:
            dict: Dictionary containing all models.
                Model values will be None if a model has not been loaded or trained.
        """
        return self.models

    def tune_model(self, model_name, param_grid, dtune, split_type = 'train', max_minutes=10, nfold=2, early_stopping_rounds=100):
        """
        Tune hyperparameters for the specified model with Optuna.
        
        Args:
            model_name (str): Name of the model to tune. 
                Must be in self.model_names.
            param_grid (dict): Dictionary of hyperparameter ranges to search.
                Example: 
                ```
                param_grid = {
                    'max_depth': ["int", {"low": 1,      "high": 10,    "log": False}], 
                    'eta': ["float", {"low": 1e-5,   "high": 1,     "log": True}]
                }
                ```
            dtune (xgboost.DMatrix or pd.DataFrame or Catalog): Data to use for tuning. 
                If DataFrame or Catalog, will be converted to DMatrix.
            max_minutes (int, default=10): Maximum duration for tuning in minutes.
            nfold (int, default=2): Number of cross-validation folds.
            early_stopping_rounds (int, default=100): Number of rounds without improvement before early stopping.
            
        Returns:
            dict or None: Dictionary containing best parameters and tuning metrics. 
                Returns None if error is encountered.
        """

        # Verify model name is valid
        if model_name not in self.models.keys():
            self.logger.error(f"Invalid model name: {model_name}. Must be one of {self.models.keys()}")
            return None

        # Validate split_type
        valid_split_types = {'train', 'val', 'validation', 'test', 'trainval'}
        if split_type not in valid_split_types:
            self.logger.error(f"Invalid split_type: '{split_type}'. Must be one of {valid_split_types}")
            return None
        
        self.logger.info(f"Starting hyperparameter tuning for model: {model_name}")
        self.logger.info(f"Max duration: {max_minutes} minutes, CV folds: {nfold}, Early stopping: {early_stopping_rounds} rounds")
        
        # Process input data based on type
        if isinstance(dtune, DMatrix):
            train_dmatrix = dtune
            self.logger.info("Using provided DMatrix for tuning")
        else:
            # Handle  Catalog
            if isinstance(dtune, Catalog):
            #if hasattr(dtune, 'get_features') and callable(getattr(dtune, 'get_features')):
                self.logger.warning(f"Catalog object passsed. Taking the features and labels of the {split_type} set stored in the passed Catalog.")

                # It's a Catalog, extract its features. By default, we take the features of the training set
                features_df = dtune.get_split_df(split_type = split_type, include_features = True, include_target = None, return_DMatrix = False)
                targets_df = dtune.get_split_df(split_type = split_type, include_features = False, include_target = model_name, return_DMatrix = False)
                self.logger.info(f"Using features from Catalog object for tuning: {features_df.shape}")

            else:
                self.logger.error(f"Unsupported data type for tuning: {type(dtune)}")
                return None
            
            try:
                train_dmatrix = DMatrix(data = features_df, label=targets_df)
                self.logger.info(f"Created DMatrix with {features_df.shape[0]} rows and {features_df.shape[1]} features")
            except Exception as e:
                self.logger.error(f"Error creating DMatrix: {str(e)}")
                return None
        
        # Validate param_grid format
        is_valid, error_message = validate_param_grid(param_grid)
        if not is_valid:
            self.logger.error(f"Invalid param_grid: {error_message}")
            return None

        # Start tuning process
        try:
            # Replace this call with your actual custom tuning function
            study = self.custom_tune_function( hp_dict = param_grid, 
                                            dist = self.target_variables[model_name],
                                            dtrain = train_dmatrix, 
                                            max_minutes = max_minutes, 
                                            nfold = nfold, 
                                            early_stopping_rounds = early_stopping_rounds
                                        )

            # Save tuning results to model info
            if study is not None:
                self.model_info[model_name]['tuned_params'] = study.best_params
                self.model_info[model_name]['tuning_score'] = study.best_value
                self.model_info[model_name]['tuning_timestamp'] = datetime.now().isoformat()
                
                self.logger.info(f"Saved tuned parameters for model: {model_name}")
                self.logger.info(f"Best parameters: {study.best_params}")
                
                return study.best_params
            else:
                self.logger.error("Tuning failed to produce valid results")
                return None
            
        except Exception as e:
            self.logger.error(f"Error during hyperparameter tuning: {str(e)}")
            return None
  

    # Retrive XGBoostLSS dist object
    @staticmethod
    def retrieve_dist(dist_name, response_fn = None, stab = None, loss_fn = None, params = None):
        """
        Retrieve a configured XGBoostLSS model.
        
        Creates and returns an XGBoostLSS object with the specified probability distribution
        and configuration parameters. 
        
        Args:
            dist_name (str): Name of the probability distribution to use. Must be one of
                the values in ALLOWED_DISTS ['ZABeta', 'Beta'].
            response_fn (str, optional): Response function for the distribution parameters.
                Common options include 'exp', 'softplus'. If None and params is provided,
                uses value from params. Defaults to None.
            stab (str, optional): Stabilization method for gradients and Hessians to improve
                model convergence. Options include 'None', 'L2', 'MAD'. If None and params
                is provided, uses value from params. Defaults to None.
            loss_fn (str, optional): Loss function to use for training. Typically 'nll' 
                (negative log-likelihood). If None and params is provided, uses value from
                params. Defaults to None.
            params (dict, optional): Dictionary containing distribution parameters. If provided,
                overrides individual parameter arguments. Expected keys: 'stabilization',
                'response_fn', 'loss_fn'. Defaults to None.
                
        Returns:
            XGBoostLSS: Configured XGBoostLSS model with the specified distribution, or None
                if an error occurs during object creation.
                
        Raises:
            Exception: If there's an error creating the XGBoostLSS object, logs the error
                and returns None.
                
        Examples:
            Create a ZABeta distribution with default parameters:
            ```python
            dist_obj = AGNBoost.retrieve_dist("ZABeta")
            ```
            
            Create a Beta distribution with custom parameters:
            ```python
            dist_obj = AGNBoost.retrieve_dist("Beta", response_fn="softplus", stab="L2")
            ```
            
            Create using a parameters dictionary:
            ```python
            params = {'response_fn': 'softplus', 'stabilization': 'None', 'loss_fn': 'nll'}
            dist_obj = AGNBoost.retrieve_dist("ZABeta", params=params)
            ```
        """
        if params is not None:
            stab = params.get('stabilization', "None")
            response_fn = params.get('response_fn', "exp")
            loss_fn =  params.get('loss_fn', 'nll')

        try:
            match dist_name:
                case "ZABeta":
                    return XGBoostLSS( ZABeta(response_fn=response_fn, stabilization= stab, loss_fn = loss_fn) )
                case "Beta":
                    return XGBoostLSS( Beta(response_fn=response_fn, stabilization= stab, loss_fn = loss_fn) )

        except Exception as e:
            self.logger.error(f"Error retrieving XGBoostLSS object: {str(e)}")
            return None


    # Define custom objective function for the model
    @staticmethod
    def custom_objective( trial, 
                          hp_dict, 
                          dist, 
                          d_train, 
                          nfold, 
                          num_boost_round, 
                          early_stopping_rounds, 
                          seed):
        """
        Custom objective function for Optuna hyperparameter optimization.
        
        Defines the optimization objective for tuning hyperparameters using Optuna.
        The function constructs hyperparameter combinations based on the provided search space,
        trains models using cross-validation, and returns the validation loss for optimization.
        Supports automatic pruning of unpromising trials and handles various parameter types
        including categorical, float, and integer parameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object used for suggesting hyperparameter values
                and managing the optimization process.
            hp_dict (dict): Dictionary defining the hyperparameter search space. Each key is a
                parameter name, and values are tuples of (param_type, param_constraints) where
                param_type is one of ['categorical', 'float', 'int', 'none'] and param_constraints
                define the search range or options.
            dist (str): Name of the probability distribution to use. Must be one of the values
                in ALLOWED_DISTS ['ZABeta', 'Beta'].
            d_train (xgboost.DMatrix): Training data in XGBoost DMatrix format for cross-validation.
            nfold (int): Number of folds for cross-validation during hyperparameter optimization.
            num_boost_round (int): Maximum number of boosting rounds to train.
            early_stopping_rounds (int): Number of rounds with no improvement after which training
                will be stopped early.
            seed (int): Random seed for reproducible results across trials.
                
        Returns:
            float: The minimum cross-validation test score (loss) achieved by the model with
                the suggested hyperparameters. Lower values indicate better performance.
            
        Notes:
            The function automatically handles special XGBoostLSS parameters ('stabilization',
            'response_fn', 'loss_fn') separately from standard XGBoost parameters to avoid
            warnings during model training. It also implements Optuna's XGBoost pruning callback
            to terminate unpromising trials early, improving optimization efficiency.
        """

        hyper_params = {}
        for param_name, param_value in hp_dict.items():

            param_type = param_value[0]
                
            if param_type == "categorical" or param_type == "none":
                hyper_params.update({param_name: trial.suggest_categorical(param_name, param_value[1])})

            elif param_type == "float":
                param_constraints = param_value[1]
                param_low = param_constraints["low"]
                param_high = param_constraints["high"]
                param_log = param_constraints["log"]
                hyper_params.update(
                    {param_name: trial.suggest_float(param_name,
                                                     low=param_low,
                                                     high=param_high,
                                                     log=param_log
                                                     )
                     })

            elif param_type == "int":
                param_constraints = param_value[1]
                param_low = param_constraints["low"]
                param_high = param_constraints["high"]
                param_log = param_constraints["log"]
                hyper_params.update(
                    {param_name: trial.suggest_int(param_name,
                                                   low=param_low,
                                                   high=param_high,
                                                   log=param_log
                                                   )
                     })
        stab = hyper_params.get('stabilization', "None")
        response_fn = hyper_params.get('response_fn', "exp")
        loss_fn =  hyper_params.get('loss_fn', 'nll')

        # Remove these from the dict so the execution doesn't raise warnings
        hyper_params.pop('stabilization', None)
        hyper_params.pop('response_fn', None)
        hyper_params.pop('loss_fn', None)

        # Make Model
        xgblss_m = AGNBoost.retrieve_dist(dist_name = dist, response_fn = response_fn, stab = stab, loss_fn = loss_fn)

        # Add pruning
        pruning_callback = [optuna.integration.XGBoostPruningCallback(trial, f"test-{xgblss_m.dist.loss_fn}")]

        tic = time.time()
        xgblss_param_tuning = xgblss_m.cv(params=hyper_params,
                                      dtrain= d_train,
                                      num_boost_round=num_boost_round,
                                      nfold=nfold,
                                      early_stopping_rounds=early_stopping_rounds,
                                      callbacks=pruning_callback,
                                      seed=seed,
                                         ) #verbose_eval=False)
                                      
        print(f'{time.time() - tic:.1f} seconds')
        
        # Add the optimal number of rounds
        opt_rounds = xgblss_param_tuning[f"test-{xgblss_m.dist.loss_fn}-mean"].idxmin() + 1
        trial.set_user_attr("opt_round", int(opt_rounds))
        
        # Extract the best score
        test_score = np.min(xgblss_param_tuning[f"test-{xgblss_m.dist.loss_fn}-mean"])
        # Replace -inf with 1e8 (to avoid -inf in the log)
        test_score = np.where(test_score == float('-inf'), float(1e8), test_score)
        
        return test_score  

    # Define custom tune function
    @staticmethod
    def custom_tune_function(hp_dict, 
                            dist,
                            dtrain,
                            n_trials = None, 
                            max_minutes = 10,
                            num_boost_round=500,
                            nfold=10,
                            early_stopping_rounds=20,
                            seed = 123,
                            metrics = None
                          ):
        """
        Perform hyperparameter optimization for AGNBoost models using Optuna. This offers the same 
        functionality of XGBoostLSS's built-in hyperparameter tuning, but also allows the tuning of 
        distributional parameters.
        
        Automatically selects the optimal hyperparameter tuning strategy based on the search
        space characteristics. Uses grid search for purely categorical parameters and Bayesian
        optimization (TPE) for mixed or continuous parameter spaces. Implements pruning to
        terminate unpromising trials early and provides comprehensive optimization results.
        
        Args:
            hp_dict (dict): Dictionary defining the hyperparameter search space. Each key is a
                parameter name, and values are tuples of (param_type, param_constraints) where
                param_type is one of ['categorical', 'float', 'int', 'none'].
            dist (str): Name of the probability distribution to use. Must be one of the values
                in ALLOWED_DISTS ['ZABeta', 'Beta'].
            dtrain (xgboost.DMatrix): Training data in XGBoost DMatrix format for cross-validation.
            n_trials (int, optional): Maximum number of optimization trials to run. If None and
                all parameters are categorical, uses grid search to try all combinations.
                Defaults to None.
            max_minutes (int, optional): Maximum time limit for optimization in minutes. 
                Optimization will stop after this duration regardless of n_trials.
                Defaults to 10.
            num_boost_round (int, optional): Maximum number of boosting rounds for each trial.
                Defaults to 500.
            nfold (int, optional): Number of folds for cross-validation during optimization.
                Defaults to 10.
            early_stopping_rounds (int, optional): Number of rounds with no improvement after
                which training will be stopped early. Defaults to 20.
            seed (int, optional): Random seed for reproducible optimization results.
                Defaults to 123.
            metrics (optional): Additional metrics to track during optimization. Currently unused.
                Defaults to None.
                
        Returns:
            optuna.Study: Completed Optuna study object containing optimization results,
                including best parameters, trial history, and performance metrics.
            
        Notes:
            The function automatically determines the optimization strategy: if all parameters
            are categorical, it uses grid search to exhaustively try all combinations. Otherwise,
            it uses Tree-structured Parzen Estimator (TPE) for Bayesian optimization. Median
            pruning is applied to terminate unpromising trials early, improving efficiency.
            The optimal number of boosting rounds is automatically determined and included
            in the returned results.
        """

        all_cat = True
        for entry in hp_dict.values():
            if entry[0] not in ['categorical', 'none']:
                all_cat = False
               
        # If all parameters are categorical, we use the grid sampler to try all combinations 
        if all_cat == True:
            hp_entries = [ *hp_dict.values() ]
            hp_ranges = [ entry[1] for entry in hp_entries]

            hp_keys = [*hp_dict]
            
            search_space = {hp_keys[i]: hp_ranges[i] for i in range(len(hp_keys))}
            sampler = optuna.samplers.GridSampler(search_space)

            num_count = np.array([len(hp_ranges[i]) for i in range(len(hp_ranges))])
            n_trials = np.prod(num_count)

        # Otherwise, take the Bayesian approach
        else:
            sampler = optuna.samplers.TPESampler(multivariate = True)

        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
        study = optuna.create_study(direction = 'minimize', pruner = pruner, sampler = sampler)


        func = lambda trial: AGNBoost.custom_objective(trial, 
                                       hp_dict = hp_dict, 
                                       dist = dist,
                                       d_train = dtrain, 
                                       nfold = nfold, 
                                       num_boost_round = num_boost_round, 
                                       early_stopping_rounds = early_stopping_rounds, 
                                       seed = seed
                                       )

        study.optimize(func, n_trials=n_trials, timeout= 60 * max_minutes, show_progress_bar= True)
            
        print("\nHyper-Parameter Optimization successfully finished.")
        print("Number of finished trials: ", len(study.trials))


        print("\tBest trial:")
        opt_param = study.best_trial
         
        # Add optimal stopping round
        opt_param.params["opt_rounds"] = study.trials_dataframe()["user_attrs_opt_round"][
            study.trials_dataframe()["value"].idxmin()]
        opt_param.params["opt_rounds"] = int(opt_param.params["opt_rounds"])

        print(f"\t\tValue: {opt_param.value}")
        print("\t\tParams:")
        for key, value in opt_param.params.items():
            print(f"\t\t\t{key}: {value}")

        return study

    @staticmethod
    def plot_eval(evals, catalog, best_iter = None):
        """
        Plot the training and validation loss curves.
        
        Args:
            evals (dict): Dict of model evaluations.
            catalog (Catalog instance): Catalog object used for training.
                Used to scale the training/validation curves according to the training/validation sizes.

        Returns:
            matplotlib.pyplot.axis: The figure axis. 
        """
        train_len = len(catalog.train_indices)
        val_len = len(catalog.val_indices)

        train_nll = np.array(evals['train']['nll']) * (val_len/train_len)
        val_nll = np.array(evals['validation']['nll'])

        fig, ax = plt.subplots()
        ax.plot(train_nll, label = 'train-nll')
        ax.plot(val_nll, label = 'validation-nll')

        if best_iter is not None:
            ax.axvline( x= best_iter, ymin = 0, ymax = 1, c = 'k', ls = '--', lw =1, label = 'best iteation')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('NLL')
        ax.legend(loc='upper left')


        return ax

    def train_model(self, model_name, dtrain, dval=None, params=None, split_type='train', 
                    num_boost_round=1000, early_stopping_rounds=None, 
                    verbose_eval=False, custom_objective=None, custom_metric=None):
        """
        Train an AGNBoost model with the given parameters.
        
        Args:
            model_name (str): Name of the model to train. 
                Must be in self.model_names.
            dtrain (xgboost.DMatrix or Catalog): Training data. 
                If a Catalog, will create a DMatrix using the specified split.
            dval (xgboost.DMatrix, bool, default=None): Validation data. 
                If a DMatrix, use directly.
                If True and dtrain is a Catalog, create a validation DMatrix from the Catalog.
                If False or None, no validation data is used.
            params (dict, default=None): Dict of model hyperparameters to use
                If None, uses saved tuned_params from model_info if available,
                otherwise uses default parameters.
            split_type (str, default='train'): Which data split to use if dtrain is a Catalog. 
                Options: 'train', 'val', 'test'.
            num_boost_round (int, default=1000): Number of boosting rounds.
            early_stopping_rounds (int or None, default=50): Number of rounds without improvement before early stopping.
                If None, no early stopping is used.
            verbose_eval (int or bool, default=100): Controls XGBoost's logging frequency. 
                If True, prints every round.
                If int, prints every verbose_eval rounds. 
                If False, no logging.
            custom_objective (callable, default=None):  Custom objective function for XGBoost.
            custom_metric (callable, default=None): Custom evaluation metric for XGBoost.
            
        Returns:
            tuple: (trained_model, training_results)

        """
        
        # Verify model name is valid
        if model_name not in self.models.keys():
            self.logger.error(f"Invalid model name: {model_name}. Must be one of {self.models.keys()}")
            return None, None

        # Validate split_type
        valid_split_types = {'train', 'val', 'validation', 'test', 'trainval'}
        if split_type not in valid_split_types:
            self.logger.error(f"Invalid split_type: '{split_type}'. Must be one of {valid_split_types}")
            return None, None

        
        # Process training data
        if isinstance(dtrain, Catalog):  # It's a Catalog
            self.logger.warning(f"Catalog object passsed. Taking the features and labels of the {split_type} set stored in the passed Catalog.")

            # It's a Catalog, extract its features. By default, we take the features of the training set
            features_df = dtrain.get_split_df(split_type = split_type, include_features = True, include_target = None, return_DMatrix = False)
            targets_df = dtrain.get_split_df(split_type = split_type, include_features = False, include_target = model_name, return_DMatrix = False)
            self.logger.info(f"Using features from Catalog object for training: {features_df.shape}")
            
            try:
                train_dmatrix = DMatrix(data = features_df, label=targets_df)
                self.logger.info(f"Created DMatrix with {features_df.shape[0]} rows and {features_df.shape[1]} features (Training)")
            
            except Exception as e:
                self.logger.error(f"Error creating DMatrix (Training): {str(e)}")
                return None, None
        
        elif isinstance(dtrain, DMatrix):  # It's already a DMatrix
            self.logger.info("Using provided training DMatrix")
            train_dmatrix = dtrain
        else:
            self.logger.error(f"dtrain must be a Catalog or xgboost.DMatrix, got {type(dtrain).__name__}")
            return None, None
        
        # Process validation data
        dval_dmatrix = None
        if dval is not None:
            if isinstance(dval, bool) and dval:  # True, and dtrain is a Catalog
                if not isinstance(dtrain, Catalog):
                    self.logger.error("Cannot create validation DMatrix: dtrain is not a Catalog")
                    return None, None
                    
                self.logger.info("Creating validation DMatrix from Catalog")

                features_df = dtrain.get_split_df(split_type = 'val', include_features = True, include_target = None, return_DMatrix = False)
                targets_df = dtrain.get_split_df(split_type = 'val', include_features = False, include_target = model_name, return_DMatrix = False)

                try:
                    val_dmatrix = DMatrix(data = features_df, label=targets_df)
                    self.logger.info(f"Created DMatrix with {features_df.shape[0]} rows and {features_df.shape[1]} features (Validation)")
            
                except Exception as e:
                    self.logger.error(f"Error creating DMatrix (Validation): {str(e)}")
                    return None, None

            elif isinstance(dval, DMatrix):  # It's already a DMatrix
                self.logger.info("Using provided validation DMatrix")
                val_dmatrix = dval

            elif not isinstance(dval, bool):  # It's neither a boolean nor a DMatrix
                self.logger.error(f"dval must be a bool or xgboost.DMatrix, got {type(dval).__name__}")
                return None, None
        

        # Set up parameters
        if params is None:
            # Try to use tuned parameters
            if 'tuned_params' in self.model_info.get(model_name, {}):
                self.logger.info(f"Using tuned parameters for model {model_name}")
                params = self.model_info[model_name]['tuned_params']
            else:
                self.logger.Error("No passed or saved parameters to use.")
                return None, None

        # If params are passed, make sure they are formatted correctly
        else:
            # Validate params format
            is_valid, error_message = validate_param_dict(params)
            if not is_valid:
                self.logger.error(f"Invalid param_grid: {error_message}")
                return None, None
        
        # Prepare evaluation list
        evals = [(train_dmatrix, 'train')]
        if val_dmatrix is not None:
            evals = [(train_dmatrix, 'train')]
            evals.append((val_dmatrix, 'validation'))
            self.logger.info("Including validation data in training")
        
        # Train the model
        self.logger.info(f"Training model {model_name} with {num_boost_round} boosting rounds")
        self.logger.info(f"Parameters: {params}")
        
        try:

            # Construct XGBoostLSS instance
            xgblss_m = AGNBoost.retrieve_dist(dist_name = self.target_variables[model_name], params = params )
            
            params.pop('stabilization', None)
            params.pop('response_fn', None)
            params.pop('loss_fn', None)

            # Store evaluations
            evals_result = {}

            # Train the model
            tic = time.time()
            xgblss_m.train(
                params,
                train_dmatrix,
                num_boost_round=num_boost_round,
                evals=evals,
                evals_result = evals_result,
                early_stopping_rounds = early_stopping_rounds,
                verbose_eval = verbose_eval
            )
            print(f'{time.time() - tic:.1f} seconds to train model')

            # Save model to instance
            self.models[model_name] = xgblss_m
            
            # Update model info
            if model_name not in self.model_info:
                self.model_info[model_name] = {}
                
            self.model_info[model_name].update({
                'training_params': params,
                'num_boost_round': num_boost_round,
                'early_stopping_rounds': early_stopping_rounds,
                'training_timestamp': datetime.now().isoformat(),
                'features': train_dmatrix.feature_names
            })
            


            # Save best iteration if early stopping was used
            if early_stopping_rounds is not None:
                self.model_info[model_name]['best_iteration'] = xgblss_m.booster.best_iteration
                self.logger.info(f"Best iteration: {xgblss_m.booster.best_iteration}")
            
            
            # Save model to disk
            self._save_model( model_name)
            
            self.logger.info(f"Successfully trained model {model_name}")
            return xgblss_m, evals_result
            
        except Exception as e:
            self.logger.error(f"Error training model {model_name}: {str(e)}")
            return None, None

    
    def _save_model(self, model_name, file_name = None):
        """
        Save a trained model and its metadata to disk using pickle serialization.
        
        Serializes the trained XGBoostLSS model along with comprehensive metadata including
        training parameters, tuning results, feature information, and timestamps. The model
        is saved as a pickle file with automatic timestamp-based naming or a custom filename.
        Creates the necessary directory structure if it doesn't exist.
        
        Args:
            model_name (str): Name of the model to save. Must be a key in the models dictionary
                and correspond to a trained model (not None).
            file_name (str, optional): Custom filename for the saved model. If None, generates
                an automatic filename using current timestamp and model name. The '.pkl' 
                extension is automatically added if not present. Defaults to None.
                
        Returns:
            bool: True if the model was saved successfully, False if an error occurred during
                the save process or if the model is invalid.
                
        Raises:
            Exception: Logs any exceptions that occur during the pickle serialization or
                file writing process and returns False.
                    
        Notes:
            The saved metadata includes:
            - Core model object and configuration
            - Feature names used for training
            - Training parameters and performance metrics
            - Hyperparameter tuning results (if available)
            - Target variable distribution information
            - Version and timestamp information
            
            The method creates a subdirectory structure based on the AGNBOOST_MODELS_DIR
            environment variable or defaults to 'models/' directory. All necessary parent
            directories are created automatically if they don't exist.
        """
        # Verify model name is valid
        if model_name not in self.models.keys():
            self.logger.error(f"Invalid model name: {model_name}. Must be one of {self.models.keys()}")
            return None, None
            
        if self.models[model_name] is None:
            self.logger.error(f"Cannot save model '{model_name}': model has not been trained")
            return False

        # Verify that file_name is a string if it is passed
        if file_name is not None and not isinstance(file_name, str):
            self.logger.error(f"Invalid value for overwrite: {overwrite}. Must be bool.")
        

        # Ensure that the sub directory for this model exists. If it does not, create it.
        model_sub_dir = os.environ.get('AGNBOOST_MODELS_DIR', f"models/{model_name}/")
        os.makedirs(model_sub_dir, exist_ok=True)

        # Create the model path using the current timestamp and model name.
        if file_name is None:
            model_path = os.path.join(self.models_dir, datetime.now().strftime("%Y_%m_%d-%p%I_%M_%S") + f"_{model_name}_model.pkl")

        # Use passed filename
        else:
            #If the passed string already has the pkl file extension
            if file_name.endswith(".pkl"):
                model_path = os.path.join(self.models_dir, file_name)
            else:   
                model_path = os.path.join(self.models_dir, f"{file_name}.pkl")
        
        # Gather model metadata
        model_metadata = {
            # Core model components
            'model': self.models[model_name],
            'model_name': model_name,
            
            # Model configuration
            'features': self.feature_names,
            
            # Version and timestamp information
            'version': '1.0',
            'save_timestamp': datetime.now().isoformat(),

            # Training parameters (if available)
            'training_params': self.model_info.get(model_name, {}).get('training_params', {}),
            'num_boost_round': self.model_info.get(model_name, {}).get('num_boost_round', None),
            'early_stopping_rounds': self.model_info.get(model_name, {}).get('early_stopping_rounds', None),
            'best_iteration': self.model_info.get(model_name, {}).get('best_iteration', None),
            'best_score': self.model_info.get(model_name, {}).get('best_score', None),
            
            # Tuning parameters (if available)
            'tuned_params': self.model_info.get(model_name, {}).get('tuned_params', None),
            'tuning_score': self.model_info.get(model_name, {}).get('tuning_score', None),
            'tuning_timestamp': self.model_info.get(model_name, {}).get('tuning_timestamp', None),
            
            # Target variables information
            'target_distribution': self.target_variables.get(model_name, None) if hasattr(self, 'target_variables') else None,
        
        }
        
        # Save model data and metadata to file
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_metadata, f)
            
            self.logger.info(f"Saved model: {model_name}")
            self.logger.info(f"  - Path: {model_path}")
            
            # Update model info with save timestamp
            if model_name not in self.model_info:
                self.model_info[model_name] = {}
            self.model_info[model_name]['save_timestamp'] = model_metadata['save_timestamp']
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {str(e)}")
            return False

    @staticmethod
    def get_available_files(path):
        """
        Get all available files in the provided path.
        
        Args:
            path (str): Path to the directory to check.

        Returns:
            list or None: list of file paths, or None if the path does not exist.
        """
        # Ensure that the sub directory for this model exists. If it does not, return None
        if os.path.exists( path ) is False:
            return None

        # Get a list of all of the files in the model directory
        files = glob.glob(os.path.join(path, "*"))
        return files


    def get_available_models(self, target_name = None):
        """
        Get all available models. Will either get all available models for the provided target,
        or will return a list of the avilable model types (e.g., fracAGN, redshift, etc.).
        
        Args:
            target_name (str, default=None): name of model type to check.
        
        Raises:
            FileNotFoundError: If the directory for the passed target does not yet exist. 

        Returns:
            list: list of available models or model types.
        """
        if target_name is not None:
            model_sub_dir = os.environ.get('AGNBOOST_MODELS_DIR', f"models/{target_name}")
            if not os.path.exists(model_sub_dir):
                raise FileNotFoundError(f'model sub directory for model {target_name} does not exist. Available options are {self.get_available_models()}')

        # If not target is passed, return a list of the available model sub directories.
        else:
            model_sub_dir = os.environ.get('AGNBOOST_MODELS_DIR', 'models/')

        return [os.path.basename(x) for x in self.get_available_files(model_sub_dir)]

    def load_model(self, model_name, file_name = None, overwrite = False):
        """
        Load a pre-trained model and its metadata from into the AGNBoost instance.
        
        Deserializes a previously saved XGBoostLSS model from pickle or gzip format and
        restores it to the AGNBoost instance. Performs comprehensive validation including
        feature compatibility checks, model data structure verification, and metadata
        restoration. Supports automatic selection of the most recent model file if no
        specific filename is provided.
        
        Args:
            model_name (str): Name of the target variable model to load. Must correspond
                to a valid model type (e.g., 'fagn', 'z_transformed').
            file_name (str, optional): Specific filename of the model to load. If None,
                automatically selects the most recently modified file in the model
                subdirectory. Can include or omit the '.pkl' extension. Supports both
                '.pkl' and '.gz' formats. Defaults to None.
            overwrite (bool, optional): Whether to overwrite an already loaded model
                in the AGNBoost instance. If False and a model is already loaded,
                the operation will fail. Defaults to False.
                
        Returns:
            bool: True if the model was loaded successfully, False if an error occurred
                during loading, validation, or if overwrite constraints were violated.
                
        Raises:
            FileNotFoundError: If the model subdirectory doesn't exist or no model files
                are found in the expected directory.
            pickle.UnpicklingError: If the file is corrupted or not a valid pickle file.
            Exception: For other unexpected errors during the loading process.
                
        Examples:
            Load the most recent model file automatically:
            ```python
            # Loads the most recently modified model file for 'fagn'
            success = agnboost.load_model('fagn')
            ```
            
            Load a specific model file:
            ```python
            # Load a specific model file
            success = agnboost.load_model('z_transformed', 'my_redshift_model.pkl')
            ```
            
            Load with filename without extension:
            ```python
            # The .pkl extension is automatically added
            success = agnboost.load_model('fagn', 'final_agn_model')
            ```
            
            Overwrite an existing loaded model:
            ```python
            # Replace currently loaded model with a different one
            success = agnboost.load_model('fagn', 'newer_model.pkl', overwrite=True)
            ```
            
            Load a compressed model:
            ```python
            # Automatically handles gzip compressed files
            success = agnboost.load_model('z_transformed', 'compressed_model.pkl.gz')
            ```
            
        Notes:
            The method performs several validation steps:
            - Verifies input parameter types and values
            - Checks model subdirectory existence  
            - Validates loaded model data structure
            - Compares feature compatibility between saved and current models
            - Ensures model names match expected targets
            
            Feature compatibility is strictly enforced - the loaded model must have been
            trained with identical features to the current AGNBoost instance. Any mismatch
            will result in loading failure with detailed logging of differences.
            
            The method supports both pickle (.pkl) and gzip-compressed (.pkl.gz) formats
            automatically based on file extension. Model metadata including training
            parameters, performance metrics, and timestamps are restored along with the
            model object.
            
            If no specific filename is provided, the method automatically selects the
            most recently modified file in the model subdirectory, making it easy to
            load the latest trained model.
        """

        # Verify model name is valid
        if not isinstance(overwrite, bool):
            self.logger.error(f"Invalid value for overwrite: {overwrite}. Must be bool.")
            return False

        # Verify that file_name is a string if it is passed
        if file_name is not None and not isinstance(file_name, str):
            self.logger.error(f"Invalid value for overwrite: {overwrite}. Must be bool.")
            return False
    
        
        self.logger.info(f"Attempting to load model: {file_name}")
        

        # Ensure that the sub directory for this model exists. If it does not, throw an error because the model cannot be loaded.
        model_sub_dir = os.environ.get('AGNBOOST_MODELS_DIR', f"models/{model_name}/")
        if os.path.exists( model_sub_dir ) is False:
            self.logger.error(f"Model sub directory for {model_name} does not exist. Model has not been trained.")
            raise FileNotFoundError(f"Model sub directory for {model_name} does not exist. Model has not been trained.")

        # If a file_name has been passed, used that
        if file_name is not None:
            model_path = os.path.join(model_sub_dir , file_name)
        else:
            # Get a list of all of the files in the model directory
            files = glob.glob(os.path.join(model_sub_dir, "*"))
            if len(files) > 0:
                # Use the most recently modifed one.
                file_name = os.path.basename(max(files, key=os.path.getmtime))
                model_path = os.path.join(model_sub_dir , file_name)
                self.logger.warning(f"No file_name passed. Using the most recently modified one instead: {file_name}.")
            else:
                raise FileNotFoundError(f"No saved models for target varibale {model_name} exist in directory {model_sub_dir}.")
                
       
        if os.path.exists(model_path):
            try:
                self.logger.info(f"Loading model: {file_name} from {model_path}")
                
                # If it is a gzip file:
                if file_name.endswith('.gz'):
                    with gzip.open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                else:
                    # If it ends with ".pkl"
                    if file_name.endswith(".pkl"):
                        with open(model_path, 'rb') as f:
                            model_data = pickle.load(f)

                    # Otherwise, try appending ".pkl" and opening it with pickle
                    else:
                        try:
                            model_path = os.path.join(model_path , ".pkl")
                            with open(model_path, 'rb') as f:
                                model_data = pickle.load(f)
                        except pickle.UnpicklingError:
                            raise pickle.UnpicklingError(f"Error unpickling the file '{file_name}'. The file might be corrupted or not a valid pickle file.")
                        except Exception as e:
                            raise Exception(f"An unexpected error occurred while loading the file '{file_name}': {e}. Ensure that desired model is actually a pickle.")
                
                # Validate model data structure
                if not isinstance(model_data, dict) or 'model' not in model_data:
                    self.logger.error(f"Invalid model data format for {model_name}. Missing 'model' key.")
                    return False
                
                # Extract model and metadata
                model = model_data['model']
                model_name = model_data['model_name']

                if model_name not in self.models.keys():
                    self.logger.warning(f"Loaded model name: {model_name} not a part of AGNBoost instance. Adding to instance.")

                elif self.models[model_name] is not None and overwrite is False:
                    self.logger.error(f"Cannot load model '{model_name}': model already been trained/loaded and overwrite is False.")
                    return False
                
                # Check feature compatibility
                model_features = model_data.get('features', [])
                if model_features and set(model_features) != set(self.feature_names):
                    self.logger.warning(
                        f"Model '{model_name}' was trained with different features:\n"
                        f"  - Model features ({len(model_features)}): {model_features}\n"
                        f"  - Current features ({len(self.feature_names)}): {self.feature_names}"
                    )
                    
                    # Log specific differences for easier debugging
                    missing_features = [f for f in model_features if f not in self.feature_names]
                    extra_features = [f for f in self.feature_names if f not in model_features]
                    
                    if missing_features:
                        self.logger.warning(f"  - Features in model but not current: {missing_features}")
                    if extra_features:
                        self.logger.warning(f"  - Features in current but not model: {extra_features}")
                    
                    self.logger.error(f"Feature mismatch for model {model_name}.")
                    self.models[model_name] = None
                    return False

                # Store model in instance
                self.models[model_name] = model
                
                # Store model metadata
                self.model_info[model_name] = {
                    k: v for k, v in model_data.items() if k != 'model'
                }
                
                # Log model details
                self.logger.info(f"Successfully loaded model: {model_name}")
                
                # Log training information if available
                if 'training_timestamp' in self.model_info[model_name]:
                    self.logger.info(f"  - Trained: {self.model_info[model_name]['training_timestamp']}")
                
                if 'best_score' in self.model_info[model_name]:
                    self.logger.info(f"  - Best score: {self.model_info[model_name]['best_score']}")
                
                if 'best_iteration' in self.model_info[model_name]:
                    self.logger.info(f"  - Best iteration: {self.model_info[model_name]['best_iteration']}")
     
            except Exception as e:
                self.logger.error(f"Error loading model at {model_path}: {str(e)}")
                return False
        else:
            self.logger.error(f"Model file not found at {model_path}")
            return False
        
        return True

    def dmatrix_validate(self, data, split_use = None):
        """
        Validate and convert input data to XGBoost DMatrix format.
        
        Handles multiple input data formats and converts them to the standardized DMatrix
        format required by XGBoost models. Supports Catalog objects with optional
        data split selection, and validates existing DMatrix objects. Performs feature
        extraction and handles missing value encoding for optimal XGBoost compatibility.
        
        Args:
            data (Catalog or xgboost.DMatrix): Input data to validate and convert.
                If Catalog, extracts features and optionally applies data splitting.
                If DMatrix, validates and passes through unchanged.
            split_use (str, optional): Specific data split to use when data is a Catalog.
                Common values include 'train', 'val', 'test', 'trainval'. If None and
                data is a Catalog, uses the entire feature dataset. Ignored if data
                is already a DMatrix. Defaults to None.
                
        Returns:
            xgboost.DMatrix or None: Validated DMatrix object ready for XGBoost operations,
                or None if validation fails or an error occurs during conversion.
        """
        # If it is a catalog and a split is specified, just use that split
        if isinstance(data, Catalog):  # It's a Catalog
            self.logger.warning(f"Catalog object passsed. Taking the features and labels of the {split_use} set stored in the passed Catalog.")

            #  If a split is specified, just use that split
            if split_use is not None:
                # It's a Catalog, extract its features. By default, we take the features of the training set
                features_df = data.get_split_df(split_type = split_use, include_features = True, include_target = None, return_DMatrix = False)
                self.logger.info(f"Using features from Catalog object {split_use} split: {features_df.shape}")

            # Otherwise just use the whole catalog
            else:
                features_df = data.get_features()
                self.logger.info(f"Using features from entirety of Catalog object: {features_df.shape}")

            try:
                dmatrix = DMatrix(data = features_df, missing = np.nan)
                self.logger.info(f"Created DMatrix with {features_df.shape[0]} rows and {features_df.shape[1]} features (Training)")
            
            except Exception as e:
                self.logger.error(f"Error creating DMatrix (Training): {str(e)}")
                return None

        # It's already a DMatrix
        elif isinstance(data, DMatrix):  
            self.logger.info("Using provided training DMatrix")
            dmatrix = data
        else:
            self.logger.error(f"data must be a Catalog or xgboost.DMatrix, got {type(data).__name__}")
            return None

        return dmatrix

    

    def predict(self, data, model_name, split_use = None, seed = 123):
        """
        Generate predictions using an internal trained XGBoostLSS model for the specified target variable.
        
        
        Args:
            data (pandas.DataFrame, Catalog, or xgboost.DMatrix): Input data for prediction.
                If Catalog, can optionally specify which data split to use. If DataFrame,
                must contain all required features. If DMatrix, used directly for prediction.
            model_name (str): Name of the specific model to use for predictions. Must
                correspond to a trained model in the AGNBoost instance (e.g., 'fagn', 
                'z_transformed').
            split_use (str, optional): Specific data split to use when data is a Catalog
                object. Common values include 'train', 'val', 'test', 'trainval'. Ignored
                if data is not a Catalog. Defaults to None.
            seed (int, optional): Random seed for reproducible prediction results. Ensures
                consistent outputs across multiple prediction runs with identical inputs.
                Defaults to 123.
                
        Returns:
            numpy.ndarray or None: Array of prediction expectation values for the target
                variable, or None if prediction fails due to data validation errors,
                model unavailability, or feature compatibility issues.
                
        Examples:
            Predict AGN fractions on test data:
            ```python
            # Using a Catalog with test split
            fagn_predictions = agnboost.predict(catalog, 'fagn', split_use='test')
            ```
            
            Predict redshifts on new data:
            ```python
            # Using a pandas DataFrame
            z_predictions = agnboost.predict(new_data_df, 'z_transformed')
            ```
            
            Predict on validation set with custom seed:
            ```python
            # Reproducible predictions on validation data
            predictions = agnboost.predict(catalog, 'fagn', split_use='val', seed=42)
            ```
            
            Handle prediction errors gracefully:
            ```python
            predictions = agnboost.predict(data, 'fagn')
            if predictions is not None:
                print(f"Generated {len(predictions)} predictions")
            else:
                print("Prediction failed - check data and model status")
            ```
            
        Notes:
            **Prediction Type**: Returns expectation values (means) of the predicted
            probability distributions rather than raw distribution parameters.
            
            When using the pre-trained models:
                For AGN models, predictions typically represent fracAGN values (0-1 range).
                For redshift models, predictions represent transformed redshift values that
                may require inverse transformation to obtain physical redshift units.
        """

        # Process the passed data data
        dmatrix = self.dmatrix_validate(data = data, split_use = split_use)
        if dmatrix is None:
            return None

        # Load the model + validate the features
        xgblss_m = self._get_model(model_name)

        pred_mu = None
        if xgblss_m is not None:
            pred_mu = predict_mean( trained_model = xgblss_m, 
                                    dist_name = self.target_variables[model_name],
                                    data = dmatrix,
                                    seed = seed, 
                                     )

            self.logger.info(f"Made predictions using model: {model_name}")

        return pred_mu


    def _get_model( self, model_name ):
        """
        Retrieve a trained model in the AGNBoost instance.
        
        Safely retrieves a trained XGBoostLSS model from the AGNBoost instance while performing
        thorough validation checks. Ensures the requested model exists, has been properly trained
        or loaded, and maintains feature compatibility with the current AGNBoost configuration.
        Provides detailed logging of any compatibility issues or access problems.
        
        Args:
            model_name (str): Name of the model to retrieve. Must be a key in the models
                dictionary and correspond to a valid target variable (e.g., 'fagn', 
                'z_transformed').
                
        Returns:
            XGBoostLSS or None: The requested trained model object ready for prediction
                or evaluation operations, or None if the model is unavailable, untrained,
                or has feature compatibility issues.
                
        Examples:
            Retrieve a trained fracAGN model:
            ```python
            fagn_model = agnboost._get_model('fagn')
            if fagn_model is not None:
                # Model is ready for predictions
                predictions = fagn_model.predict(test_data)
            ```
            
        Notes:
            The method performs several critical validation steps:
            
            **Model Existence**: Verifies the model_name exists in the AGNBoost instance's
            model registry. Invalid model names result in detailed error logging.
            
            **Training Status**: Confirms the model has been trained or loaded (not None).
            Untrained models cannot be retrieved for use.      
        """
        if model_name in self.models.keys():
            if self.models[model_name] is not None:
                    try:
                        # Load the model
                        xgblss_m = self.models[model_name]

                        # Check feature compatibility
                        model_features =  self.model_info[model_name]['features']

                        if set(model_features) != set(self.feature_names):
                            self.logger.warning(
                                f"Model '{model_name}' was trained with different features:\n"
                                f"  - Model features ({len(model_features)}): {model_features}\n"
                                f"  - Current features ({len(self.feature_names)}): {self.feature_names}"
                            )

                            # Log specific differences for easier debugging
                            missing_features = [f for f in model_features if f not in self.feature_names]
                            extra_features = [f for f in self.feature_names if f not in model_features]

                            if missing_features:
                                self.logger.warning(f"  - Features in model but not current: {missing_features}")
                            if extra_features:
                                self.logger.warning(f"  - Features in current but not model: {extra_features}")
                            
                            self.logger.error(f"Feature mismatch for model {model_name}.")
                            return None
                        
                        return xgblss_m

                    except Exception as e:
                        self.logger.error(f"Error making predictions with {model_name}: {str(e)}")
                        return None
            else:
                self.logger.error(f"Model '{model_name}' is either not trained or loaded.")
        else:
            self.logger.error(f"Model '{model_name}' is not available in this AGNBoost instance.")
            
        return None

    def prediction_uncertainty( self, uncertainty_type, catalog, model_name, split_use = None, seed = 123, M = 10, num_permutation = 100):
        """
        Estimate total predictive uncertainty  for a trained model. Will return the model uncertainty, 
        uncertainty due to photometric error, or the combination of the two depending on the value of 
        'uncertainty_type.' This allows for robust uncertainty quantification for AGNBoost estimates,
        accounting for all types of uncertainty.
        
        Args:
            uncertainty_type (str): The type of uncertainty to estimate.
                Allowed values are: 'all', 'photometric', 'model'
            catalog (Catalog instance): the Catalog object to take data from.
            model_name (str): Name of the model to retrieve. Must be a key in the models
                dictionary and correspond to a valid target variable (e.g., 'fagn', 
                'z_transformed').
            split_use (str, default=None): Specific data split to use when data is a Catalog
                object. Common values include 'train', 'val', 'test', 'trainval'. Ignored
                if data is not a Catalog. Defaults to None.
            M (int, default=10): Number of models in the virtual ensemble. Controls the
                granularity of epistemic uncertainty estimation. Higher values provide
                more detailed uncertainty estimates but increase computation time.
                Defaults to 1.
            seed (int, default = 123): Random seed for reproducible uncertainty estimates.
                Ensures consistent results across multiple runs with identical inputs.
                Defaults to 123.
            num_permutation (int, default=100): Number of Monte Carlo iterations to run 
                if calculating the uncertainty due to photometric uncertainty.
                
        Returns:
            np.ndarray or None: Array of total uncertainty estimates for each input sample.
                For single samples, returns a scalar value. For multiple samples,
                returns an array with one uncertainty value per input.
        """
        # Valdiate passed uncertainty type
        uncertainty_types_list = ['all', 'photometric', 'model']
        if isinstance(uncertainty_type, str):
            if uncertainty_type not in uncertainty_types_list:
                #self.logger.error(f"Invalid uncertainty_type {uncertainty_type} Allowed values are: {uncertainty_types_list}")
                raise ValueError(f"Invalid uncertainty_type {uncertainty_type}. Allowed values are: {uncertainty_types_list}")
        else:
            #self.logger.error(f"uncertainty_type must be a str, got {type(target_variables).__name__}")
            raise TypeError(f"target_variables must be a dictionary, got {type(target_variables).__name__}")


        # Check that passed catalog is a Catalog object
        if not isinstance(catalog, Catalog):
            #self.logger.error(f"Passed catalog must be a Catalog object, got {type(catalog).__name__}")
            raise TypeError(f"Passed catalog must be a Catalog object, got {type(catalog).__name__}")

        dmatrix = self.dmatrix_validate(data = catalog, split_use = split_use)
        if dmatrix is None:
            return None

        # Load the saved model + validate the features
        xgblss_m = self._get_model(model_name)

        match uncertainty_type:
            case 'model':
                model_uncertainty = self.model_uncertainty( model = xgblss_m,
                                                             data = dmatrix, 
                                                             dist_name = self.target_variables[model_name],
                                                             best_iter = xgblss_m.booster.best_iteration, 
                                                             M = M
                                                             )
                return model_uncertainty

            case 'photometric':
                uncertainty_from_phot_err = self.uncertainty_phot( model =xgblss_m,
                                                                    catalog = catalog,
                                                                    dist_name = self.target_variables[model_name],
                                                                    num_permutation = num_permutation,
                                                                    seed = seed
                                                                    )
                return uncertainty_from_phot_err

            case 'all':
                model_uncertainty = self.model_uncertainty( model = xgblss_m,
                                                             data = dmatrix, 
                                                             dist_name = self.target_variables[model_name],
                                                             best_iter = xgblss_m.booster.best_iteration, 
                                                             M = M
                                                             )

                uncertainty_from_phot_err = self.uncertainty_phot( model =xgblss_m,
                                                                    catalog = catalog,
                                                                    dist_name = self.target_variables[model_name],
                                                                    num_permutation = num_permutation,
                                                                    seed = seed
                                                                    )

                return np.array( [np.sqrt(model_uncertainty[i]**2 + uncertainty_from_phot_err[i]**2) for i in range( catalog.get_length() )] )

        return None


    @staticmethod
    def model_uncertainty(model: xgboostlss_model,                     
                     data: DMatrix,
                     dist_name: str,
                    best_iter: int,
                     M: int = 1,
                     seed = 123
                     ) -> np.array:
        """
        Estimate total predictive uncertainty by decomposing epistemic and aleatoric components.
        
        Implements uncertainty quantification for XGBoostLSS models using a virtual ensemble
        approach based on boosting iteration truncation. Creates multiple model snapshots
        at different training stages to approximate epistemic (model) uncertainty, while
        extracting aleatoric (data) uncertainty from the predicted distributions. Returns
        the total uncertainty as the combination of both components.
        
        Args:
            model (xgboostlss_model): Trained XGBoostLSS model object containing the
                booster, distribution, and starting values needed for prediction.
            data (xgb.DMatrix): Input data in XGBoost DMatrix format for which uncertainty
                estimates will be computed. Can contain single or multiple samples.
            dist_name (str): Name of the probability distribution used by the model.
                Must match one of the supported distribution types (e.g., 'ZABeta', 'Beta').
            best_iter (int): Best iteration number from model training, typically obtained
                from early stopping or validation-based selection. Used as the reference
                point for creating the virtual ensemble.
            M (int, optional): Number of models in the virtual ensemble. Controls the
                granularity of epistemic uncertainty estimation. Higher values provide
                more detailed uncertainty estimates but increase computation time.
                Defaults to 1.
            seed (int, optional): Random seed for reproducible uncertainty estimates.
                Ensures consistent results across multiple runs with identical inputs.
                Defaults to 123.
                
        Returns:
            np.ndarray: Array of total uncertainty estimates for each input sample.
                For single samples, returns a scalar value. For multiple samples,
                returns an array with one uncertainty value per input.
        """
        def predict_dist_trunc(dist: DistributionClass,
                     booster: xgb.Booster,
                     start_values: np.ndarray,
                     data: DMatrix,
                     iteration_range: Tuple[int, int] = (0,0),
                     ) -> pd.DataFrame:
            """
            Generate predictions from a truncated XGBoostLSS model. This serves as a way to 
            approximate the epistemci model uncertainty (i.e., uncertainty due to a lack of model
            knowledge.)
            
            Args:
                dist (DistributionClass): Distribution object defining the probability
                    distribution type and parameter structure. Contains parameter
                    definitions and response functions for transformation.
                booster (xgb.Booster): Trained XGBoost booster object containing the
                    learned model parameters and tree structure for prediction.
                start_values (np.ndarray): Starting values for each distributional
                    parameter. Used as base margins to initialize the prediction
                    process. Shape should match the number of distribution parameters.
                data (xgb.DMatrix): Input data in XGBoost DMatrix format for which
                    predictions will be generated. Must contain all required features.
                iteration_range (Tuple[int, int], optional): Range of boosting iterations
                    to use for prediction. Allows using specific tree layers or early
                    stopping points. Format is (start, end) where (0,0) uses all iterations.
                    Defaults to (0,0).
                    
            Returns:
                pd.DataFrame: DataFrame containing transformed distribution parameters
                    with columns corresponding to parameter names from the distribution's
                    parameter dictionary. Each row represents predicted parameters for
                    one input sample.
            """

            # Set base_margin as starting point for each distributional parameter. Requires base_score=0 in parameters.
            base_margin_test = (np.ones(shape=(data.num_row(), 1))) * start_values
            data.set_base_margin(base_margin_test.flatten())
            predt = np.array(booster.predict(data, output_margin=True, iteration_range = iteration_range)).reshape(-1, dist.n_dist_param)
            predt = torch.tensor(predt, dtype=torch.float32)
            
            # Transform predicted parameters to response scale
            dist_params_predt = np.concatenate(
                [
                    response_fun(
                        predt[:, i].reshape(-1, 1)).numpy() for i, (dist_param, response_fun) in
                    enumerate(dist.param_dict.items())
                ],
                axis=1,
            )
            dist_params_predt = pd.DataFrame(dist_params_predt)
            dist_params_predt.columns = dist.param_dict.keys()

            return dist_params_predt


        best_iter = best_iter - 1
        K = round(best_iter/(2*M))
      
        delta_grid = np.arange(0, best_iter/2, K, dtype = int)
        iteration_grid = np.array([best_iter - val for val in delta_grid])
        
        data_len = data.num_row()

        pred_mus = np.empty( (data_len, len(iteration_grid) )) 
        pred_stds= np.empty( (data_len, len(iteration_grid) ))
        
        for i in tqdm(range(data_len), desc = "Processing truncated model uncertainty"):
            for j in range(len(iteration_grid)):
                # Need to add 1 to iteration grid value because booster.preeict evaluates on [a, b)
                pred_params = predict_dist_trunc(dist = model.dist, booster = model.booster, start_values = model.start_values, data = data.slice([i]), iteration_range = (0, int(iteration_grid[j]) + 1) )


                pred_mus[i, j] = predict_mean( trained_model = model, 
                                            dist_name = dist_name,
                                            data = data.slice([i]),
                                            seed = seed
                                             )

                pred_stds[i, j] = predict_std( trained_model = model, 
                            dist_name = dist_name,
                            data = data.slice([i]),
                            seed = seed
                             )

        epistemic_uncert = pred_mus.std(axis = 1)
        aleatoric_uncert = np.average(pred_stds, axis = 1)

        if data_len == 1:
            return np.sqrt(pistemic_uncert[0]**2 + aleatoric_uncert[0]**2)

        return np.sqrt(epistemic_uncert**2 + aleatoric_uncert**2)

    @staticmethod
    def uncertainty_phot(model: xgboostlss_model,     
                        catalog: Catalog,
                        dist_name : str,             
                         num_permutation = 100,
                         seed = 123) -> np.ndarray:
        """
        Estimate the prediction uncertainty due to photometric uncertainty. 
        Performs monte carlo, randomly sampling the photometric bands 
        (assuming a normal flux distribution) accoridng to the photometric error bands 
        stored in the catalog object's data. The standard deviation of the monte carlo preditions
        is taken to be the prediction uncertainty due to photometric uncertainty.
        
        Args:
            model (xgboostlss.model): The trained model to use to make predictions.
            catalog (Catalog instance): the Catalog object to take data from.
            dist_name (str): The name of the probability distribution type to make predictions with
                This should correspond to the distribution stored in self.target_variables[model_name]
            seed (int, default = 123): Random seed for reproducible uncertainty estimates.
                Ensures consistent results across multiple runs with identical inputs.
                Defaults to 123.
            num_permutation (int, default=100): Number of Monte Carlo iterations to run 
                if calculating the uncertainty due to photometric uncertainty.
                
        Returns:
            np.ndarray or None: Array of the uncertainty estimates.
        """
        phot_names = catalog.get_valid_bands_list()
        err_cols = [phot + '_err' for phot in phot_names]
        data = catalog.get_data()
        phot_data = data[phot_names]
        err_data = data[err_cols]

        num_sources, num_phots = phot_data.shape

        uncert_from_phot = np.full( num_sources, np.nan )

        desc = f"Processing uncertainty due to photometric uncertainty with {num_permutation} trials per source."
        log_message( desc )
        for i in tqdm(range(num_sources), desc = None):
            mu_j_arr = np.full( num_permutation, np.nan ) 
            
            negative_trial_bands = {phot_names[i] : None for i in range(num_phots)}
            for j in range(num_permutation):
                # Create array to fill with new phots from random sampling
                phot_i_arr = np.full( num_phots, np.nan ) 

                # Iterate through each band
                for k in range(num_phots):

                    # Before sampling, we ensure that the error for this band is not negative
                    if err_data.iloc[i,k] > 0:
                        rand_phot = np.random.normal( loc = phot_data.iloc[i, k], scale = err_data.iloc[i,k])

                        negative__trial_count = 0
                        # If we get a negative flux out (due to low S/N), repeat sampling procedure unitl it is positive
                        while rand_phot < 0:
                            rand_phot = np.random.normal( loc = phot_data.iloc[i, k], scale = err_data.iloc[i,k])
                            negative__trial_count += 1

                        if negative__trial_count > 0:
                            sn = phot_data.iloc[i, k]/err_data.iloc[i,k]
                            negative_trial_bands[ phot_names[k] ] = sn

                        phot_i_arr[k] = rand_phot
                         
                    # If the error is negative, just use the existing phot
                    else:
                        phot_i_arr[k] = phot_data.iloc[i, k]

                phot_i_df = pd.DataFrame(phot_i_arr).T
                phot_i_df.columns = phot_names       

                
                feature_i_df = catalog.get_features_from_phots(phot_i_df)
                
                dmat_i = DMatrix( data = feature_i_df )
                mu_j_arr[j] = predict_mean(trained_model = model, dist_name = dist_name, data = dmat_i, seed = seed)

            uncert_from_phot[i] =  mu_j_arr.std()

            negative_bands = [ phot_names[i] for  i in range(num_phots) if negative_trial_bands[ phot_names[i] ] is not None]
            if len(negative_bands) > 0:
                sn_vals = [ negative_trial_bands[phot] for phot in negative_bands ]
                formatted_sn = [f"{sn:.2f}" for sn in sn_vals]
                mess = f"Negative monte carlo fluxes at index {i} in bands {negative_bands} with S/N {formatted_sn}. Monte carlo procedure was repeated until positive flux returned at each iteration."
                log_message( mess )

        return uncert_from_phot