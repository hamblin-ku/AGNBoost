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
#from xgboostlss.distributions import *
#from xgboostlss.model import XGBoostLSS



# chagne this to take a list of feature names
# take these from the created Catalog object
# agnboost.py
class AGNBoost:
    # Class-level logger
    logger = logging.getLogger('AGNBoost.AGNBoost')

    ALLOWED_DISTS = ['ZABeta', 'Beta']

    
    def __init__(self, feature_names=None, target_variables=None, logger=None):
        """
        Initialize the AGNBoost object.
        
        Parameters:
        -----------
        feature_names : list or None
            Feature (column) names to use for model input.
        model_names : list or None
            Names of models to load/train. If None, uses default models.
        logger : logging.Logger, default=None
            Custom logger to use. If None, uses the class logger.
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

    def tune_model(self, model_name, param_grid, dtune, split_type = 'train', max_minutes=10, nfold=2, early_stopping_rounds=100):
        """
        Tune hyperparameters for the specified model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to tune. Must be in self.model_names.
        param_grid : dict
            Dictionary of hyperparameter ranges to search.
            Example: {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.3]}
        dtune : xgboost.DMatrix or pandas.DataFrame or Catalog
            Data to use for tuning. If DataFrame or Catalog, will be converted to DMatrix.
        max_minutes : int, default=10
            Maximum duration for tuning in minutes.
        nfold : int, default=2
            Number of cross-validation folds.
        early_stopping_rounds : int, default=100
            Number of rounds without improvement before early stopping.
            
        Returns:
        --------
        dict
            Dictionary containing best parameters and tuning metrics.
        """
        #import xgboost as xgb
        #from time import time
        
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


    def train_model(self, model_name, dtrain, dval=None, params=None, split_type='train', 
                    num_boost_round=1000, early_stopping_rounds=None, 
                    verbose_eval=False, custom_objective=None, custom_metric=None):
        """
        Train an XGBoost model with the given parameters.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to train. Must be in self.model_names.
        dtrain : xgboost.DMatrix or Catalog
            Training data. If a Catalog, will create a DMatrix using the specified split.
        dval : xgboost.DMatrix, bool, or None, default=None
            Validation data. If a DMatrix, use directly.
            If True and dtrain is a Catalog, create a validation DMatrix from the Catalog.
            If False or None, no validation data is used.
        params : dict or None, default=None
            Parameters for XGBoost. If None, uses tuned_params from model_info if available,
            otherwise uses default parameters.
        split_type : str, default='train'
            Which data split to use if dtrain is a Catalog. Options: 'train', 'val', 'test'.
        num_boost_round : int, default=1000
            Number of boosting rounds.
        early_stopping_rounds : int or None, default=50
            Number of rounds without improvement before early stopping.
            If None, no early stopping is used.
        verbose_eval : int or bool, default=100
            Controls XGBoost's logging frequency. If True, prints every round.
            If int, prints every verbose_eval rounds. If False, no logging.
        custom_objective : callable or None, default=None
            Custom objective function for XGBoost.
        custom_metric : callable or None, default=None
            Custom evaluation metric for XGBoost.
            
        Returns:
        --------
        tuple
            (trained_model, training_results)
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
            if hasattr(xgblss_m, 'best_iteration'):
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
        Save a trained model and its metadata to disk.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to save.
            
        Returns:
        --------
        bool
            True if the model was saved successfully, False otherwise.
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
        if filename is None:
            model_path = os.path.join(self.models_dir, datetime.now().strftime("%Y_%m_%d-%p%I_%M_%S") + f"_{model_name}_model.pkl")

        # Use passed filename
        else:
            #If the passed string already has the pkl file extension
            if filename.endswith(".pkl"):
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
        # Ensure that the sub directory for this model exists. If it does not, return None
        if os.path.exists( path ) is False:
            return None

        # Get a list of all of the files in the model directory
        files = glob.glob(os.path.join(path, "*"))
        return files


    def get_available_models(self, target_name = None):
        if target_name is not None:
            model_sub_dir = os.environ.get('AGNBOOST_MODELS_DIR', f"models/{target_name}")
            print(model_sub_dir)
            if not os.path.exists(model_sub_dir):
                raise FileNotFoundError(f'model sub directory for model {target_name} does not exist. Available options are {self.get_available_models()}')

        # If not target is passed, return a list of the available model sub directories.
        else:
            model_sub_dir = os.environ.get('AGNBOOST_MODELS_DIR', 'models/')

        return [os.path.basename(x) for x in self.get_available_files(model_sub_dir)]

    def load_model(self, model_name, file_name = None, overwrite = False):
        """
        Load pre-trained models from disk.
        
        Args:

            file_name (str, optional): 
                Name(s) of models to load. If None, loads all models in self.model_names.
            overwrite (boo, optional): Whether to overwite already laoded model in AGNBoost instance.
                Defaults to False.

        Raises:    
           FileNotFoundError: If the sub-directory for this model type does not exist.

        Returns:
            bool: True if requested model was loaded successfully, False otherwise.
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
        Make predictions using trained models.
        
        Parameters:
        -----------
        data : DataFrame, Catalog, or dict
            Data to make predictions on. Can be a pandas DataFrame, a Catalog object,
            or a dict mapping model names to their specific data.
        model_name : str or None
            If provided, use only this model. Otherwise, use all available models.
            
        Returns:
        --------
        dict
            Dictionary of predictions for each model.
        """

        # Process the passed data data

        dmatrix = self.dmatrix_validate(data = data, split_use = split_use)
        if dmatrix is None:
            return None

        # Load the model + validate the features
        xgblss_m = self.get_model(model_name)

        pred_mu = None
        if xgblss_m is not None:
            pred_mu = predict_mean( trained_model = xgblss_m, 
                                    dist_name = self.target_variables[model_name],
                                    data = dmatrix,
                                    seed = seed, 
                                     )

            self.logger.info(f"Made predictions using model: {model_name}")

        return pred_mu


    def get_model( self, model_name ):
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
        xgblss_m = self.get_model(model_name)

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
        Function that predicts from the trained model.

        Arguments
        ---------
        model : xgblsslib.model
            Trained model.
        data : xgb.DMatrix
            Data to predict frmodel: xgblsslib.modelom.
        M   : int
            Number of desired models in virtual ensemble.

        Returns
        -------
        pred : pd.DataFrame
            Predictions.
        """
        def predict_dist_trunc(dist: DistributionClass,
                     booster: xgb.Booster,
                     start_values: np.ndarray,
                     data: DMatrix,
                     iteration_range: Tuple[int, int] = (0,0),
                     ) -> pd.DataFrame:
            """
            Function that predicts from therag trained model.

            Arguments
            ---------
            booster : xgb.Boosterdist
                Trained model.
            start_values : np.ndarray
                Starting values for each distributional parameter.
            data : xgb.DMatrix
                Data to predict from.
            iteration_range: Tuple[int,int]
                Which layer of trees to use for prediction in xgb.booster.predict

            Returns
            -------
            pred : pd.DataFrame
                Predictions.xgboost model get numeber of boostin rounds
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
    def predict_dist_trunc(dist: DistributionClass,
                     booster: xgb.Booster,
                     start_values: np.ndarray,
                     data: DMatrix,
                     iteration_range: Tuple[int, int] = (0,0),
                     ) -> pd.DataFrame:
        """
        Function that predicts from therag trained model.

        Arguments
        ---------
        booster : xgb.Boosterdist
            Trained model.
        start_values : np.ndarray
            Starting values for each distributional parameter.
        data : xgb.DMatrix
            Data to predict from.
        iteration_range: Tuple[int,int]
            Which layer of trees to use for prediction in xgb.booster.predict

        Returns
        -------
        pred : pd.DataFrame
            Predictions.xgboost model get numeber of boostin rounds
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

    @staticmethod
    def uncertainty_phot(model: xgboostlss_model,     
                        catalog: Catalog,
                        dist_name : str,             
                         num_permutation = 100,
                         seed = 123) -> np.ndarray:
        
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


    @staticmethod
    def list_available_models(models_dir=None):
        """
        List all available pre-trained models in the models directory.
        
        Parameters:
        -----------
        models_dir : str or None
            Directory to look for models. If None, uses the default models directory.
            
        Returns:
        --------
        dict
            Dictionary with model names as keys and metadata as values.
        """
        logger = logging.getLogger('AGNBoost.AGNBoost.list_available_models')
        
        # Get models directory
        if models_dir is None:
            models_dir = os.environ.get('AGNBOOST_MODELS_DIR', 'models/')
        
        if not os.path.exists(models_dir):
            logger.warning(f"Models directory '{models_dir}' does not exist.")
            return {}
        
        available_models = {}
        
        # Look for model files
        for filename in os.listdir(models_dir):
            if filename.endswith('_model.pkl'):
                model_name = filename.replace('_model.pkl', '')
                model_path = os.path.join(models_dir, filename)
                
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    # Extract metadata
                    if isinstance(model_data, dict) and 'model' in model_data:
                        metadata = {k: v for k, v in model_data.items() if k != 'model'}
                        available_models[model_name] = metadata
                    else:
                        available_models[model_name] = {'format': 'legacy'}
                        
                    logger.info(f"Found model: {model_name}")
                except Exception as e:
                    logger.warning(f"Error reading model {model_name}: {str(e)}")
        
        logger.info(f"Found {len(available_models)} available models in {models_dir}")
        return available_models