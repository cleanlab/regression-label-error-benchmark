import numpy as np 
import pandas as pd 
from autogluon.tabular import TabularPredictor

class CleanlabRegressor(TabularPredictor):
    def __init__(
        self,
        savepath = None,
        label: str = "label", 
        eval_metric: str = "r2", 
        problem_type: str = "regression",
        hyperparam: dict = {"GBM": {}},
        preset: str = "best_quality",
        verbosity: int = 1, 
        remove_per_iter: float = 2.5, 
        scoring_func = None, 
        patience = 5,
        max_trail = 100,  
        ):
        super().__init__(
            path = savepath, 
            label = label, 
            eval_metric = eval_metric,
            problem_type = problem_type,
            verbosity= verbosity,
        )
        self.savepath = savepath
        self._label = label
        self._eval_metric = eval_metric
        self._problem_type = problem_type
        self.hyperparam = hyperparam
        self.preset = preset
        self.verbosity = verbosity
        self.remove_per_iter = remove_per_iter
        self.scoring_func = scoring_func
        self.patience = patience
        self.max_trail = max_trail
    
    def fit(self, X, y):
        """
        Iteratively removes data and Returns a predictor that is fitted on subset of full dataset. 
        In case max_trail = 1, it will return a predictor fitted on full dataset. 
        """
        
        best_r2 = float("-inf")
        step_after_last_improved = 0 # keeps track of number of step since last improvement 
        self.r_squared = [] # keeps track of r2 per iteration
        
        # To keep track of removed datapoints 
        self.removed_dataframe = X_remove = pd.DataFrame(columns=X.columns)
          
        for _ in range(self.max_trail): 
            # Call fit to get predictor for current subset of data
            predictor, leaderboard = self.__fit(X,y)

            # Store r_squared values for tracking history 
            held_out_r2 = leaderboard['score_val'][0]
            self.r_squared.append(held_out_r2)
            
            # Saving current training dataframe and removed dataframe
            self.kept_dataframe = X
            self.removed_dataframe = pd.concat([self.removed_dataframe, X_remove])
            
            # Get out-of-fold predictions from current predictor
            oof_pred = predictor.get_oof_pred()
            
            # Calculate label quality score and threshold as per remove_per_iter argument 
            if self.scoring_func is None:
                label_quality_score = self.score(y, oof_pred)
            else: 
                label_quality_score = self.scoring_func(y,oof_pred)

            threshold = np.percentile(label_quality_score, self.remove_per_iter)
            to_keep_mask = (label_quality_score >= threshold)
            keep_indicies = np.where(to_keep_mask==1)[0]
            remove_indicies = np.where(to_keep_mask==0)[0]
        
            # Subsample the dataset to keep only top (100 - remove_per_iter) datapoints
            X_remove = X.iloc[remove_indicies]
            X = X.iloc[keep_indicies]
            y = y.iloc[keep_indicies]
            
            # Track the progress of the r_squared 
            if held_out_r2 > best_r2:
                best_r2 = held_out_r2
                step_after_last_improved = 0
                
                # Saving best predictor and leaderboard 
                self.best_predict = predictor
                self.best_leader = leaderboard
            else:
                step_after_last_improved += 1
            
            # If there is no change in r2 value untill `patience`, return 
            if step_after_last_improved > self.patience:
                return self
        
        # If R-squared was always improving, return after max_trail
        return self 

    def __fit(self, X,y):
        """
        Runs one iteration of fitting. 
        """
        # create dataframe from input data 
        columns = X.columns
        X = pd.DataFrame(X) 
        y = pd.DataFrame(y)
        X.columns = columns
        y.columns = [self._label]  # type: ignore

        # Get data ready in format expected by autogluon 
        training_data = pd.concat([X, y], axis=1)
        
        # Intialise to ensure that new predictor is available in every iteration in fit  
        super().__init__(
            path = self.savepath, 
            label = self._label, 
            eval_metric = self._eval_metric,
            problem_type = self._problem_type,
            verbosity= self.verbosity,
        )
        predictor = super().fit(training_data, 
                    presets= self.preset, 
                    num_stack_levels = 0, 
                    hyperparameters=self.hyperparam,
        )
        leaderboard = predictor.leaderboard(training_data, silent = True)
        return predictor, leaderboard 

    def score(self, label, predictions):
        """
        Uses residual based scoring. 
        If planning to use any other scoring technique, definition can be changed. 
        """
        residual = predictions - label
        return np.exp(-abs(residual))
    
    def predict(self, model = "LightGBM_BAG_L1"):
        """ 
        Return np.ndarray of predictions per datapoint. 
        It combines out-of-fold predictions for datapoints that were left until last iteration and
        uses LightGBM for predictions on removed datapoints during every iteration. 
        """
        # Out-of-fold predictions on datapoints that are left after max_trail or patience. 
        kept_df = self.kept_dataframe.copy()
        kept_df['predictions'] = super().get_oof_pred(model=model)
        
        # Using LightGBM to predict on datapoints that were removed. 
        removed_df = self.removed_dataframe.copy()
        removed_df['predictions'] = super().predict(removed_df, model=model)
        
        # Concat to get final dataframe with index same as original dataset. 
        final_dataframe = pd.concat([kept_df, removed_df], axis=0)
        return np.array(final_dataframe['predictions'])
    
    @property
    def r_squared_history(self):
        return self.r_squared
    
    @property
    def best_predictor(self):
        return self.best_predict
    
    @property
    def best_leaderboard(self):
        return self.best_leader
    
    @property
    def n_iteration(self):
        return len(self.r_squared_history)
    
    @property
    def get_removed_rows(self):
        return self.removed_dataframe


class AutoGluonPredictor(TabularPredictor):
    def __init__(self, 
                label= "", 
                presets = "best_quality", 
                get_oof = True,
                num_stack_levels = 0, 
                problem_type="regression", 
                eval_metric="r2", 
                path=None, 
                verbosity=2,
                hyperparameter = {"GBM": {}}, 
                ):
        super().__init__(label = label,
                        problem_type = problem_type, 
                        eval_metric=eval_metric, 
                        path= path, 
                        verbosity =verbosity
                        )
     
        self.preset = presets
        self.get_oof = get_oof
        self.num_stack_levels = num_stack_levels
        self.hyperparameter = hyperparameter
    
    def fit(self, X, y=None):
        x_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y)        
        x_df.columns = ["col"+str(i) for i in range(x_df.shape[1])]  # type: ignore
        y_df.columns = ['label']  # type: ignore

        training_data = pd.concat([x_df, y_df], axis=1)
        self.__init__(label='label', 
                get_oof=self.get_oof, 
                presets=self.preset, 
                num_stack_levels=self.num_stack_levels, 
                hyperparameter=self.hyperparameter)
    
        super().fit(training_data, 
                    presets=self.preset, 
                    num_stack_levels=self.num_stack_levels, 
                    hyperparameters = self.hyperparameter)
        return self
    
    def predict(self, data, model=None, as_pandas=True, transform_features=True):
        data = pd.DataFrame(data)
        
        if self.get_oof:
            return super().get_oof_pred()
        else:
            return super().predict(data, model, as_pandas, transform_features)
    
    def score(self, X, y):
        x_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y)        
        x_df.columns = ["col"+str(i) for i in range(x_df.shape[1])]  # type: ignore
        y_df.columns = ['label']  # type: ignore

        training_data = pd.concat([x_df, y_df], axis=1)
        self.__init__(label='label', 
                        get_oof=self.get_oof, 
                        presets=self.preset, 
                        num_stack_levels=self.num_stack_levels, 
                        hyperparameter=self.hyperparameter)
        
        super().fit(training_data, 
                    presets=self.preset, 
                    num_stack_levels=self.num_stack_levels, 
                    hyperparameters = self.hyperparameter)
        
        y_pred = self.predict(X)
        y_true = y

        RSS = ((y_true - y_pred)** 2).sum()
        TSS = ((y_true - y_true.mean()) ** 2).sum()
        R2 = 1 - (RSS/TSS)
        return R2
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        return self
    
    def get_params(self, deep = True):
        """to ensure compatibility with sklearn"""
        
        params = {}
        return params