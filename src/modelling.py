from dataclasses import dataclass, field
from typing import Dict, Callable
import math 
import numpy as np
import pandas as pd
import pmdarima as pmd
from pmdarima import model_selection
from darts import TimeSeries
from darts.metrics import mae, mape, mase, mse, ope, r2_score, rmse, rmsle
from darts.models import (ARIMA, AutoARIMA, ExponentialSmoothing, NaiveDrift,
                          NaiveSeasonal, Prophet)
from darts.utils.utils import ModelMode, SeasonalityMode
from darts.utils.model_selection import train_test_split
import warnings 
import matplotlib.pyplot as plt 

arr = np.array([(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)])
# flake8: noqa


def extract_parameters(params: str):
    print(params)

    if 'x' in params:
        param_type = 'both'
    elif '[' in params:
        param_type = 'seasonal'
    else:
        param_type = 'non-seasonal'

    parameters = params[8:]
    values = parameters.split(' ')
    #print(parameters, values, param_type)

    if 'intercept' in values:
        intercept = True 
    else:
        intercept = False

    param_values = [int(i) for i in parameters if i.isnumeric()]

    if len(param_values) == 3:
        pdq = param_values
        seasonal = [0,0,0,0]
    elif len(param_values) == 4:
        pdq = [0,0,0]
        seasonal = param_values
    else:
        pdq = param_values[0:3]
        seasonal = param_values[3:]

    return pdq, seasonal, intercept


def mean_abs_percentage_error(
    true,
    predicted
):

    return 100.0 * np.mean(np.abs((true - predicted) / true))

def model_check(mod_name: str, series: TimeSeries, seasonal_period: int) -> bool:

    if mod_name == 'arima':
        if series.n_timesteps < 36:
            return False 
        else:
            return True 
    elif mod_name == 'exponential':
        if int((series.n_timesteps -1)*0.9) < 2*seasonal_period:
            return False 
        else:
            return True 
    elif mod_name == 'prophet':
        if series.n_timesteps < 22:
            return False 
        else:
            return True 

def train_test_split_func(series: TimeSeries, train_size: float):

    train_time = series.get_timestamp_at_point(train_size)

    return series.split_before(train_time)

    

@dataclass
class modeling:

    series: TimeSeries.from_dataframe(pd.DataFrame(data=arr))
    models_dict: Dict[str, str] = field(
        default_factory=lambda: {"arima": pmd.AutoARIMA(), "exponential": ExponentialSmoothing(initialization_method=None)}
    )
    model_hyperparameters: Dict[str, str] = field(
        default_factory=lambda: {
            "prophet": {'seasonality_mode':('multiplicative','additive'),
                        'changepoint_prior_scale':[0.1,0.2],
                        'holidays_prior_scale':[0.1,0.2],
                        'n_changepoints' : [100,150]},
            "exponential": {'trend': [ModelMode.ADDITIVE, ModelMode.MULTIPLICATIVE, ModelMode.NONE],
                            'seasonal': [SeasonalityMode.ADDITIVE, SeasonalityMode.MULTIPLICATIVE, SeasonalityMode.NONE]
                            

                            }
        }
        
    )
    pred_interval: int = 12
    contractLocationID: str = ""
    ts_attributes: set = ()

    def _modeling(self):

        self.model_metrics = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for mod, mod_instantiation in (self.models_dict.items()):  # mod is key and instantiation is the value (model)

                print(mod, mod_instantiation)
                print(self.ts_attributes)
                if model_check(mod, self.series, self.ts_attributes[3]):

                    if mod == 'arima':

                        output = train_test_split_func(self.series, 0.9)
                    
                        train, val = output[0].pd_series(), output[1].pd_series()

                        model = pmd.AutoARIMA(
                                start_p=0, d=self.ts_attributes[0], start_q=0,
                                max_p=3, max_d=self.ts_attributes[0], max_q=3,
                                start_P=0, D=self.ts_attributes[1], start_Q=0,
                                max_P=3, max_D=self.ts_attributes[1], max_Q=3,
                                m=max(4,self.ts_attributes[3]), is_seasonal=self.ts_attributes[2],
                                max_order=5, stationary=False,
                                information_criterion="bic", alpha=0.05,
                                test="kpss", seasonal_test="ch", stepwise=True,
                                suppress_warnings=True, error_action="trace", trace=True, with_intercept="auto"
                            )

                        model_fit = model.fit(self.series.pd_series())
                        params = model_fit.summary().tables[0].data[1][1] + " " + model_fit.summary().tables[1].data[1][0]

                        pdq, seasonal, intercept = extract_parameters(params)
                        print(pdq, seasonal, intercept)
                        if (len(pdq)==0) & (len(seasonal)==0):
                            continue 
                            
                        arima_model = pmd.ARIMA(order=pdq, seasonal=seasonal, with_intercept=intercept)

                        cv = model_selection.RollingForecastCV(h=1, step=1, initial=train.size)
                        model_score = model_selection.cross_val_score(arima_model, self.series.pd_series(), scoring=mean_abs_percentage_error, cv=cv, verbose=2)
                        score = np.average(model_score)
                        model = arima_model

                    else:
                                
                        param_grid = self.model_hyperparameters[mod]
                        
                        if(mod=='exponential'):
                            param_grid['seasonal_periods'] = [self.ts_attributes[3]]
                        
                        if((mod=='exponential') and (self.ts_attributes[2]==False) ):
                            param_grid['seasonal'] = [SeasonalityMode.NONE]
                        
                        if((mod=='exponential') and (self.ts_attributes[0]== 0) ):
                            param_grid['trend'] = [ModelMode.NONE]
                        
                        
                        model, params, score = mod_instantiation.gridsearch(
                            parameters=param_grid,
                            series=self.series,
                            forecast_horizon=1,
                            start=0.9,
                            metric=mape,
                            reduction=np.median,
                            verbose=True,
                        )                     

                    if len(self.model_metrics) == 0:
                        self.model_metrics[mod] = {
                            "model": model,
                            "params": params,
                            "score": score,
                        }
                    else:
                        for k, v in self.model_metrics.items():
                            current_val = self.model_metrics[k]["score"]

                        if current_val > score:
                            self.model_metrics = {}
                            self.model_metrics[mod] = {
                                "model": model,
                                "params": params,
                                "score": score,
                            }
                    

        if len(self.model_metrics)==0:

            self.future_predictions = None
            self.model_name = None
            self.pred_val = None
            self.pred_score = None
            self.val_size = None
            self.conf_interval = None
            self.predictions_conf_interval = None
        else:
            self.future_predictions, self.model_name, self.pred_val, self.pred_score, self.val_size, self.conf_interval, self.predictions_conf_interval = self._output_values()

        

    def _output_values(self):

        for k, v in self.model_metrics.items():

            model = self.model_metrics[k]["model"]

            if k == 'arima':

                output = train_test_split_func(self.series, 0.9)
                train, val = output[0].pd_series(), output[1].pd_series()

                print("Train Size is: {} & Validation Size is: {}".format(train.size, val.size))

                assert (train.size + val.size == self.series.n_timesteps), "Train size + Val size do not equal time series length"
                
                #params = self.model_metrics[k]["params"]
                #pdq, seasonal, intercept = extract_parameters(params)

                """
                if self.ts_attributes[3] > 0:
                    if len(seasonal) > 3:
                        seasonal[3] = self.ts_attributes[3]
                    else:
                        seasonal.append(self.ts_attributes[3])
                """

                #arima_model = pmd.ARIMA(order=pdq, seasonal=seasonal, with_intercept=intercept)

                model_name = k 
                val_size = val.size

                model_fit = model.fit(train)
                pred, conf_interval = model.predict(val.size+self.pred_interval, return_conf_int=True)
                
                pred_val, conf_interval_pred = pred[0:val.size], conf_interval[0:val.size]
                pred_score = mape(TimeSeries.from_series(val), TimeSeries.from_series(pred_val), reduction=np.median)

                predictions, predictions_conf_interval = pred[val.size:], conf_interval[val.size:]

                model_name = k

                val_size = val.size 

                return TimeSeries.from_series(predictions), model_name, TimeSeries.from_series(pred_val), pred_score, val_size, conf_interval_pred, predictions_conf_interval

            else:

                output = train_test_split_func(self.series, 0.9)
                train, val = output[0], output[1]

                print("Train Size is: {} & Validation Size is: {}".format(train.n_timesteps, val.n_timesteps))

                assert (train.n_timesteps + val.n_timesteps == self.series.n_timesteps), "Train size + Val size do not equal time series length"
                
                backtest_cov = model.historical_forecasts(
                    self.series,
                    start=0.9,
                    forecast_horizon=1,
                    stride=1,
                    overlap_end=True,
                    retrain=True,
                    verbose=True,
                    num_samples=100
                )

                pred_val = backtest_cov.shift(-1)

                full_model_fit = model.fit(self.series)

                predictions = full_model_fit.predict(
                    self.pred_interval+1, num_samples=100
                ).shift(-1)
                
                last_val = predictions[0]
                pred_val = pred_val.append(last_val)
                predictions = predictions[1:]

                train_last_val = train[-1]
                train = train[0:train.n_timesteps-1]
                new_val = train_last_val.append(val)

                pred_score = mape(new_val, pred_val, reduction=np.median)
                model_name = k

                val_size = new_val.n_timesteps
                return predictions, model_name, pred_val, pred_score, val_size, [], []
        

