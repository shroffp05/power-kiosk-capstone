from dataclasses import dataclass, field
from typing import Dict
import math 
import numpy as np
import pandas as pd
import pmdarima as pmd
from darts import TimeSeries
from darts.metrics import mae, mape, mase, mse, ope, r2_score, rmse, rmsle
from darts.models import (ARIMA, AutoARIMA, ExponentialSmoothing, NaiveDrift,
                          NaiveSeasonal, Prophet)
from darts.utils.utils import ModelMode, SeasonalityMode
import warnings 

arr = np.array([(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)])
#metrics = np.array([("1234", "1/1/2020", 1000, 20, 10, "arima", 12.3, 2.2, "null")])

def model_check(mod_name: str, series: TimeSeries, seasonal_period: int) -> bool:

    if mod_name == 'arima':
        if series.n_timesteps < 36:
            return False 
        else:
            return True 
    elif mod_name == 'exponential':
        if seasonal_period < 2:
            return False 
        else:
            return True 
    elif mod_name == 'prophet':
        if series.n_timesteps < 22:
            return False 
        else:
            return True 

def train_test_split(series: TimeSeries, train_size: float, pandas: bool):

    len_series = series.n_timesteps
    train_size = math.floor(len_series*train_size) + 1
    val_size = math.floor(len_series*(1-train_size))

    if not pandas:
        return series[: train_size], series[train_size:] 
    else: 
        return series[: train_size].pd_series(), series[train_size:].pd_series()
    

@dataclass
class modeling:

    series: TimeSeries.from_dataframe(pd.DataFrame(data=arr))
    models_dict: Dict[str, str] = field(
        default_factory=lambda: {"arima": pmd.AutoARIMA(), "exponential": ExponentialSmoothing()}
    )
    model_hyperparameters: Dict[str, str] = field(
        default_factory=lambda: {
            "prophet": {'seasonality_mode':('multiplicative','additive'),
                        'changepoint_prior_scale':[0.1,0.2],
                        'holidays_prior_scale':[0.1,0.2],
                        'n_changepoints' : [100,150]},
            "exponential": {'trend': [ModelMode.ADDITIVE, ModelMode.MULTIPLICATIVE, ModelMode.NONE],
                            'seasonal': [SeasonalityMode.ADDITIVE, SeasonalityMode.MULTIPLICATIVE, SeasonalityMode.NONE]}
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
                
                if model_check(mod, self.series, self.ts_attributes[3]):

                    if mod == 'arima':
                        model = pmd.AutoARIMA(
                                start_p=0, d=self.ts_attributes[0], start_q=0,
                                max_p=3, max_d=self.ts_attributes[0], max_q=3,
                                start_P=0, D=self.ts_attributes[1], start_Q=0,
                                max_P=3, max_D=self.ts_attributes[1], max_Q=3,
                                m=max(4,self.ts_attributes[3]), is_seasonal=self.ts_attributes[2],
                                max_order=5, stationary=False,
                                information_criterion="bic", alpha=0.05,
                                test="kpss", seasonal_test="ch", stepwise=True,
                                suppress_warnings=True, error_action="trace", trace=False, with_intercept="auto"
                            )

                        train, val = train_test_split(self.series, 0.9, pandas=True)
                        model_fit = model.fit(train)
                        params = model_fit.get_params
                        forecast = model_fit.predict(val.size)
                        score = mape(TimeSeries.from_series(val), TimeSeries.from_series(forecast), reduction=np.median)

                    else:
                                
                        param_grid = self.model_hyperparameters[mod]

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

                        if current_val < score:
                            self.model_metrics = {}
                            self.model_metrics[mod] = {
                                "model": model,
                                "params": params,
                                "score": score,
                            }
                    

        
        self.future_predictions, self.model_name, self.pred_val, self.pred_score, self.val_size, self.conf_interval, self.predictions_conf_interval = self._output_values()

    def _output_values(self):

        for k, v in self.model_metrics.items():

            model = self.model_metrics[k]["model"]

            if k == 'arima':

                train, val = train_test_split(self.series, 0.9, pandas=True)

                print("Train Size is: {} & Validation Size is: {}".format(train.size, val.size))

                assert (train.size + val.size == self.series.n_timesteps), "Train size + Val size do not equal time series length"
                
                model_fit = model.fit(train)
                pred_val, conf_interval = model_fit.predict(val.size, return_conf_int=True)
                
                pred_score = mape(TimeSeries.from_series(val), TimeSeries.from_series(pred_val), reduction=np.median)

                full_model_fit = model.fit(self.series.pd_series())
                predictions, predictions_conf_interval = full_model_fit.predict(self.pred_interval, return_conf_int=True)
                model_name = k

                val_size = val.size 

                return TimeSeries.from_series(predictions), model_name, TimeSeries.from_series(pred_val), pred_score, val_size, conf_interval, predictions_conf_interval

            else:

                train, val = train_test_split(self.series, 0.9, pandas=False)

                print("Train Size is: {} & Validation Size is: {}".format(train.n_timesteps, val.n_timesteps))

                assert (train.n_timesteps + val.n_timesteps == self.series.n_timesteps), "Train size + Val size do not equal time series length"
                
                model_fit = model.fit(train)
                pred_val = model_fit.predict(val.n_timesteps, num_samples=100)
                pred_score = mape(val, pred_val, reduction=np.median)

                full_model_fit = model.fit(self.series)
                predictions = full_model_fit.predict(
                    self.pred_interval, num_samples=100
                )
                model_name = k

                val_size = val.n_timesteps

                return predictions, model_name, pred_val, pred_score, val_size, [], []
        

