from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd
import pmdarima as pmd
from darts import TimeSeries
from darts.metrics import mae, mape, mase, mse, ope, r2_score, rmse, rmsle
from darts.models import (ARIMA, AutoARIMA, ExponentialSmoothing, NaiveDrift,
                          NaiveSeasonal, Prophet)
from darts.utils.utils import ModelMode, SeasonalityMode

arr = np.array([(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)])
metrics = np.array(
    [("1234", "1/1/2020", 1000, 20, 10, "arima", 12.3, 2.2, "null")]
)

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
        if series.n_samples < 18:
            return False 
        else:
            return True 

    

def arima_hyperparameter_update(param_grid: dict, parameters: set) -> dict:

    n_diff = parameters[0]
    ns_diff = parameters[1]
    no_of_periods = parameters[3]

    param_grid["d"]=[int(n_diff)]

    if parameters[2]:
        seasonal_order = []
        for P in range(1,4):
            for Q in range(1,4):
                s_order = [P, ns_diff, Q, 12]
                seasonal_order.append(s_order)

        param_grid["seasonal_order"] = seasonal_order

    return param_grid

@dataclass
class modeling:

    series: TimeSeries.from_dataframe(pd.DataFrame(data=arr))
    models_dict: Dict[str, str] = field(
        default_factory=lambda: {"arima": ARIMA()}
    )
    model_hyperparameters: Dict[str, str] = field(
        default_factory=lambda: {
            "arima": {"p": [0, 1, 2, 3, 4], "d": [0], "q": [0, 1, 2, 3, 4]}
        }
        
    )
    pred_interval: int = 12
    contractLocationID: str = ""
    ts_attributes: set = ()

    def _modeling(self):

        self.model_metrics = {}
        for (
            mod,
            mod_instantiation,
        ) in (
            self.models_dict.items()
        ):  # mod is key and instantiation is the value (model)

            print(mod, mod_instantiation)
            
            if model_check(mod, self.series, self.ts_attributes[3]):
                if mod == 'arima':
                    param_grid = arima_hyperparameter_update(self.model_hyperparameters[mod], self.ts_attributes)
                else:
                    param_grid = self.model_hyperparameters[mod]

                print(param_grid)

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

        (
            self.future_predictions,
            self.model_name,
            self.pred_val,
            self.pred_score,
        ) = self._output_values()

    def _output_values(self):

        for k, v in self.model_metrics.items():

            model = self.model_metrics[k]["model"]

            len_series = self.series.n_timesteps
            train, val = (
                self.series[: len_series - 5],
                self.series[len_series - 5 :],
            )
            model_fit = model.fit(train)
            pred_val = model_fit.predict(5, num_samples=100)
            pred_score = mape(val, pred_val, reduction=np.median)

            full_model_fit = model.fit(self.series)
            predictions = full_model_fit.predict(
                self.pred_interval, num_samples=100
            )
            model_name = k

        return predictions, model_name, pred_val, pred_score

    def _cross_validation(self, series, model):
        backtests = model.backtest(
            series,
            start=0.5,
            forecast_horizon=1,
            metric=mape,
            reduction=np.median,
        )  # change forecasting horizon to 1 so it increments by 1

        return backtests
