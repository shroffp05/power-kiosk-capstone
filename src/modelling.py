import pandas as pd
import numpy as np
import math
from darts import TimeSeries
from darts.models import (
    NaiveSeasonal,
    NaiveDrift,
    Prophet,
    ExponentialSmoothing,
    ARIMA,
    AutoARIMA
)
from dataclasses import dataclass 
import math
from darts.metrics import mape, mase, mae, mse, ope, r2_score, rmse, rmsle
from dataclasses import field
from typing import Dict


arr = np.array([(1, 10),(2,20),(3,30),(4,40),(5,50)])


@dataclass
class modeling:

    series: TimeSeries.from_dataframe(pd.DataFrame(data=arr, columns=['time', 'val']), 'time', 'val')
    models_dict: Dict[str, str] = field(default_factory=lambda: {"arima": AutoARIMA()})
    pred_val: int = 1
    model_metrics: Dict[str, str] = field(default_factory=lambda: {})


    def _modeling(self):

        for mod, mod_instantiation in self.models_dict.items(): #mod is key and instantiation is the value (model)
            print(mod)
            print(mod_instantiation)
            model_score = self._cross_validation(series=self.series, model=mod_instantiation)

            self.model_metrics[mod] = model_score

        return self.model_metrics
        
    def _cross_validation(self,series,model):
        backtests = model.backtest(series,
                                start=.5,
                                forecast_horizon=1,metric = mape, reduction=np.median)# change forecasting horizon to 1 so it increments by 1
        
        return backtests


        