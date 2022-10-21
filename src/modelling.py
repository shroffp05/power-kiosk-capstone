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
    models_dict: Dict[str, str] = field(default_factory=lambda: {"exponential": ExponentialSmoothing(),
                                                                    "arima": AutoARIMA()})
    pred_val: int = 1
    model_metrics: Dict[str, str] = field(default_factory=lambda: {})
    

    
    def _modeling(self):

        for mod, mod_instantiation in self.models_dict.items(): #mod is key and instantiation is the value (model)
            model_score = _cross_validation(series=self.series, mod=mod_instantiation)
            self.model_metrics[mod] = model_score
        

    def _cross_validation(mod,series):
        backtests = mod.backtest(series,
                                start=.5,
                                forecast_horizon=1,metric = mape, reduction=np.median)# change forecasting horizon to 1 so it increments by 1
        #backtests.plot(lw=3, label='{}, MAPE={:.2f}%'.format(err)) #median absolute error. median absolute percentage error
        #mean is the ols error using nuclear distance, if you have absolute distances median is better
        #means go with square errors and median goes with aboslute errors 
        return backtests




"""
modeling(series,1)


import math
y_actual = [1,2,3,4,5]
y_predicted = [1.6,2.5,2.9,3,4.1]
 
MSE = np.square(np.subtract(y_actual,y_predicted)).mean() 
 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)
"""
