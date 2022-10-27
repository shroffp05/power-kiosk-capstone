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
metrics = np.array([('1234', '1/1/2020', 1000, 20, 10, 'arima', 12.3, 2.2, 'null')])

@dataclass
class modeling:

    series: TimeSeries.from_dataframe(pd.DataFrame(data=arr, columns=['time', 'val']), 'time', 'val')
    models_dict: Dict[str, str] = field(default_factory=lambda: {"arima": ARIMA()})
    model_hyperparameters: Dict[str, str] = field(default_factory=lambda: {'arima': {
                                                                                        'p':[0,1],
                                                                                        'd':[0,1],
                                                                                        'q':[0,1]
                                                                                        #'seasonal_order':[[0,0,0,12],[0,1,0,12]] #P,D,Q,yearly seasonality
                                                                                    }
                                                                            })
    pred_interval: int = 12
    contractLocationID: str = ""

    def _modeling(self):

        self.model_metrics = {}
        for mod, mod_instantiation in self.models_dict.items(): #mod is key and instantiation is the value (model)
            
            print(mod, mod_instantiation)

            try:
                param_grid = self.model_hyperparameters[mod]
                model, params, score = mod_instantiation.gridsearch(parameters=param_grid, 
                    series=self.series, 
                    forecast_horizon = 1, 
                    start=0.9, 
                    metric=mape,
                    reduction=np.median,
                    verbose=True)
                
            except:
                model = mod_instantiation.fit(self.series)
                params = model.model_params 
                score = self._cross_validation(self.series, model)


            if len(self.model_metrics) == 0:
                self.model_metrics[mod] = {'model':model, 'params':params, 'score':score}
            else:
                for k, v in self.model_metrics.items():
                    current_val = self.model_metrics[k]['score']

                if current_val < score:
                    self.model_metrics = {}
                    self.model_metrics[mod] = {'model':model, 'params':params, 'score':score}
            

        self.future_predictions, self.model_name, self.pred_val, self.pred_score = self._output_values()


    def _output_values(self):


        for k,v in self.model_metrics.items():

            model = self.model_metrics[k]['model']

            len_series = self.series.n_timesteps
            train, val = self.series[:len_series-5], self.series[len_series-5:]
            model_fit = model.fit(train)
            pred_val = model_fit.predict(5, num_samples=100)
            pred_score = mape(val, pred_val, reduction=np.median)

            full_model_fit = model.fit(self.series)
            predictions = full_model_fit.predict(self.pred_interval, num_samples=100)
            model_name = k 

        return predictions, model_name, pred_val, pred_score

       
    def _cross_validation(self,series,model):
        backtests = model.backtest(series,
                                start=.5,
                                forecast_horizon=1,metric = mape, reduction=np.median)# change forecasting horizon to 1 so it increments by 1
        
        return backtests
    


        