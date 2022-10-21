#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install darts


# In[4]:


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
    #Theta
)

import math
from darts.metrics import mape, mase, mae, mse, ope, r2_score, rmse, rmsle
# from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis
# from darts.dataprocessing.transformers.boxcox import BoxCox




large_N = 100



df = pd.read_csv('AirPassengers.csv')
series = TimeSeries.from_dataframe(df, 'Month', '#Passengers')


# In[12]:


#if df.shape[0] >= large_N:
#    training,validation = series.split_before(0.8)  ## change this
    #test_df = series.split_after(0.2)   ## change this
#else:
#    training,validation = series.split_before(0.7)
    #test_df = series.split_after(0.3)
    
models_dict = {
    "exponential" : ExponentialSmoothing(),
    "arima": AutoARIMA(),
    #"prophet": Prophet(),
    #"fourrier": FFT()
}

def modeling(series, pred_num):
    model_metrics = {}
    for mod, mod_instantiation in models_dict.items(): #mod is key and instantiation is the value (model)
        model_score = cross_validation(series, mod_instantiation)
        model_metrics[mod] = model_score
    return model_metrics
    

def cross_validation(series, mod):
    backtests = mod.backtest(series,
                            start=.5,
                            forecast_horizon=1,metric = mape, reduction=np.median)# change forecasting horizon to 1 so it increments by 1
    #backtests.plot(lw=3, label='{}, MAPE={:.2f}%'.format(err)) #median absolute error. median absolute percentage error
    #mean is the ols error using nuclear distance, if you have absolute distances median is better
    #means go with square errors and median goes with aboslute errors 
    return backtests



modeling(series,1)


import math
y_actual = [1,2,3,4,5]
y_predicted = [1.6,2.5,2.9,3,4.1]
 
MSE = np.square(np.subtract(y_actual,y_predicted)).mean() 
 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)

