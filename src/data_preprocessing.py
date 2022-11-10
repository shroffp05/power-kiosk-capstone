import datetime as dt
import math
import numbers
import sys
import time
from darts.utils.statistics import check_seasonality
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from scipy import signal
from scipy.stats import kruskal
from statsmodels.tsa.stattools import adfuller
import pmdarima as pmd
# flake8: noqa

def clean_data(df):
    # list of unique contract locations

    unique_ids = df.contractLocationID.unique()

    # drop all columns that are not relevant
    col_list = ["contractLocationID", "addDate", "period", "usage"]
    df = df[col_list]

    # change data type to time series
    df["addDate_ts"] = pd.to_datetime(df["addDate"])
    df["addDate_ts"] = df["addDate_ts"].dt.time
    df["period_ts"] = pd.to_datetime(df["period"])

    # initialize output dataframe
    out_df = pd.DataFrame()
    count = 1
    # loop through unique contracts
    for u_id in unique_ids:
        print('Current CLID is:%s' % u_id)
        ALPHA = 0.05
        sub_df = df[df["contractLocationID"] == u_id]
        # drop duplicates due to multiple pulls
        sub_df = sub_df.sort_values(
            ["period_ts", "addDate_ts"]
        ).drop_duplicates("period", keep="last")

        # reindex to deal with missing months
        sub_df = sub_df.set_index("period_ts")
        date_index = pd.date_range(
            start=str(sub_df.index[0]), end=str(sub_df.index[-1]), freq="MS"
        )

        df_reindex = sub_df.reindex(date_index)
        df_reindex["contractLocationID"] = df_reindex[
            "contractLocationID"
        ].replace(np.nan, df_reindex["contractLocationID"][0])
        df_reindex["addDate"] = df_reindex["addDate"].replace(
            np.nan, df_reindex["addDate"][0]
        )
        df_reindex["period"] = df_reindex["period"].replace(
            np.nan, df_reindex["period"][0]
        )
        df_reindex["addDate_ts"] = df_reindex["addDate_ts"].replace(
            np.nan, df_reindex["addDate_ts"][0]
        )
        clean_df = (
            df_reindex["usage"]
            .interpolate(method="time")
            .to_frame()
            .reset_index()
        )
        df_reindex["clean_usage"] = clean_df["usage"].values
        df_reindex = df_reindex.reset_index()
        df_reindex["period_clean"] = pd.to_datetime(df_reindex["index"])
        df_reindex = df_reindex.drop(["period"], axis=1)

        # get seasonality and stationarity flags

        # cl_stationarity, cl_seasonal_yearly = get_flags(cl_series)

        # df_reindex["stationarity_flag"] = cl_stationarity
        # df_reindex["yearly_seasonality"] = cl_seasonal_yearly
        """
        df_reindex["data_thresh_achieved"] = check_data_length(
            df_reindex["clean_usage"]
        )
        """
        df_reindex["series_len"] = [
            df_reindex.shape[0] for i in range(df_reindex.shape[0])
        ]
        df_reindex["has_zero_usage_values"] = check_zero_usage(
            df_reindex["clean_usage"]
        )
        if(df_reindex['series_len'].mean()>3):
            ts = get_TS(df_reindex)
            
            is_seasonal, mseas = seasonality_check(ts)

            y = np.asarray(df_reindex['clean_usage'])
            y= y[~np.isnan(y)]
            try:
                n_kpss = pmd.arima.ndiffs(y, alpha=ALPHA, test='kpss', max_d=2)
                n_adf = pmd.arima.ndiffs(y, alpha=ALPHA, test='adf', max_d=2)
                n_diff = max(n_adf, n_kpss)
            
                n_ocsb = pmd.arima.OCSBTest(m=max(4,mseas)).estimate_seasonal_differencing_term(y)
                ns_ch = pmd.arima.CHTest(m=max(4,mseas)).estimate_seasonal_differencing_term(y)
                ns_diff = max(n_ocsb, n_ch, is_seasonal * 1)

            except:
                
                n_diff = pmd.arima.ndiffs(y, alpha=ALPHA, test='kpss', max_d=2)
                #ns_diff = max(pmd.arima.CHTest(m=max(4,mseas)).estimate_seasonal_differencing_term(y), is_seasonal*1)
                ns_diff = is_seasonal*1

            df_reindex['seasonality_flag']= is_seasonal
            df_reindex['number_seasons']= mseas
            df_reindex['first_diff']= n_diff
            df_reindex['seasonal_diff']= ns_diff

        
            clean_df = df_reindex[
                [
                    "contractLocationID",
                    "period_clean",
                    "clean_usage",
                    "has_zero_usage_values",
                    "seasonality_flag",
                    "number_seasons",
                    "first_diff",
                    "seasonal_diff",
                    "series_len",
                ]
            ]

            out_df = pd.concat([out_df, clean_df])
            
            print("%d Contract Location ID's have been cleaned and added" % count)
            count = count + 1
    return out_df


def get_TS(df):

    ts = TimeSeries.from_dataframe(
        df,
        "period_clean",
        "clean_usage",
        fill_missing_dates=True,
        freq=None,
    )

    return ts


def get_flags(ser):
    stat = False
    seasonal = False
    result = adfuller(ser)
    if result[1] <= 0.05:
        stat = True

    # f, Pxx_den = signal.periodogram(ser)
    # per = int(1 / max(Pxx_den))
    idx = np.arange(len(ser)) % 5
    H_statistic, p_value = kruskal(ser, idx)
    if p_value <= 0.05:
        seasonal = True
    return stat, seasonal


def check_data_length(ser):
    if len(ser) >= 35:
        return 1
    else:
        return 0


def check_zero_usage(ser):
    if 0 in ser.values:
        return 1
    else:
        return 0


def seasonality_check(ser):
    ALPHA = 0.05
    for m in range(2, 25):
        is_seasonal, mseas = check_seasonality(ser, m=m, alpha=ALPHA)
        if is_seasonal:
            break

    return is_seasonal, mseas