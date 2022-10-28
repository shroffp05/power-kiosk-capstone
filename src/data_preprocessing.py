import datetime as dt
import math
import numbers
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from scipy import signal
from scipy.stats import kruskal
from statsmodels.tsa.stattools import adfuller


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

        df_reindex["data_thresh_achieved"] = check_data_length(
            df_reindex["clean_usage"]
        )
        df_reindex["has_zero_usage_values"] = check_zero_usage(
            df_reindex["clean_usage"]
        )

        clean_df = df_reindex[
            [
                "contractLocationID",
                "period_clean",
                "clean_usage",
                "data_thresh_achieved",
                "has_zero_usage_values",
            ]
        ]

        # out_df = out_df.append(clean_df)
        out_df = pd.concat([out_df, clean_df])
        print("%d Contract Location ID's have been cleaned and added" % count)
        count = count + 1
    return out_df


def get_TS(df):

    ts_df = TimeSeries.from_group_dataframe(
        df,
        "contractLocationID",
        time_col="period_ts",
        value_cols="usage",
        fill_missing_dates=True,
        freq="MS",
    )

    return ts_df


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
