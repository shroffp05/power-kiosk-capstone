import argparse
import os
import sys
from copy import deepcopy
from datetime import timedelta
from datetime import datetime  
import dateutil.relativedelta
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mape
from jinjasql import JinjaSql
from six import string_types

current_path = os.getcwd()

sys.path.insert(0, current_path + "/src")

from data_preprocessing import clean_data  # noqa: E402
from modelling import modeling  # noqa: E402
from sql_connection import connect_to_sql  # noqa: E402

pd.options.mode.chained_assignment = None  # default='warn'
# flake8: noqa

def set_param(user_arg: str) -> dict:

    """
    Function takes in the user input and creates a dictionary based on the input.
    Possible values include:
        1. The individual contract location ID
        2. A list of contract location IDs
        3. A SQL subquery that outputs all contract location IDs
    """

    params = {}

    if user_arg == "all":
        params[
            "contract_location_id"
        ] = """SELECT contractLocationID FROM ViewContractLocationUsageHistories"""
    else:
        params["contract_location_id"] = user_arg

    return params


def quote_sql_string(value):

    """
    If `value` is a string type, escapes single quotes in the string
    and returns the string enclosed in single quotes.
    """

    if isinstance(value, string_types):
        new_value = str(value)
        new_value = new_value.replace("'", "''")
        return "'{}'".format(new_value)
    return value


def apply_sql_template(template: str, parameters: dict) -> str:

    """
    Apply a JinjaSql template (string) substituting parameters (dict) and return the final SQL.
    """

    j = JinjaSql(param_style="pyformat")
    query, bind_params = j.prepare_query(template, parameters)

    if not bind_params:
        return query
    params = deepcopy(bind_params)

    for key, val in params.items():
        if (
            val
            != "SELECT contractLocationID FROM ViewContractLocationUsageHistories"
        ):
            split_val = val.split(",")
            output = ""
            for i in split_val:
                output = output + quote_sql_string(i.strip()) + ","

            params[key] = output[:-1]

    return query % params


def get_client_metrics(df: pd.DataFrame) -> list:

    """
    Function takes the timeseries dataframe as an input and returns
    the average usage of the contractLocation over the past year as well as the year before
    """

    end_date = pd.to_datetime(df["period_clean"]).max().to_pydatetime()
    start_date = end_date - dateutil.relativedelta.relativedelta(months=12)

    client_df = df.loc[
        (pd.to_datetime(df["period_clean"]) >= start_date) & (pd.to_datetime(df["period_clean"]) <= end_date),
        :,
    ]

    print("Start Date: {}".format(start_date))
    print("End Date: {}".format(end_date))

    end_date_2 = pd.to_datetime(df[
        "period_clean"
    ]).max().to_pydatetime() - dateutil.relativedelta.relativedelta(months=12)
    start_date_2 = end_date_2 - dateutil.relativedelta.relativedelta(months=12)

    client_df_2 = df.loc[
        (pd.to_datetime(df["period_clean"]) >= start_date_2)
        & (pd.to_datetime(df["period_clean"]) <= end_date_2),
        :,
    ]

    print("Start Date: {}".format(start_date_2))
    print("End Date: {}".format(end_date_2))

    return [
        client_df["clean_usage"].sum() / 12,
        client_df_2["clean_usage"].sum() / 12,
    ]  # Assuming we have atleast 24 months data


def get_forecast_metrics(
    ids: list, df: pd.DataFrame, forecast_period: int
) -> dict:

    """
    Function loops through each contract location ID in `ids` and filters the dataframe for that ID
    It creates a Timeseries object that is being used as an input for different time series models
    and outputs the best model based on Median Absolute Percentage Error.
    """

    output = pd.DataFrame()
    count = 1
    for id in ids:
        print("Contract Location ID {}: {}".format(count, id))

        filter_df = df.loc[df["contractLocationID"] == id, :]

        series = TimeSeries.from_dataframe(
            filter_df,
            "period_clean",
            "clean_usage",
            fill_missing_dates=True,
            freq=None,
        )

        print(
            "For this contract location ID, data starts from {} till {}".format(
                series.start_time(), series.end_time()
            )
        )
        n_diff, ns_diff =  filter_df["first_diff"].iloc[0], filter_df["seasonal_diff"].iloc[0]
        seasonality, no_of_seasons = filter_df["seasonality_flag"].iloc[0], filter_df["number_seasons"].iloc[0]

        model = modeling(
            series=series, contractLocationID=id, pred_interval=forecast_period, ts_attributes=(n_diff, ns_diff, seasonality, no_of_seasons)
        )

        model._modeling()

       

        future_pred = model.future_predictions
        model_name = model.model_name
        pred_val = model.pred_val
        pred_score = model.pred_score
        val_size = model.val_size

        if all([future_pred==None, model_name == None, pred_val == None, pred_score == None, val_size == None]):
            count = count + 1
            continue
        

        if len(model.predictions_conf_interval) == 0:
            future_conf_int = future_pred.quantiles_df((0.05, 0.95))
        else:
            future_conf_int = pd.DataFrame(model.predictions_conf_interval, columns=['clean_usage_0.05', 'clean_usage_0.95'])

        if len(model.conf_interval) == 0:
            pred_conf_int = pred_val.quantiles_df((0.05, 0.95))
        else:
            pred_conf_int = pd.DataFrame(model.conf_interval, columns=['clean_usage_0.05', 'clean_usage_0.95'])

        # Setting up variables for csv output

        contract_location_id = [id for i in range(val_size + forecast_period)]

        period_list = pd.date_range(
            series.end_time()
            - dateutil.relativedelta.relativedelta(
                months=val_size
            ),
            series.end_time()
            + dateutil.relativedelta.relativedelta(months=forecast_period),
            freq="M",
        ).tolist() 


        usage = filter_df.loc[
            (pd.to_datetime(filter_df["period_clean"]) >= min(period_list))
            & (pd.to_datetime(filter_df["period_clean"]) <= max(period_list)),
            "clean_usage",
        ].tolist()

        usage = usage + [0 for i in range(forecast_period)]

        client_forecast_values = get_client_metrics(filter_df)
        client_forecast = [client_forecast_values[1] for i in range(val_size)] + [
            client_forecast_values[0] for i in range(forecast_period)
        ]

        if model_name == 'arima':
            capstone_forecast = (pred_val.pd_series().tolist()
            + future_pred.pd_series().tolist())
            
        else:
            capstone_forecast = (
                pred_val.mean().pd_series().tolist()
                + future_pred.mean().pd_series().tolist()
            )
        best_model = [model_name for i in range(val_size + forecast_period)]

        client_mdape = [
            mape(
                TimeSeries.from_values(np.asarray(usage[0:val_size])),
                TimeSeries.from_values(np.asarray(client_forecast[0:val_size])),
                reduction=np.median,
            )
            for i in range(val_size)
        ] + [0 for i in range(forecast_period)]

        capstone_mdape = [pred_score for i in range(val_size)] + [
            0 for i in range(forecast_period)
        ]

        low_confidence_int = (
            pred_conf_int["clean_usage_0.05"].tolist()
            + future_conf_int["clean_usage_0.05"].tolist()
        )
        high_confidence_int = (
            pred_conf_int["clean_usage_0.95"].tolist()
            + future_conf_int["clean_usage_0.95"].tolist()
        )

        final_df = pd.DataFrame(
            columns=[
                "contractLocationID",
                "period",
                "usage",
                "client_forecast",
                "capstone_forecast",
                "best_model",
                "client_mdape",
                "capstone_mdape",
                "low_confidence_int",
                "high_confidence_int",
            ]
        )

        final_df["contractLocationID"] = contract_location_id
        final_df["period"] = period_list
        final_df["usage"] = usage
        final_df["client_forecast"] = client_forecast
        final_df["capstone_forecast"] = capstone_forecast
        final_df["best_model"] = best_model
        final_df["client_mdape"] = client_mdape
        final_df["capstone_mdape"] = capstone_mdape
        final_df["low_confidence_int"] = low_confidence_int
        final_df["high_confidence_int"] = high_confidence_int

        output = pd.concat([output, final_df])
        count = count + 1

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass a contract location ID")

    parser.add_argument(
        "--cl",
        type=str,
        help="""
            Type a contract location ID in (2356) or
            Type a list of contract location IDs comma seperated (2356, 4567, 789) or
            Type all to use all location IDs
            """,
    )

    parser.add_argument(
        "--p",
        type=int,
        default=12,
        help="""
            Type the number of months you want to predict for. Defaults to 12.
        """,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=36,
        help="""
            Type the minimum number of months the contract location's should have for training a model. Defaults to 36.
        """
        )

    args = parser.parse_args()

    print(
        """Arguments Passed:
                - Contract Location ID : {}
                - Number of forecast periods: {}
                - Minimum number of months: {}
            """.format(
            args.cl, args.p, args.n
        )
    )

    contract_location_ids = []

    with open(current_path + "/data/sql_file.txt") as f:
        sql_code = f.read()

    params = set_param(args.cl)
    sql_string = apply_sql_template(sql_code, params)
    print(sql_string)
    conn = connect_to_sql()
    conn._sql_connection()
    results_df = conn._execute_sql_statement(sql_string)
    results_df["period"] = results_df["NewPeriod"].astype(str)
    df = clean_data(results_df)
    df = df[df["has_zero_usage_values"] == 0]
    
    # df.to_csv('cleaned_database_711.csv')
    t_start = datetime.now()
    #df = pd.read_csv('cleaned_database_711.csv')

    df = df[df['series_len']>=int(args.n)]
    df = df[(df.contractLocationID != '588f179570c539450170d5375dcf0bdf') & (df.contractLocationID !='588f17956909297e01691632b1100718') 
    &(df.contractLocationID != '588f179570c539450170d539e92e0cc8')&(df.contractLocationID != '588f179570f3ee190170f783f7880304')&(df.contractLocationID != '58464f00667bb4050166a3034b4c0795')&(df.contractLocationID != 'f24bfe43ba0b4bc5a1e01db7b37827cf')]
    #df = df[df[df['contractLocationID'] == '588f179570c539450170d539e8db0cc7'].index[0]:]
    unique_clocid = df["contractLocationID"].unique().tolist()
    output = get_forecast_metrics(unique_clocid, df, int(args.p))
    #output = get_forecast_metrics(unique_clocid, df,12)
  
    time_stamp = datetime.now() 
    output.to_csv("results/predictions-{}.csv".format(time_stamp))
    #output.to_csv('predictions_11_11.csv')
    t_end = datetime.now()
    print('TOTAL TIME TAKEN TO FINISH:')
    print(t_end-t_start)
    
