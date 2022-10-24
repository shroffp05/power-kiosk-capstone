import argparse
from jinjasql import JinjaSql 
import sys 
import os 
from six import string_types
from copy import deepcopy 
from darts import TimeSeries 
import dateutil.relativedelta
import pandas as pd 
import numpy as np 

current_path = os.getcwd() 

sys.path.insert(0, current_path+"/src")

from sql_connection import connect_to_sql
from modelling import modeling 
from data_preprocessing import clean_data 

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
        params["contract_location_id"] = """SELECT contractLocationID FROM ViewContractLocationUsageHistories"""
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
    
    j = JinjaSql(param_style='pyformat')
    query, bind_params = j.prepare_query(template, parameters)

    if not bind_params:
        return query
    params = deepcopy(bind_params)

    for key, val in params.items():
        split_val = val.split(",")
        output = ""
        for i in split_val:
            output = output+quote_sql_string(i.strip())+","

        params[key] = output[:-1] 

    return query%params


def get_client_metrics(df: pd.DataFrame) -> float:

    """
    Function takes the timeseries dataframe as an input and returns
    the average usage of the contractLocation over the past year
    """

    end_date = df['period_clean'].max().to_pydatetime()
    start_date = end_date - dateutil.relativedelta.relativedelta(months=12)

    filter_df = df.loc[(df['period_clean']>=start_date) & (df['period_clean']<=end_date), :]

    print("Start Date: {}".format(start_date))
    print("End Date: {}".format(end_date))

    return filter_df["clean_usage"].sum()/12


def get_forecast_metrics(ids: list, df: pd.DataFrame) -> dict: 

    """
    Function loops through each contract location ID in `ids` and filters the dataframe for that ID
    It creates a Timeseries object that is being used as an input for different time series models 
    and outputs the best model based on Median Absolute Percentage Error.
    """
    
    output = {}
    count = 1
    for id in ids:
        print("Contract Location ID {}: {}".format(count, id))
        filter_df = df.loc[df['contractLocationID']==id, :]
        series = TimeSeries.from_dataframe(filter_df, 'period_clean', 'clean_usage', fill_missing_dates=True, freq=None)

        model = modeling(series=series)
        results = model._modeling()
        results["client_forecast"] = get_client_metrics(df)

        output[id] = results 
        count += 1

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
            """
        )

    args = parser.parse_args()

    print("Arguments Passed: {}".format(args.cl))

    contract_location_ids = []

    with open(current_path+'/data/sql_file.txt') as f:
        sql_code = f.read()
    
    params = set_param(args.cl)
    sql_string = apply_sql_template(sql_code, params)
    conn = connect_to_sql()
    conn._sql_connection()
    results_df = conn._execute_sql_statement(sql_string)
    results_df["period"] = results_df["NewPeriod"].astype(str)

    df = clean_data(results_df)
    unique_clocid = df["contractLocationID"].unique().tolist()

    output = get_forecast_metrics(unique_clocid, df)
    print(output)
    


    