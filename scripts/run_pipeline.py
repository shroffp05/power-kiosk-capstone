

# import hydra
# from omegaconf import DictConfig
# from hydra.utils import to_absolute_path as abspath
import argparse
from jinjasql import JinjaSql 
import sys 
import os 

current_path = os.getcwd() 

sys.path.insert(0, current_path+"/src")

from sql_connection import connect_to_sql

"""
@hydra.main(config_path="../config", config_name="main")
def train_model(config: DictConfig):

    input_path = abspath(config.processed.path)
    output_path = abspath(config.final.path)

    print(f"Train modeling using {input_path}")
    print(f"Model used: {config.model.name}")
    print(f"Save the output to {output_path}")
"""

def set_param(user_arg: str) -> dict:
    
    params = {}

    if user_arg == "all":
        params["contract_location_id"] = """SELECT contractLocationID FROM ViewContractLocationUsageHistories"""
    else:
        params["contract_location_id"] = user_arg

    return params


def apply_sql_template(template: str, parameters: dict) -> str:

    """
    Apply a JinjaSql template (string) substituting parameters (dict) and return the final SQL.
    """

    j = JinjaSql(param_style='pyformat')
    query, bind_params = j.prepare_query(template, parameters)
    return query%bind_params

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

    print(args.cl)

    contract_location_ids = []

    with open(current_path+'/data/sql_file.txt') as f:
        sql_code = f.read()
    
    params = set_param(args.cl)
    sql_string = apply_sql_template(sql_code, params)

    conn = connect_to_sql()
    conn._sql_connection()
    results_df = conn._execute_sql_statement(sql_string)

    print(results_df.head())


    