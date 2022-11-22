# power-kiosk-capstone


## Project structure
```bash
.
├── config                      
│   ├── main.yaml                   # Main configuration file
│   ├── model                       # Configurations for training model
│   │   ├── model1.yaml             # First variation of parameters to train model
│   │   └── model2.yaml             # Second variation of parameters to train model
│   └── process                     # Configurations for processing data
│       ├── process1.yaml           # First variation of parameters to process data
│       └── process2.yaml           # Second variation of parameters to process data
├── data            
│   ├── final                       # data after training the model
│   ├── processed                   # data after processing
│   ├── raw                         # raw data
│   └── raw.dvc                     # DVC file of data/raw
├── docs                            # documentation for your project
├── dvc.yaml                        # DVC pipeline
├── .flake8                         # configuration for flake8 - a Python formatter tool
├── .gitignore                      # ignore files that cannot commit to Git
├── Makefile                        # store useful commands to set up the environment
├── models                          # store models
├── notebooks                       # store notebooks
├── .pre-commit-config.yaml         # configurations for pre-commit
├── pyproject.toml                  # Configure black
├── requirements.txt                # requirements for pip
├── README.md                       # describe your project
├── src                             # store source code
│   ├── __init__.py                 # make src a Python module 
│   ├── process.py                  # process data before training model
│   └── train_model.py              # train model
└── tests                           # store tests
    ├── __init__.py                 # make tests a Python module 
    ├── test_process.py             # test functions for process.py
    └── test_train_model.py         # test functions for train_model.py
```

## Set up the environment
Create and activate conda environment
```bash
make conda_create
conda activate power_kiosk
make install
```

## Install ODBC Driver 
To connect your MSSQL Database with Python follow the instructions listed [here](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server?view=sql-server-ver16)

## Run the entire pipeline
To run the entire pipeline, type:

1. If running from the base folder
```bash
python scripts/run_pipeline.py --cl <contract location id(s)> 
```

2. If running from scripts folder 
```bash
python run_pipeline.py --cl <contract location id(s)>
```

### Important details about running the pipeline

The pipeline takes 3 inputs:
- `--cl`: A single contract location ID, a list of contract location IDs (comma separated, with no space) or the keyword "all" for all contract location IDs.

- `--p`: Number of periods or months you want the forecast for. Default value is set to be 12, which means the output of the pipeline will give you 12 months forecast into the future for contract location IDs in the input.

- `--n`: Minimum number of months a contract location needs to have in order for it to be part of the model. Default value is set to be 36 months. 

### Examples 

- Single contract location ID
```bash 
python script/run_pipeline.py --cl 0082c329a35944de939acdfb5975dd23
```

- Multiple contract location IDs
```bash
python script/run_pipeline.py --cl 0082c329a35944de939acdfb5975dd23,0219a6756d3e439d84f5bb5678f40499,07e2ba4b87b04684b4ea75c5654d354d,0d674bb909474caeb24cccc0d051df92
```
**Note there is no space between two contract location IDs in the input. 

- All contract location IDs
```bash
python script/run_pipeline.py --cl all
```

## Run Streamlit to visualize output

To launch the streamlit app locally, run: 

```bash
streamlit run src/build_streamlit.py
```

