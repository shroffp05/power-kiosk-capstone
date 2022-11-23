# Power Kiosk Capstone - Forecasting Energy Usage


## Set up the environment

If running on MacOS
Create and activate conda environment
```bash
make conda_create
conda activate power_kiosk
make install
```

If running on Windows
```bash
conda create --name power_kiosk python=3.9 pip
conda activate power_kiosk
conda install -c conda-forge -c pytorch u8darts-all
pip install -r requirements.txt 
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
python scripts/run_pipeline.py --cl 0082c329a35944de939acdfb5975dd23
```

- Multiple contract location IDs
```bash
python scripts/run_pipeline.py --cl 0082c329a35944de939acdfb5975dd23,0219a6756d3e439d84f5bb5678f40499,07e2ba4b87b04684b4ea75c5654d354d,0d674bb909474caeb24cccc0d051df92
```
**Note there is no space between two contract location IDs in the input. 

- All contract location IDs
```bash
python scripts/run_pipeline.py --cl all
```

- Changing the number of periods to forecast from 12 to 5 
```bash 
python scripts/run_pipeline.py --cl 0082c329a35944de939acdfb5975dd23 --p 5
```

- Changing the minimum number of months required from 36 to 30
```bash 
python scripts/run_pipeline.py --cl 0082c329a35944de939acdfb5975dd23 --n 30
```

## Run Streamlit to visualize output

To launch the streamlit app locally, run: 

```bash
streamlit run src/build_streamlit.py
```

