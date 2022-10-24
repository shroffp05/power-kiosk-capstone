import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import datetime

st.set_option("deprecation.showPyplotGlobalUse", False)


def main():
    st.title('Power Kiosk Forecasting Results')
    st.write('Please input the output csv file to visualize results')
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:

   
      df = pd.read_csv(uploaded_file)
      
      st.write(df)
      df["period_ts"] = pd.to_datetime(df["period"])
      num_cl = df['contractLocationID'].nunique()
      st.header('Number of Contract Locations Evaluated = %s ' % num_cl)
      avg_client_mdape = np.mean(df['client_mdape'].unique())
      avg_capstone_mdape = np.mean(df['capstone_mdape'].unique())
      st.header('Average Mdape for Client Forecasts = %s ' % avg_client_mdape + '%')
      st.header('Average Mdape for Capstone Forecasts = %s ' % avg_capstone_mdape + '%')
      unique_cl = df['contractLocationID'].unique()
      option = st.selectbox('Please select a Contract Location to view',
    unique_cl)

      unique_df = df[df['contractLocationID']==option]

      st.line_chart(unique_df,x='period_ts',y=['usage','client_forecast','capstone_forecast'])



if __name__ == "__main__":
    main()


