import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import datetime


st.set_option("deprecation.showPyplotGlobalUse", False)


def main():

    tcol1,tcol2 = st.columns([2,16])
    tcol1.image('src/pk-icon.jpeg',width=80)
    original_title = '<p style=" color:#1a2574; font-size: 40px; font-weight: bold; margin-top:10px">Power Kiosk Forecasting Results</p>'
    tcol2.markdown(original_title, unsafe_allow_html=True)
    sub_title = '<p style=" font-family: Courier New ;">Please input the output csv file to visualize forecasting results</p>'
    st.markdown(sub_title, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
      
      #with open('src/streamlit-design.css') as source_des:
        #st.markdown(f"<style>{source_des.read()}</style>",unsafe_allow_html=True)
   
      df = pd.read_csv(uploaded_file)
      
      st.write(df)
      df["period_ts"] = pd.to_datetime(df["period"])
      num_cl = df['contractLocationID'].nunique()
      avg_client_mdape = np.mean(df['client_mdape'].unique())
      avg_capstone_mdape = np.mean(df['capstone_mdape'].unique())
      col1, col2, col3 = st.columns(3)
      col1.metric('Number of Contract Locations',str(num_cl))
      col2.metric('Avg Mdape for Client Forecasts',str(avg_client_mdape)+'%')
      col3.metric('Avg Mdape for Capstone Forecasts',str(avg_capstone_mdape)+'%')
      unique_cl = df['contractLocationID'].unique()
      option = st.selectbox('Please select a Contract Location to view',
    unique_cl)

      unique_df = df[df['contractLocationID']==option]
      chart_data = unique_df[['period_ts','usage','client_forecast','capstone_forecast']]
      chart_data = chart_data.set_index('period_ts')
      st.line_chart(data=chart_data)
     

if __name__ == "__main__":
    main()


