import datetime

# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.pyplot import figure
st.set_option("deprecation.showPyplotGlobalUse", False)
# flake8: noqa

def main():
    
    tcol1, tcol2 = st.columns([2, 16])
    tcol1.image("src/pk-icon.jpeg", width=80)
    original_title = '<p style=" color:#1a2574; font-size: 40px; font-weight: bold; margin-top:10px">Power Kiosk Forecasting Results</p>'
    tcol2.markdown(original_title, unsafe_allow_html=True)
    sub_title = '<p style=" font-family: Courier New ;">Please input the output csv file to visualize forecasting results</p>'
    st.markdown(sub_title, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:

        # with open('src/streamlit-design.css') as source_des:
        # st.markdown(f"<style>{source_des.read()}</style>",unsafe_allow_html=True)

        df = pd.read_csv(uploaded_file)
        df2 = df.groupby(['contractLocationID']).first().reset_index()
        winners = df2[df2['capstone_mdape']<=df2['client_mdape']]
        st.write(df.head())

        df["period_ts"] = pd.to_datetime(df["period"])
        num_cl = df["contractLocationID"].nunique()

        mdape_df = df[df["client_mdape"] > 0]

        avg_client_mdape = round(
            df2.client_mdape.mean(),
            2,
        )
        avg_capstone_mdape = round(
            df2.capstone_mdape.mean(),
            2,
        )


        individual_viz = '<p style=" color:#e17326; font-size: 30px; font-weight: bold; margin-top:10px">Key Metrics</p>'
        st.markdown(individual_viz, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Contract Locations", str(num_cl))
        col2.metric(
            "Avg Mdape for baseline model", str(avg_client_mdape) + "%"
        )
        col3.metric(
            "Avg Mdape for Capstone Forecasts",
            str(avg_capstone_mdape) + "%",
            str(round(avg_client_mdape - avg_capstone_mdape, 2)) + "%",
        )
        # col4.metric('Percentage Improvement',str(avg_client_mdape-avg_capstone_mdape)+'%')

        individual_viz = '<div style="background-color:#1a2574; width:710px; height: 5px;margin-bottom:20px;"></div>'
        st.markdown(individual_viz, unsafe_allow_html=True)

        col4, col5,col6 = st.columns(3)
        
        n_winners = winners.shape[0]
        avg_client_mdape_winners = winners['client_mdape'].mean()
        avg_capstone_mdape_winners = winners['capstone_mdape'].mean()
        col4.metric("Contracts Better Than Baseline", str(n_winners))
        col5.metric("Avg Mdape For Baseline Model", str(round(avg_client_mdape_winners,2))+'%')
        col6.metric("Avg Mdape For Capstone Forecasts", str(round(avg_capstone_mdape_winners,2))+'%',
        str(round(avg_client_mdape_winners - avg_capstone_mdape_winners, 2)) + "%")

        individual_viz = '<div style="background-color:#1a2574; width:710px; height: 5px;margin-bottom:20px;"></div>'
        st.markdown(individual_viz, unsafe_allow_html=True)

        col4, col5,col6 = st.columns(3)
        losers= df2[df2['capstone_mdape']>df2['client_mdape']]
        n_losers = losers.shape[0]
        avg_client_mdape_losers = losers['client_mdape'].mean()
        avg_capstone_mdape_losers = losers['capstone_mdape'].mean()
        col4.metric("Contracts Worse Than Baseline", str(n_losers))
        col5.metric("Avg Mdape For Baseline Model", str(round(avg_client_mdape_losers,2))+'%')
        col6.metric("Avg Mdape For Capstone Forecasts", str(round(avg_capstone_mdape_losers,2))+'%',
        str(round( avg_client_mdape_losers - avg_capstone_mdape_losers, 2)) + "%")

        individual_viz = '<div style="background-color:#1a2574; width:710px; height: 5px;margin-bottom:20px;"></div>'
        st.markdown(individual_viz, unsafe_allow_html=True)

        individual_viz = '<p style=" color:#e17326; font-size: 30px; font-weight: bold; margin-top:10px">Best Model Breakdown</p>'
        st.markdown(individual_viz, unsafe_allow_html=True)

        best_mod_breakdown = winners.best_model.value_counts().reset_index()
        base = pd.DataFrame({'index':'baseline','best_model':n_losers},index=[3])
        best_mod_breakdown = pd.concat([best_mod_breakdown,base])
        fig, ax = plt.subplots()
      
        #ax.bar_label(bars)
        ax.barh(
            best_mod_breakdown['index'],
            best_mod_breakdown["best_model"],
            label="Actual Historical Usage",
            color = ('#1a2574','#1a2574','#1a2574','#e17326'),
        
        )
       

        plt.xticks(rotation=45)
        plt.xlabel("Count")
        plt.ylabel("Best Model")
        
        st.pyplot(fig)

        individual_viz = '<p style=" color:#e17326; font-size: 30px; font-weight: bold; margin-top:10px">Contract Location Level Results</p>'
        st.markdown(individual_viz, unsafe_allow_html=True)

        view = st.selectbox("Select Contract Category to View",('Contracts better than baseline', 'Contracts worse than baseline'))


        
        start_df = df2

        if(view=='Contracts better than baseline'):
          start_df = start_df[start_df['capstone_mdape']<=start_df['client_mdape']]
        if(view=='Contracts worse than baseline'):
          start_df = start_df[start_df['capstone_mdape']>start_df['client_mdape']]
        
        filters = st.multiselect('Select Filters',['best_model=arima','best_model=prophet'
        ,'best_model=exponential'])

         
        for i in filters:

          if(i=='best_model=arima'):
            start_df = start_df[start_df['best_model']=='arima']
          if(i=='best_model=prophet'):
            start_df = start_df[start_df['best_model']=='prophet']
          if(i=='best_model=exponential'):
            start_df = start_df[start_df['best_model']=='exponential']
          


        unique_cl = start_df["contractLocationID"].unique()
        option = st.selectbox(
            "Please select a Contract Location to view", unique_cl
        )

        unique_df = df[df["contractLocationID"] == option]
        unique_df["usage"] = unique_df["usage"].replace(0, np.nan)
        st.write(unique_df.head())
        chart_data = unique_df[
            [
                "period_ts",
                "usage",
                "client_forecast",
                "capstone_forecast",
                "low_confidence_int",
                "high_confidence_int",
                "best_model",
                "client_mdape",
                "capstone_mdape",
            ]
        ]
        chart_data = chart_data.set_index("period_ts")
        # st.line_chart(data=chart_data)

        fig, ax = plt.subplots()
        ax.plot(
            chart_data.index,
            chart_data["usage"],
            label="Actual Historical Usage",
            linewidth=2.5,
        )
        ax.plot(
            chart_data.index,
            chart_data["capstone_forecast"],
            label="Capstone Forecast",
            linewidth=2.5,
        )
        ax.plot(
            chart_data.index,
            chart_data["client_forecast"],
            label="Power Kiosk Forecast",
        )
        ax.plot(
            chart_data.index,
            chart_data["low_confidence_int"],
            label="Confidence Interval Lower Bound",
            linestyle="dashed",
        )
        ax.plot(
            chart_data.index,
            chart_data["high_confidence_int"],
            label="Confidence Interval Upper Bound",
            linestyle="dashed",
        )
        ax.fill_between(
            chart_data.index,
            chart_data["low_confidence_int"],
            chart_data["high_confidence_int"],
            color="b",
            alpha=0.1,
            linestyle="dashed",
        )
        plt.xticks(rotation=45)
        plt.grid()
        plt.xlabel("Period")
        plt.ylabel("Monthly Electricity Usage")
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=3,
            fancybox=True,
            shadow=True,
        )
        st.pyplot(fig)

        icol1, icol2, icol3 = st.columns(3)
        icol1.metric("Best Model", chart_data["best_model"][0])
        icol2.metric(
            "Baseline Mdape",
            str(round(chart_data["client_mdape"][0], 2)) + "%",
        )
        icol3.metric(
            "Capstone Mdape",
            str(round(chart_data["capstone_mdape"][0], 2)) + "%",
            str(
                round(
                    chart_data["client_mdape"][0]
                    - chart_data["capstone_mdape"][0],
                    2,
                )
            )
            + "%",
        )


if __name__ == "__main__":
    main()
