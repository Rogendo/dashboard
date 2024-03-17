import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from streamlit_extras.colored_header import colored_header
from streamlit_extras.card import card
import pandas as pd
from ydata_profiling import ProfileReport
import time
#from streamlit_elements import elements, mui, html
import os
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import sklearn.metrics as metrics

# Set page config
st.set_page_config(
    page_title="Electricity Consumption Data Analysis",
    page_icon="random",
    initial_sidebar_state='expanded',
    menu_items={
        'Get Help': 'mailto:faochieng@kabarak.ac.ke',
        'report a bug': 'mailto:faochieng@kabarak.ac.ke',
        "About": "This website is created by Fredrick Ochieng showcasing a deep exploratory data analysis of three zones in Morocco",
    }
)
@st.cache_data
def get_df():
    return load_data()

@st.cache_data
def get_profile_report(df):
    return ProfileReport(df, title="Electricity consumption Profile Report")

def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('powerconsumption.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    return df


@st.cache_data
def convert_df(df_frame):
    return df_frame.to_csv().encode('utf-8')

def create_card(image_path, title, description):
    # HTML/CSS for styling the card
    st.markdown(
        f"""
        <style>
            .card {{
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
                background-color: #f5f5f5;
                margin-bottom: 20px;
            }}
            .card-title {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin-bottom: 10px;
            }}
            .card-description {{
                font-size: 16px;
                color: #666;
            }}
        </style>
        <div class="card">
            <img src="./{image_path}" alt="chart" style="width:100%">
            <div class="card-title">{title}</div>
            <div class="card-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    # Main function to run the Streamlit app
    st.sidebar.markdown("---")
    with st.sidebar:
        menu = option_menu(
            menu_icon="cast",
            menu_title="Menu",
            options=["Analysis", "Data", "Predictions", "About", "Contact us"],
            icons=["house", "database-gear", "graph-up", "info-circle", "envelope"],
            default_index=0,
        )
    df = load_data()
    df = create_features(df)

    df['SMA10'] = df['PowerConsumption_Zone1'].rolling(10).mean()
    df['SMA15'] = df['PowerConsumption_Zone1'].rolling(15).mean()
    df['SMA30'] = df['PowerConsumption_Zone1'].rolling(30).mean()

    # Extract month from Datetime column
    # df['month'] = pd.to_datetime(df['Datetime']).dt.month

    # Create a filtering interface
    selected_month = st.sidebar.selectbox('Select Month', df['month'].unique())

    # Filter the dataset based on the selected month
    filtered_df = df[df['month'] == selected_month]

    if menu == "Analysis":
        st.title("Power Consumption EDA Dashboard")
        st.markdown("---")
        
        b1, b2, b3, b4 = st.columns(4)
        b1.image(Image.open('assets/img.jpeg'))
        b2.image(Image.open('assets/img7.jpeg'))
        b3.image(Image.open('assets/img5.jpeg'))
        b4.image(Image.open('assets/img13.jpeg'))

        col = st.columns(2)

        st.sidebar.subheader("More chart options")
        if st.sidebar.checkbox("Correlation Heatmap"):
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(df.corr(), annot=True, ax=ax, cmap='vlag', fmt='.1g', annot_kws={'fontsize': 14, 'fontweight': 'regular'}, xticklabels=['Temp', 'Hum', 'Wind', 'Gen Diff Flows', 'Diff Flows', 'Power Z1', 'Power Z2', 'Power Z3'], yticklabels=['Temp', 'Hum', 'Wind', 'Gen Diff Flows', 'Diff Flows', 'Power Z1', 'Power Z2', 'Power Z3'])
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            st.pyplot(fig)

        if st.sidebar.checkbox("Histogram of Power Consumption"):
            fig4, ax = plt.subplots(figsize=(20, 10))

            st.subheader("Histogram of Power Consumption")
            plt.hist(df["PowerConsumption_Zone3"], bins=20)
            plt.xlabel("Power Consumption (Zone 3)")
            plt.ylabel("Frequency")
            st.pyplot(fig4)
        
        if st.sidebar.checkbox("All zones"):
            fig, ax = plt.subplots(figsize=(30, 20))
            st.subheader("Power Consumption in KW against time in  the 3 Zones")

            zone1 = sns.boxplot(data=df, x='hour', y='PowerConsumption_Zone1', palette='Oranges', showfliers=False)
            zone2 = sns.boxplot(data=df, x='hour', y='PowerConsumption_Zone2', palette='Reds', showfliers=False)
            zone3 = sns.boxplot(data=df, x='hour', y='PowerConsumption_Zone3', palette='Blues', showfliers=False)

            plt.suptitle('KW by Hour', fontsize=15)
            plt.xlabel('hour', fontsize=12)
            plt.ylabel('Power Consumption in KW', fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            st.pyplot(fig)
        
                
                
            # Plotting based on the filtered data
        if not filtered_df.empty:
            # Plot Temperature vs. Power Consumption_Zone1
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=filtered_df, x='Temperature', y='PowerConsumption_Zone1', ax=ax)
            plt.title('Temperature vs. Power Consumption (Zone 1)')
            st.pyplot(fig)        
            # Close the Matplotlib figure to release memory
            plt.close(fig)
            
            # Plot Temperature vs. Power Consumption_Zone2
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=filtered_df, x='Temperature', y='PowerConsumption_Zone2', ax=ax)
            plt.title('Temperature vs. Power Consumption (Zone 2)')
            st.pyplot(fig)
            # Close the Matplotlib figure to release memory
            plt.close(fig)
            
            # Plot Temperature vs. Power Consumption_Zone2
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=filtered_df, x='Temperature', y='PowerConsumption_Zone3', ax=ax)
            plt.title('Temperature vs. Power Consumption (Zone 3)')
            st.pyplot(fig)
            # Close the Matplotlib figure to release memory
            plt.close(fig)
            
            
            # Plot Humidity vs. Power Consumption_Zone1
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=filtered_df, x='Humidity', y='PowerConsumption_Zone1', ax=ax)
            plt.title('Humidity vs. Power Consumption (Zone 1)')
            st.pyplot(fig)
            # Close the Matplotlib figure to release memory
            plt.close(fig)
            
            
            # Plot Humidity vs. Power Consumption_Zone2
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=filtered_df, x='Humidity', y='PowerConsumption_Zone2', ax=ax)
            plt.title('Humidity vs. Power Consumption (Zone 2)')
            st.pyplot(fig)
            # Close the Matplotlib figure to release memory
            plt.close(fig)
            
            
            # Plot Humidity vs. Power Consumption_Zone3
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=filtered_df, x='Humidity', y='PowerConsumption_Zone3', ax=ax)
            plt.title('Humidity vs. Power Consumption (Zone 3)')
            st.pyplot(fig)
            # Close the Matplotlib figure to release memory
            plt.close(fig)
        else:
            st.write("No data available for the selected month.")

        with col[0]:
            st.markdown('#### Temperature by Hour')


            fig2, ax = plt.subplots(figsize=(20, 10))

            sns.boxplot(data=df, x='hour', y='Temperature', palette = 'Blues', showfliers=False)

            plt.suptitle('Temperature by Hour', fontsize=15)
            plt.xlabel('hour', fontsize=12)
            plt.ylabel('Temperature in °C', fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            #Generating plot
            st.pyplot(fig2)
            # Close the Matplotlib figure to release memory
            plt.close(fig2)
            st.image(Image.open("important-feature.png"))
            

        with col[1]:
            # st.subheader("Histogram of Power Consumption")
            # plt.hist(df["PowerConsumption_Zone3"], bins=20)
            # plt.xlabel("Power Consumption (Zone 3)")
            # plt.ylabel("Frequency")
            # st.pyplot(fig)

            st.markdown('#### Humidity by Hour')

            fig3, ax = plt.subplots(figsize=(20, 10))

            sns.boxplot(data=df, x='hour', y='Humidity', palette = 'Greens', showfliers=False)

            plt.suptitle('Humidity by Hour', fontsize=15)
            plt.xlabel('hour', fontsize=12)
            plt.ylabel('Humidity in %', fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            #Generating plot
            st.pyplot(fig3)
            # Close the Matplotlib figure to release memory
            plt.close(fig3)
            #Printing predictions on chart to visually assess accuracy
                
            image_path = os.path.abspath("https://www.mongodb.com/docs/charts/chart-types/")
            create_card(image_path, "Chart 1 Title", "Description of chart 1.")
            
            # create_card("important-feature.png", "Chart 3 Title", "Description of chart 3.")

        # st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, ax=ax, cmap='vlag', fmt='.1g', annot_kws={'fontsize': 14, 'fontweight': 'regular'}, xticklabels=['Temp', 'Hum', 'Wind', 'Gen Diff Flows', 'Diff Flows', 'Power Z1', 'Power Z2', 'Power Z3'], yticklabels=['Temp', 'Hum', 'Wind', 'Gen Diff Flows', 'Diff Flows', 'Power Z1', 'Power Z2', 'Power Z3'])
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        st.pyplot(fig)
        plt.close(fig)

        
            # Display multiple cards
        create_card("/home/rogendo/Desktop/dashboard/octoberdatavspredictiondata.png", "Chart 1 Title", "Description of chart 1.")
        create_card("/home/rogendo/Desktop/dashboard/octoberdatavspredictiondata.png", "Chart 2 Title", "Description of chart 2.")
        create_card("/home/rogendo/Desktop/dashboard/octoberdatavspredictiondata.png", "Chart 3 Title", "Description of chart 3.")
   
        # download notebook for further analysis
        st.subheader("For Further Analysis, Download the Notebook bellow... happy debugging!!")
        with open("profilereport.html", "rb") as file:
            st.download_button(label="Notebook",
                    data=file,
                    file_name="time-series-forecasting-on-power-consumption.ipynb",
                    mime="Electricity_Forecasting_Notebook/notebook"
                )
    if menu == "Data":
        df = load_data()
        profile = ProfileReport(df, title="Electricity consumption Profile Report")

        col1, col2 = st.columns(2)
        with col1:
            with open("profilereport.html", "rb") as file:
                st.download_button(
                    label="Download Profile Report",
                    data=file,
                    file_name="profilereport.html",
                    mime="Electricity_profilereport/html"
                )
            
        st.subheader("Data Card")
        st.write(filtered_df.head(20))
        
        csv = convert_df(df)
        with col2:
            st.download_button(
                label="Download Dataset as CSV",
                data=csv,
                file_name="powerconsumption.csv",
                mime='powerconsumption/csv',
            )
        
        
        if st.sidebar.checkbox("Summary Statistics"):
            st.subheader("Summary Card")
            st.write(df.describe())

    if menu == "About":
        st.title("About Page")
        st.markdown("---")
        about_page = """
        <div class="about">
            <h3>Introduction</h3>
            <p>The project's goal is to leverage time series analysis to predict energy consumption in 10-minute windows for the city of Tétouan in Morocco.</p>
            <h3>Context</h3>
            <p>
    According to a 2014 Census, Tétouan is a city in Morocco with a population of 380,000 people, occupying an area of 11,570 km². The city is located in the northern portion of the country and it faces the Mediterranean sea. The weather is particularly hot and humid during the summer.

    According to the state-owned ONEE, the “National Office of Electricity and Drinking Water”, Morocco’s energy production in 2019 came primarily from coal (38%), followed by hydroelectricity (16%), fuel oil (8 %), natural gas (18%), wind (11%), solar (7%), and others (2%) [1].

    Given the strong dependency on non-renewable sources (64%), forecasting energy consumption could help the stakeholders better manage purchases and stock. On top of that, Morocco’s plan is to reduce energy imports by increasing production from renewable sources. It’s common knowledge that sources like wind and solar present the risk of not being available all year round. Understanding the energy needs of the country, starting with a medium-sized city, could be a step further in planning these resources.
    </p>
        <h3>Data</h3>
        <p>The Supervisory Control and Data Acquisition System (SCADA) of Amendis, a public service operator, is responsible for collecting and providing the project’s power consumption data. The distribution network is powered by 3 zone stations, namely: Quads, Smir and Boussafou. The 3 zone stations power 3 different areas of the city, this is why we have three potential target variables.

    The data, which you can find at  <a href="https://www.kaggle.com/datasets/fedesoriano/electric-power-consumption">this link</a>, has 52,416 energy consumption observations in 10-minute windows starting from January 1st, 2017, all the way through December 30th (not 31st) of the same year. Some of the features are:</p>
    <ul>
    <li>Date Time: Time window of ten minutes.</li>
    <li>Temperature: Weather Temperature in °C</li>
    <li>Humidity: Weather Humidity in %</li>
    <li>Wind Speed: Wind Speed in km/h</li>
    <li>Zone 1 Power Consumption in KiloWatts (KW)</li>
    <li>Zone 2 Power Consumption in KW</li>
    <li>Zone 3 Power Consumption in KW</li>
    </ul>
        </div>
        """
        st.markdown(about_page, unsafe_allow_html=True)

        
    elif menu == "Contact us":
        st.title("Get in touch")
        contact_form = """
        <form action="https://formsubmit.co/faochieng@kabarak.ac.ke" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here"></textarea>
            <button type="submit">Send</button>
        </form>
        """

        st.markdown(contact_form, unsafe_allow_html=True)

# Use Local CSS File
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css")



if __name__ == "__main__":
    main()
