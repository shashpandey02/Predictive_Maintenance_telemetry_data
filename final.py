import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

# Removed deprecated or unsupported option
# st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Predictive Model", page_icon="📈")
def page1():
    st.title(":blue[DATA COLLECTION AND PRE-PROCESSING]")

    # Load data
    df = pd.read_csv("Datasets/PdM_telemetry.csv")
    df1 = pd.read_csv("Datasets/PdM_machines.csv")
    df2 = pd.read_csv("Datasets/PdM_failures.csv")
    df3 = pd.read_csv("Datasets/PdM_errors.csv")
    df4 = pd.read_csv("Datasets/PdM_maint.csv")

    # Display datasets
    st.subheader("Telemetry Data")
    st.dataframe(df)

    st.subheader("Machines Data")
    st.dataframe(df1)

    st.subheader("Failure Data")
    st.dataframe(df2)

    st.subheader("Errors Data")
    st.dataframe(df3)
    
    st.subheader("Maintenance Data")
    st.dataframe(df4)
    df['datetime'] = pd.to_datetime(df['datetime'])

    telemetry_daily = df.groupby(['machineID', pd.Grouper(key='datetime', freq='D')]).sum().reset_index()

    telemetry_daily['pressure'] = telemetry_daily['pressure'] / 24
    telemetry_daily['volt'] = telemetry_daily['volt'] / 24
    telemetry_daily['vibration'] = telemetry_daily['vibration'] / 24
    telemetry_daily['rotate'] = telemetry_daily['rotate'] / 24

    telemetry_daily = telemetry_daily.dropna()

    st.subheader("Telemetry Daily Data")
    st.dataframe(telemetry_daily)

    # Pre-processing
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['datetime'].dt.year != 2016]
    df = df.sort_values(by='datetime')

    #Load Dataset
    tele = pd.read_csv("Datasets/PdM_telemetry.csv")
    st.divider()
    st.header(":blue[Outlier Detection and Replacement]")
    st.write("Dataset")
    st.dataframe(tele.head())

    ## Analyze the 'vibration' column
    st.subheader("Boxplot Analysis")

    # Access the 'pressure' column
    df_pressure = tele['pressure']

    # Compute the median
    median_pressure = df_pressure.median()
    st.write(f":blue[Median Pressure: {median_pressure}]")

    # Create the boxplot using Matplotlib + Seaborn
    fig, ax = plt.subplots()
    sns.boxplot(x=df_pressure, ax=ax)
    st.pyplot(fig)

     # Calculate the boundaries for outliers detection in 'vibration'

    Q1_vibration = tele['pressure'].quantile(0.25)

    Q3_vibration = tele['pressure'].quantile(0.75)

    IQR_vibration = Q3_vibration - Q1_vibration

    Lower_Fence_pressure = Q1_vibration - (1.5 * IQR_vibration)

    Upper_Fence_pressure = Q3_vibration + (1.5 * IQR_vibration)

    

    # Update the 'vibration' column based on the boundaries, Replace with median

    tele["pressure"] = np.where(tele["pressure"] >= Upper_Fence_pressure, tele['pressure'].median(), tele['pressure'])

    tele["pressure"] = np.where(tele["pressure"] <= Lower_Fence_pressure, tele['pressure'].median(), tele['pressure'])
    df_pressure = tele['pressure']
    median_pressure = df_pressure.median()

    # Box plot after outlier replacement
    st.subheader("Boxplot after outlier removal")
    fig, ax = plt.subplots()
    sns.boxplot(x=df_pressure, ax=ax)
    st.pyplot(fig)

    # Pressure Boundaries
    st.subheader("Pressure Outliers Boundaries")
    st.write("Min:", Lower_Fence_pressure)
    st.write("Max:", Upper_Fence_pressure)
    st.write("Median:", median_pressure)


    st.divider()
    st.title(" :blue[Trend over the year]")
    st.subheader("Hourly Variation")
    st.subheader("Variation of Pressure over Time")

    # Create the figure
    fig = plt.figure(figsize=(20, 10))
    plt.plot(df['datetime'], df['pressure'], color='blue')
    plt.xlabel("Datetime")
    plt.ylabel("Pressure")
    plt.title("Pressure Variation over Time")

    # Display in Streamlit
    st.pyplot(fig)

    max_pressure = df['pressure'].max()
    min_pressure = df['pressure'].min()

    max_pressure_date = df.loc[df['pressure'].idxmax(), 'datetime']
    min_pressure_date = df.loc[df['pressure'].idxmin(), 'datetime']

    st.write("Maximum Pressure:", max_pressure)
    st.write("Date of Maximum Pressure:", max_pressure_date)
    st.write("Minimum Pressure:", min_pressure)
    st.write("Date of Minimum Pressure:", min_pressure_date)

    st.subheader("Statistical Summary of Telemetry Attributes")
    ####### TABLE
    data = {
        'Attribute': ['Pressure', 'Vibration','Voltage','Rotation'],
        'Max': ['185.951997730866','76.7910723016723','255.124717259791','695.020984403396'],
        'Date (Max)':['2015-04-04 21:00:00','2015-04-06 04:00:00','2015-11-16 07:00:00','2015-10-27 22:00:00'],
        'Min': ['51.2371057734253','14.877053998383','97.333603782359','138.432075304341'],
        'Date (Min)':['2015-09-22 00:00:00','2015-07-04 06:00:00','2015-08-31 04:00:00','2015-09-25 08:00:00']
    }

    df = pd.DataFrame(data)
    st.table(df.style.set_properties(**{'text-align': 'center'}).set_table_styles([{
        'selector': 'th',
        'props': [('text-align', 'center')]
    }]))

def page2():
    st.title(":blue[MACHINES, ERRORS & FAILURES]")
    merged_df = pd.read_csv("Datasets/PdM_failures.csv")
    df1 = pd.read_csv("Datasets/PdM_machines.csv")
    merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])
    merged_df['failure'] = merged_df['failure'].astype(str)

    label_encoder = LabelEncoder()
        
    merged_df['failure_encoded'] = label_encoder.fit_transform(merged_df['failure'])+1
    # st.write(merged_df.head())
        
    machfail = pd.merge(df1, merged_df)
    machfail['datetime'] = pd.to_datetime(machfail['datetime'])

    # Calculate the minimum datetime value
    min_datetime = machfail['datetime'].min()

    # Convert datetime to days
    machfail['days'] = ((machfail['datetime'] - min_datetime).dt.days)+2

    # Display the updated DataFrame
    st.subheader("Failure trend for Machine")

    machine_id = st.number_input("Enter the machine ID:", value=1, min_value=1)
    machine_df = machfail[machfail['machineID'] == machine_id]

    machine_id_1_df = merged_df[merged_df['machineID'] == machine_id]

    # Group by 'failure' and count occurrences
    failure_trends = machine_id_1_df.groupby('failure')['failure_encoded'].size()

    fig, ax = plt.subplots(figsize=(10, 6))
    failure_trends.plot(kind='bar', color='red', ax=ax)
    ax.set_title(f"Frequency of Errors for Machine ID - {machine_id}")
    ax.set_ylabel("Count")
    ax.set_xlabel("Failure Type")

    # Display in Streamlit
    st.pyplot(fig)


    failure_trend = machine_df.groupby('datetime')['failure_encoded'].sum()
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(failure_trend.index, failure_trend.values, marker='o')
    ax.set_title(f"Failure Trend for Machine ID - {machine_id}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Failure')
    ax.grid(True)   
    st.pyplot(fig)

    machine_id_1_df = machfail[machfail['machineID'] == machine_id]
    machine_id_1_df = machine_id_1_df.drop(['model', 'age', ], axis=1)  # Drop 'model' and 'age' columns

    machine_id_1_df['Difference'] = machine_id_1_df['days'].diff()
    st.write(machine_id_1_df)
    sum_diff=machine_id_1_df['Difference'].sum()
    mean_diff=sum_diff/(len(machine_id_1_df)-1)
    st.write(f":blue[Average number of days between two failures for Machine ID: {machine_id} -]", int(mean_diff))

    ###############################################
    st.divider()

    df = pd.read_csv("Datasets/PdM_errors.csv")
    df = df.sort_values(by='errorID')
    # Convert the 'datetime' column to datetime format

    df['datetime'] = pd.to_datetime(df['datetime'])
    # Filter out rows where the year is not 2016

    df = df[df['datetime'].dt.year != 2016]
    # Encode the 'errorID' column using label encoding
    label_encoder = LabelEncoder()
    df['errorID_encoded'] = label_encoder.fit_transform(df['errorID']) + 1
    errors = df.groupby('errorID')['errorID_encoded'].size()

    plt.figure(figsize=(10, 6))

    errors.plot(kind='bar', color='red')
    st.subheader("Frequency of Errors")
    plt.title('Frequency of Errors')
    plt.xlabel('ErrorID')
    plt.ylabel('No. of Errors')
    st.pyplot(plt)

    st.subheader('Monthly Variation of Errors')
    # Load data

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['month'] = df['datetime'].dt.month
    monthly_errors = df.groupby('month')['errorID_encoded'].size()

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun','Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Create the plot

    plt.figure(figsize=(20, 10))
    plt.plot(monthly_errors.index, monthly_errors.values, color='blue', marker='o')
    plt.title('Monthly Variation of Errors')
    plt.xlabel('Month')
    plt.ylabel('Errors')
    plt.xticks(monthly_errors.index, month_names)
    plt.grid(True)

    # Display the plot using Streamlit
    st.pyplot(plt)

    avg_errors = df.groupby('machineID')['errorID_encoded'].size()
    plt.figure(figsize=(20,6))
    plt.title('Variation of Errors by machine ID')
    avg_errors.plot(kind='bar')
    st.pyplot(plt)

    avg_errors = df.drop('errorID_encoded', axis=1).groupby('machineID').size()
    top_3_most_errors = avg_errors.nlargest(3)

    # Find the top 3 machine IDs with the least errors
    top_3_least_errors = avg_errors.nsmallest(3)

    # Rename the column headers
    top_3_most_errors = top_3_most_errors.rename_axis('Machine ID').reset_index(name='No of errors')
    top_3_least_errors = top_3_least_errors.rename_axis('Machine ID').reset_index(name='No of errors')

    # Print the results
    st.write("Top 3 Machine IDs with Most Errors:")
    st.write(top_3_most_errors)

    st.write("Top 3 Machine IDs with Least Errors:")
    st.write(top_3_least_errors)
def page3():
    st.subheader(":blue[Variation of Telemetry Attributes wrt Age of Machine]")
    ## MACHINES DATASET
    df1 = pd.read_csv("Datasets/PdM_machines.csv")
    # st.header("MACHINES DATASET")

    # Read the CSV files
    df = pd.read_csv("Datasets/PdM_telemetry.csv")

    # Convert 'datetime' column to datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Compute daily averages for telemetry data
    telemetry_daily = df.groupby(['machineID', pd.Grouper(key='datetime', freq='D')]).sum().reset_index()

    # Divide telemetry values by 24 to get daily averages
    telemetry_daily['pressure'] = telemetry_daily['pressure'] / 24
    telemetry_daily['volt'] = telemetry_daily['volt'] / 24
    telemetry_daily['vibration'] = telemetry_daily['vibration'] / 24
    telemetry_daily['rotate'] = telemetry_daily['rotate'] / 24

    # Drop rows with missing values
    telemetry_daily = telemetry_daily.dropna()

    # Merge telemetry data with machines data
    merged_df = pd.merge(telemetry_daily, df1, on=['machineID'], how='left')

    # Display the shape and head of the merged DataFrame
    # st.write("Shape of merged DataFrame:", merged_df.shape)
    st.write("Merged DataFrame:")
    st.write(merged_df.head())

    # Mean Pressure Variation by Age
    mean_pressure_by_age = merged_df.groupby('age')['pressure'].mean()
    plt.plot(mean_pressure_by_age.index, mean_pressure_by_age.values, marker='o')
    plt.title('Mean Pressure Variation by Age')
    plt.xlabel('Age')
    plt.ylabel('Mean Pressure')
    st.pyplot(plt)
    
    # Mean Rotation Variation by Age
    mean_rotation_by_age = merged_df.groupby('age')['rotate'].mean()
    plt.plot(mean_rotation_by_age.index, mean_rotation_by_age.values, marker='o')
    plt.title('Mean rotation Variation by Age')
    plt.xlabel('Age')
    plt.ylabel('Mean rotation')
    st.pyplot(plt)
    
    # Mean Voltage variation by age
    mean_voltage_by_age = merged_df.groupby('age')['volt'].mean()
    plt.plot(mean_voltage_by_age.index, mean_voltage_by_age.values, marker='o')
    plt.title('Mean voltage Variation by Age')
    plt.xlabel('Age')
    plt.ylabel('Mean voltage')
    st.pyplot(plt)
    
    # Mean Vibration variation by age
    mean_vibration_by_age = merged_df.groupby('age')['vibration'].mean()
    plt.plot(mean_vibration_by_age.index, mean_vibration_by_age.values, marker='o')
    plt.title('Mean vibration Variation by Age')
    plt.xlabel('Age')
    plt.ylabel('Mean vibration')
    st.pyplot(plt)
    

    st.divider()
    st.subheader("Variation of Failure Components wrt Telemetry Attributes")

    # Load the telemetry, failures, and additional dataset

    df = pd.read_csv("Datasets/PdM_telemetry.csv")

    df2 = pd.read_csv("Datasets/PdM_failures.csv")

    df1 = pd.read_csv("Datasets/PdM_machines.csv")


    df3 = pd.merge(df, df2, how='outer') # telemetry and failure
    df4 = pd.merge(df3, df1) # telemetry, failure and machines 
    df5=pd.merge(df,df2) #telemetry and failures
    mer = pd.merge(df5,df1) #df5 and machines 
    
    df3['failure'] = df3['failure'].astype(str)
    df4['failure'] = df4['failure'].astype(str)
    
    #VARIATION OF FAILURE COMPONENTS WITH TELEMETRY 
    plt.figure(figsize=(10, 6))
    plt.scatter(df5['failure'], df5['pressure'], s=50, alpha=0.5)
    plt.title('Variation of Pressure wrt Failure Components')
    plt.xlabel('Failure Component')
    plt.ylabel('Pressure')
    plt.grid(True)
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    plt.scatter(df5['failure'], df5['rotate'], s=50, alpha=0.5)
    plt.title('Variation of Rotation wrt Failure Components')
    plt.xlabel('Failure Component')
    plt.ylabel('Rotation')
    plt.grid(True)
    st.pyplot(plt)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df5['failure'], df5['volt'], s=50, alpha=0.5)
    plt.title('Variation of Voltage wrt Failure Components')
    plt.xlabel('Failure Component')
    plt.ylabel('Voltage')
    plt.grid(True)
    st.pyplot(plt)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df5['failure'], df5['vibration'], s=50, alpha=0.5)
    plt.title('Variation of Rotation wrt Failure Components')
    plt.xlabel('Failure Component')
    plt.ylabel('Vibration')
    plt.grid(True)
    st.pyplot(plt)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(mer['failure'], mer['age'], s=50, alpha=0.5)
    plt.title('Variation of Rotation wrt Failure Components')
    plt.xlabel('Failure Component')
    plt.ylabel('Age')
    plt.grid(True)
    st.pyplot(plt)
        
    

# Run the function
#page1()
# Create a dictionary to map page names to their corresponding functions
pages = {
    "Telemetry": page1,
    "Errors": page2,
    "Machines, Failure + Telemetry": page3, 
    # "Model Building": page4,
    # "Failure prediction":page5
}

# Create a sidebar or navigation menu
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Display the selected page
pages[selection]()
