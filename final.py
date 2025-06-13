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
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

@st.cache_resource
def train_models(X_train, y_train):
    
    smote = SMOTE(random_state=42, n_jobs=-1)  # Use all CPU cores
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_classifier.fit(X_train, y_train)

    svc_pipeline = Pipeline([
        ("scaler", StandardScaler()),  # Helps with SVM performance
        ("svc", SVC(kernel="linear", probability=True, random_state=42))
    ])
    svc_pipeline.fit(X_train, y_train)

    logreg = LogisticRegression(solver='saga', class_weight='balanced', max_iter=300, penalty='l2')
    logreg.fit(X_train, y_train)

    ensemble = VotingClassifier(estimators=[
        ('rf', rf_classifier), ('svc', svc_pipeline), ('logreg', logreg)
    ], voting='soft')
    ensemble.fit(X_train, y_train)

    return rf_classifier, svc_pipeline, logreg, ensemble


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
    ax.set_title(f"Frequency of Failure for Machine ID - {machine_id}")
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
    #############################################################################################################################
    telemetry_df = pd.read_csv("Datasets/PdM_telemetry.csv")
    failure_df = pd.read_csv("Datasets/PdM_failures.csv")
    
    telemetry_df['datetime'] = pd.to_datetime(telemetry_df['datetime'])
    failure_df['datetime'] = pd.to_datetime(failure_df['datetime'])
    

    
    failure_df['failure_flag'] = 1


    # Merge telemetry with failure on datetime and machineID, left join to keep all telemetry rows
    m_df = telemetry_df.merge(
        failure_df[['datetime', 'machineID', 'failure_flag']], 
        on=['datetime', 'machineID'], 
        how='left'
    )

    # Fill NaN failure_flag with 0 (means no failure at that time)
    m_df['failure_flag'] = m_df['failure_flag'].fillna(0).astype(int)

    # Rename failure_flag to failure for clarity
    m_df = m_df.rename(columns={'failure_flag': 'failure'})
    # st.write(merge_df)
    
    m_df['datetime'] = pd.to_datetime(m_df['datetime'])
    m_df.set_index('datetime', inplace=True)
    
    X = m_df.drop(['failure','machineID'], axis=1)
    Y = m_df['failure']
    
    # Count total failures per machine
    failures_per_machine = m_df.groupby('machineID')['failure'].sum().reset_index()

    # Top 3 machines with max failures
    top_3_max = failures_per_machine.sort_values(by='failure', ascending=False).head(3)

    # Top 3 machines with min failures (excluding machines with zero failures if you want)
    top_3_min = failures_per_machine[failures_per_machine['failure'] > 0].sort_values(by='failure', ascending=True).head(3)

    st.write("Top 3 machines with max failures:\n", top_3_max)
    st.write("\nTop 3 machines with min failures:\n", top_3_min)

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
    plt.figure() # start a new figure
    plt.plot(mean_pressure_by_age.index, mean_pressure_by_age.values, marker='o')
    plt.title('Mean Pressure Variation by Age')
    plt.xlabel('Age')
    plt.ylabel('Mean Pressure')
    st.pyplot(plt)
    
    # Mean Rotation Variation by Age
    mean_rotation_by_age = merged_df.groupby('age')['rotate'].mean()
    plt.figure() # start a new figure
    plt.plot(mean_rotation_by_age.index, mean_rotation_by_age.values, marker='o', color='red')
    plt.title('Mean rotation Variation by Age')
    plt.xlabel('Age')
    plt.ylabel('Mean rotation')
    st.pyplot(plt)
    
    # Mean Voltage variation by age
    mean_voltage_by_age = merged_df.groupby('age')['volt'].mean()
    plt.figure() # start a new figure
    plt.plot(mean_voltage_by_age.index, mean_voltage_by_age.values, marker='o', color='green')
    plt.title('Mean voltage Variation by Age')
    plt.xlabel('Age')
    plt.ylabel('Mean voltage')
    st.pyplot(plt)
    
    # Mean Vibration variation by age
    mean_vibration_by_age = merged_df.groupby('age')['vibration'].mean()
    plt.figure() # start a new figure
    plt.plot(mean_vibration_by_age.index, mean_vibration_by_age.values, marker='o', color='orange')
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
    
def page4():
    st.title(":blue[MODEL BUILDING]")
    df = pd.read_csv("Datasets/PdM_telemetry.csv")
    df5 = pd.read_csv("Datasets/PdM_failures.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])

    #Group by machineID and daily frequency, and calculate the daily averages
    telemetry_daily = df.groupby(['machineID', pd.Grouper(key='datetime', freq='D')]).sum().reset_index()
    telemetry_daily['pressure'] = telemetry_daily['pressure']/24
    telemetry_daily['volt'] = telemetry_daily['volt']/24
    telemetry_daily['vibration'] = telemetry_daily['vibration']/24
    telemetry_daily['rotate'] = telemetry_daily['rotate']/24

    #drop any rows with missing values
    telemetry_daily = telemetry_daily.dropna()
    # Replae failure column with binary values
    df5['failure'] = df5['failure'].replace(['comp1', 'comp2', 'copm3', 'comp4'], 1)

    # Convert failure column to string type
    df5['failure'] = df5['failure'].astype(str)
    df5['datetime'] = pd.to_datetime(df5['datetime'])
    df5.set_index('datetime', inplace= True)

    # Group by machineId  and daily frequency, calculate daily sum
    df5 = df5.groupby(['machineID', pd.Grouper(freq='D')]).sum()
    df5 = df5.reset_index()

    # Normalize the datetime column
    df5['datetime'] = df5['datetime'].dt.normalize()

    # Merging datasets
    merge_df = pd.merge(telemetry_daily, df5, on=['machineID', 'datetime'], how='left')
    st.write('Shape of dataset after merging:', merge_df.shape)

    merge_df['failure'] = merge_df['failure'].astype(str)

    # Replace known 'nan' strings with actual NaN
    merge_df['failure'] = merge_df['failure'].replace('nan', np.nan)

    # Any non-null value (non-zero, string, comp label etc.) → 1
    merge_df['failure'] = merge_df['failure'].apply(lambda x: 0 if pd.isna(x) or x == '0' else 1)

    # Now it's safe to convert to int
    merge_df['failure'] = merge_df['failure'].astype(int)


    # Show data
    st.write(merge_df)
    st.divider()
    # Time and index prep
    merge_df['datetime'] = pd.to_datetime(merge_df['datetime'])
    merge_df.set_index('datetime', inplace=True)
    


    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_test = X_test.drop(columns=['machineID'])
    # SMOTE
    # smote = SMOTE(random_state=42)
    # X_resampled, Y_resampled = smote.fit_resample(X_train, Y_train)
    # rf, svc, lr, ensemble_model = train_models(X_train, Y_train)
    # joblib.dump(rf, "rf_model.pkl")
    # joblib.dump(svc, "svc_pipeline.pkl")
    # joblib.dump(lr, "logreg_model.pkl")
    # joblib.dump(ensemble_model, "ensemble_model.pkl")
    rf = joblib.load('models/rf_model.pkl') 
    
    # RANDOM FOREST CLASSIFIER
    st.header("Random Forest Classifier")
    y_pred = rf.predict(X_test)
    ac = accuracy_score(Y_test, y_pred)
    st.write("Accuracy: ", ac)
    
    # confusion Matrix
    # cm = confusion_matrix(Y_test, y_pred)
    # st.write(cm)
    

    st.divider()
    # SVC
    svc = joblib.load('models/svc_pipeline.pkl') 
    st.header("Support Vector Classifier")
    y_pred = svc.predict(X_test)
    ac = accuracy_score(Y_test, y_pred)
    st.write("Accuracy: ", ac)
    # confusion Matrix
    # cm = confusion_matrix(Y_test, y_pred)
    # st.write(cm)

    # LOGISTIC REGRESSION : NEWTON METHOD
    st.divider()
    lr = joblib.load('models/logreg_model.pkl')
    st.header("Logistic Regression - netwon-cg solver")
    y_pred = lr.predict(X_test)
    ac = accuracy_score(Y_test, y_pred)
    st.write("Accuracy: ", ac)
    # confusion Matrix
    # cm = confusion_matrix(Y_test, y_pred)
    # st.write(cm)
    
    # Ensemble Model
    st.divider()
    ensemble_model = joblib.load('models/ensemble_model.pkl')
    st.header('Ensemble Model')
    y_pred = ensemble_model.predict(X_test)
    ac = accuracy_score(Y_test, y_pred)
    st.write("Accuracy: ", ac)
    
    # confusion Matrix
    # cm = confusion_matrix(Y_test, y_pred)
    # st.write(cm)
    

    # # Train RandomForest
    # rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
    # rf_classifier.fit(X_train, Y_train)
    # y_pred = rf_classifier.predict(X_test)

    # # Metrics
    # # accuracy = accuracy_score(Y_test, y_pred)
    # # st.write("Accuracy: ", accuracy)

    # # cm = confusion_matrix(Y_test, y_pred)
    # # st.write("Confusion Matrix:")
    # # st.write(cm)
    # # # classification report
    # # st.text("Classification Report:")
    # # st.text(classification_report(Y_test, y_pred, target_names=['No Failure', 'Failure']))
    
    # # SMOTE to generate more failures synthetically
    # smote = SMOTE(random_state=42)
    # X_resampled, y_resampled = smote.fit_resample(X_train, Y_train)
    # rf_classifier.fit(X_resampled, y_resampled)
    # y_pred = rf_classifier.predict(X_test)
    # # Accuracy
    # accuracy = accuracy_score(Y_test, y_pred)
    # cm = confusion_matrix(Y_test, y_pred)
    # st.write(accuracy)
    # st.write(cm)

    # # Training Accuracy
    # # y_train_pred = rf_classifier.predict(X_train)
    # # ac1 = accuracy_score(Y_train, y_train_pred)
    # # st.write('Training Accuracy: ', ac1)

    # st.divider()
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    # # Apply SMOTE again on this split
    # smote = SMOTE(random_state=0)
    # X_resampled_svc, y_resampled_svc = smote.fit_resample(X_train, Y_train)
    # # Create a SVC support vector classifier
    # sv_classifier = SVC(kernel='rbf', random_state=0, probability=True)
    # sv_classifier.fit(X_resampled_svc, y_resampled_svc)
    # y_pred = sv_classifier.predict(X_test)
    
    # # Calculate accuracy 
    # accuracy = accuracy_score(Y_test, y_pred)
    # st.header("Support Vector Classifier")
    # st.write('Accuracy: ', accuracy)
    # # st.divider()
  
    # #confusion matrix
    # cm = confusion_matrix(Y_test, y_pred)
    # st.write(cm)
    
    # #LOGISTIC REGRESSION - NEWTON CG 
    # st.divider()
    # st.header('Logistic Regression - netwon-cg solver')
    # # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # smote = SMOTE(random_state=42)
    # X_resampled_lr, y_resampled_lr = smote.fit_resample(X_train, Y_train)
    # lr = LogisticRegression(solver='newton-cg', class_weight='balanced', max_iter=1000)
    # lr.fit(X_resampled_lr, y_resampled_lr)
    
    # y_pred = lr.predict(X_test)
    # accuracy = accuracy_score(Y_test, y_pred)
    # st.write('Accuracy: ', accuracy)
    # cm = confusion_matrix(Y_test, y_pred)
    # st.write(cm)
    
    # # After confusion matrix
    # # st.text("Classification Report:")
    # # st.text(classification_report(Y_test, y_pred, target_names=['No Failure', 'Failure']))
    
    # st.header(':blue[Ensemble Model]')
    
    # # Define base models
    # rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
    # sv_classifier = SVC(kernel='rbf', probability=True, random_state=0)  
    # lr = LogisticRegression(solver='newton-cg', class_weight='balanced', max_iter=1000)
    
    # smote = SMOTE(random_state=42)
    # X_resampled, y_resampled = smote.fit_resample(X_train, Y_train)
    
    # # Fit models
    # rf_classifier.fit(X_resampled, y_resampled)
    # sv_classifier.fit(X_resampled, y_resampled)
    # lr.fit(X_resampled, y_resampled)
    
    # # Combine into a voting classifier (hard voting by default)
    # ensemble_model = VotingClassifier(estimators=[
    #     ('rf', rf_classifier),
    #     ('svc', sv_classifier),
    #     ('lr', lr)
    # ], voting='soft') # soft voting uses predicted probabilities
    
    # ensemble_model.fit(X_resampled, y_resampled)
    
    # y_pred = ensemble_model.predict(X_test)
    # acc = accuracy_score(Y_test, y_pred)
    # cm = confusion_matrix(Y_test, y_pred)
    # st.write("Accuracy: ", acc)
    # st.write(cm)
    
def page5():
    st.title(':blue[FAILURE PREDICTION]')

    df = pd.read_csv("Datasets/PdM_telemetry.csv")
    df5 = pd.read_csv("Datasets/PdM_failures.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])

    #Group by machineID and daily frequency, and calculate the daily averages
    telemetry_daily = df.groupby(['machineID', pd.Grouper(key='datetime', freq='D')]).sum().reset_index()
    telemetry_daily['pressure'] = telemetry_daily['pressure']/24
    telemetry_daily['volt'] = telemetry_daily['volt']/24
    telemetry_daily['vibration'] = telemetry_daily['vibration']/24
    telemetry_daily['rotate'] = telemetry_daily['rotate']/24

    #drop any rows with missing values
    telemetry_daily = telemetry_daily.dropna()
    # Replae failure column with binary values
    df5['failure'] = df5['failure'].replace(['comp1', 'comp2', 'copm3', 'comp4'], 1)

    # Convert failure column to string type
    df5['failure'] = df5['failure'].astype(str)
    df5['datetime'] = pd.to_datetime(df5['datetime'])
    df5.set_index('datetime', inplace= True)

    # Group by machineId  and daily frequency, calculate daily sum
    df5 = df5.groupby(['machineID', pd.Grouper(freq='D')]).sum()
    df5 = df5.reset_index()

    # Normalize the datetime column
    df5['datetime'] = df5['datetime'].dt.normalize()

    # Merging datasets
    merge_df = pd.merge(telemetry_daily, df5, on=['machineID', 'datetime'], how='left')
    # Convert failure column to string
    merge_df['failure'] = merge_df['failure'].astype(str)

    # Replace known 'nan' strings with actual NaN
    merge_df['failure'] = merge_df['failure'].replace('nan', np.nan)

    # Any non-null value (non-zero, string, comp label etc.) → 1
    merge_df['failure'] = merge_df['failure'].apply(lambda x: 0 if pd.isna(x) or x == '0' else 1)

    # Now it's safe to convert to int
    merge_df['failure'] = merge_df['failure'].astype(int)

    # Time and index prep
    merge_df['datetime'] = pd.to_datetime(merge_df['datetime'])
    merge_df.set_index('datetime', inplace=True)

    # Feature/target split
    X = merge_df[['volt', 'rotate', 'pressure', 'vibration']]
    Y = merge_df['failure']

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    st.write(df.describe())


    # Get user input
    pressure = st.number_input("Pressure", min_value=0.0)
    volt = st.number_input("Volt", min_value=0.0)
    vibration = st.number_input("Vibration", min_value=0.0)
    rotate = st.number_input("Rotate", min_value=0.0)

    user_input = np.array([[volt, rotate, pressure, vibration]])
    ensemble_model = joblib.load('models/ensemble_model.pkl')
    # rf, svc, lr, ensemble_model = train_models(X_train, Y_train)
    if st.button("Predict"):
        if not (volt > 0 and rotate > 0 and pressure > 0 and vibration > 0):
            st.error("❌ One or more telemetry values are out of expected range.")
        else:
            y_pred = ensemble_model.predict(user_input)
            probability = ensemble_model.predict_proba(user_input)[0][1] # Probability of class 1

            if y_pred[0] == 1:
                st.error(f"⚠️ Failure Predicted with Probability: **{probability:.2%}**")
            else:
                st.success(f"✅ No Failure Predicted. Failure Probability: **{probability:.2%}**")

def page6(): 
    st.header(":blue[Failure Prediction by MachineID]")
    
    telemetry_df = pd.read_csv("Datasets/PdM_telemetry.csv")
    failure_df = pd.read_csv("Datasets/PdM_failures.csv")
    
    telemetry_df['datetime'] = pd.to_datetime(telemetry_df['datetime'])
    failure_df['datetime'] = pd.to_datetime(failure_df['datetime'])
    

    
    failure_df['failure_flag'] = 1


    # Merge telemetry with failure on datetime and machineID, left join to keep all telemetry rows
    merge_df = telemetry_df.merge(
        failure_df[['datetime', 'machineID', 'failure_flag']], 
        on=['datetime', 'machineID'], 
        how='left'
    )

    # Fill NaN failure_flag with 0 (means no failure at that time)
    merge_df['failure_flag'] = merge_df['failure_flag'].fillna(0).astype(int)

    # Rename failure_flag to failure for clarity
    merge_df = merge_df.rename(columns={'failure_flag': 'failure'})
    st.write(merge_df)
    
    merge_df['datetime'] = pd.to_datetime(merge_df['datetime'])
    merge_df.set_index('datetime', inplace=True)
    
    X = merge_df.drop(['failure','machineID'], axis=1)
    Y = merge_df['failure']
    
    # Count total failures per machine
    failures_per_machine = merge_df.groupby('machineID')['failure'].sum().reset_index()

    # Top 3 machines with max failures
    top_3_max = failures_per_machine.sort_values(by='failure', ascending=False).head(3)

    # Top 3 machines with min failures (excluding machines with zero failures if you want)
    top_3_min = failures_per_machine[failures_per_machine['failure'] > 0].sort_values(by='failure', ascending=True).head(3)

    st.write("Top 3 machines with max failures:\n", top_3_max)
    st.write("\nTop 3 machines with min failures:\n", top_3_min)

    

    
    
    
    



    

    

# Run the function
#page1()
# Create a dictionary to map page names to their corresponding functions
pages = {
    "Telemetry": page1,
    "Errors": page2,
    "Machines, Failure + Telemetry": page3, 
    "Model Building": page4,
    "Failure prediction":page5,
    "Error prediction":page6
}

# Create a sidebar or navigation menu
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Display the selected page
pages[selection]()
