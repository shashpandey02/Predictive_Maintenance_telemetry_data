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

def train_models(X_train, y_train):
    
    smote = SMOTE(random_state=42) 
    X_resampled, Y_resampled = smote.fit_resample(X_train, y_train)
    
    rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_classifier.fit(X_resampled, Y_resampled)

    svc_pipeline = Pipeline([
        ("scaler", StandardScaler()),  # Helps with SVM performance
        ("svc", SVC(kernel="linear", probability=True, random_state=42))
    ])
    svc_pipeline.fit(X_resampled, Y_resampled)

    logreg = LogisticRegression(
        solver='saga',
        class_weight='balanced',
        max_iter=3000,  # Increase to 1000
        penalty='l2'
    )

    logreg.fit(X_resampled, Y_resampled)

    ensemble = VotingClassifier(estimators=[
        ('rf', rf_classifier), ('svc', svc_pipeline), ('logreg', logreg)
    ], voting='soft')
    ensemble.fit(X_resampled, Y_resampled)

    return rf_classifier, svc_pipeline, logreg, ensemble


def main():
    
    # df = pd.read_csv("Datasets/PdM_telemetry.csv")
    # df5 = pd.read_csv("Datasets/PdM_failures.csv")
    # df['datetime'] = pd.to_datetime(df['datetime'])
    
    # #Group by machineID and daily frequency, and calculate the daily averages
    # telemetry_daily = df.groupby(['machineID', pd.Grouper(key='datetime', freq='D')]).sum().reset_index()
    # telemetry_daily['pressure'] = telemetry_daily['pressure']/24
    # telemetry_daily['volt'] = telemetry_daily['volt']/24
    # telemetry_daily['vibration'] = telemetry_daily['vibration']/24
    # telemetry_daily['rotate'] = telemetry_daily['rotate']/24
    
    #  #drop any rows with missing values
    # telemetry_daily = telemetry_daily.dropna()
    # # Replae failure column with binary values
    # df5['failure'] = df5['failure'].replace(['comp1', 'comp2', 'comp3', 'comp4'], 1)

    # # Convert failure column to string type
    # df5['failure'] = df5['failure'].astype(str)
    # df5['datetime'] = pd.to_datetime(df5['datetime'])
    # df5.set_index('datetime', inplace= True)

    # # Group by machineId  and daily frequency, calculate daily sum
    # df5 = df5.groupby(['machineID', pd.Grouper(freq='D')]).sum()
    # df5 = df5.reset_index()

    # # Normalize the datetime column
    # df5['datetime'] = df5['datetime'].dt.normalize()

    # # Merging datasets
    # merge_df = pd.merge(telemetry_daily, df5, on=['machineID', 'datetime'], how='left')

    # # Convert failure column to string
    # merge_df['failure'] = merge_df['failure'].astype(str)

    # # Replace known 'nan' strings with actual NaN
    # merge_df['failure'] = merge_df['failure'].replace('nan', np.nan)

    # # Any non-null value (non-zero, string, comp label etc.) → 1
    # merge_df['failure'] = merge_df['failure'].apply(lambda x: 0 if pd.isna(x) or x == '0' else 1)

    # # Now it's safe to convert to int
    # merge_df['failure'] = merge_df['failure'].astype(int)
    
    # # Time and index prep
    # merge_df['datetime'] = pd.to_datetime(merge_df['datetime'])
    # merge_df.set_index('datetime', inplace=True)

    # # Feature/target split
    # X = merge_df.drop(['failure','machineID'], axis=1)
    # Y = merge_df['failure']
    # Split data
   

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
    
    
    merge_df['datetime'] = pd.to_datetime(merge_df['datetime'])
    merge_df.set_index('datetime', inplace=True)

    X = merge_df.drop(['failure','machineID'], axis=1)
    y = merge_df['failure']

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf, svc, lr, ensemble_model = train_models(X_train, Y_train)
    joblib.dump(rf, "rf_model.pkl")
    joblib.dump(svc, "svc_pipeline.pkl")
    joblib.dump(lr, "logreg_model.pkl")
    joblib.dump(ensemble_model, "ensemble_model.pkl")
    
if __name__ == "__main__":
    main()
   
    