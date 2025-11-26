import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import os
import requests

DATA_DIR = 'data'

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def load_breast_cancer_data():
    ensure_data_dir()
    print("Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    filepath = os.path.join(DATA_DIR, 'breast_cancer.csv')
    df.to_csv(filepath, index=False)
    print(f"Saved Breast Cancer data to {filepath}")
    return df

def load_heart_disease_data():
    ensure_data_dir()
    print("Downloading Heart Disease dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    try:
        df = pd.read_csv(url, names=column_names, na_values='?')
        # Binary classification: 0 = no disease, 1-4 = disease
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        
        filepath = os.path.join(DATA_DIR, 'heart_disease.csv')
        df.to_csv(filepath, index=False)
        print(f"Saved Heart Disease data to {filepath}")
        return df
    except Exception as e:
        print(f"Error downloading Heart Disease data: {e}")
        return None

def load_diabetes_data():
    ensure_data_dir()
    print("Downloading Diabetes dataset...")
    # Pima Indians Diabetes Dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    try:
        df = pd.read_csv(url, names=column_names)
        df.rename(columns={'Outcome': 'target'}, inplace=True)
        
        filepath = os.path.join(DATA_DIR, 'diabetes.csv')
        df.to_csv(filepath, index=False)
        print(f"Saved Diabetes data to {filepath}")
        return df
    except Exception as e:
        print(f"Error downloading Diabetes data: {e}")
        return None

if __name__ == "__main__":
    load_breast_cancer_data()
    load_heart_disease_data()
    load_diabetes_data()
