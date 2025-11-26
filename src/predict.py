import pandas as pd
import numpy as np
import os
from preprocessing import DataPreprocessor
from models import ModelFactory

DATA_DIR = 'data'

HEART_DISEASE_DESCRIPTIONS = {
    'age': 'Age (years)',
    'sex': 'Sex (1 = male, 0 = female)',
    'cp': 'Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)',
    'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Serum cholestoral (mg/dl)',
    'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
    'restecg': 'Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: LV hypertrophy)',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina (1 = yes, 0 = no)',
    'oldpeak': 'ST depression induced by exercise relative to rest',
    'slope': 'Slope of the peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)',
    'ca': 'Number of major vessels (0-3) colored by flourosopy',
    'thal': 'Thal (3: normal, 6: fixed defect, 7: reversable defect)'
}

DIABETES_DESCRIPTIONS = {
    'Pregnancies': 'Number of times pregnant',
    'Glucose': 'Plasma glucose concentration',
    'BloodPressure': 'Diastolic blood pressure (mm Hg)',
    'SkinThickness': 'Triceps skin fold thickness (mm)',
    'Insulin': '2-Hour serum insulin (mu U/ml)',
    'BMI': 'Body mass index (weight in kg/(height in m)^2)',
    'DiabetesPedigreeFunction': 'Diabetes pedigree function',
    'Age': 'Age (years)'
}

def get_user_input(feature_names, descriptions=None):
    print("\nPlease enter the following information:")
    input_data = {}
    for feature in feature_names:
        prompt = f"{feature}: "
        if descriptions and feature in descriptions:
            prompt = f"{descriptions[feature]}: "
            
        while True:
            try:
                value = float(input(prompt))
                input_data[feature] = value
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
    return pd.DataFrame([input_data])

def predict_heart_disease():
    filepath = os.path.join(DATA_DIR, 'heart_disease.csv')
    if not os.path.exists(filepath):
        print("Heart Disease dataset not found. Please run data_loader.py first.")
        return

    print("Training Heart Disease model (Random Forest)...")
    df = pd.read_csv(filepath)
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Fit preprocessor on all data
    preprocessor = DataPreprocessor()
    preprocessor.imputer.fit(X)
    X_imputed = preprocessor.imputer.transform(X)
    preprocessor.scaler.fit(X_imputed)
    X_scaled = preprocessor.scaler.transform(X_imputed)
    
    model = ModelFactory.get_model('random_forest')
    model.fit(X_scaled, y)
    
    print("\n--- Enter Patient Data ---")
    user_input_df = get_user_input(X.columns, HEART_DISEASE_DESCRIPTIONS)
    
    # Transform input
    user_input_imputed = preprocessor.imputer.transform(user_input_df)
    user_input_scaled = preprocessor.scaler.transform(user_input_imputed)
    
    prediction = model.predict(user_input_scaled)[0]
    probability = model.predict_proba(user_input_scaled)[0][1]
    
    print(f"\nPrediction: {'Heart Disease Detected' if prediction == 1 else 'No Heart Disease'}")
    print(f"Probability: {probability:.2f}")

def predict_diabetes():
    filepath = os.path.join(DATA_DIR, 'diabetes.csv')
    if not os.path.exists(filepath):
        print("Diabetes dataset not found.")
        return

    print("Training Diabetes model (Random Forest)...")
    df = pd.read_csv(filepath)
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    preprocessor = DataPreprocessor()
    preprocessor.imputer.fit(X)
    X_imputed = preprocessor.imputer.transform(X)
    preprocessor.scaler.fit(X_imputed)
    X_scaled = preprocessor.scaler.transform(X_imputed)
    
    model = ModelFactory.get_model('random_forest')
    model.fit(X_scaled, y)
    
    print("\n--- Enter Patient Data ---")
    user_input_df = get_user_input(X.columns, DIABETES_DESCRIPTIONS)
    
    user_input_imputed = preprocessor.imputer.transform(user_input_df)
    user_input_scaled = preprocessor.scaler.transform(user_input_imputed)
    
    prediction = model.predict(user_input_scaled)[0]
    probability = model.predict_proba(user_input_scaled)[0][1]
    
    print(f"\nPrediction: {'Diabetes Detected' if prediction == 1 else 'No Diabetes'}")
    print(f"Probability: {probability:.2f}")

def main():
    while True:
        print("\n=== Disease Prediction System ===")
        print("1. Predict Heart Disease")
        print("2. Predict Diabetes")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            predict_heart_disease()
        elif choice == '2':
            predict_diabetes()
        elif choice == '3':
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
