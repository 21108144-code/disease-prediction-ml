import streamlit as st
import pandas as pd
import numpy as np
import os
from src.preprocessing import DataPreprocessor
from src.models import ModelFactory

DATA_DIR = 'data'

def load_data(dataset_name):
    filepath = os.path.join(DATA_DIR, dataset_name)
    if not os.path.exists(filepath):
        return None
    return pd.read_csv(filepath)

def train_model(df, model_name='random_forest'):
    X = df.drop(columns=['target'])
    y = df['target']
    
    preprocessor = DataPreprocessor()
    preprocessor.imputer.fit(X)
    X_imputed = preprocessor.imputer.transform(X)
    preprocessor.scaler.fit(X_imputed)
    X_scaled = preprocessor.scaler.transform(X_imputed)
    
    model = ModelFactory.get_model(model_name)
    model.fit(X_scaled, y)
    
    return model, preprocessor

def main():
    st.set_page_config(page_title="Disease Prediction System", page_icon="🏥", layout="wide")
    
    st.title("🏥 Disease Prediction System")
    st.markdown("Predict the likelihood of diseases based on medical data.")
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a Disease", ["Home", "Heart Disease", "Diabetes", "Breast Cancer"])
    
    if app_mode == "Home":
        st.image("https://img.freepik.com/free-vector/medical-healthcare-blue-background_1017-26807.jpg", use_container_width=True)
        st.markdown("""
        ### Welcome to the Disease Prediction System
        
        This application uses Machine Learning to predict the probability of:
        - **Heart Disease**
        - **Diabetes**
        - **Breast Cancer**
        
        👈 **Select a disease from the sidebar to get started.**
        """)

    elif app_mode == "Heart Disease":
        st.header("❤️ Heart Disease Prediction")
        df = load_data('heart_disease.csv')
        
        if df is None:
            st.error("Dataset not found. Please run data loader first.")
            return

        model, preprocessor = train_model(df)
        
        st.subheader("Enter Patient Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
            cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], 
                            format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"}[x])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
            
        with col2:
            chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[1, 0], format_func=lambda x: "True" if x == 1 else "False")
            restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], 
                                 format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 2: "LV Hypertrophy"}[x])
            thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
            
        with col3:
            exang = st.selectbox("Exercise Induced Angina", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST", options=[1, 2, 3], 
                               format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x])
            ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
            thal = st.selectbox("Thal", options=[3, 6, 7], 
                              format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversable Defect"}[x])
        
        if st.button("Predict Heart Disease"):
            input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                                    columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
            
            # Transform
            input_imputed = preprocessor.imputer.transform(input_data)
            input_scaled = preprocessor.scaler.transform(input_imputed)
            
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            if prediction == 1:
                st.error(f"⚠️ High Risk of Heart Disease Detected (Probability: {probability:.2%})")
            else:
                st.success(f"✅ Low Risk of Heart Disease (Probability: {probability:.2%})")

    elif app_mode == "Diabetes":
        st.header("🩸 Diabetes Prediction")
        df = load_data('diabetes.csv')
        
        if df is None:
            st.error("Dataset not found.")
            return

        model, preprocessor = train_model(df)
        
        st.subheader("Enter Patient Details")
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
            glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
            blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
            skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
            
        with col2:
            insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            
        if st.button("Predict Diabetes"):
            input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], 
                                    columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
            
            input_imputed = preprocessor.imputer.transform(input_data)
            input_scaled = preprocessor.scaler.transform(input_imputed)
            
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            if prediction == 1:
                st.error(f"⚠️ High Risk of Diabetes Detected (Probability: {probability:.2%})")
            else:
                st.success(f"✅ Low Risk of Diabetes (Probability: {probability:.2%})")

    elif app_mode == "Breast Cancer":
        st.header("🎗️ Breast Cancer Prediction")
        df = load_data('breast_cancer.csv')
        
        if df is None:
            st.error("Dataset not found.")
            return
            
        model, preprocessor = train_model(df)
        
        st.info("Note: The Breast Cancer dataset has 30 features. For simplicity, this demo uses the mean values for most features and lets you adjust key ones.")
        
        # Simplified input for demo purposes (or we can list all 30, but that's a lot for a UI)
        # Let's just show a few key ones for the demo, or maybe just a "Run on Test Data" button?
        # Better: Let's list the top 5-6 features if possible, or just use a random sample from test set to demonstrate.
        # For a real app, we'd need all inputs. Let's provide inputs for 'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'.
        # And fill the rest with mean values from the dataset.
        
        st.subheader("Enter Key Tumor Characteristics")
        col1, col2 = st.columns(2)
        
        with col1:
            mean_radius = st.number_input("Mean Radius", value=float(df['mean radius'].mean()))
            mean_texture = st.number_input("Mean Texture", value=float(df['mean texture'].mean()))
            mean_perimeter = st.number_input("Mean Perimeter", value=float(df['mean perimeter'].mean()))
            
        with col2:
            mean_area = st.number_input("Mean Area", value=float(df['mean area'].mean()))
            mean_smoothness = st.number_input("Mean Smoothness", value=float(df['mean smoothness'].mean()))
            
        if st.button("Predict Breast Cancer"):
            # Create a row with all features set to mean
            input_row = df.drop(columns=['target']).mean().to_frame().T
            
            # Update with user values
            input_row['mean radius'] = mean_radius
            input_row['mean texture'] = mean_texture
            input_row['mean perimeter'] = mean_perimeter
            input_row['mean area'] = mean_area
            input_row['mean smoothness'] = mean_smoothness
            
            input_imputed = preprocessor.imputer.transform(input_row)
            input_scaled = preprocessor.scaler.transform(input_imputed)
            
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            # Target 0 = Malignant, 1 = Benign in sklearn dataset usually? 
            # Let's check. load_breast_cancer description:
            # class: WDBC-Malignant, WDBC-Benign
            # target: 0 = malignant, 1 = benign (Wait, sklearn usually maps 0 to malignant?)
            # Actually, let's verify. 
            # In sklearn: malignant is usually class 0, benign is class 1.
            # But let's check the data loader.
            # df['target'] = data.target
            # If prediction == 0 -> Malignant (Bad)
            # If prediction == 1 -> Benign (Good)
            
            if prediction == 0:
                st.error(f"⚠️ Malignant Tumor Detected (Probability: {1-probability:.2%})")
            else:
                st.success(f"✅ Benign Tumor (Probability: {probability:.2%})")

if __name__ == "__main__":
    main()
