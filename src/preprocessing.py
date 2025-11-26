import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def preprocess(self, df, target_column='target', test_size=0.2, random_state=42):
        """
        Preprocesses the dataframe: imputes missing values, scales features, and splits into train/test.
        """
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Handle missing values
        X_imputed = self.imputer.fit_transform(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test

def load_and_preprocess(filepath, target_column='target'):
    df = pd.read_csv(filepath)
    preprocessor = DataPreprocessor()
    return preprocessor.preprocess(df, target_column)
