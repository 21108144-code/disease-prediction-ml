import os
import pandas as pd
from preprocessing import load_and_preprocess
from models import ModelFactory
from evaluate import evaluate_model, print_metrics

DATA_DIR = 'data'
DATASETS = ['breast_cancer.csv', 'heart_disease.csv', 'diabetes.csv']

def main():
    models = ModelFactory.get_all_models()
    
    for dataset_name in DATASETS:
        filepath = os.path.join(DATA_DIR, dataset_name)
        if not os.path.exists(filepath):
            print(f"Dataset {dataset_name} not found. Skipping.")
            continue
            
        print(f"\n{'='*20} Processing {dataset_name} {'='*20}")
        
        try:
            X_train, X_test, y_train, y_test = load_and_preprocess(filepath)
            
            for model_name in models:
                print(f"\nTraining {model_name}...")
                model = ModelFactory.get_model(model_name)
                model.fit(X_train, y_train)
                
                metrics = evaluate_model(model, X_test, y_test)
                print_metrics(metrics)
                
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")

if __name__ == "__main__":
    main()
