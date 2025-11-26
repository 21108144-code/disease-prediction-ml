# Disease Prediction System 🏥

A machine learning application to predict the likelihood of **Heart Disease**, **Diabetes**, and **Breast Cancer** based on medical data.

## Features
- **Multi-Disease Prediction**: Support for three major diseases.
- **Machine Learning Models**: Utilizes Logistic Regression, SVM, Random Forest, and XGBoost.
- **Interactive Frontend**: User-friendly web interface built with Streamlit.
- **CLI Support**: Command-line scripts for training and prediction.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/21108144-code/disease-prediction-ml.git
    cd disease-prediction-ml
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Setup
Download the required datasets:
```bash
python src/data_loader.py
```

### 2. Training
Train the models and evaluate performance:
```bash
python src/train.py
```

### 3. Web Interface (Recommended)
Launch the interactive web app:
```bash
streamlit run app.py
```

### 4. CLI Prediction
Run predictions from the command line:
```bash
python src/predict.py
```

## Project Structure
- `data/`: Dataset storage.
- `src/`: Source code for data loading, preprocessing, modeling, and evaluation.
- `app.py`: Streamlit frontend application.
- `requirements.txt`: Python dependencies.

## Technologies
- Python
- Scikit-learn
- XGBoost
- Pandas & NumPy
- Streamlit
