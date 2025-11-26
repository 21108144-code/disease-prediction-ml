from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class ModelFactory:
    @staticmethod
    def get_model(model_name, random_state=42):
        if model_name == 'logistic_regression':
            return LogisticRegression(random_state=random_state, max_iter=1000)
        elif model_name == 'svm':
            return SVC(probability=True, random_state=random_state)
        elif model_name == 'random_forest':
            return RandomForestClassifier(random_state=random_state)
        elif model_name == 'xgboost':
            return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    @staticmethod
    def get_all_models():
        return ['logistic_regression', 'svm', 'random_forest', 'xgboost']
