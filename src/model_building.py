import os
import logging
import pickle
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train, y_train, params: dict) -> RandomForestClassifier:
    """Train the RandomForestClassifier model."""
    try:
        clf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            random_state=params['random_state']
        )
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')
        return clf
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        params = load_params('params.yaml')['model_building']

        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.drop(columns=['label']).values
        y_train = train_data['label'].values

        clf = train_model(X_train, y_train, params)

        save_model(clf, 'models/model.pkl')
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
