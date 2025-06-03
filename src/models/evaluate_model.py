import pandas as pd 
import numpy as np
from joblib import load
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data():
    """
    load data 
    """
    X_train = pd.read_csv('data/processed_data/X_train.csv')
    X_test  = pd.read_csv('data/processed_data/X_test.csv')
    y_train = pd.read_csv('data/processed_data/y_train.csv')
    y_test  = pd.read_csv('data/processed_data/y_test.csv')
    y_train = np.ravel(y_train)
    y_test  = np.ravel(y_test)
    return X_train, X_test, y_train, y_test

def main():
    #load data
    model = load("models/trained_xgb_model.joblib")
    X_train, X_test, y_train, y_test = load_data()
    #make prediction
    y_pred = model.predict(X_test)
    # Evaluate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    # store results in dictionairy
    metrics = {
        "mse": mse,
        "mae": mae,
        "r2": r2
    }
    # save metrics as json file
    metrics_path = Path("metrics/scores.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=4))
    
if __name__ == "__main__":
    main()