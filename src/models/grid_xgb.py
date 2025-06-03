import os
import pickle

import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


def load_data():
    """
    Load data for gridsearch
    """
    X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
    X_test = pd.read_csv('data/processed_data/X_test_scaled.csv')
    y_train = pd.read_csv('data/processed_data/y_train.csv')
    y_test = pd.read_csv('data/processed_data/y_test.csv')
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    return X_train, X_test, y_train, y_test

def run_gridsearch(X_train, y_train):
    """
    gridsearch xgboost
    """
    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    param_grid_xgb = {
            "n_estimators":  [50, 100, 150, 200, 250],
            "max_depth":     [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3]
    }
    
    grid_xgb = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid_xgb,
            scoring="r2",
            cv=5,
            n_jobs=-1,
            verbose=2
    )
    grid_xgb.fit(X_train, y_train)
    return grid_xgb
def save_params(best_params: dict, filepath="models/best_params.pkl"):
    """
    Speichert das Dictionary best_params als Pickle in filepath.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(best_params, f)
    print(f"➜ Parameter gespeichert unter: {filepath}")
def main():
    # 0. Daten laden
    X_train, X_test, y_train, y_test = load_data()
    # 1. GridSearchCV ausführen
    print("▶ Starte GridSearchCV auf XGBRegressor …")
    grid = run_gridsearch(X_train, y_train)
    print("✅ GridSearch abgeschlossen.")
    # 2. Beste Parameter ausgeben
    best_params = grid.best_params_
    best_score_cv = grid.best_score_
    print(f"Beste XGBoost-Parameter: {best_params}")
    print(f"Bestes CV-R²-Score: {best_score_cv:.4f}")
    # 3. Parameter speichern (ohne Versionierung)
    save_params(best_params, filepath="models/best_params.pkl")
if __name__ == "__main__":
    main()
