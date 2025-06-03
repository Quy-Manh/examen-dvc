import os
import pandas as pd 
import numpy as np
import joblib
import pickle
import sklearn
from xgboost import XGBRegressor

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

def run_save_model(X_train, y_train):
    """
    upload best params from gridseach and train model xgboost
    """
    #get best params
    params_path = 'models/best_params.pkl'  
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"best_params.pkl not found: {params_path}")
    with open(params_path, 'rb') as f:
        best_params = pickle.load(f)
    print("Upload XGBoost-params:", best_params)

    #initilize model 
    xgb_model = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    #train model
    xgb_model.fit(X_train, y_train)
    
    #save model
    model_filename = './models/trained_xgb_model.joblib'
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    joblib.dump(xgb_model, model_filename)
    print(f"XGBoost model saved as: {model_filename}")

def main():
    # load data
    X_train, X_test, y_train, y_test = load_data()
    # run and save model
    run_save_model(X_train, y_train)

if __name__ == "__main__":
    main()
