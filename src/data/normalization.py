import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

#load train & test sets
X_train = pd.read_csv('data/processed_data/X_train.csv')
X_test = pd.read_csv('data/processed_data/X_test.csv')
#MinMaxScaler as data are not normal distributed
scaler = MinMaxScaler()
#fit and transform train and test set
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#save as df
X_train_scaled_df=pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df=pd.DataFrame(X_test_scaled, columns=X_test.columns)

#save scaled version in processed_data
X_train_scaled_df.to_csv("data/processed_data/X_train_scaled.csv", index=False)
X_test_scaled_df.to_csv("data/processed_data/X_test_scaled.csv", index=False)
