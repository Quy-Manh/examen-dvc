import pandas as pd
import os
from sklearn.model_selection import train_test_split

#read data
df=pd.read_csv("./data/raw_data/raw.csv")

#split in feature X and target y
X=df.drop(["date", "silica_concentrate"],axis=1)
y=df["silica_concentrate"]

#data splitting in train and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

#safe files 
output_folderpath="./data/processed_data"
for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
    output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
    file.to_csv(output_filepath, index=False)