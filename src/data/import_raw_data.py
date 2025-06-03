import os
import requests

# url of raw data
url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
# define target folder
output_dir = "./data/raw_data"
os.makedirs(output_dir, exist_ok=True)
#
output_path = os.path.join(output_dir, "raw.csv")

#download raw data
response = requests.get(url)
response.raise_for_status()  # if connection error
with open(output_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:  
            f.write(chunk)
print(f"raw data uploaded and saved here: {output_path}")
