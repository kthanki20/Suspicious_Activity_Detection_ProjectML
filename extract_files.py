import zipfile
import os

#This is used unziped the datasets and store in data_extracted folder.
zip_file_path = r'C:\Users\eTech\Desktop\ML Project\Datasets.zip'
extracted_dir = r'C:\Users\eTech\Desktop\ML Project\Data_Extracted'

os.makedirs(extracted_dir, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)

print(f"Files extracted to {extracted_dir}")