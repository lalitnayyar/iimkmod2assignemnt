import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

def download_titanic_dataset():
    """
    Download the Titanic dataset from Kaggle using the Kaggle API.
    Requires Kaggle API credentials to be set up.
    """
    try:
        # Initialize the Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        print("Downloading Titanic dataset from Kaggle...")
        
        # Download the competition files
        api.competition_download_files('titanic', path='.')
        
        # Extract the zip file
        with zipfile.ZipFile('titanic.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        
        # Remove the zip file
        os.remove('titanic.zip')
        
        if os.path.exists('train.csv'):
            print("Successfully downloaded train.csv!")
            print(f"File size: {os.path.getsize('train.csv')} bytes")
        else:
            print("Error: train.csv was not found after extraction")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nTo use the Kaggle API, you need to:")
        print("1. Create a Kaggle account at https://www.kaggle.com")
        print("2. Go to your account settings (https://www.kaggle.com/account)")
        print("3. Scroll to 'API' section and click 'Create New API Token'")
        print("4. This will download 'kaggle.json'. Place this file in:")
        print("   - Windows: %USERPROFILE%\\.kaggle\\")
        print("5. Run this script again after setting up the credentials")

if __name__ == "__main__":
    download_titanic_dataset()
