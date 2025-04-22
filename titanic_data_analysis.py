import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_understand_data(file_path):
    """Load and analyze the raw data"""
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # 1. Understanding Raw Data
    print("\n1. UNDERSTANDING RAW DATA")
    print("-" * 50)
    print(f"Total number of records: {len(df)}")
    print("\nColumns and their descriptions:")
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")
    
    # Missing values analysis
    print("\nMissing values in each column:")
    print(df.isnull().sum())
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate records: {duplicates}")
    
    return df

def clean_data(df):
    """Clean the dataset by handling missing values and removing non-essential columns"""
    print("\n2. DATA CLEANING")
    print("-" * 50)
    
    # Create a copy to avoid modifying original data
    df_cleaned = df.copy()
    
    # Handle missing values
    # Age: Use median for missing values
    df_cleaned['Age'] = df_cleaned['Age'].fillna(df_cleaned['Age'].median())
    
    # Cabin: Most values are missing, create a binary feature indicating if cabin is known
    df_cleaned['Has_Cabin'] = df_cleaned['Cabin'].notna().astype(int)
    df_cleaned.drop('Cabin', axis=1, inplace=True)
    
    # Embarked: Fill with mode as it has very few missing values
    df_cleaned['Embarked'] = df_cleaned['Embarked'].fillna(df_cleaned['Embarked'].mode()[0])
    
    # Remove non-essential columns
    columns_to_drop = ['Name', 'Ticket', 'PassengerId']
    df_cleaned.drop(columns_to_drop, axis=1, inplace=True)
    
    print("Columns removed:", columns_to_drop)
    print("Remaining columns:", df_cleaned.columns.tolist())
    
    return df_cleaned

def transform_data(df):
    """Apply data transformation techniques"""
    print("\n3. DATA TRANSFORMATION")
    print("-" * 50)
    
    df_transformed = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = ['Sex', 'Embarked']
    
    for col in categorical_columns:
        df_transformed[col] = le.fit_transform(df_transformed[col])
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_columns = ['Age', 'Fare']
    df_transformed[numerical_columns] = scaler.fit_transform(df_transformed[numerical_columns])
    
    print("Transformed columns:")
    print("Categorical:", categorical_columns)
    print("Numerical:", numerical_columns)
    
    return df_transformed

def generate_insights(original_df, cleaned_df, transformed_df):
    """Generate insights and create visualizations"""
    print("\n4. INSIGHTS AND REFLECTION")
    print("-" * 50)
    
    print("Data cleaning summary:")
    print(f"Original shape: {original_df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Transformed shape: {transformed_df.shape}")
    
    # Save insights to a text file
    with open('data_analysis_insights.txt', 'w') as f:
        f.write("Titanic Dataset Analysis Insights\n")
        f.write("================================\n\n")
        f.write("1. Data Quality:\n")
        f.write(f"- Original dataset had {len(original_df)} records\n")
        f.write(f"- Missing values were handled for Age and Embarked columns\n")
        f.write("- Cabin column was transformed into a binary feature due to high missing values\n\n")
        f.write("2. Feature Engineering:\n")
        f.write("- Created 'Has_Cabin' feature to capture cabin information\n")
        f.write("- Removed non-essential columns: Name, Ticket, PassengerId\n")
        f.write("- Encoded categorical variables: Sex, Embarked\n")
        f.write("- Standardized numerical features: Age, Fare\n")

def main():
    # File path to your dataset
    file_path = 'train.csv'  # Make sure this file is in the same directory
    
    try:
        # Execute all steps
        original_df = load_and_understand_data(file_path)
        cleaned_df = clean_data(original_df)
        transformed_df = transform_data(cleaned_df)
        generate_insights(original_df, cleaned_df, transformed_df)
        
        # Save the final processed dataset
        transformed_df.to_csv('processed_titanic_data.csv', index=False)
        print("\nProcessing completed successfully!")
        print("Check 'data_analysis_insights.txt' for detailed insights")
        print("Processed data saved as 'processed_titanic_data.csv'")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file {file_path}")
        print("Please make sure to download the Titanic dataset (train.csv) from Kaggle and place it in the same directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
