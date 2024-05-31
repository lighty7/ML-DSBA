import pandas as pd

def explore_dataset(file_path):
    # Check if the file is a CSV or Excel file
    if file_path.endswith('.csv'):
        # Load CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        # Load Excel file into a pandas DataFrame
        df = pd.read_excel(file_path)
    else:
        print("Unsupported file format. Please provide a CSV or Excel file.")
        return

    # Display basic information about the DataFrame
    print("Dataset information:")
    print(df.info())
    
    # Display the first few rows of the DataFrame
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    # Display summary statistics for numerical columns
    print("\nSummary statistics:")
    print(df.describe())
    
    # Display unique values for categorical columns
    print("\nUnique values for categorical columns:")
    for column in df.select_dtypes(include='object').columns:
        print(f"{column}: {df[column].unique()}")

# Example usage
file_path = './iris_sample_100.csv'  # Change this to the path of your CSV or Excel file
explore_dataset(file_path)
