import pandas as pd
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

def preprocess_dataset(df):
    # Handle missing data (Iris dataset doesn't have missing values, but we'll simulate some)
    df.iloc[::10, 0] = float('NaN')
    # Simulate missing values in the first column
    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df[df.columns])
    # Encode categorical variable (if applicable)
    # Since Iris dataset doesn't have categorical variables, we'll skip this step
    # Perform feature scaling
    scaler = StandardScaler()
    df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
    return df

# Preprocess Iris dataset
preprocessed_df = preprocess_dataset(iris_df)

# Display preprocessed dataset
print("Preprocessed dataset:")
print(preprocessed_df.head())