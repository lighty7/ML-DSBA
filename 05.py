import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_dataset(file_path):
    # Load the dataset into a pandas DataFrame
    df = pd.read_csv(file_path)  # Assuming it's a CSV file, change accordingly if it's an Excel file

    # Plot scatter plots
    sns.pairplot(df, hue='species')
    plt.title("Pairplot of the Dataset")
    plt.show()

    # Plot bar chart for categorical column (assuming the 'species' column is categorical)
    if 'species' in df.columns:
        sns.countplot(x='species', data=df)
        plt.title("Bar Chart of Species Column")
        plt.xlabel('Species')
        plt.ylabel("Count")
        plt.show()
    else:
        print("No categorical column found to plot bar chart.")

# Example usage
file_path = './iris_sample_100.csv'  # Change this to the path of your CSV file
visualize_dataset(file_path)
