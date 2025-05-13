import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def analyze_iris_data():
    """
    Loads, explores, analyzes, and visualizes the Iris dataset.
    """
    try:
        # --- Task 1: Load and Explore the Dataset ---
        print("--- Task 1: Load and Explore the Dataset ---")

        # Load the Iris dataset
        iris = load_iris(as_frame=True)
        df = iris.frame

        # Display the first few rows
        print("\nFirst 5 rows of the dataset:")
        print(df.head())

        # Explore the structure of the dataset
        print("\nDataset information:")
        df.info()

        # Check for missing values
        print("\nMissing values:")
        print(df.isnull().sum())

        # Clean the dataset (no missing values in Iris, but demonstrating the concept)
        if df.isnull().sum().sum() > 0:
            print("\nHandling missing values...")
            # Example: Filling missing numerical values with the mean
            for col in df.select_dtypes(include=['number']).columns:
                df[col].fillna(df[col].mean(), inplace=True)
            # Example: Dropping rows with any missing values
            # df.dropna(inplace=True)
            print("Missing values after cleaning:")
            print(df.isnull().sum())
        else:
            print("\nNo missing values to handle in the Iris dataset.")

        # --- Task 2: Basic Data Analysis ---
        print("\n--- Task 2: Basic Data Analysis ---")

        # Compute basic statistics of numerical columns
        print("\nBasic statistics of numerical columns:")
        print(df.describe())

        # Perform groupings on the 'target' (species) column and compute the mean of other columns
        print("\nMean of features grouped by species:")
        print(df.groupby('target').mean())

        # Identify any patterns or interesting findings
        print("\nInteresting Findings:")
        print("- The 'sepal length' generally appears larger than 'sepal width'.")
        print("- 'Petal length' and 'petal width' show greater variability compared to sepal dimensions.")
        print("- There are noticeable differences in the average feature values across the three Iris species.")

        # --- Task 3: Data Visualization ---
        print("\n--- Task 3: Data Visualization ---")

        # 1. Line chart (mean feature values per species)
        mean_values_species = df.groupby('target').mean()
        plt.figure(figsize=(10, 6))
        for column in mean_values_species.columns:
            plt.plot(mean_values_species.index, mean_values_species[column], label=column)
        plt.title('Mean Feature Values per Iris Species')
        plt.xlabel('Species (0, 1, 2)')
        plt.ylabel('Mean Value (cm)')
        plt.xticks(mean_values_species.index, iris.target_names)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 2. Bar chart (average petal length per species)
        plt.figure(figsize=(8, 6))
        sns.barplot(x='target', y='petal length (cm)', data=df, errorbar=None)
        plt.title('Average Petal Length per Iris Species')
        plt.xlabel('Species')
        plt.ylabel('Average Petal Length (cm)')
        plt.xticks([0, 1, 2], iris.target_names)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

        # 3. Histogram (distribution of sepal width)
        plt.figure(figsize=(8, 6))
        sns.histplot(df['sepal width (cm)'], kde=True)
        plt.title('Distribution of Sepal Width')
        plt.xlabel('Sepal Width (cm)')
        plt.ylabel('Frequency')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

        # 4. Scatter plot (sepal length vs. petal length)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', data=df)
        plt.title('Sepal Length vs. Petal Length')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.legend(title='Species', labels=iris.target_names)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print("\n--- Analysis Complete ---")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    analyze_iris_data()