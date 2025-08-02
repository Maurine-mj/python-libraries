# Iris Dataset Analysis and Visualization Assignment

# Importing Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# -----------------------------
# Task 1: Load and Explore the Dataset
# -----------------------------

try:
    # Load the Iris dataset from sklearn
    iris_raw = load_iris()
    # Convert to pandas DataFrame
    df = pd.DataFrame(data=iris_raw.data, columns=iris_raw.feature_names)
    df['species'] = pd.Categorical.from_codes(iris_raw.target, iris_raw.target_names)

    print("First five rows of the dataset:")
    print(df.head())

    print("\nDataset Information:")
    print(df.info())

    print("\nMissing values in the dataset:")
    print(df.isnull().sum())

except FileNotFoundError:
    print("Dataset not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")

# -----------------------------
# Task 2: Basic Data Analysis
# -----------------------------

print("\nBasic Statistical Summary:")
print(df.describe())

print("\nMean values grouped by species:")
grouped_means = df.groupby('species').mean()
print(grouped_means)

# Interesting finding
print("\nObservation: Versicolor and Virginica have closer petal sizes than Setosa.")

# -----------------------------
# Task 3: Data Visualization
# -----------------------------

# Set Seaborn style
sns.set(style="whitegrid")

# 1. Line Chart (not time series, but showing trends across index for one feature)
plt.figure(figsize=(10, 5))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.plot(subset.index, subset['sepal length (cm)'], label=species)
plt.title("Sepal Length Trend Across Samples")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar Chart: Average petal length per species
plt.figure(figsize=(7, 5))
grouped_means['petal length (cm)'].plot(kind='bar', color=['skyblue', 'orange', 'green'])
plt.title("Average Petal Length per Species")
plt.ylabel("Petal Length (cm)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 3. Histogram: Distribution of sepal length
plt.figure(figsize=(7, 5))
sns.histplot(df['sepal length (cm)'], bins=10, kde=True, color='purple')
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='deep')
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.tight_layout()
plt.show()

# -----------------------------
# Final Observations
# -----------------------------
print("\nKey Findings:")
print("1. Setosa species is clearly separable from the others based on petal measurements.")
print("2. Sepal and petal lengths are positively correlated.")
print("3. Petal length has the highest variation across species.")
