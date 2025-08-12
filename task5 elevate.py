import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For better visuals
sns.set(style="whitegrid", palette="muted")
# Replace 'data.csv' with your dataset path
df = pd.read_csv("C:\\Users\\Devansh Tyagi\\Downloads\\train.csv")

# Check first few rows
df.head()

# Shape of dataset
print("Shape:", df.shape)

# Data types and non-null values
df.info()

# Summary statistics
df.describe()

# Missing values
df.isnull().sum()

# Value counts for categorical variables
for col in df.select_dtypes(include='object'):
    print(f"\nValue counts for {col}:\n", df[col].value_counts())
# Histograms for numeric columns
df.hist(bins=20, figsize=(15, 10))
plt.suptitle('Histograms of Numerical Columns')
plt.show()

# Boxplots for numeric columns
for col in df.select_dtypes(include=np.number):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
# Scatterplots for two numerical variables
sns.scatterplot(data=df, x='col1', y='col2', hue='target')
plt.title('Col1 vs Col2')
plt.show()

# Boxplots for numerical vs categorical
sns.boxplot(x='CategoryColumn', y='NumericColumn', data=df)
plt.title('NumericColumn by CategoryColumn')
plt.show()

# Pairplot
sns.pairplot(df, hue='target')
plt.show()
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
