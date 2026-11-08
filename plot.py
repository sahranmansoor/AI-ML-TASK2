import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("Titanic-Dataset.csv")

    sns.set_theme(style="whitegrid")

    # histograms for all numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(data=df, x=col, bins=100, kde=True, ax=axes[i], edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

    # Boxplot for all numeric columns
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(data=df, y=col, ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
