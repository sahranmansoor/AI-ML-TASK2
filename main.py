import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("Titanic-Dataset.csv")

    # Summary statistics
    print(df.describe())

    sns.set_theme(style="whitegrid")


    # histograms for all numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    print(numeric_cols)

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

    # correlation matrix for feature relationships
    corr_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr_matrix, 
        annot=True,          
        cmap='coolwarm',     
        fmt=".2f",           
        linewidths=0.5       
    )

    plt.title('Correlation Matrix of Titanic Features', fontsize=14)
    plt.show()

    # pairplot for feature relationships
    features_to_plot = ['Survived', 'Pclass', 'Age', 'Fare']

    sns.pairplot(
        data=df[features_to_plot], 
        hue='Survived',            
        palette='husl',            
        diag_kind='kde',           
        plot_kws={'alpha': 0.6}    
    )

    plt.suptitle('Pairplot of Key Features by Survival Status', y=1.02, fontsize=14)
    plt.show()

if __name__ == "__main__":
    main()
