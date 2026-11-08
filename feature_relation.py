import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("Titanic-Dataset.csv")

    sns.set_theme(style="whitegrid")

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
