import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("Titanic-Dataset.csv")


    # Pattern 1: Does Age affect survival?
    bins = [0, 12, 18, 35, 60, 100]
    labels = ['Child (0-12)', 'Teen (13-18)', 'Adult (19-35)', 'Middle-Aged (36-60)', 'Senior (60+)']

    # 2. Assign each passenger to an age group
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

    # 3. Calculate survival rate for each group
    print(df.groupby('AgeGroup', observed=True)['Survived'].mean().round(3))

    # Pattern 2: Does class affect survival?
    print("\nSurvival rate by Pclass:")
    print(df.groupby('Pclass')['Survived'].mean().round(3))

    # Pattern 3: Did having family help or hurt?
    print("\nSurvival rate by traveling alone vs with family:")
    df['Alone'] = (df['SibSp'] + df['Parch']) == 0
    print(df.groupby('Alone')['Survived'].mean().round(3))

    # Create a 1x3 grid for side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # TREND 1: Age (Non-linear pattern — children survived more)
    sns.kdeplot(data=df[df['Survived'] == 1], x='Age', ax=axes[0], 
                label='Survived', fill=True, alpha=0.5, color='green')
    sns.kdeplot(data=df[df['Survived'] == 0], x='Age', ax=axes[0], 
                label='Died', fill=True, alpha=0.5, color='red')
    axes[0].set_title('Trend 1: Age Distribution by Survival')
    axes[0].set_xlabel('Age')
    axes[0].legend()

    # TREND 2: Fare (Higher fare = higher survival)
    sns.kdeplot(data=df[df['Survived'] == 1], x='Fare', ax=axes[1], 
                label='Survived', fill=True, alpha=0.5, color='green')
    sns.kdeplot(data=df[df['Survived'] == 0], x='Fare', ax=axes[1], 
                label='Died', fill=True, alpha=0.5, color='red')
    axes[1].set_title('Trend 2: Fare Distribution by Survival')
    axes[1].set_xlabel('Fare')
    axes[1].set_xlim(0, 300)  # Zoom in to ignore extreme outliers
    axes[1].legend()

    # TREND 3: Pclass (Lower class number = higher survival)
    sns.kdeplot(data=df[df['Survived'] == 1], x='Pclass', ax=axes[2], 
                label='Survived', fill=True, alpha=0.5, color='green')
    sns.kdeplot(data=df[df['Survived'] == 0], x='Pclass', ax=axes[2], 
                label='Died', fill=True, alpha=0.5, color='red')
    axes[2].set_title('Trend 3: Passenger Class by Survival')
    axes[2].set_xlabel('Pclass (1=1st, 2=2nd, 3=3rd)')
    axes[2].legend()

    plt.tight_layout()
    plt.suptitle('Key Trends: How Features Relate to Survival', fontsize=14, y=1.02)
    plt.show()

if __name__ == "__main__":
    main()
