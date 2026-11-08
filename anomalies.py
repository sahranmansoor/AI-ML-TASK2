import pandas as pd

def main():
    df = pd.read_csv("Titanic-Dataset.csv")

    # Missing Data
    missing_pct = (df.isnull().sum() / len(df)) * 100
    print(missing_pct[missing_pct > 0].sort_values(ascending=False))

    # Outlier Analysis
    cols = ["Age", "Fare", "SibSp", "Parch"]

    for c in cols:
        Q1 = df[c].quantile(0.25)
        Q3 = df[c].quantile(0.75)

        IQR = Q3 - Q1

        lo = Q1 - 1.5 * IQR
        up = Q3 + 1.5 * IQR

        outliers = df[(df[c] < lo) | (df[c] > up)]

        print(f"\n({c})")
        print("lower: ", lo)
        print("upper: ", up)
        print("OUTLIERS: ", len(outliers))

if __name__ == "__main__":
    main()
