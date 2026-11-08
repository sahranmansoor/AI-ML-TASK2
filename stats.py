import pandas as pd

def main():
    df = pd.read_csv("Titanic-Dataset.csv")

    # Summary statistics
    print(df.describe())

if __name__ == "__main__":
    main()
