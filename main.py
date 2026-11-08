import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("Titanic-Dataset.csv")

    # Summary statistics
    print(df.describe())

if __name__ == "__main__":
    main()
