# Task 2: Exploratory Data Analysis (EDA) — Titanic Dataset

## 🎯 Objective
Understand the Titanic dataset using descriptive statistics and visualizations to uncover patterns, trends, anomalies, and feature relationships before modeling.

## 🛠️ Tools Used
- **Python** (Pandas, Matplotlib, Seaborn)
- **Dataset**: Titanic Dataset (`Titanic-Dataset.csv`)

## 🔍 Steps Performed

### 1. Summary Statistics (`stats.py`)
Generated mean, median, std, min, max, and quartiles for all numeric features using `df.describe()`.

### 2. Histograms & Boxplots (`plots.py`)
Plotted distributions and outlier views for all numeric columns: `Survived`, `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`.

### 3. Feature Relationships (`feature_relation.py`)
- **Correlation Heatmap**: Identified linear relationships between numeric features.
- **Pairplot**: Visualized feature interactions, colored by `Survived` status.

### 4. Anomaly Detection (`anomalies.py`)
- **Missing Data**: Identified columns with null values.
- **Outliers**: Used IQR method to detect extreme values in `Age`, `Fare`, `SibSp`, `Parch`.

### 5. Patterns & Trends (`patterns_trends.py`)
- Binned `Age` into 5 categories (Child, Teen, Adult, Middle-Aged, Senior).
- Analyzed survival by `Pclass` and by traveling alone vs. with family.
- Plotted KDE trends comparing survivors vs. non-survivors across `Age`, `Fare`, and `Pclass`.

## 📊 Key Findings

### Missing Data
| Column    | % Missing  |
|-----------|------------|
| Cabin     | [77.1]%    |
| Age       | [19.8]%    |
| Embarked  | [0.22]%    |

### Outliers Detected (IQR Method)
| Feature | Lower Bound   | Upper Bound  | # Outliers|
|---------|---------------|--------------|-----------|
| Age     | [ -6.6]       | [64.8]       | [ 11]     |
| Fare    | [-26.7]       | [65.6]       | [116]     |
| SibSp   | [ -1.5]       | [ 2.5]       | [ 46]     |
| Parch   | [    0]       | [   0]       | [213]     |

### Strong Correlations
- `Pclass` ↔ `Fare`: **[-0.55]** (strong negative → multicollinearity risk)
- `Survived` ↔ `Pclass`: **[-0.34]** (moderate negative)
- `Survived` ↔ `Fare`: **[0.26]** (moderate positive)

### Survival Patterns
| Group                  | Survival Rate  |
|------------------------|----------------|
| 1st Class              | [0.63]         |
| 3rd Class              | [0.23]         |
| Child (0–12)           | [0.58]         |
| Senior (60+)           | [0.22]         |
| Traveling Alone        | [0.50]         |
| With Family            | [0.30]         |

### Key Trends (from KDE Plots)
1. **Age**: Survivors show a clear spike at ages [0]–[12], indicating children were prioritized.
2. **Fare**: Survivors cluster at [higher/lower] fares, confirming the positive correlation with survival.
3. **Pclass**: Class [1] passengers had visibly higher survival rates than Class [3].

## 💡 Feature-Level Inferences
1. **Age** has a non-linear relationship with survival — correlation missed it, but KDE plots revealed it.
2. **Fare** is heavily right-skewed with extreme outliers — must be handled before modeling.
3. **Pclass** and **Fare** are strongly correlated — potential multicollinearity for linear models.
4. **Cabin** has [77.1]% missing data — likely best dropped during preprocessing.
