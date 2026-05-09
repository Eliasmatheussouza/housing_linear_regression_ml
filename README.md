# 🏠 California Housing Price Prediction — Linear Regression (scikit-learn)

A supervised machine learning project using **Linear Regression** to predict housing prices in California, based on the 1990 U.S. Census dataset available in scikit-learn.

---

## 📌 Objective

Predict the median house value (`MedHouseVal`) for California block groups using census features, evaluate model performance with error metrics, and visualize the regression fit.

---

## 📊 Dataset

| Attribute | Detail |
|---|---|
| Source | 1990 U.S. Census — California Housing (`sklearn.datasets.fetch_california_housing`) |
| Instances | 20,640 rows |
| Features | 8 numeric attributes |
| Granularity | Block group (smallest geographical unit published by the U.S. Census) |

**Features:** `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`

---

## ⚙️ Tech Stack

- **Language:** Python
- **Libraries:** scikit-learn, pandas, matplotlib

---

## 🔍 Methodology

The notebook covers two modeling experiments:

**Experiment 1 — Simple Linear Regression (toy dataset)**

A small dataset with 5 points is used to introduce core concepts. A baseline straight line (`y = x`) is compared against the fitted regression line, and both are evaluated with MAE and MSE.

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X.values.reshape(-1, 1), y)
a = reg.coef_[0]   # slope
b = reg.intercept_ # intercept
```

| Model | MAE | MSE |
|---|---|---|
| Baseline (y = x) | 0.28 | 0.108 |
| Linear Regression | 0.224 | 0.078 |

**Experiment 2 — Univariate Regression on California Housing**

`MedInc` (median income) is used as the sole feature to predict `MedHouseVal`. Dataset is split 67/33 for train/test.

| Metric | Value |
|---|---|
| MAE | ~0.627 |
| MSE | ~0.703 |

**Experiment 3 — Multivariate Regression on California Housing**

All 8 features are used. Results are nearly identical to the univariate model, suggesting that `MedInc` is the dominant predictor and that the relationship is non-linear in nature — pointing to opportunities for improvement with tree-based models.

| Metric | Value |
|---|---|
| MAE | ~0.627 |
| MSE | ~0.703 |

---

## 📈 Results

Scatter plots of `y_pred` vs `y_test` were generated for both California Housing experiments. The dispersion pattern illustrates the limitations of a linear model for this dataset.

---

## 📁 Structure

```
housing_linear_regression_ml/
├── Linear Regression (sklearn).ipynb   # Main notebook
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/Eliasmatheussouza/housing_linear_regression_ml.git

# Install dependencies
pip install scikit-learn pandas matplotlib

# Open the notebook
jupyter notebook "Linear Regression (sklearn).ipynb"
```
