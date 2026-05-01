# DATA3402.Spring.2026: Abrar Elahi
# Raisin Variety Classification

This project aims to apply Random Forest Classification and Logistic Regression to correctly identify two varieties of raisins based on the Raisin Binary Classification dataset: https://www.kaggle.com/datasets/nimapourmoradi/raisin-binary-classification

---

## Overview

The goal of this project is to build a binary classification model that distinguishes between two raisin varieties: Besni and Kecimen. Each raisin is described by seven physical measurements like Area, Perimeter, Eccentricity, etc. A Logistic Regression baseline is compared against a Random Forest classifier, using Accuracy and ROC-AUC as the primary metrics. Logistic Regression is the best model and achieved **92.8% accuracy** and **0.96 AUC** on the held-out test set, with Random Forest performing comparably at 87.2% accuracy and 0.94 AUC.

---

## Summary of Workdone

### Data

- **Type:** Tabular CSV — each row represents one raisin grain with 7 numerical features and one class label
- **Input:** Area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, Extent, Perimeter
- **Output:** Binary class label — Besni (1) or Kecimen (0)
- **Size:** 900 samples, balanced (450 per class), no missing values
- **Split:** 540 train · 180 validation · 180 test (stratified)

---

### Preprocessing / Clean Up

- No missing values or duplicate rows were found
- IQR-based outliers were detected in several features but retained, as they represent real structural variation
- The target column `Class` was label-encoded (Kecimen = 0, Besni = 1)
- All 7 numerical features were standardized using `StandardScaler` (zero mean, unit variance), which is important for Logistic Regression's gradient-based solver

---

### Data Visualization:

1. **Feature Distribution Histograms** — shows which features separate the two classes best. `Area` and `Perimeter` have the clearest gap between Besni and Kecimen, while `Extent` barely separates at all. Best visualization for understanding the data.
<img width="806" height="465" alt="image" src="https://github.com/user-attachments/assets/113785f7-4625-439c-bece-25fb58dbf724" />
<img width="835" height="459" alt="image" src="https://github.com/user-attachments/assets/c61e6ad2-ce5c-47bd-8073-b7d70e43aad8" />


2. **Correlation Heatmap** — reveals the >0.97 correlation between the four size features. Important because it explains why a simple model like Logistic Regression still performs well — the classes are almost linearly separable.
<img width="1236" height="884" alt="image" src="https://github.com/user-attachments/assets/c9789f0e-bbd7-4299-a193-665ea8be1573" />

3. **ROC Curves** — directly compares both models. Logistic Regression's curve hugs the top-left corner more tightly (AUC 0.959 vs 0.939), making this the clearest visualization of which model wins.
<img width="859" height="719" alt="image" src="https://github.com/user-attachments/assets/69794e09-ed89-46a7-a866-896bade5a612" />

4. **Confusion Matrices** — shows exactly where each model makes mistakes. Both models struggle slightly with the same boundary cases, confirming the overlap seen in `Eccentricity` and `Extent`.
<img width="1641" height="576" alt="image" src="https://github.com/user-attachments/assets/b9df50df-cabd-4a59-ae39-d931f56c18a9" />

---

## Problem Formulation

- **Input:** 7 scaled numerical features per raisin grain
- **Output:** Binary label — 0 (Kecimen) or 1 (Besni)

### Models

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear baseline; fast, interpretable, and well-suited to scaled numerical features. Uses L2 regularization by default. |
| **Random Forest** | Ensemble of 100 decision trees; captures non-linear boundaries and feature interactions without requiring scaling. |

- **Loss / Optimizer:** Logistic Regression uses log-loss minimized via the `lbfgs` solver. Random Forest uses Gini impurity for splitting.
- **Hyperparameters:** `max_iter=1000` for LR convergence; `n_estimators=100` and `random_state=42` for RF reproducibility.

---

## Training

- **Software:** Python 3.12, scikit-learn, pandas, matplotlib
- **Hardware:** Standard laptop CPU — no GPU required
- **Training time:** Under 5 seconds for both models combined
- **Training curves:** Not applicable — both models are non-iterative (fit in one pass); there are no loss-vs-epoch curves to plot
- **Early stopping:** Not needed for either model at this dataset size

---

## Performance Comparison

**Primary Metrics:**
- **Accuracy** — fraction of correct predictions
- **ROC-AUC** — quality of class separation (0.5 = random, 1.0 = perfect)

| Model               | Split      | Accuracy | ROC-AUC |
|---------------------|------------|----------|---------|
| Logistic Regression | Validation | 85.56%   | 0.9244  |
| Logistic Regression | Test       | 92.78%   | 0.9589  |
| Random Forest       | Validation | 85.56%   | 0.9239  |
| Random Forest       | Test       | 87.22%   | 0.9391  |

**Best Model:** Logistic Regression — 92.78% accuracy and 0.959 AUC on the test set.

Visualizations in the notebook include ROC curves and confusion matrices for both models on the test set.

---

## Conclusions

- Logistic Regression outperformed Random Forest on both accuracy and AUC on the test set, suggesting the decision boundary between the two raisin types is largely linear
- The four size-related features (Area, ConvexArea, MajorAxisLength, Perimeter) are the strongest predictors — Besni raisins are physically larger
- The jump from validation (85.6%) to test accuracy (92.8%) for Logistic Regression is likely due to favorable random split variance rather than overfitting
- The dataset's perfect class balance made evaluation straightforward with no need for resampling techniques

---

## Future Work

- **Hyperparameter tuning** — grid search over LR regularization strength (`C`) and RF depth/leaf size to squeeze out additional performance
- **Additional models** — try SVM with RBF kernel or Gradient Boosting such as XGBoost
- **Feature engineering** — ratio features like Area/Perimeter or PCA to address high multicollinearity
- **Official Kaggle submission** — upload the generated CSV to obtain a leaderboard score for direct comparison

---

## How to Reproduce Results

1. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

2. **Download the data** from [Kaggle](https://www.kaggle.com/datasets/nimapourmoradi/raisin-binary-classification) and place `Raisin_Dataset.csv` in the same directory as the notebook.

3. **Run the notebook** — open `Raisin_Classification_Elahi.ipynb` and run all cells in order (Kernel → Restart & Run All).

4. The submission file `raisin_kaggle_submission.csv` will be saved to the same directory automatically.

---

## Overview of Files in Repository

| File | Description |
|------|-------------|
| `README.md` | Project summary, conclusions, and reproduction steps |
| `Raisin_Classification_Elahi.ipynb` | Main notebook — data loading, EDA, preprocessing, training, and evaluation |
| `Raisin_Dataset.csv` | Raw dataset downloaded from Kaggle (900 samples, 7 features + label) |
| `raisin_kaggle_submission.csv` | Generated submission file with predictions and probabilities |

---

## Software Setup

**Required packages:**
```
pandas
numpy
matplotlib
scikit-learn
```

Install all at once:
```bash
pip install pandas numpy matplotlib scikit-learn
```

Clone the repo, install the dependencies, and run the notebook.

---

## Citations

- Cinar, I., Koklu, M., & Tasdemir, S. (2020). *Classification of Raisin Grains Using Machine Vision and Artificial Intelligence Methods*. Selcuk University, Konya, Turkey.
  Dataset: https://www.kaggle.com/datasets/nimapourmoradi/raisin-binary-classification

- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825–2830.
  https://scikit-learn.org
