# DATA3402.Spring.2026
Raisin Variety Classification
This project aims to apply Random Forest Classification and Logistic Regression to correctly identify two varieties of raisins based on the Raisin Binary Classification dataset: https://www.kaggle.com/datasets/nimapourmoradi/raisin-binary-classification.

Overview:
The task is to classify raisin grains into one of two Turkish cultivars — Besni or Kecimen — using seven physical measurements extracted from images (Area, Perimeter, Eccentricity, etc.). The approach formulates this as a supervised binary classification problem, comparing a Logistic Regression baseline against a Random Forest classifier. Our best model (Logistic Regression) achieved 92.8% accuracy and 0.959 AUC on the held-out test set, outperforming Random Forest which achieved 87.2% accuracy and 0.939 AUC.

Summary of Workdone
Data:
Type: Tabular CSV — each row represents one raisin grain with 7 numerical morphological features and one class label.
Input: Area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, Extent, Perimeter
Output: Binary class label — Besni (1) or Kecimen (0)
Size: 900 samples, perfectly balanced (450 per class), no missing values
Split: 540 train · 180 validation · 180 test (stratified)

Clean Up:
No missing values or duplicate rows were found — no imputation needed.
IQR-based outliers were detected in several features but retained, as they represent real morphological variation.
The target column Class was label-encoded (Kecimen = 0, Besni = 1).
All 7 numerical features were standardized using StandardScaler (zero mean, unit variance), which is important for Logistic Regression's gradient-based solver.

Data Visualization:
Overlapping histograms for each feature split by class reveal that Area, ConvexArea, MajorAxisLength, and Perimeter show strong separation between the two varieties — Besni raisins are consistently larger.
Eccentricity and Extent overlap heavily, making them weaker individual predictors.
A correlation heatmap shows that Area, ConvexArea, MajorAxisLength, and Perimeter are highly correlated (>0.97), meaning they carry largely redundant information.

Problem Formulation:
Input: 7 scaled numerical features per raisin grain
Output: Binary label — 0 (Kecimen) or 1 (Besni)
Models:
Logistic Regression — linear baseline; fast, interpretable, and well-suited to scaled numerical features. Uses L2 regularization by default.
Random Forest — ensemble of 100 decision trees; captures non-linear boundaries and feature interactions without requiring scaling.
Loss/Optimizer — Logistic Regression uses log-loss minimized via the lbfgs solver. Random Forest uses Gini impurity for splitting.
Hyperparameters: max_iter=1000 for LR convergence; n_estimators=100 and random_state=42 for RF reproducibility.

Training:
Software: Python 3.12, scikit-learn, pandas, matplotlib
Hardware: Standard laptop CPU — no GPU required
Training time: Under 5 seconds for both models combined
No training curves — both models are non-iterative (fit in one pass); there are no loss-vs-epoch curves to plot
No early stopping needed for either model at this dataset size

Performance Comparison:
Primary metrics: Accuracy (fraction of correct predictions) and ROC-AUC (quality of class separation; 0.5 = random, 1.0 = perfect).
ModelSplitAccuracyROC-AUCLogistic RegressionValidation0.85560.9244Logistic RegressionTest0.92780.9589Random ForestValidation0.85560.9239Random ForestTest0.87220.9391
Visualizations in the notebook include ROC curves and confusion matrices for both models on the test set.

Conclusions:
Logistic Regression outperformed Random Forest on both accuracy and AUC on the test set, suggesting the decision boundary between the two raisin types is largely linear.
The four size-related features (Area, ConvexArea, MajorAxisLength, Perimeter) are the strongest predictors — Besni raisins are physically larger.
The jump from validation (85.6%) to test accuracy (92.8%) for Logistic Regression is likely due to favorable random split variance rather than overfitting.
The dataset's perfect class balance made evaluation straightforward with no need for resampling techniques.

Future Work:
Hyperparameter tuning — grid search over LR regularization strength (C) and RF depth/leaf size to squeeze out additional performance.
Additional models — try SVM with RBF kernel or Gradient Boosting (XGBoost / LightGBM).
Feature engineering — ratio features like Area/Perimeter or PCA to address high multicollinearity.
Cross-validation — replace single validation split with k-fold CV for more robust performance estimates.
Official Kaggle submission — upload the generated CSV to obtain a leaderboard score for direct comparison.


How to Reproduce Results:

Install dependencies:

bash   pip install pandas numpy matplotlib scikit-learn

Download the data from Kaggle and place Raisin_Dataset.csv in the same directory as the notebook.
Run the notebook — open Raisin_Classification.ipynb and run all cells in order (Kernel → Restart & Run All).
The submission file raisin_kaggle_submission.csv will be saved to the same directory automatically.

Overview of Files in Repository
FileDescriptionRaisin_Classification.ipynbMain notebook — data loading, EDA, preprocessing, training, and evaluationRaisin_Dataset.csvRaw dataset downloaded from Kaggle (900 samples, 7 features + label)raisin_kaggle_submission.csvGenerated submission file with predictions and probabilitiesREADME.mdThis file

Software Setup
Required packages:
pandas
numpy
matplotlib
scikit-learn
Install all at once:
bashpip install pandas numpy matplotlib scikit-learn
No custom package installation is needed — just clone the repo, install the dependencies, and run the notebook.

