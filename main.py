# ==========================================
# - Luis Rodriguez
# - lrodriguezgie@outlook.com
# - WIZELINE - ML Modeling Challenge
#
# Tasks
# Using the programming language and libraries of your choice, your tasks are the following:
# 1. Preprocess the features if necessary (justify if not).
# 2. Select a subset of features (justify if not).
# 3. Train a model using the training data set.
# 4. Perform the model metrics that you consider necessary or best to evaluate the
# performance of the model you just trained. Beware of overfitting. The target has some
# noise, even if you had the exact noiseless function you would get around 0.92 R^2.
# 5. Predict the target values with your model for the blind test dataset.

# Submission
# 1. Your code for the challenge, with brief comments explaining its purpose.
# 2. Your 200 predictions for the blind test in csv format with a single column target_pred.
# ==========================================

# ==========================================
# NOTE: I decided to submit the challenge in a .py file due it is the easies way
# to run in almost any device with python installed.
# NOTE: Through the challenge some decision where taken due the lack of meaning
# of the feature 0 to 19 and where taken as pure numeric values
# but in an applied problem some procedures would be different with better understanding of the problem context.
# ==========================================



# ==========================================
# Load Libraries and Packages
# ==========================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import randint
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
from sklearn.model_selection import KFold

# ==========================================
#  Load Datasets of training_data and blind_test_data
# ==========================================

training_data = pd.read_csv('training_data.csv')
print(training_data.head())

blind_test_data = pd.read_csv('blind_test_data.csv')
print(blind_test_data.head())

# ==========================================
#  Exploratory Data Analysis
# ==========================================
# - Check Missing Values
# - Check Duplicated Values
# - Check Unique Values
# - Statistically inspection of the data set, box plot, variance values,
# count, mean, std, min, 25th/50th/75th percentiles, and max for every column
# - Check Variance: Removes features that barely change across the dataset due a
#  feature with near-zero variance carries almost no information for a model.

print(training_data.isnull().sum())
print(blind_test_data.isnull().sum())

print(training_data.duplicated().sum())
print(training_data.nunique())

training_data.boxplot(figsize=(15,6))
plt.show()

variance_values = training_data.var().sort_values(ascending=False)
print("Variance")
print(variance_values)
print("Standard Deviation")
# print(variance_values.std())
print(training_data.std())
print("Description")
training_data.describe()

X = training_data.drop(columns=["target"])

selector = VarianceThreshold(threshold=0.01)

selector.fit(X)

removed_features = X.columns[~selector.get_support()]
selected_features = X.columns[selector.get_support()]

print("Removed features:")
print(removed_features)

print("\nSelected features:")
print(selected_features)

# NOTE: No features were dropped according to the threshold

# RESULTS:
# Data Quality: No missing data, No duplicates, No invalid ranges
# Feature Quality: All features vary sufficiently, No near-constant features, Continuous numeric inputs
# Distribution Quality: No obvious extreme outliers, balanced distributions

# ==========================================
#  Correlation Analysis
# ==========================================

corr_matrix = training_data.corr()

target_corr = corr_matrix["target"].sort_values(
    ascending=False
)

print(target_corr)

plt.figure(figsize=(14,10))

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    annot=False
)

plt.show()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(training_data)
print(X_scaled)

# RESULTS: These are meaningful correlations:
# feature_2 → 0.55
# feature_13 → 0.40
# feature_9  → 0.36
# feature_11 → 0.32
# The rest of the features have weak linear correlation.
# “Correlation analysis revealed that a subset of features exhibited moderate linear
# relationships with the target variable, particularly feature_2, feature_13, feature_9, and feature_11.
# However, several variables showed weak linear correlation, suggesting that nonlinear
# relationships and feature interactions may be present.
# Therefore, will go directly to test ML models starting with Random Forest Model.”

# NOTE: I didn't do any PCA due the dataset shows:
# - Low multicollinearity
# - Low redundancy
# - Features contribute largely distinct information

# ==========================================
#  Random Forest Model
# ==========================================

X = training_data.drop(columns=["target"])
y = training_data["target"]

X_test = blind_test_data.copy()

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# Use the model over the VALID set

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

preds = model.predict(X_val)

r2 = r2_score(y_val, preds)
rmse = mean_squared_error(
    y_val,
    preds
) ** 0.5

mae = mean_absolute_error(y_val, preds)

print("R2:", r2)
print("RMSE:", rmse)
print("MAE:", mae)

# Use the model over the TRAIN set
train_preds = model.predict(X_train)

train_r2 = r2_score(y_train, train_preds) # R2 TRAIN set
val_r2 = r2_score(y_val, preds)  # R2 VALID set

print("Train R2:", train_r2)
print("Validation R2:", val_r2)

# R2: 0.7187375930948523
# RMSE: 2.512907277292944
# MAE: 2.011719188368962
# Train R2: 0.963389266736701
# Validation R2: 0.7187375930948523 <- First Attempt Result

# RESULTS: The Random Forest model achieved strong training performance  0.96 but substantially
# lower validation performance 0.72, indicating overfitting. This suggests that the unconstrained
# tree depth allowed the model to capture noise in the training set rather than generalizable patterns.

# ==========================================
#  Hyperparameters Grid Optimization
# ==========================================
# NOTE: In order to see if the model R2 can be improved based on tuning the hyperparameters I implemented a Grid Search

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

param_dist = {
    "n_estimators": randint(100, 600),
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": randint(2, 30),
    "min_samples_leaf": randint(1, 15),
    "max_features": ["sqrt", "log2", 1.0]
}

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

train_preds = best_model.predict(X_train)
val_preds = best_model.predict(X_val)

print("Best parameters:")
print(search.best_params_)

print("Train R2:", r2_score(y_train, train_preds))
print("Validation R2:", r2_score(y_val, val_preds))

print("Validation MAE:", mean_absolute_error(y_val, val_preds))
print("Validation RMSE:", mean_squared_error(y_val, val_preds) ** 0.5)

# Best parameters:
# {'max_depth': 15, 'max_features': 1.0, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 200}
# Train R2: 0.9587457280473752
# Validation R2: 0.7186007746108329
# Validation MAE: 2.0187553554607716
# Validation RMSE: 2.5135183976462576

# RESULTS: Adjusting the hyperparameters adding some grid search doesn't add any significant improvement in the model
# performance, so the basic model just perform well enough, implying that the next best step would be using another model.

# ==========================================
#  Gradient Boosting Model
# ==========================================

# I decided to try Gradient Models due

# Iteratively corrects errors.
# Learns smoother functions,
# Captures subtle nonlinearities,
# Achieves lower bias.
# Especially on structured/tabular data.

gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_preds = gb_model.predict(X_val)

gb_r2 = r2_score(y_val, gb_preds)
gb_rmse = mean_squared_error(y_val, gb_preds) ** 0.5
gb_mae = mean_absolute_error(y_val, gb_preds)

gb_train_preds = gb_model.predict(X_train)
gb_train_r2 = r2_score(y_train, gb_train_preds)

print("Gradient Boosting Results")
print("Train R2:", gb_train_r2)
print("Validation R2:", gb_r2)
print("RMSE:", gb_rmse)
print("MAE:", gb_mae)

# Gradient Boosting Results
# Train R2: 0.9676237526005519
# Validation R2: 0.8188311304390964
# RMSE: 2.0167980223940414
# MAE: 1.6419196787103612

# ==========================================
# XGBOOST Model
# ==========================================

from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_val)

xgb_r2 = r2_score(y_val, xgb_preds)
xgb_rmse = mean_squared_error(y_val, xgb_preds) ** 0.5
xgb_mae = mean_absolute_error(y_val, xgb_preds)

xgb_train_preds = xgb_model.predict(X_train)
xgb_train_r2 = r2_score(y_train, xgb_train_preds)

print("XGBoost Results")
print("Train R2:", xgb_train_r2)
print("Validation R2:", xgb_r2)
print("RMSE:", xgb_rmse)
print("MAE:", xgb_mae)

# XGBoost Results
# Train R2: 0.9702934658982967
# Validation R2: 0.8416638002543254
# RMSE: 1.8854313678873886
# MAE: 1.5394258608284732

# RESULTS: Ensemble boosting methods substantially outperformed Random Forest performance 0.719 to 0.842.
# This strongly suggests: residual structure exists, sequential correction is effective, boosting matches the dataset better.
# Using the XGBOOST Model I will try to increase the R2 making a more careful tuning.
# But first I will validate this last result.

# ==========================================
# Cross Validation, Importance and Residual Analysis
# ==========================================

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    xgb_model,
    X,
    y,
    cv=5,
    scoring="r2"
)

print(scores)
print("Mean R2:", scores.mean())

# [0.84986997 0.8526065  0.84894349 0.81555682 0.86736608]
# Mean R2: 0.8468685725881387

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": xgb_model.feature_importances_
})

print(
    importance.sort_values(
        by="importance",
        ascending=False
    )
)

#        feature  importance
# 2    feature_2    0.309813
# 13  feature_13    0.158933
# 11  feature_11    0.137120
# 9    feature_9    0.134146
# 18  feature_18    0.073143
# 8    feature_8    0.015820
# 15  feature_15    0.014379
# 19  feature_19    0.014191
# 10  feature_10    0.014169
# 16  feature_16    0.013801
# 14  feature_14    0.013462
# 17  feature_17    0.013305
# 4    feature_4    0.012706
# 12  feature_12    0.011940
# 3    feature_3    0.011657
# 6    feature_6    0.011329
# 1    feature_1    0.010529
# 5    feature_5    0.010509
# 7    feature_7    0.009827
# 0    feature_0    0.009224

# RESULTS: The XGBOOST Model generalizes consistently showing performance is stable across splits making it robust
# Also I demonstrated it some features had almost zero linear correlation,
# still have nontrivial importance showing nonlinear interactions exist and that few variables
# contain most predictive signal, others contribute marginal refinements.

# Residuals
residuals = y_val - xgb_preds
print(residuals)

# Residual plot
plt.figure(figsize=(8,6))

plt.scatter(
    xgb_preds,
    residuals,
    alpha=0.7
)

plt.axhline(
    0,
    color='red',
    linestyle='--'
)

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Analysis")

plt.show()

plt.figure(figsize=(8,6))

plt.hist(
    residuals,
    bins=30
)

plt.title("Residual Distribution")

plt.show()

plt.figure(figsize=(8,6))

plt.scatter(
    X_val["feature_2"],
    residuals,
    alpha=0.7
)

plt.axhline(
    0,
    color='red',
    linestyle='--'
)

plt.xlabel("feature_2")
plt.ylabel("Residuals")

plt.title("Residuals vs feature_2")

plt.show()

# RESULTS: Analyzing the plots we can understand the following:
# - Model captured main nonlinear structure
# - No obvious remaining functional trend
# - Residuals mostly behave like noise
# - Low prediction bias
# - The remaining gap to 0.92 may mostly be noise

# Early Stopping.
# NOTE: At this stage I tried to implement a early stopping in order to reduce residuals noise and create a smother optimization
# but there were no significant improvement.

# ==========================================
# Investigate Interaction Features and Model Order Reduction
# ==========================================

# NOTE: At this point it was quite difficult to improve the R2 parameter,
# adjusting hyperparameters or making modifications to the XGBOOST Model,
# so what I decided to do is adding different most important features interactions and make a
# manual order reduction of the model that may be causing the extra noise.
# After several iterations based on the model performance I keep the feature 2, 9, 10, 11, 13, 17, 18 and 19.
# Also, introducing the no linear interaction features feature_2*feature_11 and feature_2*feature_2.

top_features = [
    #"feature_0",
    #"feature_1",
    "feature_2",
    #"feature_3",
    #"feature_4",
    #"feature_5",
    #"feature_6",
    #"feature_7",
    #"feature_8",
    "feature_9",
    "feature_10",
    "feature_11",
    #"feature_12",
    "feature_13",
    #"feature_14",
    #"feature_15",
    #"feature_16",
    "feature_17",
    "feature_18",
    "feature_19"
]

X_small = X[top_features].copy()

X_small["f2_f11"] = (
    X_small["feature_2"]
    * X_small["feature_11"]
)

X_small["feature_2_sq"] = (
    X_small["feature_2"] ** 2
)

X_train_small, X_val_small, y_train_small, y_val_small = train_test_split(
    X_small,
    y,
    test_size=0.2,
    random_state=42
)

xgb_small_model = XGBRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42
)

xgb_small_model.fit(
    X_train_small,
    y_train_small,
    eval_set=[(X_val_small, y_val_small)],
    verbose=False
)

small_preds = xgb_small_model.predict(X_val_small)

print("R2:", r2_score(y_val_small, small_preds))
print("RMSE:", mean_squared_error(y_val_small, small_preds) ** 0.5)
print("MAE:", mean_absolute_error(y_val_small, small_preds))

train_preds_small = xgb_small_model.predict(X_train_small)

print("Train R2:", r2_score(y_train_small, train_preds_small))
print("Validation R2:", r2_score(y_val_small, small_preds))

scores_small = cross_val_score(
    xgb_small_model,
    X_small,
    y,
    cv=5,
    scoring="r2"
)

print(scores_small)
print("Mean R2 with engineered small features:", scores_small.mean())

importance = pd.DataFrame({
    "feature": X_small.columns,
    "importance": xgb_small_model.feature_importances_
})

print(
    importance.sort_values(
        by="importance",
        ascending=False
    )
)

# R2: 0.8671811394798613
# RMSE: 1.7268336167108025
# MAE: 1.4056640144331056
# Train R2: 0.9943875722264717
# Validation R2: 0.8671811394798613
# [0.88833148 0.86632104 0.87144053 0.84198188 0.89075932]
# Mean R2 with engineered small features: 0.8717668485144348

#         feature  importance
# 7        f2_f11    0.248583
# 0     feature_2    0.176920
# 3    feature_13    0.158587
# 8  feature_2_sq    0.139905
# 1     feature_9    0.128335
# 5    feature_18    0.080992
# 2    feature_11    0.039857
# 6    feature_19    0.013573
# 4    feature_17    0.013247

# RESULTS: This approach generated a new improvement of the R2 from 0.84 to 0.87.
# The only step I would think of implementing was improve the hyperparameter tuning using RandomizedSearchCV to
# systematically searching for the best combination of hyperparameters

# ==========================================
# RandomizedSearchCV Tuning
# ==========================================

xgb_base = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

param_dist = {
    "n_estimators": randint(500, 4000),
    "learning_rate": uniform(0.005, 0.05),
    "max_depth": randint(2, 6),
    "min_child_weight": randint(1, 10),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "gamma": uniform(0, 0.5),
    "reg_alpha": uniform(0, 1),
    "reg_lambda": uniform(0.5, 3)
}

search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=100,
    scoring="r2",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_small, y)

print("Best CV R2:", search.best_score_)
print("Best parameters:")
print(search.best_params_)

best_xgb = search.best_estimator_

best_xgb.fit(X_train_small, y_train_small)

best_preds = best_xgb.predict(X_val_small)

print("Validation R2:", r2_score(y_val_small, best_preds))
print("RMSE:", mean_squared_error(y_val_small, best_preds) ** 0.5)
print("MAE:", mean_absolute_error(y_val_small, best_preds))

train_preds = best_xgb.predict(X_train_small)

print("Train R2:", r2_score(y_train_small, train_preds))
print("Validation R2:", r2_score(y_val_small, best_preds))

scores_best = cross_val_score(
    best_xgb,
    X_small,
    y,
    cv=5,
    scoring="r2"
)

print(scores_best)
print("Mean R2 with engineered best parameters:", scores_best.mean())

# Validation R2: 0.8638964913557643
# RMSE: 1.7480557534347811
# MAE: 1.4105232646622379
# Train R2: 0.9727132383341226
# Validation R2: 0.8638964913557643
# [0.89609965 0.87306149 0.87846485 0.84817831 0.90129284]
# Mean R2 with engineered best parameters: 0.8794194296424103

# RESULTS:

# The best model achieved a cross-validated R2 of approximately 0.87.
# The gap between 0.87 and 0.92 should not be interpreted as simple underperformance.
# Although the theoretical upper bound is around 0.92 due to noise in the target, reaching that value would require
# recovering the underlying noiseless data-generating function almost perfectly.
# Further improvements may be possible through additional tuning or feature engineering,
# but they are expected to be marginal and improvement becomes increasingly
# difficult because the remaining variance is less explainable.


# ==========================================
# PREPARE BLIND TEST FEATURES
# ==========================================

X_blind_small = blind_test_data[top_features].copy()

# Same engineered features used in training
X_blind_small["f2_f11"] = (
    X_blind_small["feature_2"]
    * X_blind_small["feature_11"]
)

X_blind_small["feature_2_sq"] = (
    X_blind_small["feature_2"] ** 2
)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Store predictions from each fold
blind_preds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_small)):

    print(f"\nTraining Fold {fold + 1}")

    # Split data
    X_train_fold = X_small.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]

    X_val_fold = X_small.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]

    # Create model using best hyperparameters
    fold_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_estimators=3824,
        learning_rate=0.014789556739464822,
        max_depth=2,
        min_child_weight=2,
        subsample=0.6284754593840916,
        colsample_bytree=0.9844762255295657,
        gamma=0.45267532097803187,
        reg_alpha=0.0944429607559284,
        reg_lambda=2.5490203202490704,
        n_jobs=-1
    )

    # Train fold model
    fold_model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False
    )

    # Predict blind test
    fold_blind_preds = fold_model.predict(X_blind_small)

    # Store predictions
    blind_preds.append(fold_blind_preds)

blind_preds = np.array(blind_preds)

final_blind_preds = blind_preds.mean(axis=0)

submission = pd.DataFrame({
    "target": final_blind_preds
})

print("\nFinal Predictions:")
print(submission.head())

# Save predictions
submission.to_csv(
    "blind_test_predictions.csv",
    index=False
)

print("\nFile saved as:")
print("blind_test_predictions.csv")