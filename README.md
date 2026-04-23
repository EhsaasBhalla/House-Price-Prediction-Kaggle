# 🏠 House Price Prediction Challenge

## 📋 Project Overview

This project builds a robust **regression pipeline** to estimate the median house value (`TargetPrice`) in various California districts. Acting as a data consultant for a real estate investment firm, the goal is to predict property values while navigating real-world data challenges like Gaussian noise, missing records, and obfuscated feature names.

* **Target Variable:** `TargetPrice` (representing the median house value in units of $100,000).
* **Evaluation Metric:** Root Mean Squared Error (RMSE).

---

## 📂 Dataset

The dataset features demographic and structural attributes for specific property clusters:
* **Training Data (`estate_train.csv`)**: 16,512 instances, 12 features
* **Test Data (`estate_test.csv`)**: 4,128 instances for leaderboard submission
* **Key Features**: 
  * `PropertyID`, `IncomeLevel`, `PropertyAge`, `TotalRooms`, `TotalBedrooms`
  * `NeighborhoodPop`, `AvgOccupancy`, `RoomsPerHousehold`, `BedroomsRatio`

---

## 🔍 Workflow

### 1. Data Exploration & Basic Info
* Checked dataset structural overview via `.info()` and `.describe()`
* Verified data scales: Significant scale differences observed (e.g., `NeighborhoodPop` vs `IncomeLevel`), indicating the need for robust feature scaling.
* Visualized correlation heatmaps and price distributions to identify feature importance and relationships.

### 2. Handling Missing Values & Duplicates
* **Duplicates:** 0 duplicate rows detected in both train and test records.
* **Missing Data:** `PropertyAge` had **1,313 missing values** in the training set and 355 in the test set. 
* **Imputation Strategy:** Utilized **K-Nearest Neighbors (KNN) Imputation** to carefully estimate and fill the missing housing age data based on spatial/structural similarities, maximizing accuracy over basic median/mean fills.

### 3. Data Preprocessing & Feature Engineering
* **Feature Scaling:** Handled extreme outliers (e.g. anomalous occupancy numbers) and properly scaled attributes.
* Formatted features like `RoomsPerHousehold` and `BedroomsRatio`.

### 4. Train-Validation Split
* Split the dataset into 80% training data and 20% validation data to evaluate baseline and tuned model predictions safely before full-scale retraining.

---

## 🤖 Models Trained

A broad assortment of regression models were explored to establish strong baselines:
* Linear Regression & Ridge Regression
* Decision Tree Regressor
* SVR (Support Vector Regressor)
* KNN (K-Nearest Neighbors Regressor)
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

---

## ⚙️ Hyperparameter Tuning

Applied exhaustive **RandomizedSearchCV** to hyper-tune the most promising ensemble/tree algorithms:
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

---

## 📈 Model Performance Results

After training and tuning, all models were evaluated against the validation set using **RMSE**, **MAE**, and **R²** variance scores.

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| **XGBoost (Tuned)** | **0.1359** | **0.0939** | **0.8567** |
| Gradient Boosting (Tuned) | 0.1382 | 0.0943 | 0.8518 |
| XGBoost (Base) | 0.1384 | 0.0956 | 0.8515 |
| Gradient Boosting (Base) | 0.1406 | 0.0977 | 0.8465 |
| Random Forest (Tuned) | 0.1439 | 0.0988 | 0.8394 |
| Random Forest (Base) | 0.1439 | 0.0988 | 0.8393 |
| SVR | 0.1600 | 0.1200 | 0.7907 |
| KNN | 0.1700 | 0.1200 | 0.7838 |
| Decision Tree | 0.1900 | 0.1300 | 0.7246 |
| Ridge Regression | 0.2100 | 0.1600 | 0.6722 |
| Linear Regression | 0.2100 | 0.1600 | 0.6721 |

---

## 🏆 Model Selection

* **Best Model:** **XGBoost (Tuned)** was selected for yielding the lowest RMSE (`0.1359`) and highest test variance coverage (`85.67%`).
* **Final Action:** The best performing XGBoost model was completely retrained on the full unified dataset (training + validation) to maximize learning capacity, prior to blind test application.

---

## 📤 Final Output

* Iterated over the 4,128 test items to predict continuous values.
* Created the final submission output mapping `PropertyID` to the predicted continuous `TargetPrice`.
  ```text
  submission.csv
  ```

---

## 🚀 Conclusion

This project highlights a complete regression analytical pipeline capable of penetrating noisy PropTech data. Through meticulous handling of missing variables (via KNN) and gradient optimization, the final model consistently restricts mean absolute errors to less than $9,400 (MAE: `0.0939` * $100k) amidst a volatile target spectrum. 

https://www.kaggle.com/competitions/house-price-prediction-iitm/data
