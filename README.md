# Bulldozer Price Prediction

Machine learning project for predicting the auction sale prices of used heavy equipment using historical transaction data.

The project focuses on tabular data preprocessing, feature engineering, comparing tree-based regression algorithms, and evaluating predictive performance.

## Project Overview

The objective is to estimate equipment sale prices based on machine characteristics and historical auction information.

The project includes:

- Exploratory data analysis and missing value assessment
- Feature engineering using historical sale dates
- Categorical feature preprocessing
- Training and comparison of multiple regression models
- Ensemble predictions
- Evaluation using Root Mean Squared Logarithmic Error (RMSLE)

## Models

Three machine learning algorithms were implemented:

- **Random Forest Regressor**
- **XGBoost Regressor**
- **CatBoost Regressor**

The target variable was transformed using `log1p`, with model predictions converted back to the original price scale using `expm1`.

A simple weighted ensemble of Random Forest and CatBoost predictions was also evaluated.

## Feature Engineering

The preprocessing stage includes:

- Extracting year, month, and day of the week from sale dates
- Handling missing numerical and categorical values
- Creating missing-value indicators
- Applying one-hot encoding and categorical encoding
- Comparing different preprocessing approaches for tree-based models

## Evaluation

The dataset was sorted chronologically to evaluate model performance on later observations.

The primary evaluation metric is **RMSLE**, which measures differences between logarithmically transformed actual and predicted prices.

The project was developed as an educational ML experiment. Some preprocessing decisions and model settings were explored using the available labeled data; therefore, the reported holdout results should not be interpreted as an entirely independent final test.

## Technologies

Python, pandas, NumPy, scikit-learn, XGBoost, CatBoost.

## Running the Project

1. Download the original dataset from the Kaggle Blue Book for Bulldozers competition.
2. Place the required CSV files in the `data/` directory.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Run:

   `python bulldozer_prediction.py`

The project expects the original training and testing files and a separate CSV containing the corresponding test labels, as described in the source code.

## Future Improvements

- Train-only preprocessing and feature selection
- Separate validation and final testing periods
- Systematic hyperparameter optimization
- Evaluation of alternative ensemble strategies
- Feature importance analysis
