import pandas as pd
from mpmath.functions.functions import expm1
from sympy.codegen import Print
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

df_train = pd.read_csv(DATA_DIR / "Train_bulldozer.csv")
df_valid_X = pd.read_csv(DATA_DIR / "Valid_bulldozer.csv")
df_valid_y = pd.read_csv(DATA_DIR / "ValidSolution.csv")
df_test = pd.read_csv(DATA_DIR / "Test_bulldozer.csv")
df_test_y = pd.read_csv(DATA_DIR / "test_predictions.csv")
df_test_y["SalePrice"] = df_test_y["SalesPrice"]
df_test_y.drop(["SalesPrice"], axis=1, inplace=True)
df_test = pd.merge(df_test, df_test_y, on="SalesID")
median = df_train["MachineHoursCurrentMeter"].median()
median1 = df_train['auctioneerID'].median()
df_train = pd.concat([df_train, df_test], ignore_index=True)
print(df_test.info())
print(df_train.info())
df_test["saledate"] = pd.to_datetime(df_test["saledate"])
df_train["saledate"] = pd.to_datetime(df_train["saledate"])
bip = []

for i in df_train.columns:
    bip.append(df_train[i].isna().sum()/len(df_train[i]))
print(bip)
df_analysis_nan = pd.DataFrame({"Columns" : list(df_train.columns),
                                "share_of_nan": bip})
print(df_analysis_nan.head(52))
spisok = []
for i in range(len(df_analysis_nan['Columns'])):
    if df_analysis_nan['share_of_nan'][i] >0.6:
        spisok.append(df_analysis_nan['Columns'][i])
print(spisok)



print(df_train.info())
df_cat = df_train.copy()
df_cat.drop(["SalesID", "MachineID", "fiProductClassDesc"], axis=1, inplace=True)
cat_cols = ["ModelID", "datasource"]
cat_cols_1 = df_cat.select_dtypes(include=["object", "string", "category"]).columns.to_list()

for i in cat_cols_1:
    cat_cols.append(i)
print("cat_cols", cat_cols)
for i in cat_cols:
    df_cat[i] = df_cat[i].fillna("missing").astype(str)
df_cat["year"] = df_cat["saledate"].dt.year
df_cat["month"] = df_cat["saledate"].dt.month
df_cat["dow"] = df_cat["saledate"].dt.dayofweek
df_cat = df_cat.sort_values(by=["saledate"], ascending=True)
print(df_cat.info())
df_cat.drop("saledate", axis=1, inplace=True)
cutoff = -12457

y_cat = df_cat["SalePrice"]
y_cat_tr = y_cat[:cutoff]
y_cat_test = y_cat[cutoff:]
y_cat_log = np.log1p(y_cat)
y_cat_log_tr = y_cat_log[:cutoff]
y_cat_log_test = y_cat_log[cutoff:]
X_cat = df_cat.drop(["SalePrice"], axis=1)
X_cat_tr = X_cat[:cutoff]
X_cat_test = X_cat[cutoff:]
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_log_error
ct = CatBoostRegressor(random_seed=42)
ct.fit(X_cat_tr, y_cat_log_tr, cat_features = cat_cols)
y_cat_pred_log = ct.predict(X_cat_test)
y_cat_pred = np.expm1(y_cat_pred_log)

print("rmsle_cat", root_mean_squared_log_error(y_cat_pred, y_cat_test))

print("df_cat_info", df_cat.info())

print(spisok)
df_train["MachineHoursCurrentMeter"] = df_train["MachineHoursCurrentMeter"].fillna(median)

print(df_train.info())
print(df_train.columns)
s = []
for i in df_train.columns:
    if (df_train[i].isna()).sum()>1:
        s.append(i)
print(s)
for i in s:
    df_train[f'{i}_has_data'] = df_train[i].isna().astype(int)
df_train['auctioneerID'] = df_train['auctioneerID'].fillna(median1)
for i in s:
    if pd.api.types.is_numeric_dtype(df_train[i]):
        df_train[i] = df_train[i].fillna(df_train[i].median())
    else:
        df_train[i] = df_train[i].fillna("Unknown")
print(df_train.info())
print(df_train.head())
df_train = df_train.drop(columns=["fiProductClassDesc"])
for i in s:
    spisok.append(i)
print(spisok)
spisok.remove('MachineHoursCurrentMeter')
spisok.remove('auctioneerID')

df_train = df_train.drop(columns=["fiModelDesc"])
spisok.append("state")
df_train = df_train.drop(columns=["fiBaseModel"])
df_train = df_train.drop(columns=["ProductGroup"])
df_train = df_train.drop(columns=["ProductGroupDesc"])
print(len(spisok))
#dummies = pd.get_dummies(df_train, columns=spisok).astype(int)
#print(dummies.info())
a=[]
for i in df_train.columns:
    a.append({"Columns": i,
              "Number_of_unique": df_train[i].nunique()})
a=pd.DataFrame(a)

df_train = df_train.drop(columns=["SalesID", "MachineID"])
df_train["year"] = df_train["saledate"].dt.year
df_train["month"] = df_train["saledate"].dt.month
df_train["dow"] = df_train["saledate"].dt.dayofweek
df_train = df_train.sort_values("saledate")
df_train = df_train.drop(columns=["saledate"])
listik = []
for i in df_train.columns:
    if type(df_train[i][1]) == str:
        listik.append(i)
print(listik)
print(a.head(87))
print(df_train.info())
a=a.drop([0,1,2])
print(a["Columns"])
b=[]
for t in range(len(a["Columns"])):
    if a["Number_of_unique"].iloc[t] < 10:
        b.append(a["Columns"].iloc[t])
print(b)
X_train = df_train.drop(columns=["SalePrice"])
print(X_train.head(5))
X_train = pd.get_dummies(X_train, columns=list(b), dtype= int)
print(X_train.info())
for col in X_train.select_dtypes(include=["object", "string", "category"]).columns:
    X_train[col] = X_train[col].astype("category").cat.codes
print(X_train.info())

y = df_train["SalePrice"]
y_log = np.log1p(y)
X_tr = X_train[:cutoff]
y_tr = y_log[:cutoff]
X_test = X_train[cutoff:]
y_test = y[cutoff:]
#rf top features: n_estimators = 80, max_samples=256720, max_features=0.5
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_log_error
dict = []
#params = {"n_estimators": [2000],
          #"learning_rate": [0.1],
         # "colsample_bytree": [0.6]
         # }
rf = RandomForestRegressor(n_estimators=80, max_samples=256720, max_features=0.5, n_jobs=-1)
rf.fit(X_tr, y_tr)
pred_rf_log = rf.predict(X_test)
pred_rf = np.expm1(pred_rf_log)
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.15, colsample_bytree=0.6, n_jobs=-1)
xgb.fit(X_tr.values, y_tr)
pred_xgb_log = xgb.predict(X_test.values)
pred_xgb = np.expm1(pred_xgb_log)

pred = 0.5*pred_rf + 0.5*y_cat_pred
print("rmsle_final", root_mean_squared_log_error(y_test, pred))

from sklearn.model_selection import RandomizedSearchCV
#rand_search = RandomizedSearchCV(estimator=xgb, param_distributions=params, n_jobs=-1, cv=5, verbose=1, scoring='neg_mean_squared_error')
#rand_search.fit(X_tr.values, y_tr)
#print("Best_params", rand_search.best_params_)
#best_model = rand_search.best_estimator_

#y_pred_log = best_model.predict(X_test.values)
#y_pred = np.expm1(y_pred_log)
#print("rmsle", root_mean_squared_log_error(y_test, y_pred))


#Best_params {'n_estimators': 1000, 'learning_rate': 0.15, 'colsample_bytree': 0.6}
#rmsle 0.27007102966308594