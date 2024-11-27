import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Loard Data
# data = pd.read_csv("diabetes.csv", names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"])
data = pd.read_csv("diabetes.csv",header=0)
# # check
# print(data.head())

x = data.drop("Outcome", axis=1)
y = data["Outcome"] # []:retruns Series, [[]]:retruns DataFrame
# # check``
# print(x.head())
# print(y.head())

# データ分け(Train, Test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

# 学習する
# xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)

# パラメータを利用して学習させる
params = {"eta":[0.1,0.3,0.9],"max_depth":[2,4,6,8]}

# GridSearchCVはCross検証を利用して最適なパラメータを探す
xgb_grid = GridSearchCV(
    estimator = xgb.XGBClassifier(),
    param_grid=params
)
xgb_grid.fit(x_train, y_train)

for key, value in xgb_grid.best_params_.items():
    print(key, value)

# テストする、評価する
xgb_pred = xgb_grid.predict(x_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
print("******************************")
print(f"XGBoost Accuracy: {xgb_acc:.4f}")