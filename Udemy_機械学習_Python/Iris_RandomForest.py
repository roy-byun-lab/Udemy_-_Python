import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 1. Loard Iris 
iris = load_iris()
x = iris.data  # 独立変数(Feature)
y = iris.target  # 従属変数 (Label)
# # check
# print(iris.head())
# print(x.head())
# print(y.head())

# 2. データ分け(Train-Set, Test-Set)
    # random_state: ランダムに振り分け(任意設定)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
# # Data-Scaling
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# 3. 学習
    # n_estimators: Decision Tree Num
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# 4. 予測
y_pred = model.predict(x_test)

# 5. 評価
accuracy = accuracy_score(y_test, y_pred)
print(f"正解率: {accuracy:.4f}")

# # 7. 備考
# print("\nConfusion_matrix:")
# print(confusion_matrix(y_test, y_pred))

# print("\n分類レポート:")
# print(classification_report(y_test, y_pred))

