import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

# 学習する
# SVC_model = SVC(kernel='linear', C=1, gamma='scale')
SVC_model = SVC()
SVC_model.fit(x_train, y_train)

# テストする、評価する
    # 予測度確認
y_pred = SVC_model.predict(x_test)

# 答え合わせをして正解率を表示する
print("******************************")
print("正解率を",accuracy_score(y_test, y_pred))