import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Loard Data
iris = pd.read_csv("iris.data", names = ["Sepal Length","Sepal width","Petal length","Petal width","Class"],header=None)
# # check
# print(iris.head())

# データ分け(Train-Set, Test-Set)
# y = iris.loc[:,"Class"]
# x = iris.loc[:,"Sepal Length","Sepal width","Petal length","Petal width"]
y = iris["Class"]
x = iris[["Sepal Length", "Sepal width", "Petal length", "Petal width"]]
# # Check
# print(x.head())
# print(y.head())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)

# 学習する
SVC_model = SVC()
SVC_model.fit(x_train, y_train)

# テストする、評価する
    # 予測度確認
y_pred = SVC_model.predict(x_test)

# 答え合わせをして正解率を表示する
print("■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■")
print("正解率を",accuracy_score(y_test, y_pred))