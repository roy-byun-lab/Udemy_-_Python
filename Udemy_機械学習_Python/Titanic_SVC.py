import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Loard Data
data = pd.read_csv("Titanic _train.csv",header=0)
# # Check
# print(data.head())

# 1.前処理
    # 名前削除
    # Ticket削除
    # Cabin削除
data = data.drop(["Name","Ticket","Cabin"], axis=1)
# # Check
# print(data.head())

    # 欠損データ処理
#Check
# print("処理前:",len(data))
data = data.dropna(subset=["Embarked"])
data = data.dropna()
# # 損失データを埋める
# imputer = SimpleImputer(strategy='mean')
# data = imputer.fit_transform(data)
# print("処理後:",len(data))
    # 性別変換
    # Embarked変換
label_encoder = LabelEncoder()
data["Sex"] = label_encoder.fit_transform(data["Sex"])
data["Embarked"] = label_encoder.fit_transform(data["Embarked"])
# # Check
# print(data.head())

# 2.データ分け
y = data["Survived"]
x = data.drop(["Survived"], axis=1)
# # Check
# print(x.head())
# print(y.head())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)

# 3.学習する
SVC_model = SVC()
SVC_model.fit(x_train, y_train)

# 4.テストする、評価する
    # 予測度確認
y_pred = SVC_model.predict(x_test)

    # 答え合わせをして正解率を表示する
print("■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■")
print("正解率を",accuracy_score(y_test, y_pred))