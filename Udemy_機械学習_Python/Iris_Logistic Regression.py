import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# 1. Irisデータセットの読み込み
# データを読み込みます。IrisデータセットのURLから直接読み込むことができます。
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Class"]
iris = pd.read_csv(url, names=columns)

# 2. データの準備
# 特徴量とターゲットに分割
X = iris.drop("Class", axis=1)
y = iris["Class"]

# ラベルを数値に変換
    # "Iris-setosa", "Iris-versicolor", "Iris-virginica"を 0, 1, 2に変換
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. ロジスティック回帰のモデル
    # 学習する
        # max_iter: gradient descentの繰り返し
log_reg_model = LogisticRegression(max_iter=200)
log_reg_model.fit(X_train, y_train)

# 4. 評価する
log_reg_pred = log_reg_model.predict(X_test)

log_reg_acc = accuracy_score(y_test, log_reg_pred)
print("■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■")
print(f"Logistic Regression Accuracy: {log_reg_acc:.4f}")

