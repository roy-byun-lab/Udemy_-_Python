import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Input

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
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. ニューラルネットワークのモデル
nn_model = Sequential()

# layer作成
    # 64,32,3:出力数
    # input_dim=X_train.shape[1]: 入力データの特徴数(4個："Sepal Length", "Sepal Width", "Petal Length", "Petal Width")
    # activation: 活性化関数
# nn_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
nn_model.add(Input(shape=(X_train.shape[1],)))
nn_model.add(Dense(64, activation='relu'))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(3, activation='softmax'))  # 3クラス分類

# 4.学習
    # loss: 損失関数. 多重クラス分類で整数型のラベルには"sparse_categorical_crossentropy"をよく利用する
    # optimizer: 最適化
    # metrics：評価指標
nn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

# 5.評価
nn_loss, nn_acc = nn_model.evaluate(X_test, y_test, verbose=0)
print("■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■")
print(f"Neural Network Accuracy: {nn_acc:.4f}")
print(f"loss: {nn_loss:.4f}")
