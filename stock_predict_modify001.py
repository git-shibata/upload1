# Nikkei平均予測

# 前準備
# x-data.csvとt-data.csvを用意する。　メモ帳でUTF-8で保存する
# x-data.csv: ヒストリーデータから自然対数をとり、自然対数の差分をB列に置く
# t-data.csv：　自然対数の騰落をカテゴリー化する。
#　=(前日自然対数－当日自然対数)x100
# エクセルの場合、関数で0,1に割り振る。=IF(AND(0.0<=B3, B3<0.995033085), 1, 0)
# csvで保存すると関数は消えてしまう。

# モジュールやライブラリー（大雑把に言えば機能部品）の組み込み部
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

from __future__ import print_function
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils import plot_model

from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_uniform
from keras.initializers import orthogonal
from keras.initializers import TruncatedNormal

 	
# 学習データ
dataframe1 = csv.reader(open('x-data.csv', 'r', encoding='UTF-8'))
up_down_data1 = [ v for v in dataframe1]
up_down_data2 = np.array(up_down_data1)   # Numpy配列形式に変換
up_down_data3 = up_down_data2[1:]         # 見出し行を外す [1:]１行目（１つめの配列）以外
up_down_data = up_down_data3[:, 1:].astype(np.float)  # 2列目以降を抜き出してfloat変換 [:, 1:]全部の配列の1列目以外
print('騰落データ(up_down_data.shape)の配列形式=', up_down_data.shape)

# ラベルデータ
# 1％以上／0％以上／-1％以上／-1％未満
dataframe2 = csv.reader(open('t-data.csv', 'r', encoding='UTF-8'))
label_data1 = [ v for v in dataframe2]
label_data2 = np.array(label_data1)   # Numpy配列形式に変換
label_data3 = label_data2[1:]         # 見出し行を外す [1:]１行目（１つめの配列）以外
label_data = label_data3[:, 1:].astype(np.float)  # 2列目以降を抜き出してfloat変換 [:, 1:]全部の配列の1列目以外

print('ラベルデータ(label_data.shape)の配列形式=', label_data.shape)

# パラメーター設定
maxlen = 80                           # 入力系列数（80個で切断する？）
n_in = up_down_data.shape[1]          # 騰落データ（＝入力）の列数、[1]は２次元目の配列を指す
n_out = label_data.shape[1]           # ラベルデータ（=出力）の列数、[1]は２次元目の配列を指す
len_seq = up_down_data.shape[0] - maxlen + 1    # 騰落データの行数、[0]は１次元目の配列を指す

# 途中で必要になる配列の準備
data = []                             # 空の配列の作成
target = []                           # 空の配列の作成
for i in range(0, len_seq):           # 騰落データの数分繰り返す
  data.append(up_down_data[i:i+maxlen, :])   # data配列にｉからｉ＋maxlen分の長さの要素を加えていく
  target.append(label_data[i+maxlen-1, :])   # target配列にi+maxlen-1目の行（次の日の騰落）のone hotを加えていく

x = np.array(data).reshape(len(data), maxlen, n_in)    # data配列（騰落データ）を成形してXに格納
t = np.array(target).reshape(len(data), n_out)         # target配列（ラベルデータ）を成形してｔに格納

print('騰落データの配列(data)の形式：', x.shape, '　ラベルデータの配列(target)の形式：', t.shape)

# 騰落とラベルのデータを分割して訓練用データとテスト用データを生成
n_train = int(len(data)*0.9)               # 分割して学習用にする配列の長さの指定
x_train, x_test = np.vsplit(x, [n_train])  # 騰落データを訓練用とテスト用に分割
t_train, t_test = np.vsplit(t, [n_train])  # ラベルデータを訓練用とテスト用に分割

print('分割後の騰落データの配列形式：', x_train.shape, x_test.shape, '　分割後のラベルデータのの配列形式：', t_train.shape, t_test.shape)

# 予測クラス（Prediction）の初期値をｓｅｌｆセットに格納 	
class Prediction :
  def __init__(self, maxlen, n_hidden, n_in, n_out):
    self.maxlen = maxlen
    self.n_hidden = n_hidden
    self.n_in = n_in
    self.n_out = n_out

# モデル生成関数の定義
  def create_model(self):
    model = Sequential()
    model.add(LSTM(self.n_hidden, batch_input_shape = (None, self.maxlen, self.n_in),
             kernel_initializer = glorot_uniform(seed=20170719), 
             recurrent_initializer = orthogonal(gain=1.0, seed=20170719), 
             dropout = 0.5, 
             recurrent_dropout = 0.5))
    model.add(Dropout(0.5))
    model.add(Dense(self.n_out, 
            kernel_initializer = glorot_uniform(seed=20170719)))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer = "RMSprop", metrics = ['categorical_accuracy'])
    return model

# 学習関数の定義
  def train(self, x_train, t_train, batch_size, epochs) :
    early_stopping = EarlyStopping(patience=0, verbose=1)
    model = self.create_model()
    model.fit(x_train, t_train, batch_size = batch_size, epochs = epochs, verbose = 1,
          shuffle = True, callbacks = [early_stopping], validation_split = 0.1)
    return model

# 学習モデルの内部パラメーター
n_hidden = 60     # 出力次元
epochs = 2      # エポック数（何世代計算するか）
batch_size = 100   # ミニバッチサイズ

# 予測クラスの初期値セットの生成
prediction = Prediction(maxlen, n_hidden, n_in, n_out)

# 学習関数の実行
model = prediction.train(x_train, t_train, batch_size, epochs)

# テスト結果の算出
score = model.evaluate(x_test, t_test, batch_size = batch_size, verbose = 1)
print("score:", score)

# 正答率、準正答率の集計
preds = model.predict(x_test)     # x_testの騰落データでの予測結果をpreds変数に格納
correct = 0                       # correct変数を0に
semi_correct = 0                  # semi_correct変数を0に
for i in range(len(preds)):       # preds変数の要素の数分実行する
  pred = np.argmax(preds[i,:])    # predにpreds配列中のｉ番目の要素セット中最大のものが何番目か(0~3)を返す
  targ = np.argmax(t_test[i,:])   # 同じくtarg変数にi番目の要素セット中の最大のもの（1）が何番目か(0~3)を返す
  if pred == targ :               # 予測とラベルが同じなら正解！
    correct += 1                  # 正解の数をカウントアップする
  else :
    if pred+targ == 1 or pred+targ == 5 :    # pred, targがイコールでなく0か１なら、"騰"は合っている、2か3なら"落"は合っている
      semi_correct += 1           # 準正解の数をカウントアップする

print("正答率:", 1.0 * correct / len(preds))
print("準正答率（騰落は当たっていた）:", 1.0 * (correct+semi_correct) / len(preds))

# 明日を予測
x_predict = x[-1, :]                               # 最後の配列だけ取り出す
x_predict = np.delete(x_predict, 0)                # 先頭の要素を削除
x_predict = np.append(x_predict, 0)                # 0を追記
x_predict = np.reshape(x_predict, (1, 80, 1))      # 形式を３次元に
tomorrow = model.predict(x_predict)                # 予測実施
print('明日の予想( 1%以上騰)', np.round(tomorrow[:,0], 3), '%')
print('明日の予想( 0%以上騰)', np.round(tomorrow[:,1], 3), '%')
print('明日の予想(-1%未満落)', np.round(tomorrow[:,2], 3), '%')
print('明日の予想(-1%以上落)', np.round(tomorrow[:,3], 3), '%')

### save weights
# json_string = model.to_json()
# open('nikkei_stock_model.json', 'w').write(json_string)
# model.save_weights('nikkei_stock_weights.h5')