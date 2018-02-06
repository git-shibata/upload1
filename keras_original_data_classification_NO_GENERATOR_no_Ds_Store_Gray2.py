# Keras ディレクトリーからファイルを読み込んで判定する

import keras as K
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# パラメーター
nb_classes = 4               # 判定するラベルの数
nb_pictures = 480            # 画像の枚数
img_rows, img_cols = 28, 28  # 画像サイズ
batch_size = 4               #　画像の枚数を割り切れる数がよいらしい
dropout = 0.2                # 過学習を避けるために0を織り込む確率
nb_neuron = 32               # ニューロンの数
channel = 1                  # RGB:3 Grayscale:1
epochs = 8                   # 世代数
input_shape = channel, img_rows, img_cols
pic_folder = 'guitar_pic_28x28'

# 最適化アルゴリズムの選択　SGD, Adadelta, Adamax, Adam, Adagrad, RMSprop, Nadam
opt_arg = Nadam()
# モデル評価んのロス関数 hinge, binary_crossentropy, categorical_crossentropy
loss_arg = 'binary_crossentropy'

# 学習用のデータ配列を作る.
image_list = []
label_list = []

# 判定ラベルをフォルダー名で規定する
label1 = 'LPG'
label2 = 'PRS'
label3 = 'STR'
label4 = 'TEL'

dir1 = "others" # dir1の初期値設定

# 画像を読み込む
for dir in os.listdir(pic_folder + "/train"):
    dir != dir1
    dir1 = pic_folder + "/train/" + dir 
    label = 0

    if dir == label1:     # ギブソンレスポール
        label = 0
    elif dir == label2:   # ポールリードスミス
        label = 1
    elif dir == label3:   # フェンダーストラトキャスター
        label = 2
    elif dir == label4:   # フェンダーテレキャスター
        label = 3

    for file in os.listdir(dir1):
        base, ext = os.path.splitext(file)    # os.path.splitext関数は、pathをbase（拡張子以外の部分）とext（ピリオドを含む拡張子）に分割して、タプルで返す
        if ext == '.jpg':
            # 配列label_listに正解ラベルを追加(ラベル名１:0 ラベル名２:1 ラベル名３:2 ラベル名４:3)
            label_list.append(label)
            filepath = dir1 + "/" + file
            # 画像をグレースケールにしてimg_cols x imag_rowspixelに変換
            image = np.array(Image.open(filepath).convert("L").resize((img_rows, img_cols)))
            # 配列をリシェイプ 
            image = image.reshape(img_rows, img_cols, channel)
            # 出来上がった配列をimage_listに追加。
            image_list.append(image / 255.)

# kerasに渡すためにnumpy配列に変換
image_list = np.array(image_list)

# ラベルの配列を1と0からなるラベル配列に変更
# 数値のone hot配列への変換(0 -> [1,0,0], 1 -> [0,1,0] という感じ）
Y = to_categorical(label_list)

# モデルを生成してニューラルネットを構築
model = Sequential()
model.add(Conv2D(nb_neuron, (3, 3), padding='same', input_shape=(img_rows, img_cols, channel)))
model.add(Activation("relu"))
model.add(Conv2D(nb_neuron, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(dropout))

model.add(Flatten())

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(dropout))

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(dropout))

model.add(Dense(nb_classes)) # この値がラベルの数（判定カテゴリー数） ／ここより上の隠れ層は交換可能
model.add(Activation("softmax"))

model.summary()              # モデル構造の簡易表示

# モデルをコンパイル
model.compile(loss=loss_arg, optimizer=opt_arg, metrics=["accuracy"])
# 学習を実行。10%は評価に使用。
predict = model.fit(image_list, Y, epochs=epochs, batch_size=batch_size, validation_split=0.1)
print(predict.history)
# 正答率
plt.plot(predict.history['acc'])
plt.plot(predict.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# plt.xlim([0, epochs])
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# ロス率
plt.plot(predict.history['loss'])
plt.plot(predict.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xlim([0, epochs])
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# テスト用ディレクトリの画像判定し、正解率を表示する。
total = 0.
ok_count = 0.

dir1 = "others" # dir1の初期値設定

for dir in os.listdir(pic_folder + "/validation"):
    dir != dir1
    dir1 = pic_folder + "/validation/" + dir 
    label = 0

    if dir == label1:
        label = 0
    elif dir == label2:
        label = 1
    elif dir == label3:
        label = 2
    elif dir == label4:
        label = 3

    for file in os.listdir(dir1):
        base, ext = os.path.splitext(file)    # os.path.splitext関数は、pathをbase（拡張子以外の部分）とext（ピリオドを含む拡張子）に分割して、タプルで返す
        if ext == '.jpg':
            label_list.append(label)
            filepath = dir1 + "/" + file
            # 画像をグレースケールにしてimg_cols x imag_rowspixelに変換
            image = np.array(Image.open(filepath).convert("L").resize((img_rows, img_cols)))
            # 配列をリシェイプ 
            image = image.reshape(img_rows, img_cols, channel)
			# 正規化したimageで予測
            result = model.predict_classes(np.array([image / 255.]))
            print("label:", label, "result:", result[0])
            # ファイルを開いて、表示
            # im = Image.open(filepath)
            # plt.imshow(im)
            if label == 0:
                print('ギブソンレスポール')
            elif label == 1:
                print('ポールリードスミス')
            elif label == 2:
                print('フェンダーストラトキャスター')
            elif label == 3:
                print('フェンダーテレキャスター')
            total += 1. 

            if label == result[0]:
                ok_count += 1.

print("正解率: ", ok_count / total * 100, "%")

### save weights
# json_string = model.to_json()
# open('guitar_no-gen_model.json', 'w').write(json_string)
# model.save_weights('guitar_no-gen_weights.h5')