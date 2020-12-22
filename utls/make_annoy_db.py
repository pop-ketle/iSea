import os
import cv2
import numpy as np
import pandas as pd
import configparser
from annoy import AnnoyIndex
from datetime import datetime, timedelta

# import tensorflow as tf
from keras.models import model_from_json
from keras.backend import clear_session


'''Autoencoderの中間表現ベクトル空間に対してannoyのdbを生成する
'''

def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        # yield root # ファイルのみ返す(ディレクトリは返さない)
        for file in files:
            if file=='.DS_Store':
                continue
            yield os.path.join(root, file)

def img_preprocess(path, size):
    # 画像のパスが与えられると処理を行う
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(size, size))
    img = img.astype('float32')/255.
    return img

def daterange(_start, _end):
    for n in range((_end - _start).days+1):
        yield _start + timedelta(n)

###########################

# configparserの宣言とiniファイルの読み込み
config_ini = configparser.ConfigParser()
config_ini.read('./config.ini', encoding='utf-8')

# 環境変数を設定
DATABASE_PATH = config_ini['DEFAULT']['DATABASE_PATH']
DB_PATH = config_ini['DEFAULT']['FISH_DB_PATH']
# annoy周りのパラメータ
METRIC  = 'euclidean'
N_TREES = 10
# 読み込む画像サイズ
SIZE = 512


# 事前に作成しておいた512x512x3のサイズの画像を入力として受け取り、32x32x8まで落とすAutoencoderのEncoder部分を読み込む
model_path = os.path.join(DATABASE_PATH, 'Autoencoder_models', '512x512x3to32x32x8', 'encoder512x512')
encoder = model_from_json(open(model_path+'.json').read())
encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
encoder.load_weights(model_path+'.h5')
# encoder._make_predict_function() # これ入れると"<tensor> is not an element of this graph"が解決する
print('--- Model Loading Finished! ---')

idx, dates, imgs = [], [], []
for year in range(2010, 2018+1):
    start = datetime.strptime('{}-{:0=2}-{:0=2}'.format(year,1,1), '%Y-%m-%d')
    end   = datetime.strptime('{}-{:0=2}-{:0=2}'.format(year,12,31), '%Y-%m-%d')

    for i, date in enumerate(daterange(start, end)):
        date = str(date).replace(' 00:00:00', '')
        _year, _month, _day = date.split('-')
        img_path = os.path.join(DATABASE_PATH, 'satellite_images', _year, '7Wc', f'7Wc_{_year}{_month}{_day}0000.png')

        if os.path.isfile(img_path):
            idx.append(i)
            dates.append(date)
            imgs.append(img_preprocess(img_path, SIZE))
imgs = np.array(imgs)

# 32x32x8の中間表現に変換
encoded = encoder.predict(imgs)
# annoyのdbに入れるために整形
encoded = encoded.reshape(-1, 32*32*8)

# annoyのdbを作る
annoy_db = AnnoyIndex(32*32*8, metric=METRIC)
for i, latent in enumerate(encoded):
    # annoyのdbにデータを入れていく
    annoy_db.add_item(i, latent)

annoy_db.build(n_trees=N_TREES) # annoyのビルド

# annoyのdbの保存
output_path = os.path.join(DATABASE_PATH, 'annoy_db32x32x8')
os.makedirs(output_path, exist_ok=True)
annoy_db.save(os.path.join(output_path, f'{METRIC}_{N_TREES}trees.ann'))

# annoyのインデックスと対応する日付の表をcsvで保存する
df = pd.DataFrame()
df['annoy_idx'] = idx
df['date'] = dates
df.to_csv(os.path.join(output_path, f'{METRIC}_{N_TREES}trees.csv'), index=False)