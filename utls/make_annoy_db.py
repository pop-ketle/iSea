import os
import configparser
from annoy import AnnoyIndex

# import tensorflow as tf
from keras.models import model_from_json
from keras.backend import clear_session


'''Autoencoderの中間表現ベクトル空間に対してannoyのdbを生成する
'''

# configparserの宣言とiniファイルの読み込み
config_ini = configparser.ConfigParser()
config_ini.read('./config.ini', encoding='utf-8')

# 環境変数を設定
DATABASE_PATH = config_ini['DEFAULT']['DATABASE_PATH']
DB_PATH = config_ini['DEFAULT']['FISH_DB_PATH']

print(DATABASE_PATH)
print(DB_PATH)