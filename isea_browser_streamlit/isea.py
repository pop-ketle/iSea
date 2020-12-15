import pickle
import sqlite3
import numpy as np
import pandas as pd
import configparser
import streamlit as st
from datetime import datetime as dt

import plotly.graph_objs as go


def select_place_method_species(methods_dict, species_dict):
    '''漁港、漁業手法、魚種を選択するサイドバーを作り、選択された漁港、漁業手法、魚種を返す
    Args:
        methods_dict(dict): f'{湾名}'をキー、キーに対する漁業手法が値のdict
        species_dict(dict): f'{湾名}:{漁業手法}'をキー、キーに対する魚種が値のdict
    Returns:
        place(str): 選択された湾名
        method(str): 選択された漁業手法
        species(str): 選択された魚種
    '''

    place = st.sidebar.selectbox(
        '場所選択',
        list(methods_dict.keys()))

    method = st.sidebar.selectbox(
        '漁業手法選択',
        list(methods_dict[place]))

    species= st.sidebar.selectbox(
        '漁業手法選択',
        list(species_dict[f'{place}:{method}']))

    return place, method, species

def make_timeline_catch_graph(place, method, species, db_path):
    '''漁獲量の時系列グラフを表示
    Args:
        place(str): 選択された湾名
        method(str): 選択された漁業手法
        species(str): 選択された魚種
        db_path(str): 漁獲データのデータが置いてあるdbへのパス
    '''
    # 文字列で時系列グラフの開始と終了を定義
    start_date, end_date = '2012-01-01', '2018-12-31'
    # 文字列をdatetime objectに変換
    start_date, end_date = dt.strptime(start_date, '%Y-%m-%d'), dt.strptime(end_date, '%Y-%m-%d')

    # dbとのコネクトを確立
    con = sqlite3.connect(db_path)
    c   = con.cursor()

    sql = f'''
        SELECT * from data
        WHERE 場所=='{place}'
            and 漁業種類=='{method}'
            and 魚種=='{species}'
            and 日付 BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY 日付
    '''
    c.execute(sql)
    data = [list(s) for s in c]

    # dbのカラム名を列名にして、pd.DataFrame化
    data_df = pd.DataFrame(data, columns=[s[0] for s in con.execute('select * from data').description])
    data_df['日付'] = data_df['日付'].astype('datetime64[ns]')
    print(data_df)

    print(place)

    # データベースに無い日付を補完するためのpd.DataFrameを作成して補完
    _df = pd.DataFrame(pd.date_range(start_date, end_date, freq='D'), columns=['日付'])
    data_df = pd.merge(_df, data_df, how='outer', on='日付')
    data_df['場所'].fillna(place, inplace=True)
    data_df['漁業種類'].fillna(method, inplace=True)
    data_df['魚種'].fillna(method, inplace=True)

    print(data_df)






def main():
    # configparserの宣言とiniファイルの読み込み
    config_ini = configparser.ConfigParser()
    config_ini.read('./config.ini', encoding='utf-8')

    # 環境変数を設定
    DATABASE_PATH = config_ini['DEFAULT']['DATABASE_PATH']
    DB_PATH = config_ini['DEFAULT']['FISH_DB_PATH']

    # 漁港・漁業手法・魚種・画像の辞書形式のデータを読み込む
    with open(DATABASE_PATH+'method_dict.pickle', 'rb') as f: methods_dict = pickle.load(f)
    with open(DATABASE_PATH+'species_dict.pickle', 'rb') as f: species_dict = pickle.load(f)
    with open(DATABASE_PATH+'group_dict.pkl', 'rb') as f: img_group_dict = pickle.load(f)

    np.set_printoptions(suppress=True) # 指数表記にしない
    print('============== Initialized ==============')

    # print(methods_dict)

    # print(list(methods_dict.keys()))

    # print(species_dict)

    # print(img_group_dict)

    # サイドバーで漁港、漁業手法、魚種を選択
    place, method, species = select_place_method_species(methods_dict, species_dict)

    # 漁獲量の時系列グラフをPlotlyで表示
    make_timeline_catch_graph(place, method, species, DB_PATH)




if __name__ == "__main__":
    main()