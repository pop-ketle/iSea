import time
import pickle
import sqlite3
import itertools
import numpy as np
import pandas as pd
import configparser
import streamlit as st
from datetime import datetime as dt
from datetime import timedelta

import plotly.graph_objs as go


def daterange(_start, _end):
    for n in range((_end - _start).days+1):
        yield _start + timedelta(n)

def select_place_method_species(methods_dict, species_dict):
    '''漁港、漁業手法、魚種を選択するサイドバーを作り、選択された漁港、漁業手法、魚種を返す
    Args:
        methods_dict(dict): f'{湾名}'をキー、キーに対する漁業手法が値のdict
        species_dict(dict): f'{湾名}:{漁業手法}'をキー、キーに対する魚種が値のdict
    Returns:
        selected_places(list): 選択された湾名(複数)
        selected_methods(list): 選択された漁業手法(複数)
        selected_species(list): 選択された魚種(複数)
    '''
    # 複数条件を入れて組み合わせで様々な条件について見れるようにしたい

    # 全漁港
    all_places = list(methods_dict.keys())

    # 全漁業手法
    all_methods = []
    for method in methods_dict.values():
        for m in method:
            all_methods.append(m)
    all_methods = list(set(all_methods))

    # 全魚種
    all_species = []
    for species in species_dict.values():
        for s in species:
            all_species.append(s)
    all_species = list(set(all_species))

    selected_places = st.sidebar.multiselect(
        '場所選択', all_places,
    )
    selected_methods = st.sidebar.multiselect(
        '漁業手法選択', all_methods,
    )
    selected_species = st.sidebar.multiselect(
        '漁業手法選択', all_species,
    )

    st.write(f'You selected: {selected_places}, {selected_methods}, {selected_species}')

    return selected_places, selected_methods, selected_species

def make_data_df(place, method, species, start_date, end_date, db_path):
    '''place, method, species, start_date, end_dateに対応するデータをpd.DataFrameの形式で返す
    Args:
        place(str): 選択された湾名
        method(str): 選択された漁業手法
        species(str): 選択された魚種
        start_date(datetime object): 持ってくるデータの期間の開始日
        end_data(datetime object): 持ってくるデータの期間の最終日
        db_path(str): 漁獲データのデータが置いてあるdbへのパス
    Returns:
        data_df: 入力した条件に対応するpd.DataFrame
                 水揚量が全部欠損値、つまり漁に行ったデータがなかった時は空pd.DataFrameを返す)
    '''
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

    # データベースに無い日付を補完するためのpd.DataFrameを作成して補完
    _df = pd.DataFrame(pd.date_range(start_date, end_date, freq='D'), columns=['日付'])
    data_df = pd.merge(_df, data_df, how='outer', on='日付')
    data_df['場所'].fillna(place, inplace=True)
    data_df['漁業種類'].fillna(method, inplace=True)
    data_df['魚種'].fillna(species, inplace=True)

    # 水揚量が全部欠損値、つまり寮に行ったデータがなかった時
    if data_df['水揚量'].isnull().sum()==len(data_df):
        return pd.DataFrame()
    else:
        # '水揚量'の欠損値を-1で埋める
        data_df['水揚量'].fillna(-1, inplace=True)

        # 型をキャストする(特に、水揚量に'7,216'のようにコンマが入っててうざいのでそれをなんとかする)
        data_df['水揚量'] = data_df['水揚量'].astype(str).map(lambda x: x.replace(',', ''))
        data_df['水揚量'] = data_df['水揚量'].astype(float)
        data_df['高値'] = data_df['高値'].replace('', np.nan) # '2,000'のようなコンマ入り数字とは別に空の''文字列があったので、それをnp.nanに置換
        data_df['高値'] = data_df['高値'].astype(str).map(lambda x: x.replace(',', ''))
        data_df['高値'] = data_df['高値'].astype(float)
        return data_df

def make_plotly_graph(df):
    '''pd.DataFrameから時系列の漁獲量折れ線グラフを生成
    '''
    start_date, end_date = df['日付'].min(), df['日付'].max()
    place, method, species = df['場所'][0], df['漁業種類'][0], df['魚種'][0]

    # 漁獲量の折れ線グラフ
    return go.Scatter(
        x=list(daterange(start_date, end_date)),
        y=df['水揚量'],
        mode='lines+markers',
        name=f'{place}-{method}-{species}',
        opacity=0.6,
        line=dict(width=2.0),
        marker=dict(size=6),
    )

@st.cache # threshold変わらず、サンプリング数が変わることがあると思うのでキャッシュ化に意味がありそうなので採用！
def divide_pn_df(df, target_col, threshold):
    '''pd.cutでthresholdを境にpositive, negativeを分割して返す
    '''
    df['positive_negative'] = pd.cut(df['水揚量'], bins=[-1, threshold, df['水揚量'].max()], labels=['negative','positive'])
    positive_df = df[df['positive_negative']=='positive']
    negative_df = df[df['positive_negative']=='negative']
    return positive_df, negative_df

@st.cache # サンプリング数が変わらず、他条件が変わることがあると思うのでキャッシュ化 NOTE: あんまりキャッシュ化の効果ないかも
def sampling_data(df, n_samples):
    df = df.sort_values('日付', ascending=False).reset_index() # 新しいデータから順にサンプリングしてきたいので降順にする
    # サンプリングするデータのインデックス
    sampling_idx = [(len(df)//n_samples) * i for i in range(n_samples)]
    # サンプリングされてきたデータ
    return df.iloc[sampling_idx]


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

    # 文字列で表示するデータの開始と終了を定義 NOTE: 固定値になる気がするので大文字変数にした方がいいかも
    start_date_str, end_date_str = '2012-01-01', '2018-12-31'
    # 文字列をdatetime objectに変換
    start_date, end_date = dt.strptime(start_date_str, '%Y-%m-%d'), dt.strptime(end_date_str, '%Y-%m-%d')

    np.set_printoptions(suppress=True) # 指数表記にしない
    st.title('iSea: 海況と漁獲データの結びつけによる関連性の可視化')
    st.write(f'{start_date_str}~{end_date_str} までの期間のデータを対象とします')
    print('============== Initialized ==============')


    # サイドバーで漁港、漁業手法、魚種を選択(複数選択可)
    place, method, species = select_place_method_species(methods_dict, species_dict)

    # 選択した複数条件の全ての組み合わせを取得
    p = itertools.product(place, method, species)
    comb = [v for v in p]
    assert len(comb)==len(place)*len(method)*len(species), '条件組み合わせ生成エラー'

    # 選択した複数条件の全ての組み合わせに対応するpd.DataFrameのリストを得る NOTE: 内包表記で書いた方が早いんだろうけど、長くなってみにくそう
    data_dfs = []
    for p, m, s in comb: # 場所、漁法、魚種
        data_dfs.append(make_data_df(p, m, s, start_date, end_date, DB_PATH))

    # tracesにグラフのデータを詰めていく
    traces = [make_plotly_graph(_df) for _df in data_dfs if len(_df)!=0]

    # 漁獲量の最大値を取得する
    if len(data_dfs)==0:
        max_catch = 1
    else:
        max_catch = int(max([_df['水揚量'].max() for _df in data_dfs if len(_df)!=0]))

    # 0~漁獲量の最大値までの間で、ポジネガを分ける閾値を決める
    # threshold = st.sidebar.slider('漁獲量の閾値設定',  min_value=0, max_value=max_catch, step=1, value=0) # スライドバーは、ユーザビリティにかけるのでなし
    threshold = st.sidebar.number_input('閾値選択', min_value=0, max_value=max_catch, value=int(max_catch//2), step=10000)
    x_range = list(daterange(start_date, end_date))

    traces.append(go.Scatter(x=x_range, y=[threshold]*len(x_range), name='閾値', marker_color='rgba(124,117,117,1.0)'))
    st.write(f'閾値: {threshold}')

    n_samples = st.sidebar.number_input('サンプリング数', min_value=1, value=5, step=1)

    # ポジティブデータとネガティブデータに分割
    for df in data_dfs:
        if len(df)==0: continue # 念の為
        print(df)

        # 漁に行かなかったデータ以外を持ってくる(つまり、漁に行かなかったデータを除外する)
        droped_df = df[df['水揚量']!=-1]

        # 水揚量で閾値を境に、ポジティブネガティブに2分割する
        positive_df, negative_df = divide_pn_df(droped_df, '水揚量', threshold)

        # 日付でソートして、n_samples分区切って、区切った点をサンプリングしてくる
        p_sampling_df = sampling_data(positive_df, n_samples)
        n_sampling_df = sampling_data(negative_df, n_samples)
        
        # サンプリングしたデータをPlotlyの時系列グラフにプロット
        _place, _method, _species = p_sampling_df['場所'].values[0], p_sampling_df['漁業種類'].values[0], p_sampling_df['魚種'].values[0]
        traces.append(go.Scatter(x=p_sampling_df['日付'], y=p_sampling_df['水揚量'], mode='markers', name=f'{_place}-{_method}-{_species}-ポジティブ', marker_color='rgba(255,0,0,.8)', marker_size=15))
        traces.append(go.Scatter(x=n_sampling_df['日付'], y=n_sampling_df['水揚量'], mode='markers', name=f'{_place}-{_method}-{_species}-ネガティブ', marker_color='rgba(0,0,255,.8)', marker_size=15))



    ##########################

    # NOTE: Plotlyのグラフ生成は出来るだけ後ろに回した方が嬉しそう
    # 漁獲量の時系列グラフをPlotlyで表示
    if len(traces)!=0:
        # Plotly、漁獲量時系列グラフのレイアウトの指定
        layout = go.Layout(xaxis=dict(title='日付', type='date', dtick='M6', tickformat='%Y-%m-%d'),
                        yaxis=dict(title='漁獲量(kg)'),
                        xaxis_rangeslider_visible=True,
                        width=900, height=750,
                        clickmode='select+event',)

        fig = dict(data=traces, layout=layout)
        st.plotly_chart(fig)
        # st.plotly_chart(fig, use_container_width=True) # Trueだとカラム幅にサイズが自動調整されるんだけど、それだとちょっと小さい





if __name__ == "__main__":
    main()