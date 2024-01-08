import os
import math
import pytz
import pickle
import sqlite3
import itertools
import numpy as np
import pandas as pd
import configparser
from PIL import Image
from datetime import datetime as dt
from datetime import timedelta
from datetime import time

import plotly.graph_objs as go

import streamlit as st
import streamlit.components.v1 as components
from streamlit.hashing import _CodeHasher
try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server


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

def make_data_df(place, method, species, min_date, max_date, db_path):
    '''place, method, species, min_date, max_dateに対応するデータをpd.DataFrameの形式で返す
    Args:
        place(str): 選択された湾名
        method(str): 選択された漁業手法
        species(str): 選択された魚種
        min_date(datetime object): 持ってくるデータの期間の開始日
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
            and 日付 BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY 日付
    '''
    c.execute(sql)
    data = [list(s) for s in c]

    # dbのカラム名を列名にして、pd.DataFrame化
    data_df = pd.DataFrame(data, columns=[s[0] for s in con.execute('select * from data').description])
    data_df['日付'] = data_df['日付'].astype('datetime64[ns]')

    # データベースに無い日付を補完するためのpd.DataFrameを作成して補完
    _df = pd.DataFrame(pd.date_range(min_date, max_date, freq='D'), columns=['日付'])
    data_df = pd.merge(_df, data_df, how='outer', on='日付')
    data_df['場所'].fillna(place, inplace=True)
    data_df['漁業種類'].fillna(method, inplace=True)
    data_df['魚種'].fillna(species, inplace=True)

    # 水揚量が全部欠損値、つまり漁に行ったデータがなかった時
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
    min_date, max_date = df['日付'].min(), df['日付'].max()
    place, method, species = df['場所'][0], df['漁業種類'][0], df['魚種'][0]

    # 漁獲量の折れ線グラフ
    return go.Scatter(
        x=list(daterange(min_date, max_date)),
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
    if df['水揚量'].max()<threshold: # dfの最大値が閾値以下の時はpositiveに空pd.DataFrameを返して、negativeにdfを返す
        df['positive_negative'] = 'negative'
        return pd.DataFrame(), df
    else:
        df['positive_negative'] = pd.cut(df['水揚量'], bins=[-1, threshold, df['水揚量'].max()], labels=['negative','positive'])
        positive_df = df[df['positive_negative']=='positive']
        negative_df = df[df['positive_negative']=='negative']
        return positive_df, negative_df

@st.cache # サンプリング数が変わらず、他条件が変わることがあると思うのでキャッシュ化 NOTE: あんまりキャッシュ化の効果ないかも
def sampling_data(df, n_samples):
    df = df.sort_values('日付', ascending=False).reset_index() # 新しいデータから順にサンプリングしてきたいので降順にする
    # サンプリングするデータのインデックス
    sampling_idx = sorted(list(set([(len(df)//n_samples) * i for i in range(n_samples)])))
    # サンプリングされてきたデータ
    return df.iloc[sampling_idx]

# @st.cache
def get_img_dict(dates, database_path):
    '''日付のpd.Series(dates)を受け取り、それと対応するbaseエンコードされた画像を値、日付をキーとするdictを返す
    '''
    def get_img_from_db(date, database_path):
        '''日付を与えると対応するbaseエンコードされた画像がdbから引っ張られてくる
        '''
        con = sqlite3.connect(database_path+'encoded_img.db')
        c   = con.cursor()
        c.execute(f'SELECT encoded_img from encoded_imgs WHERE date=="{date}"')
        img_b64data = [list(s)[0] for s in c]
        # 画像のデータがなかったときは何もしない
        if len(img_b64data)!=0:
            con.close()
            return img_b64data[0]

    b64_img_dict = {date: get_img_from_db(date, database_path) for date in set(dates.astype(str).values)}
    return b64_img_dict

def render_sampling_img(img_dict, df, title, n_col):
    '''baseエンコードされた画像を値、日付をキーとするdictを受け取って画像をテーブルで表示するhtml stringを書いてレンダリングする
    Args:
        img_dict(dict): baseエンコードされた画像を値、日付をキーとするdict
        df(pd.DataFrame): 表示したいデータのpd.DataFrame
        title(str): 表示するテーブルのタイトル
        n_col(int): 表示するテーブルのカラム数(ここで改行する)
    '''
    html_text = f"""
        <table border="1">
        <thead>
        <tr valign="top">
            <th colspan={len(img_dict)*2}>{title}</th>
        </tr>
        </thead>
    """
    html_text += '<tbody> <form action="/" method="POST" enctype="multipart/form-data" target="_blank" name="select_image">'
    for i, (k, v) in enumerate(img_dict.items()):
        # 日付と画像の紐づいたボタンを作るhtml stringを書く
        if i!=0 and i%n_col==0: # テーブルを改行する
            html_text += '<tr>'
        html_text += '<td>'
        html_text += f'<input type="hidden" name="date" value="{k}">'
        html_text += f'<input type="image" alt="not found image of {k}" src="{v}" width="110" height="150" style="margin-top: 10px; vertical-align: bottom;">'
        html_text += '</td>'

        html_text += f"""
            <td>
            日付: <br>{k}<br>
            漁獲量:<br>{df[df['日付']==k]['水揚量'].values[0]}(kg)<br>
            </td>
        """

        if i!=(n_col-1) and i%n_col==(n_col-1):
            html_text +='</tr>'
    html_text += '</form></tbody></table>'
    components.html(html_text, width=1200, height=220*math.ceil(len(img_dict)/n_col), scrolling=True)

# @st.cache(suppress_st_warning=True)
def render_nn_img(df, title, n_col, DATABASE_PATH):
    '''与えられたdfと対応する画像を表示する
    Args:
        df(pd.DataFrame): 近傍データのpd.DataFrame
        title(str): 表題のタイトル(テーブルみたいに並べるので、そこの名前にする)
        n_col(int): テーブルの横の列数
        DATABASE_PATH(str): データ保管所のパス(基本 '../datasets/')
    Returns:
        selected_date(None or str): Noneか、ボタンが押されたらその日の日付
        b64_img_dict(dict): 日付がキー、その日のbase64エンコードされた画像が値のdict
    '''
    st.markdown('---')
    st.write(title)

    b64_img_dict = get_img_dict(df['日付'], DATABASE_PATH)

    # 列数n_colのテーブルのようにデータを並べる
    col_no, selected_date = 0, None
    for i, (k, v) in enumerate(b64_img_dict.items()):
        if col_no%n_col==0: # n_col列ごとに改行(新しくカラムを生成する)を行う
            col_obj_ls = st.beta_columns(n_col)
        col_no = col_no%n_col # 改行に合わせてカラムのナンバーの調整

        with col_obj_ls[col_no]:
            html_text = f'<img src="{v}" width="110" height="150"/>'
            components.html(html_text, width=120, height=170)

            button_text = f'''
                日付: {k} \n
                漁獲量: {int(df[df["日付"]==k]["水揚量"].values[0])}(kg)
            '''
            st.write(button_text)

            if st.button(f'選択: {i}', key=f'{title}{i}'):
                selected_date = k

        col_no+=1
    return selected_date, b64_img_dict


def main():
    # configparserの宣言とiniファイルの読み込み
    config_ini = configparser.ConfigParser()
    config_ini.read('./config.ini', encoding='utf-8')

    # 環境変数を設定
    DATABASE_PATH = config_ini['DEFAULT']['DATABASE_PATH']
    DB_PATH = config_ini['DEFAULT']['FISH_DB_PATH']

    # 漁港・漁業手法・魚種・画像の辞書形式のデータを読み込む
    with open(DATABASE_PATH+'method_dict.pkl', 'rb') as f: methods_dict = pickle.load(f)
    with open(DATABASE_PATH+'species_dict.pkl', 'rb') as f: species_dict = pickle.load(f)
    with open(DATABASE_PATH+'group_dict.pkl', 'rb') as f: img_group_dict = pickle.load(f)

    # 文字列で表示するデータの開始と終了を定義 NOTE: 固定値になる気がするので大文字変数にした方がいいかも
    min_date_str, max_date_str = '2012-01-01', '2018-12-31'
    # 文字列をdatetime objectに変換
    min_date, max_date = dt.strptime(min_date_str, '%Y-%m-%d'), dt.strptime(max_date_str, '%Y-%m-%d')

    # NOTE: 日付選択できるようにしようとしたら意外と沼ったので後回し
    # print(min_date, max_date, type(min_date), type(max_date))
    # selected_date = st.sidebar.date_input(
    #     "表示したい期間を入力してください",
    #     [min_date, max_date],
    #     min_value=min_date,
    #     max_value=max_date
    # )
    # min_date, max_date = selected_date[0], selected_date[1]
    # # min_data = dt.combine(min_date, time())
    # # min_date = pytz.timezone('Asia/Tokyo').localize(dt_native)
    # print(min_date, max_date, type(min_date), type(max_date))

    np.set_printoptions(suppress=True) # 指数表記にしない
    st.set_page_config(page_title='iSea: 海況と漁獲データの結びつけによる関連性の可視化', page_icon=None, layout='wide', initial_sidebar_state='auto')
    st.title('iSea: 海況と漁獲データの結びつけによる関連性の可視化')
    st.write(f'{min_date_str}~{max_date_str} までの期間のデータを対象とします。')

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
        data_dfs.append(make_data_df(p, m, s, min_date, max_date, DB_PATH))

    # tracesにグラフのデータを詰めていく
    traces = [make_plotly_graph(_df) for _df in data_dfs if len(_df)!=0]

    # 漁獲量の最大値を取得する
    if len(data_dfs)==0:
        max_catch = 1
    else:
        tmp = [_df['水揚量'].max() for _df in data_dfs if len(_df)!=0]
        if len(tmp)==0: # 全部空のpd.DataFrameだったとき 0
            max_catch = 0
        else:
            max_catch = int(min(tmp)) # 各DFの中でも最小値を決めることで、ビニングするときにthresholdが最大値を超えないことを期待する

    # 0~漁獲量の最大値までの間で、ポジネガを分ける閾値を決める
    # threshold = st.sidebar.slider('漁獲量の閾値設定',  min_value=0, max_value=max_catch, step=1, value=0) # スライドバーは、ユーザビリティにかけるのでなし
    threshold = st.sidebar.number_input('閾値選択', min_value=0, max_value=max_catch, value=int(max_catch//2), step=10000)
    x_range = list(daterange(min_date, max_date))

    traces.append(go.Scatter(x=x_range, y=[threshold]*len(x_range), name='閾値', marker_color='rgba(0,255,0,1.0)'))
    st.write(f'閾値: {threshold}')

    n_samples = st.sidebar.number_input('サンプリング数', min_value=1, value=5, step=1)

    # 漁獲量時系列グラフ表示部
    # ポジティブデータとネガティブデータに分割して、サンプリングしたデータをPlotlyにプロット
    for df in data_dfs:
        if len(df)==0: continue # 念の為

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

    # ---
    st.markdown('---')

    # 画像表示部
    # ポジティブデータとネガティブデータに分割して、サンプリングした画像データを表示
    for df in data_dfs:
        if len(df)==0: continue # 念の為

        # 漁に行かなかったデータ以外を持ってくる(つまり、漁に行かなかったデータを除外する)
        droped_df = df[df['水揚量']!=-1]

        # 水揚量で閾値を境に、ポジティブネガティブに2分割する
        positive_df, negative_df = divide_pn_df(droped_df, '水揚量', threshold)

        # 日付でソートして、n_samples分区切って、区切った点をサンプリングしてくる
        p_sampling_df = sampling_data(positive_df, n_samples)
        n_sampling_df = sampling_data(negative_df, n_samples)
        
        # サンプリングされてきた画像表示部分
        for i, sampling_df in enumerate([p_sampling_df, n_sampling_df]):
            if i==0: # ポジティブデータなら
                title = f'{df["場所"][0]} {df["漁業種類"][0]} {df["魚種"][0]} ポジティブデータ'
            else:
                title = f'{df["場所"][0]} {df["漁業種類"][0]} {df["魚種"][0]} ネガティブデータ'

            st.markdown('---')
            st.write(title)

            # サンプリングされた画像のbaseエンコードされた画像を値、日付をキーとするdictを取得
            b64_img_dict = get_img_dict(sampling_df['日付'], DATABASE_PATH)

            # baseエンコードされた画像を値、日付をキーとするdictから情報を受け取ってhtml stringを作成してレンダリングする
            render_sampling_img(b64_img_dict, sampling_df, title, 5)

    
    # -------テストコード-------
    st.markdown('---')
    st.markdown('---')
    st.markdown('---')


    # 画像表示部
    # ポジティブデータとネガティブデータに分割して、サンプリングした画像データを表示
    for df in data_dfs:
        if len(df)==0: continue # 念の為

        # 漁に行かなかったデータ以外を持ってくる(つまり、漁に行かなかったデータを除外する)
        droped_df = df[df['水揚量']!=-1]

        # 水揚量で閾値を境に、ポジティブネガティブに2分割する
        positive_df, negative_df = divide_pn_df(droped_df, '水揚量', threshold)

        # 日付でソートして、n_samples分区切って、区切った点をサンプリングしてくる
        p_sampling_df = sampling_data(positive_df, n_samples)
        n_sampling_df = sampling_data(negative_df, n_samples)
        
        # サンプリングされてきた画像表示部分 HACK: ここら辺、近傍でも似た処理するのでうまく関数化できそう(ボタンの処理が面倒ではある)
        for i, sampling_df in enumerate([p_sampling_df, n_sampling_df]):
            if i==0: # ポジティブデータなら
                title = f'{df["場所"][0]} {df["漁業種類"][0]} {df["魚種"][0]} ポジティブデータ'
            else:
                title = f'{df["場所"][0]} {df["漁業種類"][0]} {df["魚種"][0]} ネガティブデータ'

            # TODO: 
            # 画像をn_col列のテーブル形式のように表示する
            global selected_date
            selected_date, b64_img_dict = render_nn_img(sampling_df, title, 6, DATABASE_PATH)
            print('0', selected_date)

            # try: # ボタンが押された時だけ(selected_dateが定義されてる時だけ)作動 似ている画像を表示
            # 選択された日付がNoneなら処理を飛ばす
            if selected_date is None: continue

            st.markdown(f'# {selected_date}の画像を選択！')
            
            html_text = f'<img src="{b64_img_dict[selected_date]}" width="300" height="350"/>'
            components.html(html_text, width=310, height=370)

            # 近傍の画像の日付を取得、それと対応するpd.DataFrameを取得する
            nn_dates = img_group_dict[f'{selected_date}:group_date']
            nn_df = df[df['日付'].isin(nn_dates)]

            # 近傍データを漁に行かなかったデータを除外してと、漁に行ったデータを分離する
            removed_nn_df = nn_df[nn_df['水揚量']==-1]
            nn_df = nn_df[nn_df['水揚量']!=-1]

            # 漁に行った近傍データをポジティブ・ネガティブに分離する
            positive_nn_df, negative_nn_df = divide_pn_df(nn_df, '水揚量', threshold)

            for i, sampling_df in enumerate([positive_nn_df, negative_nn_df, removed_nn_df]):
                if i==0: # ポジティブデータなら
                    title = f'{df["場所"][0]} {df["漁業種類"][0]} {df["魚種"][0]} 近傍ポジティブデータ'
                elif i==1: # ネガティブデータなら
                    title = f'{df["場所"][0]} {df["漁業種類"][0]} {df["魚種"][0]} 近傍ネガティブデータ'
                else:
                    title = f'{df["場所"][0]} {df["漁業種類"][0]} {df["魚種"][0]} 近傍漁に行かなかったデータ'

                # 画像をn_col列のテーブル形式のように表示する
                print('1', selected_date)
                nn_selected_date, b64_img_dict = render_nn_img(sampling_df, title, 10, DATABASE_PATH)
                print('2', selected_date, nn_selected_date)

            #         st.markdown('---')
            #         st.write(title)

            #         b64_img_dict = get_img_dict(sampling_df['日付'], DATABASE_PATH)

            #         # n_col列ごとに改行したい
            #         N_COL, col_no = 10, 0
            #         for j, (k, v) in enumerate(b64_img_dict.items()):
            #             if col_no%N_COL==0: # N_COL列ごとに改行を行う
            #                 col_obj_ls = st.beta_columns(N_COL)
            #             col_no = col_no%N_COL # 改行に合わせてカラムのナンバーを合わせる

            #             with col_obj_ls[col_no]:
            #                 html_text = f'<img src="{v}" width="80" height="120"/>'
            #                 components.html(html_text, width=90, height=140)

            #                 st.write(f'日付: {k}')
            #                 st.write(f'漁獲量: {int(sampling_df[sampling_df["日付"]==k]["水揚量"].values[0])}(kg)')

            #             col_no+=1

            #     # del selected_date # 同じ動作を繰り返さないように未定義にする
            #     selected_date = None

            # # except NameError: # ボタンが押されてない時は未定義なので何もしない
            # #     pass



    st.balloons()
    st.error('This is an error')
    st.warning('This is a warning')
    st.info('This is a purely informational message')
    st.success('This is a success message!')


    placeholder = st.empty()
    placeholder.text("Hello")
    placeholder.line_chart({"data": [1, 5, 2, 6]})

    col1, col2, col3, col4, col5, col6, col7 = st.beta_columns(7)

    with col1:
        st.header("A cat")
        st.image("https://static.streamlit.io/examples/cat.jpg", use_column_width=True)

    with col2:
        st.header("A dog")
        st.image("https://static.streamlit.io/examples/dog.jpg", use_column_width=True)

    with col3:
        st.header("An owl")
        st.write('the GREATEST HERO of China')
        st.header("An owl2")
        st.image("https://static.streamlit.io/examples/owl.jpg", use_column_width=True)
        if st.button('Say hello'):
            st.write('Why hello there')
        else:
            st.write('gg')

    with col4:
        st.header("An owl")
        st.write('the GREATEST HERO of China')
        st.header("An owl2")
        st.image("https://static.streamlit.io/examples/owl.jpg", use_column_width=True)
        if st.button('Say hello4'):
            st.write('Why hello there4')
        else:
            st.write('gg')
    with col5:
        st.header("An owl")
        st.write('the GREATEST HERO of China')
        # st.header("An owl2")
        # st.image("https://static.streamlit.io/examples/owl.jpg", use_column_width=True)
        if st.button('Say hello5'):
            st.write('Why hello there5')
        else:
            st.write('gg')
    with col6:
        st.header("An owl")
        st.write('the GREATEST HERO of China')
        st.header("An owl2")
        st.image("https://static.streamlit.io/examples/owl.jpg", use_column_width=True)
        if st.button('Say hello6'):
            st.write('Why hello there6')
            st.image("https://static.streamlit.io/examples/owl.jpg", use_column_width=True)

        else:
            st.write('gg')
    with col6:
        st.header("An owl")
        st.write('the GREATEST HERO of China')
        st.header("An owl2")
        for i in range(5):
            st.image(["https://static.streamlit.io/examples/owl.jpg"], use_column_width=True)
            st.write(f'image{i+1}!')
        if st.button('Say hello7'):
            st.write('Why hello there7')
            st.image("https://static.streamlit.io/examples/owl.jpg", use_column_width=True)

        else:
            st.write('gg')



if __name__ == "__main__":
    main()