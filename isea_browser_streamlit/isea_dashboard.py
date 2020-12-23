import os
import cv2
import math
import pytz
import base64
import pickle
import sqlite3
import itertools
import numpy as np
import pandas as pd
import configparser
from PIL import Image
from io import BytesIO
from annoy import AnnoyIndex
from datetime import datetime as dt
from datetime import timedelta
from datetime import time

import matplotlib.pyplot as plt
import plotly.graph_objs as go

from keras.models import model_from_json
from keras.backend import clear_session

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


class _SessionState: # from: https://qiita.com/niship2/items/f0c825c6f0d291583b27
    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)

def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session

def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state

##########################################

def img_preprocess(path, size):
    # 画像のパスが与えられると処理を行う
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(size, size))
    img = img.astype('float32')/255.
    return img

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

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

##########################################

def main():
    state = _get_state()
    pages = {
        "データ入力": img_upload_page,
        "類似画像表示": render_nn_page,
        "データ可視化": out_page
    }

    #st.sidebar.title(":floppy_disk: Page states")
    page = st.sidebar.radio("ページ選択", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()



def img_upload_page(state):
    uploaded_file = st.file_uploader('Choose a image file')

    if uploaded_file is not None:
        img = Image.open(uploaded_file)

        # base64エンコード
        buf = BytesIO()
        img.save(buf, format='png')
        # バイナリデータをbase64でエンコードし、それをさらにutf-8でデコードしておく
        img_b64str = base64.b64encode(buf.getvalue()).decode("utf-8")
        # image要素のsrc属性に埋め込めこむために、適切に付帯情報を付与する
        img_b64 = f'data:image/png;base64,{img_b64str}'

        st.markdown(f'# Upload img: {uploaded_file.name}')
        components.html(f'<img src="{img_b64}" width="320" height="400"/>', width=330, height=410)

        state.upload_img      = img
        state.upload_img_b64  = img_b64
        state.upload_filename = uploaded_file.name

def render_nn_page(state):
    """
　　　#受け入れたい変数を ~~=state.~~ で受付る。
    dataframe1 = state.dataframe1
    var1 = state.var1
    ~~~
    またはそのままstate.dataframe1
    state.var1
　　で使う。
    """
    upload_img      = state.upload_img
    upload_img_b64  = state.upload_img_b64
    upload_filename = state.upload_filename

    st.markdown(f'# upload img: {upload_filename}')
    components.html(f'<img src="{upload_img_b64}" width="320" height="400"/>', width=330, height=410)

    # TODO: ここキャッシュ化できそう
    # 事前に作成しておいた512x512x3のサイズの画像を入力として受け取り、32x32x8まで落とすAutoencoderのEncoder部分を読み込む
    model_path = os.path.join(DATABASE_PATH, 'Autoencoder_models', '512x512x3to32x32x8', 'encoder512x512')
    encoder = model_from_json(open(model_path+'.json').read())
    encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    encoder.load_weights(model_path+'.h5')
    # encoder._make_predict_function() # これ入れると"<tensor> is not an element of this graph"が解決する
    print('--- Model Loading Finished! ---')

    img = pil2cv(upload_img)
    img = cv2.resize(img,(512, 512))
    img = img.astype('float32')/255.

    # annoyのdbに合わせて整形
    encoded = encoder.predict(np.array([img]))
    encoded = encoded.reshape(32*32*8)

    # annoyのdbを作ってインデックスをロードしてくる
    annoy_db = AnnoyIndex(32*32*8, metric='euclidean')
    annoy_db.load(os.path.join(DATABASE_PATH, 'annoy_db32x32x8', 'euclidean_10trees.ann'))
    annoy_df = pd.read_csv(os.path.join(DATABASE_PATH, 'annoy_db32x32x8', 'euclidean_10trees.csv'))

    idx, dis = annoy_db.get_nns_by_vector(encoded, 21, search_k=-1, include_distances=True)

    nn_df = annoy_df.iloc[idx]
    nn_df['dis'] = dis
    nn_dict = get_img_dict(nn_df['date'], DATABASE_PATH)

    st.markdown('---')
    st.markdown('# 近傍画像データ表示部')

    st.write(nn_df)
    st.markdown('---')

    # 近傍画像表示部
    # 列数n_colのテーブルのようにデータを並べる
    n_col = 6
    for i, (k, v) in enumerate(nn_dict.items()):
        if i%n_col==0: # n_col列ごとに改行(新しくカラムを生成する)を行う
            col_obj_ls = st.beta_columns(n_col)
        i = i%n_col # 改行に合わせてカラムのナンバーの調整

        with col_obj_ls[i]:
            st.write(k)
            components.html(f'<img src="{v}" width="110" height="150"/>', width=120, height=170)
            st.markdown('---')

    state.nn_dates    = nn_df['date']
    state.nn_distance = nn_df['dis']

def out_page(state):
    nn_dates    = state.nn_dates
    nn_distance = state.nn_distance

    # とりあえず近傍の日付と類似度(距離の情報を表示)
    _df = pd.concat([nn_dates, nn_distance], axis=1)
    # _df = _df.reset_index().drop('index', axis=1)
    st.write(_df.reset_index().drop('index', axis=1))

    # dbとのコネクトを確立
    con = sqlite3.connect(DB_PATH)
    c   = con.cursor()

    datas = []
    for date in nn_dates:
        sql = f'''
            SELECT * from data
            WHERE 日付=="{date}"
        '''
        c.execute(sql)
        data = [list(s) for s in c]
        datas += data

    # dbのカラム名を列名にして、pd.DataFrame化
    data_df = pd.DataFrame(datas, columns=[s[0] for s in con.execute('select * from data').description])

    # 型をキャストする(特に、水揚量に'7,216'のようにコンマが入っててうざいのでそれをなんとかする)
    data_df['日付']  = data_df['日付'].astype('datetime64[ns]')
    data_df['水揚量'] = data_df['水揚量'].astype(str).map(lambda x: x.replace(',', ''))
    data_df['水揚量'] = data_df['水揚量'].astype(float)
    data_df['高値']  = data_df['高値'].replace('', np.nan) # '2,000'のようなコンマ入り数字とは別に空の''文字列があったので、それをnp.nanに置換
    data_df['高値']  = data_df['高値'].astype(str).map(lambda x: x.replace(',', ''))
    data_df['高値']  = data_df['高値'].astype(float)

    _df = pd.DataFrame(data_df.groupby(['日付','漁業種類'])['水揚量'].agg(['sum']).reset_index())
    _df = _df.rename(columns={'sum': '日付:漁業種類:SUM'})
    data_df = pd.merge(data_df, _df, on=['日付','漁業種類'], how='left')
    print(_df)
    data_df['単価水揚量'] = data_df['水揚量'] / data_df['隻数']

    st.line_chart(_df)

    st.write(data_df)

    selected_places = st.sidebar.multiselect(
        '場所選択', list(set(data_df['場所'])),
    )
    selected_species = st.sidebar.multiselect(
        '漁業手法選択', list(set(data_df['魚種'])),
    )

    st.write(selected_places, selected_species)

    # 選択した複数条件の全ての組み合わせを取得
    p = itertools.product(selected_places, selected_species)
    comb = [v for v in p]
    st.write(comb)
    print(comb)
    traces = []
    for c in comb:
        place, species = c[0], c[1]
        print(place, species)

        _df = data_df[(data_df['場所']==place) &  (data_df['魚種']==species)]
        st.write(_df)

        fig, ax = plt.subplots()
        ax.hist(_df['水揚量'], bins=1000)

        st.pyplot(fig)

    #     traces.append(go.Histogram(x=_df['水揚量'], xbins=dict(start=0, end=_df['水揚量'].max(), size=10000)))
    
    # # レイアウトの指定
    # layout = go.Layout(
    #     xaxis = dict(title="value", dtick=10),     # dtick でラベルの表示間隔
    #     yaxis = dict(title="count"),
    #     bargap = 0.1) # 棒の間隔

    # fig = dict(data=traces, layout=layout)
    # st.plotly_chart(fig)
    # # offline.iplot(fig)



    # data = go.Histogram(x=X, 
    #                 xbins=dict(start=0, end=101, size=10)) 

if __name__ == "__main__":
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

    np.set_printoptions(suppress=True) # 指数表記にしない
    st.set_page_config(page_title='iSea: 海況と漁獲データの結びつけによる関連性の可視化', page_icon=None, layout='wide', initial_sidebar_state='auto')
    st.title('iSea: 海況と漁獲データの結びつけによる関連性の可視化')
    st.write(f'{min_date_str}~{max_date_str} までの期間のデータを探索対象とします。')
    print('============== Initialized ==============')

    main()

