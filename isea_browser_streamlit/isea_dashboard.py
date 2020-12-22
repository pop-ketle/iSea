import os
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
        img = f'data:image/png;base64,{img_b64str}'

        st.markdown('# upload img')
        components.html(f'<img src="{img}" width="320" height="400"/>', width=330, height=410)
        state.uploaded_img = img

    # #受け渡したい変数をstate.~~で入れて、
    # state.dataframe1 = dataframe1
    # state.var1 = var1

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
    uploaded_img = state.uploaded_img

    st.markdown('# upload img')
    components.html(f'<img src="{uploaded_img}" width="320" height="400"/>', width=330, height=410)



def out_page(state):
    pass

if __name__ == "__main__":
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
    min_date_str, max_date_str = '2012-01-01', '2018-12-31'
    # 文字列をdatetime objectに変換
    min_date, max_date = dt.strptime(min_date_str, '%Y-%m-%d'), dt.strptime(max_date_str, '%Y-%m-%d')

    np.set_printoptions(suppress=True) # 指数表記にしない
    st.set_page_config(page_title='iSea: 海況と漁獲データの結びつけによる関連性の可視化', page_icon=None, layout='wide', initial_sidebar_state='auto')
    st.title('iSea: 海況と漁獲データの結びつけによる関連性の可視化')
    st.write(f'{min_date_str}~{max_date_str} までの期間のデータを探索対象とします。')
    print('============== Initialized ==============')

    main()

