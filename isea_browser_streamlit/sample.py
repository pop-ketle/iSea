import streamlit as st
import pandas as pd
import numpy as np

"""
# 初めての Streamlit
データフレームを表として出力できます:
"""

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})

df

"""
# グラフ描画の例
"""

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])

st.line_chart(chart_data)

"""
# 地図を描画
"""

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [35.68109, 139.76719],
    columns=['lat', 'lon'])

st.map(map_data)

"""
# ウィジェットの例
"""

if st.checkbox("チェックボックス"):
    st.write("チェックが入りました。")

selection = st.selectbox("セレクトボックス", ["1", "2", "3"])
st.write(f"{selection} を選択")

"""
## プログレスバーとボタン
"""

import time

if st.button("ダウンロード"):
    text = st.empty()
    bar = st.progress(0)

    for i in range(100):
        text.text(f"ダウンロード中 {i + 1}/100")
        bar.progress(i + 1)
        time.sleep(0.01)