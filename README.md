# タイトル
iSea: 海況と漁獲データの結びつけによる関連性の可視化 https://github.com/pop-ketle/iSea

# ファイルの説明
## config.ini
設定ファイル: とりあえずよく使いそうなパスを書いておいた。

# 動かし方
## 仮想環境構築
- git clone
```zsh
$ git clone https://github.com/pop-ketle/iSea.git
```

- 作業ディレクトリの移動
```zsh
$ cd iSea
```

- 仮想環境作成
```zsh
$ python3 -m venv isea_venv
```

- 仮想環境に入る
```zsh
$ . isea_venv/bin/activate
```

## 必要ライブラリをインストール
```zsh
$ pip install -r requirements.txt
```

## datasetsをダウンロード
datasetsのファイルをiSea直下においてください。  
Github経由の人は、datasetsのファイルはGithubに上げてないので、欲しい人は連絡してください。連絡もらったら考えます。


---
# Flask基盤のアプリケーション
## iSea(ブラウザ)を起動
- フォルダを移動
```zsh
$ cd isea_browser
```

- アプリをラン
```zsh
$ python isea.py
```

- ローカルホストにアクセス
基本、http://localhost:5000/ に立つと思います。

# Streamlit基盤のアプリケーション
- フォルダを移動
```zsh
$ cd isea_browser_streamlit
```

- アプリをラン
```zsh
$ streamlit run isea_dashboard.py
```

- ローカルホストにアクセス
基本、http://localhost:8501 に立つと思います。


# 備考
- スクレイピングするスクリプト
scraperフォルダはまだ調整中なので触らないでください。漁獲情報のスクレイピング自体は多分できます。

