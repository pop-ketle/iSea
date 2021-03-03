# [iSea: 海況と漁獲データの結びつけによる関連性の可視化](https://github.com/pop-ketle/iSea)

# 紹介動画
- [iSea: 海況と漁獲データの結びつけによる関連性の可視化](https://youtu.be/nqoquktgO8g)
- [iSea: Streamlit App](https://youtu.be/jHwsYuPuvw4)

# フォルダ構成
```
.
├── README.md # これ
├── config.ini # 設定ファイル: とりあえずよく使いそうなパスを書いた
├── crawler # [いわて大漁ナビ](https://www.suigi.pref.iwate.jp/)からデータをスクレイピングしてくるためのコード類
│   ├── README.md
│   ├── img_crawler.py
│   ├── shikyo_crawler.py
│   ├── shikyo_daily_urls_dict.pkl
│   ├── shikyo_monthly_urls_dict.pkl
│   ├── suion_crawler.py
│   ├── suion_daily_urls_dict.pkl
│   └── suion_monthly_urls_dict.pkl
├── datasets # データ置き場
│   ├── Autoencoder_models
│   ├── data.db
│   ├── db_archives
│   ├── encoded_img.db
│   ├── group_dict.pkl
│   ├── method_dict.pkl
│   ├── satellite_images
│   ├── simirality_dicts
│   └── species_dict.pkl
├── isea_browser # メインの製作物 Flask実装
│   ├── __pycache__
│   ├── isea.py
│   ├── myfunc.py
│   ├── static
│   └── templates
├── isea_browser_streamlit  # streamlitを用いたサブの製作物
│   ├── config.ini
│   ├── isea.py # 練習
│   ├── isea_dashboard.py
│   └── sample.py # サンプル
├── isea_venv # pyenvの仮想環境
│   ├── bin
│   ├── etc
│   ├── include
│   ├── lib
│   ├── pyvenv.cfg
│   └── share
├── requirements.txt # 依存関係ライブラリ
└── utls # 修論書く際に使った小さな実験のファイル
    ├── README.md
    ├── canny_edge.py
    ├── color_hist.py
    ├── config.ini
    └── make_annoy_db.py
```

# 動かし方
# 1. 環境構築
## git clone
```zsh
$ git clone https://github.com/pop-ketle/iSea.git
```

## 作業ディレクトリの移動
```zsh
$ cd iSea
```

## 仮想環境作成
```zsh
$ python3 -m venv isea_venv
```

## 仮想環境に入る
```zsh
$ . isea_venv/bin/activate
```

## 必要ライブラリをインストール
```zsh
$ pip install -r requirements.txt
```

## (datasetsをダウンロード)
datasetsのファイルをiSea直下においてください。  
Github経由の人は、datasetsのファイルはGithubに上げてないので、欲しい人は連絡してください。連絡もらったら考えます。

## 備考
2021/3/3現在、M1 Macだとうまくtensorflowを動かせなかったので、tensorflowを使うisea_dashboard.pyの動作については保証しません。


---
# 2. Flask基盤のアプリケーション
# iSea(ブラウザ)を起動
## フォルダを移動
```zsh
$ cd isea_browser
```

## アプリをラン
```zsh
$ python isea.py
```

## ローカルホストにアクセス
基本、http://localhost:5000/ に立つと思います。

# 3. Streamlit基盤のアプリケーション
## フォルダを移動
```zsh
$ cd isea_browser_streamlit
```

## アプリをラン
```zsh
$ streamlit run isea_dashboard.py
```

## ローカルホストにアクセス
基本、http://localhost:8501 に立つと思います。

---

# 備考
- 連絡先: gshrhg@gmail.com  
ご不明な点あれば、連絡いただければできる限り対応します