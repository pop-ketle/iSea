# タイトル
iSea: 海況と漁獲データの結びつけによる関連性の可視化

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
datasetsのファイルは、上げてないので、欲しい人は連絡してください。連絡もらったら考えます。

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

## 備考
- スクレイピングするスクリプト
scraperフォルダはまだ調整中なので触らないでください。漁獲情報のスクレイピング自体は多分できます。

