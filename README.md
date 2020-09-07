# タイトル
iSea: 海況と漁獲データの結びつけによる関連性の可視化

# 動かし方
## 仮想環境構築
- git clone
```zsh
$ git clone https://github.com/pop-ketle/iSea.git
```

- 仮想環境作成
```zsh
$ python -m venv isea
```

- 仮想環境に入る
```zsh
$ . bin/activate
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

<!-- ## 備考
### スクレイピングするスクリプトはまだ調整中なので触らないでください。 -->
<!-- git 管理におかなければいいか -->

