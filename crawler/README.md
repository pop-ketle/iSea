# shikyo_crawler.py
[いわて大漁ナビ市況日報一覧](https://www.suigi.pref.iwate.jp/shikyo/monthly/list)から各リンクに飛んでいき、飛んだ先で漁獲データ(市況データ)をスクレイピングしてきて、datasets/data.dbのsqlite3のデータベース'data'に入れる。

## 中間生成物
既にスクレイピングしたデータがあるURLを保存しておいて差分を見ることで、差分だけ更新できるようにする。

- shikyo_daily_urls_dict.pkl
[いわて大漁ナビ市況日報一覧](https://www.suigi.pref.iwate.jp/shikyo/monthly/list)にある各月へのリンクのURLを値、f'{year}-{month}'をキーとした辞書

- shikyo_monthly_urls_dict.pkl
[いわて大漁ナビ市況日報一覧](https://www.suigi.pref.iwate.jp/shikyo/monthly/list)から飛んだ先の各湾各日ごとのリンクのURLを値、f'{place}-{year}-{month}-{day}'をキーとした辞書

# suion_crawler.py
[いわて大漁ナビ定置水温日報一覧](https://www.suigi.pref.iwate.jp/teichi/monthly/list/daily)から各リンクに飛んでいき、飛んだ先で水温データをスクレイピングしてきて、datasets/data.dbのsqlite3のデータベース'suion_data'に入れる。

## 中間生成物
既にスクレイピングしたデータがあるURLを保存しておいて差分を見ることで、差分だけ更新できるようにする。

- suion_daily_urls_dict.pkl
[いわて大漁ナビ定置水温日報一覧](https://www.suigi.pref.iwate.jp/teichi/monthly/list/daily)にある各月へのリンクのURLを値、f'{year}-{month}'をキーとした辞書

- suion_monthly_urls_dict.pkl
[いわて大漁ナビ定置水温日報一覧](https://www.suigi.pref.iwate.jp/teichi/monthly/list/daily)から飛んだ先の各日ごとのリンクのURLを値、f'{year}-{month}-{day}'をキーとした辞書

# img_crawler.py
[衛星画像一覧](http://www.suigi.pref.iwate.jp/satellite/monthly/list)から引数'img_class'の衛星画像をdatasets/satellite_images/下のディレクトリと比較して、データセットにないデータだけとってくるようにする。(できるだけ。年単位で見ているため、既にあるデータセットのうち、一番最新の年は必ず全部再スクレイピングする。)