import os
import time
import glob
import requests
import lxml.html
import configparser
from retrying import retry
from datetime import datetime,date,timedelta

'''以下のURLから画像をスクレイピングしてくる
http://www.suigi.pref.iwate.jp/satellite/monthly/list
'''

# init
# configparserの宣言とiniファイルの読み込み
config_ini = configparser.ConfigParser()
config_ini.read('../config.ini', encoding='utf-8')

TEMPORARY_ERROR_CODES = (400,500,502,503,504) # 一時的なエラーを表すステータスコード
DATABASE_PATH         = config_ini['DEFAULT']['DATABASE_PATH']

# スクレイピングする先のURLを保存したdict
MONTHLY_URLS_DICT = './img_monthly_urls_dict.pkl'
DAILY_URLS_DICT   = './img_daily_urls_dict'

def daterange(_start, _end):
    for n in range((_end - _start).days):
        yield _start + timedelta(n)

###################################
# スクレイピング用の処理
@retry(stop_max_attempt_number=3,wait_exponential_multiplier=1000)
def fetch(url):
    '''指定したURLを取得してResponseオブジェクトを返す。一時的なエラーが起きた場合は最大3回リトライする
    '''
    print(f'Retrieving: {url} ...')
    response = requests.get(url)
    print(f'Status: {response.status_code}')
    if response.status_code not in TEMPORARY_ERROR_CODES:
        return response # 一時的なエラーでなければresponseを返す

    # 一時的なエラーの場合は例外を発生させてリトライする
    raise Exception(f'Temporary Error: {response.status_code}')

def status_code_error(response):
    '''ステータスコードに応じたエラー処理
    '''
    print('Success!') if 200 <= response.status_code < 300 else print('Error!')

def scrape_img(img_class):
    '''以下のURLから画像をスクレイピングしてくる http://www.suigi.pref.iwate.jp/satellite/monthly/list
    '''
    # 衛星画像一覧に飛ぶ
    res  = fetch('http://www.suigi.pref.iwate.jp/satellite/monthly/list')
    root = lxml.html.fromstring(res.content)
    root.make_links_absolute(res.url) # すべてのリンクを絶対アドレスに変換する

    # id="contents"のリンク、つまり各月のページへのリンクのリストを取得 ex: 'https://www.suigi.pref.iwate.jp/satellite/daily/list/2020/9'
    monthly_urls = [a.get('href') for a in root.cssselect('#contents a')]
    # 岩手大漁ナビに何年の画像データがあるのか確認する
    site_exist_years  = set([url.split('/')[-2] for url in monthly_urls])

    # ディレクトリを見て、何年のデータをもう既に持っているか確認する
    img_dirs = glob.glob(DATABASE_PATH+'satellite_images/**/', recursive=True)
    data_exist_years = set([img_dir.split('/')[-3] for img_dir in img_dirs if img_class in img_dir])

    # 画像データを持ってない年のlist
    diff_exist_years = list(site_exist_years - data_exist_years)
    diff_exist_years = sorted([int(year) for year in diff_exist_years], reverse=True)
    # 画像データを持ってない一番若い年の一年前の分もスクレイピングし直すことで抜けがないようにする
    diff_exist_years.append(diff_exist_years[-1] - 1)

    start = datetime.strptime(f'{diff_exist_years[-1]}-01-01', '%Y-%m-%d')
    end   = datetime.strptime(f'{diff_exist_years[0]}-12-31', '%Y-%m-%d')
    for date in daterange(start, end):
        # time.sleep(1) # 1sウェイト

        date = str(date).replace(' 00:00:00', '')
        year, month, day = date.split('-')

        path_template = f'https://www.suigi.pref.iwate.jp/satellite/daily/image/list/{year}/{month}/{day}'

        # path_template先の画像をの中からimg_class(ex:'7Wc')の画像へのリンクを抜き出す
        img_res  = fetch(path_template)
        root     = lxml.html.fromstring(img_res.content)
        img_urls = [a.get('href') for a in root.cssselect('#contents a') if img_class in a.get('href')]

        try: # サムネ、文字のリンクからダウンロードしたい画像へのリンク先へ飛ぶ
            img_res = fetch('https://www.suigi.pref.iwate.jp'+img_urls[0])
            root    = lxml.html.fromstring(img_res.content)
        except IndexError: # 画像がない時は飛ばす
            continue

        # 取り込む画像のファイル名
        filename = [a.get('src') for a in root.cssselect('img') if img_class in a.get('src')][0]

        # 保存するファイル名、保存先のディレクトリを作成
        save_path = f'{DATABASE_PATH}satellite_images/{year}/{img_class}/{img_class}_{year}{month}{day}0000.png'
        save_dir  = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        # 画像の保存
        with open(save_path, 'wb') as f:
            req = requests.get('https://www.suigi.pref.iwate.jp'+filename)
            f.write(req.content)

if __name__ == "__main__":
    scrape_img(img_class='7Wc')