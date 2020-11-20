import os
import re
import time
import pickle
import zipfile
import sqlite3
import requests
import lxml.html
import datetime
import configparser
from tqdm import tqdm
from retrying import retry

'''
水温のデータをスクレイピングしてくる
http://www.suigi.pref.iwate.jp/teichi/monthly/list/daily
'''

# init
# configparserの宣言とiniファイルの読み込み
config_ini = configparser.ConfigParser()
config_ini.read('../config.ini', encoding='utf-8')

TEMPORARY_ERROR_CODES = (400,500,502,503,504) # 一時的なエラーを表すステータスコード
DATABASE_PATH         = config_ini['DEFAULT']['DATABASE_PATH']

ARCHIVE_FOLDER = DATABASE_PATH+'db_archives/' # 漁業データのデータベースの保存先
MAX_CNT        = 3 # データベースの最大アーカイブ数

# スクレイピングする先のURLを保存したdict
MONTHLY_URLS_DICT = './suion_monthly_urls_dict.pkl'
DAILY_URLS_DICT   = './suion_daily_urls_dict.pkl'

###################################

def archive_db():
    '''もしもの時のために、data.dbを、最大MAX_CNT個、ARCHIVE_FOLDERの中に保存する
    '''
    # 念のため、保存したいdata.dbが存在するか確認
    if not os.path.exists(DATABASE_PATH+'data.db'):
        print(f'データベースファイル({DATABASE_PATH}data.db)が存在しないため、アーカイブはされません。')
        pass

    # 保存先フォルダ生成
    if not os.path.exists(ARCHIVE_FOLDER): os.makedirs(ARCHIVE_FOLDER)

    # 保存しておくデータベース名のtimestamp
    now = datetime.datetime.now()
    timestamp = f'{now.year}{now.month}{now.day}{now.hour}{now.minute}{now.second}'

    # ファイルをzip化して保存して初期化する
    with zipfile.ZipFile(f'{ARCHIVE_FOLDER}data{timestamp}.db.zip', "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(f'{DATABASE_PATH}data.db', f'{DATABASE_PATH}data.db')

    # MAX_CNTよりアーカイブが多い場合、古いデータを削除する
    # ファイル名とファイル作成日時を取得、ファイル作成日時でソートして古い順に削除
    files = [[file, os.path.getctime(ARCHIVE_FOLDER+file)] for file in os.listdir(ARCHIVE_FOLDER)]
    files.sort(key=lambda x:x[1], reverse=True)
    for i, file in enumerate(files):
        if i > MAX_CNT-1:
            print(f'delete {file[0]}')
            os.remove(ARCHIVE_FOLDER+file[0])

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

def make_urls_dict():
    '''いわて大漁ナビの市況日報データへ各日毎のページへのurlを辞書形式でまとめたものを作成
        suion_monthly_urls_dict: f'{year}-{month}'をキー、各月へのurlを値としたdictを作成して.pkl形式で保存しておく
        suion_daily_urls_dict:   f'{year}-{month}-{day}'をキー、urlを値としたdictを作成して.pkl形式で保存しておく
        suion_monthly_urls_dictとsuion_daily_urls_dictの差分を見て最低限のクロウルだけで済ませる
    '''
    # MONTHLY_URLS_DICT、DAILY_URLS_DICT ファイルがすでに存在している場合はそれをロード、ない場合は新規作成
    if os.path.exists(MONTHLY_URLS_DICT):    
        with open(MONTHLY_URLS_DICT, 'rb') as f: monthly_urls_dict = pickle.load(f)
    else:
        monthly_urls_dict = dict()

    if os.path.exists(DAILY_URLS_DICT):    
        with open(DAILY_URLS_DICT, 'rb') as f: daily_urls_dict = pickle.load(f)
    else:
        daily_urls_dict = dict()

    start_time = datetime.datetime.now() # かかった時間計算用

    # 漁業データの市況日報一覧に飛ぶ
    res  = fetch('http://www.suigi.pref.iwate.jp/teichi/monthly/list/daily')
    root = lxml.html.fromstring(res.content)
    root.make_links_absolute(res.url) # すべてのリンクを絶対アドレスに変換する

    # id="contents"のリンク、つまり各月のページへのリンクのリストを取得 ex: 'https://www.suigi.pref.iwate.jp/teichi/daily/list/1994/12'
    monthly_urls = [a.get('href') for a in root.cssselect('#contents a')]

    # new_monthly_urls_dictに各月のページへのurlを格納
    # これをすでにクロウル済みのurlを保存してあるmonthly_urls_dictと比較して差分をとる
    # これで更新分(データにない月)だけクロウルすることができる
    new_monthly_urls_dict  = {f'{url.split("/")[-2]}-{url.split("/")[-1]}': url for url in monthly_urls}
    diff_monthly_urls_dict = dict(new_monthly_urls_dict.items() - monthly_urls_dict.items())
    monthly_urls_dict.update(new_monthly_urls_dict)

    for url in tqdm(diff_monthly_urls_dict.values(), total=len(diff_monthly_urls_dict)):
        # time.sleep(1) # 1sウェイト
        
        res  = fetch(url) # ここで月ごとの日報一覧に飛ぶ
        root = lxml.html.fromstring(res.content)
        root.make_links_absolute(res.url) # すべてのリンクを絶対アドレスに変換する

        daily_urls = [a.get('href') for a in root.cssselect('#contents a')]

        # 既にクロウル済みのurlを保存してあるdaily_urls_dictと比較して差分をとり、更新分だけクロウルする
        # HACK: ここら辺の処理を市況の方にも入れられる気がする
        new_daily_urls_dict  = {f'{url.split("/")[-3]}-{url.split("/")[-2]}-{url.split("/")[-1]}': url for url in daily_urls}
        diff_daily_urls_dict = dict(new_daily_urls_dict.items() - daily_urls_dict.items())
        daily_urls_dict.update(new_daily_urls_dict)

    with open(MONTHLY_URLS_DICT, 'wb') as f: pickle.dump(monthly_urls_dict, f)
    with open(DAILY_URLS_DICT, 'wb') as f: pickle.dump(daily_urls_dict, f)

    end_time = datetime.datetime.now()
    print(f'Time to make .pkl file: {end_time - start_time}')

def web_crawl_and_scrape(reset=False):
    '''岩手大漁ナビのサイトをクロールする(水温)
    Args:
        reset(bool): Falseなら何もしない、Trueならデータベース削除して初期化
    '''
    # 初期化フラグがTrueなら初期化
    if reset:
        os.remove(DATABASE_PATH+'data.db')

    con = sqlite3.connect(DATABASE_PATH+'data.db')
    c   = con.cursor()
    create_table = '''
        CREATE TABLE IF NOT EXISTS suion_data (
            日付 text,
            時間 text,
            場所 text,
            水温 int,
            primary key(日付,時間,場所)
        )'''
    con.execute(create_table)
    con.commit()
    con.close()

    # NOTE: 一応ファイルがない場合に備えて例外処理書いた方がいい気もするけど、前の関数で.pkl作成するから大丈夫だろの精神
    with open(DAILY_URLS_DICT, 'rb') as f: daily_urls_dict = pickle.load(f)

    # 差分だけ追加するために、すでにデータベースに存在している'年-月-日'のsetを作る
    con = sqlite3.connect(DATABASE_PATH+'data.db')
    c   = con.cursor()
    c.execute('SELECT 日付 from suion_data GROUP BY 日付')
    exist_date = set([f'{s[0].split("-")[0]}-{s[0].split("-")[1]}-{s[0].split("-")[2]}' for s in c])
    con.close()

    # keyの差分をとることで、データベースにないデータだけスクレイピングする
    # HACK: 処理上ソートする意味自体はないけど、古い時のデータが完全にない時もちゃんと処理できるように古いのでチェックしたかった
    diff_daily_urls_key = sorted(set(set([key for key in daily_urls_dict.keys()]) - exist_date))

    # スクレイピング部分
    for key in diff_daily_urls_key:
        # time.sleep(1) # 1sウェイト

        url = daily_urls_dict[key]

        res  = fetch(url) # ここで日ごとの水温データに飛ぶ
        root = lxml.html.fromstring(res.content)
        status_code_error(res)

        # ヘッダーの前年・当年を除く
        places = [place.text_content() for place in root.cssselect('.daily_report_detail_table_th') if '湾' in place.text_content()]
        places = [re.sub(r'\([^)]*\)', '', s) for s in places] # (*m)の表記を消す

        times = [time.text_content() for time in root.cssselect('.daily_report_detail_table_td_side')]
        times = [re.sub(r'\s+', ' ', s).strip() for s in times] # '\xa023などの整形用の空白を消す

        # 偶数のデータだけ持ってくることで当年のデータを持ってくる
        suion_data = [suion.text_content() for i,suion in enumerate(root.cssselect('.daily_report_detail_table_td')) if i%2==0]
        suion_data = [re.sub(r'\s+', ' ', s).strip() for s in suion_data] # '\xa023などの整形用の空白を消す

        div_suion = [suion_data[i::len(places)] for i in range(len(places))] # 場所ごとに水温を分割した二重リストにする

        # [日付、時間、場所、水温]
        for place, suions in zip(places, div_suion):
            for time, suion in zip(times, suions):
                if suion=='': continue # 水温の情報がないものはいらないので飛ばす NOTE: 欠損値なら欠損値で情報な気もするので、いれておくか...?

                data = [key, place]
                data.append(time)
                data.append(suion)
                data[1], data[2] = data[2], data[1] # [日付、時間、場所、水温]に合わせるためスワップ

                # データベースに突っ込んでく
                con = sqlite3.connect(DATABASE_PATH+'data.db')
                c   = con.cursor()
                sql = 'INSERT OR REPLACE INTO suion_data VALUES (?,?,?,?)'
                c.execute(sql, data)
                con.commit()
                con.close()


if __name__ == "__main__":
    # もしもの時のために、data.dbを、最大MAX_CNT個、ARCHIVE_FOLDERの中に保存する
    archive_db()

    # いわて大漁ナビの水温データへ各日毎のページへのurlを辞書形式でまとめたものを作成
    make_urls_dict()

    # 岩手大漁ナビのサイトをクロールする reset=Trueならデータベースを削除して初期化
    web_crawl_and_scrape(reset=False)