import os
import time
import pickle
import zipfile
import sqlite3
import datetime
import requests
import lxml.html
import configparser
from tqdm import tqdm
from retrying import retry

'''
サイトが更新されてきれいになってるっぽいのでスクレイピングするプログラムを新しく組む
NOTE: 差分を見て月単位でスクレイピングするようになっているので、バグがあった場合、意図せず日単位だとデータに抜けが生じる可能性があるかも
https://www.suigi.pref.iwate.jp/
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
MONTHLY_URLS_DICT = './shikyo_monthly_urls_dict.pkl'
DAILY_URLS_DICT   = './shikyo_daily_urls_dict.pkl'

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
        shikyo_monthly_urls_dict: f'{year}-{month}'をキー、各月へのurlを値としたdictを作成して.pkl形式で保存しておく
        shikyo_daily_urls_dict:   f'{place}-{year}-{month}-{day}'をキー、urlを値としたdictを作成して.pkl形式で保存しておく
        shikyo_monthly_urls_dictとshikyo_daily_urls_dictの差分を見て最低限のクロウルだけで済ませる
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
    res  = fetch('http://www.suigi.pref.iwate.jp/shikyo/monthly/list')
    root = lxml.html.fromstring(res.content)
    root.make_links_absolute(res.url) # すべてのリンクを絶対アドレスに変換する

    # id="contents"のリンク、つまり各月のページへのリンクのリストを取得 ex: https://www.suigi.pref.iwate.jp/shikyo/daily/list/2020/1
    monthly_urls = [a.get('href') for a in root.cssselect('#contents a')]

    # new_monthly_urls_dictに各月のページへのurlを格納
    # これをすでにクロウル済みのurlを保存してあるmonthly_urls_dictと比較して差分をとる
    # これで更新分(データにない月)だけクロウルすることができる
    new_monthly_urls_dict  = {f'{url.split("/")[-2]}-{url.split("/")[-1]}': url for url in monthly_urls}
    diff_monthly_urls_dict = dict(new_monthly_urls_dict.items() - monthly_urls_dict.items())
    monthly_urls_dict.update(new_monthly_urls_dict)

    for url in tqdm(diff_monthly_urls_dict.values(), total=len(diff_monthly_urls_dict)):
        time.sleep(1) # 1sウェイト
        
        res  = fetch(url) # ここで月ごとの日報一覧に飛ぶ
        root = lxml.html.fromstring(res.content)
        root.make_links_absolute(res.url) # すべてのリンクを絶対アドレスに変換する

        # tableのheader情報を取得(''要素があるのでそれを除く)
        columns = [place.text_content() for place in root.cssselect('th') if place.text_content()!='']
        # header情報を日付のリストと、湾名のリストに分離する
        days, places   = [s for s in columns if '日' in s], [s.replace('日', '') for s in columns if '日' not in s]

        # 対象の日のデータ(url)が存在するかのマルバツのリスト
        is_exist_daily_data = [s.text_content().replace(' ', '').replace('\n', '') for s in root.cssselect('td')]

        # 日毎の市況データのurlがない日があるためにデータがある時のデータだけを抽出できるようにする
        daily_urls = [a.get('href') for a in root.cssselect('#contents a')] # 各日毎の市況データのurlリスト

        daily_url_cnt = 0
        for i, is_exist in enumerate(is_exist_daily_data):
            if is_exist=='×': continue

            # [-1]は湾のidになる HACK: これでdict作っても良いかも
            place = places[i%len(places)]
            year, month, day = daily_urls[daily_url_cnt].split('/')[-4], daily_urls[daily_url_cnt].split('/')[-3], daily_urls[daily_url_cnt].split('/')[-2]
            daily_urls_dict[f'{place}-{year}-{month}-{day}'] = daily_urls[daily_url_cnt]

            daily_url_cnt+=1

    with open(MONTHLY_URLS_DICT, 'wb') as f: pickle.dump(monthly_urls_dict, f)
    with open(DAILY_URLS_DICT, 'wb') as f: pickle.dump(daily_urls_dict, f)

    end_time = datetime.datetime.now()
    print(f'Time to make .pkl file: {end_time - start_time}')

def web_crawl_and_scrape(reset=False):
    '''岩手大漁ナビのサイトをクロールする
    Args:
        reset(bool): Falseなら何もしない、Trueならデータベース削除して初期化
    '''
    # 初期化フラグがTrueなら初期化
    if reset:
        os.remove(DATABASE_PATH+'data.db')

    con = sqlite3.connect(DATABASE_PATH+'data.db')
    c   = con.cursor()
    create_table = '''
        CREATE TABLE IF NOT EXISTS data (
            場所 text, 日付 text, 漁業種類 text,
            隻数 int, 魚種 text, 規格 text,
            本数 int, 水揚量 int,
            高値 int, 平均値 int, 安値 int,
            primary key(場所,日付,漁業種類,魚種)
        )'''
    c.execute(create_table)
    con.commit()
    con.close()

    # NOTE: 一応ファイルがない場合に備えて例外処理書いた方がいい気もするけど、前の関数で.pkl作成するから大丈夫だろの精神
    with open(DAILY_URLS_DICT, 'rb') as f: daily_urls_dict = pickle.load(f)

    # 差分だけ追加するために、すでにデータベースに存在している'年-月'のsetを作る
    # FIXME: 結局これだと全部なめてる気がするけど、湾ごとにデータを格納する必要があるのでそのための処理が面倒なので後回し
    con = sqlite3.connect(DATABASE_PATH+'data.db')
    c   = con.cursor()
    c.execute('SELECT 日付 from data GROUP BY 日付')
    exist_date = set([f'{s[0].split("-")[0]}-{s[0].split("-")[1]}' for s in c])
    con.close()

    # スクレイピング部分
    for key, url in tqdm(daily_urls_dict.items(), total=len(daily_urls_dict)):
        # time.sleep(1) # 1sウェイト

        place, year, month, day = key.split('-')

        # 存在確認、既にデータベースに'{year}-{month}'のデータが存在している時は、
        # その月はすでにスクレイピング済みだとみなして飛ばす
        # NOTE: 月単位でスクレイピングしていることになり、日にち(day)は考慮していないので、足りない日付がある可能性あり
        # TODO: 日付単位でチェックするようにしたらいいのでは？ 速度とのトレードオフになる？
        if f'{year}-{month}' in exist_date:
            continue

        res  = fetch(url) # ここで日ごとの市況データに飛ぶ
        root = lxml.html.fromstring(res.content)
        status_code_error(res)

        # 列名
        columns = [s.text_content().replace('（', '(').replace('）',')') for s in root.cssselect('.shikyo_daily_report_th')]
        # 漁業手法
        methods = [s.text_content() for s in root.cssselect('.shikyo_daily_report_td_fishing_cd')]

        # 漁業手法のリスト中が''で埋まってるため、行を作るために、''を漁業手法で埋めて行を作成する
        m = methods[0]
        for i in range(len(methods)):
            if methods[i]=='':
                methods[i] = m
            else:
                m = methods[i]
        
        # 漁獲のデータ
        data = [c.text_content().replace('\xa0', '') for c in root.cssselect('.shikyo_daily_report_td')]
        # columnsに合わせて、二重リストの入れ子構造にする 一つのリストが一つの行を意味する
        datas = []
        for i, d in enumerate(data):
            if i%(len(columns)-1)==0:
                if i!=0: datas.append(tmp)
                tmp = []
            tmp.append(d)
        datas.append(tmp) # 一番最後のがappendされないので

        # 各リストが一つの行を意味する二重リストを作成する
        for method, data in zip(methods, datas):
            data = [method] + data
            # データがない時
            if len(set(data))==3 and '' in set(data): continue

            # 漁業データをdict形式で扱う
            data_dict = dict([(c,d) for c,d in zip(columns, data)])
            data_dict['場所'] = place
            data_dict['日付'] = f'{year}-{month:>02}-{day:>02}'
            if '規格' not in data_dict: data_dict['規格'] = ''
            if '本数' not in data_dict: data_dict['本数'] = ''

            con = sqlite3.connect(DATABASE_PATH+'data.db')
            c   = con.cursor()
            sql = 'INSERT OR REPLACE INTO data VALUES (?,?,?,?,?,?,?,?,?,?,?)'
            c.execute(sql, [data_dict['場所'], data_dict['日付'], data_dict['漁業種類'], data_dict['隻数'], data_dict['魚種(銘柄)'], data_dict['規格'], data_dict['本数'], data_dict['水揚量(kg)'], data_dict['高値(円/kg)'], data_dict['平均値(円/kg)'], data_dict['安値(円/kg)']])
            con.commit()
            con.close()

###################################

if __name__ == "__main__":
    # もしもの時のために、data.dbを、最大MAX_CNT個、ARCHIVE_FOLDERの中に保存する
    archive_db()

    # いわて大漁ナビの市況日報データへ各日毎のページへのurlを辞書形式でまとめたものを作成
    make_urls_dict()

    # 岩手大漁ナビのサイトをクロールする reset=Trueならデータベースを削除して初期化
    # 差分を見て月単位でスクレイピングするようになっているので、バグがあった場合、意図せず日単位だとデータに抜けが生じる可能性があるかも
    web_crawl_and_scrape(reset=False)