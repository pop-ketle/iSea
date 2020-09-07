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
https://www.suigi.pref.iwate.jp/
'''

# init
# configparserの宣言とiniファイルの読み込み
config_ini = configparser.ConfigParser()
config_ini.read('../config.ini', encoding='utf-8')

TEMPORARY_ERROR_CODES = (400,500,502,503,504) # 一時的なエラーを表すステータスコード
DATABASE_PATH = config_ini['DEFAULT']['DATABASE_PATH']
URLS_DICT_MAKE_FLAG = False


def init_db(db_path, time_stamp, filename='data.db', db_init_flag=False):
    '''データベースを作成する、すでにdbが存在する場合は過去のものをMAX_CNT分はzipして保存する
    '''
    if not os.path.exists(db_path+'old_db/'): os.makedirs(db_path+'old_db/')

    # すでに.dbが存在する時の処理
    if os.path.exists(db_path + filename):
        # ファイルをzip化して保存して初期化する
        with zipfile.ZipFile(f'{db_path}old_db/{filename}{time_stamp}', "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(db_path + filename, f'{db_path}old_db/{filename}{time_stamp}')

        # MAX_CNTより古いデータを削除する
        MAX_CNT = 3
        files = [[file, os.path.getctime(db_path+'old_db/'+file)] for file in os.listdir(db_path+'old_db/')]
        files.sort(key=lambda x:x[1], reverse=True)
        for i, file in enumerate(files):
            if i > MAX_CNT-1:
                print(f'delete {file[0]}')
                os.remove(db_path+'old_db/'+file[0])

    # 初期化フラグがTrueなら初期化
    if db_init_flag:
        os.remove(db_path + filename)
    else:
        con = sqlite3.connect(db_path + filename)
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

def make_urls_dict(output_filename='urls_dict.pkl'):
    '''いわて大漁ナビの市況日報データへ各日毎のページへのurlをまとめたい
        f'{place}-{year}-{month}-{day}'をキー、urlを値としてdictを作成して.pkl形式で保存する
        TODO: 更新の際は差分だけdictに追加するような関数を組みたいけど、そこまで処理時間かかるわけでもないので、面倒なのでパスで
    '''
    # 漁業データの市況日報一覧に飛ぶ
    res  = fetch('http://www.suigi.pref.iwate.jp/shikyo/monthly/list')
    root = lxml.html.fromstring(res.content)
    root.make_links_absolute(res.url) # すべてのリンクを絶対アドレスに変換する

    # id="contents"のリンク、つまり各月のページへのリンクのリストを取得
    monthly_urls = [a.get('href') for a in root.cssselect('#contents a')]

    urls_dict = dict()
    for url in tqdm(monthly_urls):
        time.sleep(1) # 1sウェイト
        
        res  = fetch(url) # ここで月ごとの日報一覧に飛ぶ
        root = lxml.html.fromstring(res.content)
        root.make_links_absolute(res.url) # すべてのリンクを絶対アドレスに変換する
        
        # tableのheader情報を取得(''要素があるのでそれを除く)
        columns = [place.text_content() for place in root.cssselect('th') if place.text_content()!='']
        # 日付のリストと、湾名のリストに分離する
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
            urls_dict[f'{place}-{year}-{month}-{day}'] = daily_urls[daily_url_cnt]

            daily_url_cnt+=1
    with open('./'+output_filename, 'wb') as f: pickle.dump(urls_dict, f)

    
def web_crawl_and_scrape():
    '''岩手大漁ナビのサイトをクロールする
    '''
    # f'{place}-{year}-{month}-{day}'をキー、urlを値としたdictがないときはFLAGをTrueにして作成
    if URLS_DICT_MAKE_FLAG:
        make_urls_dict(output_filename='urls_dict.pkl')

    # f'{place}-{year}-{month}-{day}'をキー、urlを値としたdictの読み込み
    with open('./urls_dict.pkl', 'rb') as f: urls_dict = pickle.load(f)

    # スクレイピング部分
    for key, url in tqdm(urls_dict.items()):
        # time.sleep(1) # 1sウェイト

        place, year, month, day = key.split('-')

        res  = fetch(url) # ここで日ごとの市況データに飛ぶ
        root = lxml.html.fromstring(res.content)
        status_code_error(res)

        # 列名
        columns = [s.text_content().replace('（', '(').replace('）',')') for s in root.cssselect('.shikyo_daily_report_th')]

        # 漁業手法
        methods = [s.text_content() for s in root.cssselect('.shikyo_daily_report_td_fishing_cd')]

        # 行を作るために、''を漁業手法で埋めて行を作成する
        m = methods[0]
        for i in range(len(methods)):
            if methods[i]=='': methods[i] = m
            else: m = methods[i]

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

            con = sqlite3.connect(DATABASE_PATH + 'data.db')
            c   = con.cursor()
            sql = 'INSERT OR REPLACE INTO data VALUES (?,?,?,?,?,?,?,?,?,?,?)'
            c.execute(sql, [data_dict['場所'], data_dict['日付'], data_dict['漁業種類'], data_dict['隻数'], data_dict['魚種(銘柄)'], data_dict['規格'], data_dict['本数'], data_dict['水揚量(kg)'], data_dict['高値(円/kg)'], data_dict['平均値(円/kg)'], data_dict['安値(円/kg)']])
            con.commit()
            con.close()


if __name__ == "__main__":
    now = datetime.datetime.now()
    time_stamp = f'{now.year}-{now.month}-{now.day}-{now.hour}{now.minute}{now.second}'

    init_db(DATABASE_PATH, time_stamp)

    web_crawl_and_scrape()
