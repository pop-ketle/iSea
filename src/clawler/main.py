import os
# import time
# import pickle
# import zipfile
import sqlite3
# import datetime
import requests
import lxml.html
import configparser
# from tqdm import tqdm
from retrying import retry
from pathlib import Path


"""[いわて大漁ナビ](https://www.suigi.pref.iwate.jp/)のデータのクロウラー
"""

class DbInit():
    """dbの初期化処理を行うクラス
    """
    def __init__(self):
        root = Path('../../')
        config_ini = configparser.ConfigParser()
        config_ini.read(root / Path('./configs/config.ini'), encoding='utf-8')

        self.DB_FOLDER = root / Path(config_ini['DEFAULT']['DB_FOLDER'])
        self.DB_NAME = config_ini['DEFAULT']['DB_NAME']

    def create_table(self):
        """テーブルを作成する
        """
        con = sqlite3.connect(self.DB_FOLDER / 'data.db')
        c   = con.cursor()

        create_table = """
            CREATE TABLE IF NOT EXISTS shikyo (
            place text,
            date text,
            fishing_type text,
            num_of_ship int,
            species text,
            catch int,
            high_price int,
            mean_price int,
            low_price int,
            primary key(place, date, fishing_type, species)
            )
            """
        c.execute(create_table)
        con.commit()
        con.close()

        # create_table = '''
            # CREATE TABLE IF NOT EXISTS data (
            #     場所 text, 日付 text, 漁業種類 text,
            #     隻数 int, 魚種 text, 規格 text,
            #     本数 int, 水揚量 int,
            #     高値 int, 平均値 int, 安値 int,
            #     primary key(場所,日付,漁業種類,魚種)
            # )'''
        


    pass

class Clawler():
    """クロウラークラス
    """
    def __init__(self):
        self.TEMPORARY_ERROR_CODES = (400,500,502,503,504) # 一時的なエラーを表すステータスコード

    @retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000)
    def fetch(self, url):
        """指定したURLを取得してResponseオブジェクトを返す。一時的なエラーが起きた場合は最大3回リトライする
        """
        print(f'Retrieving: {url} ...')
        response = requests.get(url)
        print(f'Status: {response.status_code}')
        if response.status_code not in self.TEMPORARY_ERROR_CODES:
            return response # 一時的なエラーでなければresponseを返す

        # 一時的なエラーの場合は例外を発生させてリトライする
        raise Exception(f'Temporary Error: {response.status_code}')

    def get(self):
        # 漁業データの市況日報一覧に飛ぶ
        res  = self.fetch('http://www.suigi.pref.iwate.jp/shikyo/monthly/list')
        root = lxml.html.fromstring(res.content)
        root.make_links_absolute(res.url) # すべてのリンクを絶対アドレスに変換する

        print(root)

        # id="contents"のリンク、つまり各月のページへのリンクのリストを取得 ex: https://www.suigi.pref.iwate.jp/shikyo/daily/list/2020/1
        monthly_urls = [a.get('href') for a in root.cssselect('#contents a')]
        print(monthly_urls)




def main():
    """メイン処理 スクレイピングを行い、データを取得する
    """
    db_init = DbInit()
    db_init.create_table()

    exit()

    clawler = Clawler()
    clawler.get()


if __name__ == "__main__":
    main()