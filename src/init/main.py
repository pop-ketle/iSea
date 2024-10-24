import os
import sqlite3
import configparser
from pathlib import Path


"""DBを初期化して、sqlite DBを作成する
"""

root = Path(__file__).parent.parent.parent

class DbInit():
    """dbの初期化処理を行うクラス
    """
    def __init__(self):
        config_ini = configparser.ConfigParser()
        config_ini.read(root / Path('./configs/config.ini'), encoding='utf-8')

        self.DB_FOLDER = root / Path(config_ini['DEFAULT']['DB_FOLDER'])
        self.DB_NAME = config_ini['DEFAULT']['DB_NAME']

    def create_table(self):
        """テーブルを作成
        """
        def _make_table(path):
            with open(path, 'r') as f:
                create_table = f.read()

            con = sqlite3.connect(self.DB_FOLDER / 'data.db')
            c   = con.cursor()

            c.execute(create_table)
            con.commit()
            con.close()

        _make_table(root / Path('./src/sqls/init_shikyo_table.sql'))
        _make_table(root / Path('./src/sqls/init_shikyo_table.sql'))


def main():
    db_init = DbInit()
    db_init.create_table()



if __name__ == "__main__":
    main()