from pprint import pprint
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta

import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as offline
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

def daterange(_start, _end):
    for n in range((_end - _start).days+1):
        yield _start + timedelta(n)

def nibutan(ls, catch):
    # [漁獲量, 日付連番, 漁獲量でソートした連番]の二重リストと漁獲量を与えて、漁獲量がどこに挿入できるのか二分探索する
    # TODO: 漁獲量でソートして渡そう！
    lo, hi = 0, len(ls)
    while lo<hi:
        mid = (lo+hi)//2
        if ls[mid][0]<catch: lo = mid+1
        else: hi = mid
    return lo

def get_fish_data(db_path, place, method, species, start, end):
    '''
    dbにアクセスして漁獲量の情報を時系列順に得る
    Arg:
        db_path(str): コネクトするdbのパス
        place  (str): 漁港
        method (str): 漁業手法
        species(str): 魚種
        start(datetime object): 検索の開始位置
        end(datetime object): 検索の終了位置
    Return:
        catch_dateidx_fishidx: ['漁獲量', '日付連番', '日付', '漁獲量でソートした連番']の日付連番で昇順にソート済み二重リスト
    '''
    # dbとのコネクトを確立
    con = sqlite3.connect(db_path)
    c   = con.cursor()

    columns = [s[0] for s in con.execute('select * from data').description] # dbのカラム名
    catch_dateidx_fishidx = []
    # 日付によってデータがない時があるので
    for i,date in enumerate(daterange(start, end)):
        date = str(date).replace(' 00:00:00', '')

        sql = f"""
            SELECT * from data
            WHERE 場所=='{place}'
                and 漁業種類=='{method}'
                and 魚種=='{species}'
                and 日付=='{date}'
        """
        c.execute(sql)
        data = [list(s) for s in c]
        if len(data)==0:
            catch_dateidx = [-1, i, date] # 漁獲量を0じゃなくて-1にしてみる
        else:
            data = dict([(_c,_d) for _c,_d in zip(columns,data[0])])
            # 漁獲量が''かつ他のデータもほぼ''の時がある、この時どうするかの処理
            # TODO: 正直隻数はあまり信用できる情報ではないが、これがないときは漁にもいかなかったと考えて-1を与える
            # 隻数があったときは漁に行ったと判断し0にする
            if data['水揚量']=='': # 漁のデータがあっても漁獲量の部分が空のことがある
                if data['隻数']=='':
                    catch_dateidx = [-1, i, date]
                else:
                    catch_dateidx = [0, i, date]
            else:
                catch_dateidx = [int(data['水揚量']), i, date]
        catch_dateidx_fishidx.append(catch_dateidx)

    # 漁獲量でソートした連番をつける
    catch_dateidx_fishidx.sort(key=lambda x: (x[0],x[1])) # 漁獲量でソート(漁獲量が同じときは日付連番でソートされるはず)
    for i,catch_dateidx in enumerate(catch_dateidx_fishidx):
        catch_dateidx.append(i) # 漁獲量でのデータ連番を与える

    con.close()  # dbをクローズ
    return sorted(catch_dateidx_fishidx, key=lambda x: (x[1])) # 日付連番でソートして返す



# '''
# init
# '''
# db_path = '../fish/data.db'
# place, method, species = '大船渡', '定置網', 'サバ類'
# catch_min, catch_max = 100000, 250000 # 漁獲量下限上限のスライダ-の値になる予定、intで頼む
# start = datetime.strptime('{}-{:0=2}-{:0=2}'.format(2012,1,1), '%Y-%m-%d')
# end   = datetime.strptime('{}-{:0=2}-{:0=2}'.format(2017,12,31), '%Y-%m-%d')

# '''
# process
# '''
# data = get_fish_data(db_path, place, method, species, start, end) # [漁獲量, 日付連番, '日付', 漁獲量でソートした連番]の日付連番で昇順にソート済み二重リスト
# print(data)

# # データとして与えやすそうなのでpd DataFrameにしてみる
# data_df = pd.DataFrame(data, columns=['漁獲量', '日付連番', '日付', '漁獲量連番'])
# print(data_df)

# date_trace = go.Scatter(x=list(daterange(start, end)), y=data_df['漁獲量'], name=f'{place}-{method}-{species}-漁獲量')
# min_trace = go.Scatter(x=list(daterange(start, end)), y=[catch_min]*len(data), name='下限')
# max_trace = go.Scatter(x=list(daterange(start, end)), y=[catch_max]*len(data), name='上限')

# # レイアウトの指定
# layout = go.Layout(xaxis = dict(title = 'date', type='date', dtick = 'M6', tickformat='%Y-%m-%d'),
#                 yaxis = dict(title = 'value'),
#                 xaxis_rangeslider_visible=True)
#                 # yaxis_rangeslider_visible=True)

# fig = dict(data = [date_trace, min_trace, max_trace], layout = layout)

# offline.iplot(fig)
