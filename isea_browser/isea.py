import os
import cv2
import time
import pickle
import shutil
import base64
import sqlite3
import numpy as np
import pandas as pd
import configparser
from tqdm import tqdm
from io import BytesIO
from colour import Color
import concurrent.futures
from datetime import datetime
from datetime import timedelta
from PIL import Image, ImageFilter, ImageDraw

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.offline as offline
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from flask import Flask, render_template
from flask import request

import myfunc

def daterange(_start, _end):
    for n in range((_end - _start).days+1):
        yield _start + timedelta(n)

def nibutan(ls, val, c):
    # cの列でvalに対して二分探索を行う、事前にcの列に対してソートをしておく必要がある
    lo, hi = 0, len(ls)
    while lo<hi:
        mid = (lo+hi)//2
        if float(ls[mid][c])<val: lo = mid+1 # よくわからないけどdetailsのデータで漁に行っていないデータを除外するために使いたい時、'numpy.str_'のことがあるのでfloatでキャスト
        else: hi = mid
    return lo

# def clustering_samples(data, img_data_path, num_sample):
#     '''
#     クラスタリングして表示する画像を選ぶ
#     # FIXME: ここら辺どこかで画像のshapeを保存しておくべき
#     # FIXME: やっぱり画像データの読み込み部分が速度のネックかなと
#     Arg:
#         data: 探索したい範囲のデータ ['漁獲量', '日付連番', '日付', '漁獲量でソートした連番']
#         img_data_path(str): 画像データが入った辞書のpklデータが保存されている階層のフォルダパス
#     Return:
#         shape(tuple): 画像のshape(いくつか違うのが入る可能性もあるが一つ目のshapeで保存してしまおう)
#         imgs(dict): 日付がキー、画像のflaatenが値
#         center_keys(list): クラスタリングした中心のインデックスを返す
#     '''
#     imgs = dict()
#     init_flag = True
#     for _d in data:
#         _year,_month,_day = _d[2].split('-') # 年、月、日
        
#         with open(img_data_path+f'img_dict{_year}.pkl', 'rb') as f:
#             img = pickle.load(f)
#             if init_flag:
#                 init_flag = False
#                 shape = img[_d[2]].shape
#             if _d[2] in img and _d[2]!=None: # _d[2]は日付の情報、この日付のキーがある時のみ処理を行う
#                 imgs[_d[2]] = img[_d[2]].flatten()

#     # サンプリングしたい数以下しかデータがないときの処理
#     if len(data)<=num_sample:
#         keys = [d[2] for d in data] # サンプリングしたい数以下しかデータが範囲にないときは全データの日付を返す
#         return shape, imgs, keys
    
#     # クラスタリングをする
#     clusters = KMeans(n_clusters=num_sample)
#     pred = clusters.fit_predict(list(imgs.values())) # predは使ってないけど上でfit_predictも一緒にやろうとしたらエラーが出たので

#     # クラスターの中心の配列を得て、センターのキー(日付)を返す
#     centers = clusters.cluster_centers_
#     labels = clusters.labels_
#     center_keys = []
#     for label in set(labels): # 違うクラスタとして早く出てきた順に決めてしまおう
#         for l,k in zip(labels,imgs.keys()):
#             if l==label:
#                 center_keys.append(k)
#                 break
#     # print('++++++++++++++')
#     # print(imgs)
#     # print(centers)
#     # center_keys = [k for k, v in imgs.items() if v in centers] # 複数出てくる
#     print(center_keys)
#     # print(len(centers),len(center_keys))
#     return shape, imgs, center_keys

def get_img_path(path_temp, center_keys):
    '''
    日付と画像のフォルダへのパスをくっつけて取ってくる
    Arg:
        path_temp(str): 画像のフォルダのパス
        center_keys(list): str方の日付の入ったリスト
    Return:
        imh_path(list): 画像へのパス
    '''
    img_path = []
    for _keys in center_keys:
        _year,_month,_day = _keys.split('-')
        
        file_path = path_temp + f'{_year}/7Wc/7Wc_{_year}{_month}{_day}0000.png'
        if not os.path.isfile(file_path):
            file_path = path_temp + f'{_year}/7Wc/7Wc_{_year}{_month}{_day}0000.jpg'
            if not os.path.isfile(file_path): file_path = None # 一応jpgファイルの存在確認にもしておいてないならNone

        img_path.append(file_path)
    return img_path

def get_img_array(dates):
    '''日付のリストを入れると各日付ごとの画像をbaseエンコード、utf-8デコードしたものを持ってきてhtmlに埋め込めるようにする
    '''
    def get_img(date, img_data):
        # 日付を入れると日付画像をbaseエンコード、utf-8デコードしたものをdbから引っ張ってくる
        c.execute(f'SELECT encoded_img from encoded_imgs WHERE date=="{date}"')
        img_b64data = [list(s)[0] for s in c]
        
        # とりあえず画像のデータがなかったときはNoneを返してみる
        img_b64data = None if len(img_b64data)==0 else img_b64data[0]
        img_data.append(img_b64data)

    # executor = concurrent.futures.ThreadPoolExecutor(max_workers=2) # マルチスレッド化
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=2) # マルチコア化

    con = sqlite3.connect(DATABASE_PATH + 'encoded_img.db')
    c   = con.cursor()

    img_data = []
    for date in tqdm(dates): executor.submit(get_img(date, img_data))

    con.close()
    return img_data

def get_catch_array(dates, place, method, species, DB_PATH):
    # 日付のリストと湾、漁法、魚種を引数に与えると各日付ごとの画像をキャッシュに入れたリストを返す
    con = sqlite3.connect(DB_PATH)
    c   = con.cursor()

    catches = []
    for date in dates:
        c.execute(f'SELECT 水揚量 from data WHERE 日付=="{date}" and 場所=="{place}" and 漁業種類=="{method}" and 魚種=="{species}"')
        catch = [list(s)[0] for s in c]
        catch = -1 if len(catch)==0 else catch[0] # 漁獲量データが出てこないときは-1にしておく
        catches.append(catch)
    catches = [c if c!='' else -1 for c in catches] # 漁獲量が''の時があるので-1に変換しておく
    con.close()
    return catches


app = Flask(__name__)

'''
with open('/Volumes/HDD/datasets/method_dict.pickle', 'rb') as f:
    methods_dict = pickle.load(f)
with open('/Volumes/HDD/datasets/species_dict.pickle', 'rb') as f:
    species_dict = pickle.load(f)
with open(f'/Volumes/HDD/datasets/FishMap/group_dicts/32x32x8/group_dict.pkl', 'rb') as f:
    img_group_dict = pickle.load(f)

DB_PATH = '../fish/data.db'
# BASE_PATH = '/Volumes/HDD/datasets/'
np.set_printoptions(suppress=True) # 指数表記にしない
# num_sample = 3

print('init test')
print(methods_dict)
print(species_dict)
# print(img_group_dict)
'''

@app.route('/', methods=['GET', 'POST'])
def index(): # 最初に動く
        return render_template('index.html',
                            message='場所を選んでください',
                            places=methods_dict.keys())

@app.route('/select_method', methods=['GET', 'POST'])
def get_method():
    place = request.form.get('place')
    return render_template('select_method.html',
                        title = 'sample(post)',
                        message = f'[{place}]選択、漁業手法を選んでください',
                        methods=methods_dict[f'{place}'],
                        place=place)

@app.route('/select_species', methods=['GET', 'POST'])
def get_species():
    place  = request.form.get('place')
    method = request.form.get('method')
    return render_template('select_species.html',
                    title = 'sample(post)',
                    message = f'[{place}-{method}]選択、魚種を選んでください',
                    species_ls=species_dict[f'{place}:{method}'],
                    place=place, method=method)

@app.route('/simple_graph', methods=['GET', 'POST'])
def make_graph():
    print(request.form)
    place     = request.form.get('place')
    method    = request.form.get('method')
    species   = request.form.get('species')
    
    startdate = str(request.form.get('startdate')) if request.form.get('startdate') is not None else '2012-01-01'
    enddate   = str(request.form.get('enddate'))   if request.form.get('enddate') is not None else '2018-12-31'

    (sy,sm,sd), (ey,em,ed) = str(startdate).split('-'), str(enddate).split('-')
    start = datetime.strptime(f'{sy}-{sm}-{sd}', '%Y-%m-%d')
    end   = datetime.strptime(f'{ey}-{em}-{ed}', '%Y-%m-%d')
    
    print(startdate, enddate)


    threshold  = int(request.form.get('threshold'))  if request.form.get('threshold') is not None else ''
    num_sample = int(request.form.get('num_sample')) if request.form.get('num_sample') is not None else 5 # サンプリング画像数

    print(f'{place}-{method}-{species} 閾値{threshold}(kg)')


    traces = []
    data = myfunc.get_fish_data(DB_PATH, place, method, species, start, end)
    # データとして与えやすそうなのでpd DataFrameにしてみる
    data_df = pd.DataFrame(data, columns=['漁獲量', '日付連番', '日付', '漁獲量連番'])
    # 漁獲量の折れ線グラフ
    traces.append(go.Scatter(x=list(daterange(start, end)), y=data_df['漁獲量'], mode='lines+markers', name=f'{place}-{method}-{species}-漁獲量',
                    line=dict(color='rgba(100,149,237,1.0)',
                            width=2.0),
                    marker=dict(color='rgba(100,149,237,0.9)',
                            size=6)
                            ))

    # あとで漁獲量の上限加減のインデックスを得るのに便利なのでソートしておく
    data.sort(key=lambda x: (x[0],x[1])) # 漁獲量でソートしたものを与える(漁獲量が同じときは日付連番でソートされるはず)
    # TODO: ポジネガで二つに分ける

    # print(data)
    if not threshold=='': # ポジティブネガティブの閾値が設定された場合
        traces.append(go.Scatter(x=list(daterange(start, end)), y=[threshold]*len(data), name='閾値'))
        idx = nibutan(data, threshold, 0)

        # ポジネガで分ける
        positive_data, negative_data = data[idx:], data[:idx]

        # 漁に行っていないデータを除外する
        idx = nibutan(negative_data, 0, 0)
        negative_data, removed_data = negative_data[idx:], negative_data[:idx] # 漁に行かなかったデータをremoved_dataとする
        print(f'要素数:合計{len(data)}, ポジティブ数:{len(positive_data)}, ネガティブ数:{len(negative_data)}, 漁に行かなかったデータ数{len(removed_data)}')

        # HACK: 分けたデータに対して日付でソート、サンプル数分適当に区切って、区切ったところを代表値にすればいいんじゃないかな
        positive_data.sort(key=lambda x: x[2], reverse=True) # 日付でソート
        negative_data.sort(key=lambda x: x[2], reverse=True) # 日付でソート

        # HACK: ここら辺なんか関数化したいね
        # ポジティブデータ
        positive_date_idx = [(len(positive_data)//num_sample) * i for i in range(num_sample)]
        if len(positive_date_idx)<=len(positive_data):
            positive_date  = [positive_data[i][2] for i in positive_date_idx] # 日付のデータ
            positive_catch = [positive_data[i][0] for i in positive_date_idx] # 漁獲量のデータ
        else: # サンプリングしてきたい数のほうが多い場合
            positive_date  = [d[2] for d in positive_data]
            positive_catch = [d[0] for d in positive_data]
        positive_imgs = get_img_array(positive_date) # 日付のデータを入れると画像を読み込んでくる
        positive_data = list(zip(positive_imgs, positive_catch, positive_date))
        traces.append(go.Scatter(x=positive_date, y=positive_catch, mode='markers', name='ポジティブデータサンプル', marker_color='rgba(255,0,0,.8)', marker_size=15))

        # ネガティブデータ
        negative_date_idx = [(len(negative_data)//num_sample) * i for i in range(num_sample)]
        if len(negative_date_idx)<=len(negative_data):
            negative_date  = [negative_data[i][2] for i in negative_date_idx] # 日付のデータ
            negative_catch = [negative_data[i][0] for i in negative_date_idx] # 漁獲量のデータ
        else: # サンプリングしてきたい数のほうが多い場合
            negative_date  = [d[2] for d in negative_data]
            negative_catch = [d[0] for d in negative_data]
        negative_imgs = get_img_array(negative_date) # 日付のデータを入れると画像を読み込んでくる
        negative_data = list(zip(negative_imgs, negative_catch, negative_date))
        traces.append(go.Scatter(x=negative_date, y=negative_catch, mode='markers', name='ネガティブデータサンプル', marker_color='rgba(0,0,255,.8)', marker_size=15))
    else:
        positive_data = None
        negative_data = None

    # レイアウトの指定
    layout = go.Layout(xaxis=dict(title='日付', type='date', dtick='M6', tickformat='%Y-%m-%d'),
                    yaxis=dict(title='漁獲量(kg)'),
                    xaxis_rangeslider_visible=True,
                    width=1000, height=750,
                    clickmode='select+event',)
                    # yaxis_rangeslider_visible=True)

    fig = dict(data=traces, layout=layout)
    with open('./static/js/plotly_click.js') as f: plotly_click_js = f.read()
    plotly_div = plotly.io.to_html(
        fig,
        include_plotlyjs=True,
        post_script=plotly_click_js,
        full_html=False, # 完全なHTMLファイルではなく、一つの<div>要素のみを含んだ文字列で埋め込む
    )
    # plotly.offline.plot(fig, filename='./templates/simple_plotly_graph.html', auto_open=False)

    return render_template('simple_graph.html',
                title='sample(post)',
                message=f'[{place}-{method}-{species}]選択',
                place=place, method=method, species=species,
                startdate=startdate, enddate=enddate,
                threshold=str(threshold), num_sample=str(num_sample),
                plotly_div=plotly_div,
                positive_data=positive_data, negative_data=negative_data
                )


@app.route('/img_details', methods=['GET', 'POST'])
def show_img_details():
    print(request.form)
    place, method, species = request.form.get('place'), request.form.get('method'), request.form.get('species')

    sim_metric = str(request.form.get('sim_metric')) if request.form.get('sim_metric') is not None else 'scaled_avg'
    startdate = str(request.form.get('startdate')) if request.form.get('startdate') is not None else '2012-01-01'
    enddate   = str(request.form.get('enddate'))   if request.form.get('enddate') is not None else '2018-12-31'

    (sy,sm,sd), (ey,em,ed) = str(startdate).split('-'), str(enddate).split('-')
    start = datetime.strptime(f'{sy}-{sm}-{sd}', '%Y-%m-%d')
    end   = datetime.strptime(f'{ey}-{em}-{ed}', '%Y-%m-%d')

    threshold = int(request.form.get('threshold'))
    date      = request.form.get('date')
    print(f'{place}-{method}-{species} 閾値{threshold}(kg)')

    data = myfunc.get_fish_data(DB_PATH, place, method, species, start, end)
    data_df = pd.DataFrame(data, columns=['漁獲量', '日付連番', '日付', '漁獲量連番']) # データとして与えやすそうなのでpd DataFrameにしてみる

    # 漁獲量の折れ線グラフ
    traces = []
    traces.append(go.Scatter(x=list(daterange(start, end)), y=data_df['漁獲量'], mode='lines+markers', name=f'{place}-{method}-{species}-漁獲量',
                    line=dict(color='rgba(100,149,237,1.0)',
                            width=2.0),
                    marker=dict(color='rgba(100,149,237,0.9)',
                            size=6)))
    traces.append(go.Scatter(x=list(daterange(start, end)), y=[threshold]*len(data), name='閾値'))    

    # 各データが[[類似度順連番, 画像のエンコードしたやつ, 漁獲量, 日付, 類似度],...]の二重リストにまとめる
    nn_idx     = [i+1 for i in range(20)]
    nn_dates   = img_group_dict[f'{date}:group_date']
    nn_imgs    = get_img_array(nn_dates)
    nn_catches = get_catch_array(nn_dates, place, method, species, DB_PATH)
    nn_dis     = [round(i, 2) for i in img_group_dict[f'{date}:group_dis']]
    nn_data    = [[idx, imgs, int(str(catch).replace(',', '')), date, dis] for idx,imgs,catch,date,dis in zip(nn_idx, nn_imgs, nn_catches, nn_dates, nn_dis)]

    seed_img, nn_data = [nn_data[0]], nn_data[1:] # 代表値(seed_img)と近傍(nn_data)を分離
    nn_data.sort(key=lambda x:x[2]) # 漁獲量でソート

    idx = nibutan(nn_data, threshold, 2)
    # ポジティブネガティブでデータを分ける
    nn_data = np.array(nn_data)
    nn_positive_data, nn_negative_data = nn_data[idx:], nn_data[:idx]

    # 漁に行っていないデータを除外する
    idx = nibutan(nn_negative_data, 0, 2)
    nn_negative_data, removed_data = nn_negative_data[idx:], nn_negative_data[:idx] # 漁に行かなかったデータをremoved_dataとする
    print(f'要素数:合計{len(nn_data)}, ポジティブ数:{len(nn_positive_data)}, ネガティブ数:{len(nn_negative_data)}, 漁に行かなかったデータ数{len(removed_data)}')

    # HACK: ここら辺関数化できそう
    if nn_positive_data.size!=0:
        positive_idx = [int(i)-2 for i in nn_positive_data[:,0]] # あとで類似度を見る時に使う 0インデックスじゃないっぽい(No.で使うつもりだったから上で1足してる上に、一つ目をseedとして抜いてるので-2)
        text = [f'No.{_idx} 日付:{_date} 漁獲量:{_catch}' for _idx,_catch,_date in zip(nn_positive_data[:,0], nn_positive_data[:,2], nn_positive_data[:,3])] # 各自のラベル
        traces.append(go.Scatter(x=nn_positive_data[:,3],  y=nn_positive_data[:,2], text=text,  mode='markers', name='ポジティブ類似画像', marker_color='rgba(255,0,0,.8)', marker_size=15))
        nn_positive_data  = list(nn_positive_data)
    else: nn_positive_data = None; positive_idx = []

    if nn_negative_data.size!=0:
        negative_idx = [int(i)-2 for i in nn_negative_data[:,0]] # あとで類似度を見る時に使う 0インデックスじゃないっぽい(No.で使うつもりだったから上で1足してる上に、一つ目をseedとして抜いてるので-2)
        text = [f'No.{_idx} 日付:{_date} 漁獲量:{_catch}' for _idx,_catch,_date in zip(nn_negative_data[:,0], nn_negative_data[:,2], nn_negative_data[:,3])] # 各自のラベル
        traces.append(go.Scatter(x=nn_negative_data[:,3],  y=nn_negative_data[:,2], text=text,  mode='markers', name='ネガティブ類似画像', marker_color='rgba(0,0,255,.8)', marker_size=15))
        nn_negative_data  = list(nn_negative_data)
    else: nn_negative_data = None; negative_idx = []

    if removed_data.size!=0:
        removed_idx = [int(i)-2 for i in removed_data[:,0]] # あとで類似度を見る時に使う 0インデックスじゃないっぽい(No.で使うつもりだったから上で1足してる上に、一つ目をseedとして抜いてるので-2)
        text = [f'No.{_idx} 日付:{_date} 漁獲量:{_catch}' for _idx,_catch,_date in zip(removed_data[:,0], removed_data[:,2], removed_data[:,3])] # 各自のラベル
        traces.append(go.Scatter(x=removed_data[:,3],  y=removed_data[:,2], text=text,  mode='markers', name='漁に行かなかった類似画像', marker_color='rgba(0,0,0,.8)', marker_size=15))
        removed_data  = list(removed_data)
    else: removed_data = None; removed_idx = []

    # TODO: 選択画像に対してハイライトなどの処理
    s_year, s_month, s_day = str(seed_img[0][3]).split('-') # 選択画像の日付を年、月、日に分解
    print(s_year,s_month,s_day)
    # 画像の類似度辞書を読み込み
    with open(DATABASE_PATH + f'simirality_dicts/256x256x3/{s_year}/{seed_img[0][3]}.pkl', 'rb') as f: simirality_dict = pickle.load(f)

    # 画像の部位毎の類似度のリストが欲しい[[(画像部位1の類似度のリスト) S_11,S_12,...],[(画像部位2の類似度のリスト) S_21,S_22,...],...]
    img_parts_simiralities = np.array([simirality_dict[seed_img[0][3] + f':ssim_psnr_rmse_{img_no}'] for img_no in range(48)])
    print(img_parts_simiralities)
    print(img_parts_simiralities.shape)
    print(positive_idx, negative_idx, removed_idx, len(positive_idx), len(negative_idx), len(removed_idx))
    # print(nn_negative_data)
    # positive_img_parts_simiralities = np.array([d for i,d in enumerate(img_parts_simiralities) if str(i) in set(positive_idx)])
    # negative_img_parts_simiralities = np.array([d for i,d in enumerate(img_parts_simiralities) if str(i) in set(negative_idx)])
    positive_img_parts_simiralities = np.array([img_parts_simiralities[:,idx,:] for idx in positive_idx])
    negative_img_parts_simiralities = np.array([img_parts_simiralities[:,idx,:] for idx in negative_idx])
    print('test', img_parts_simiralities.shape, positive_img_parts_simiralities.shape, negative_img_parts_simiralities.shape)
    # (48, 19, 4) (1, 48, 4) (16, 48, 4)　-> (48, 19, 4) (1, 48, 4) (16, 48, 4)
    if len(positive_img_parts_simiralities)>0:
        positive_img_parts_simiralities = positive_img_parts_simiralities.transpose((1,0,2))
    if len(negative_img_parts_simiralities)>0:
        negative_img_parts_simiralities = negative_img_parts_simiralities.transpose((1,0,2))
    print('test', img_parts_simiralities.shape, positive_img_parts_simiralities.shape, negative_img_parts_simiralities.shape)

    def get_parts_simiralities(img_parts_simiralities):
        '''
        画像の部位ごとに類似度の平均を得る(ポジティブとネガティブで分離して類似度を得たかったので作った)
        Arg:
            img_parts_simiralities: 画像の類似度の二重リスト[[ssim,psnr,rmse,avg],...を画像の部位数分]
        Return:
            simiralities: [[画像の部位連番、類似度の平均],..]の二重リスト(もちろん画像の部位数分ある)
        '''
        ssim = [[i,np.mean(d)] for i,d in enumerate(img_parts_simiralities[:,:,0])]
        psnr = [[i,np.mean(d)] for i,d in enumerate(img_parts_simiralities[:,:,1])]
        rmse = [[i,np.mean(d)] for i,d in enumerate(img_parts_simiralities[:,:,2])]
        avg  = [[i,np.mean(d)] for i,d in enumerate(img_parts_simiralities[:,:,3])]

        # FIXME: psnrで0徐算してinfになることがある 要相談
        scaler = MinMaxScaler() # 0~1で正規化した上で平均を出す
        ssim_scaled = scaler.fit_transform(img_parts_simiralities[:,:,0])
        psnr_scaled = scaler.fit_transform(img_parts_simiralities[:,:,1])
        rmse_scaled = scaler.fit_transform(img_parts_simiralities[:,:,2])
        ssim_scaled = [np.mean(ls) for ls in ssim_scaled]
        psnr_scaled = [np.mean(ls) for ls in psnr_scaled]
        rmse_scaled = [np.mean(ls) for ls in rmse_scaled]
        avg_scaled  = [[i,(s+p+r)/3] for i,(s,p,r) in enumerate(zip(ssim_scaled, psnr_scaled, rmse_scaled))]
        return ssim, psnr, rmse, avg, avg_scaled

    ssim, psnr, rmse, avg, avg_scaled = get_parts_simiralities(img_parts_simiralities)
    # NOTE: ここで類似度に何を使うか選べる
    print('selected metric:', sim_metric)
    if sim_metric=='scaled_avg': simirality_metric = avg_scaled[:]
    elif sim_metric=='ssim': simirality_metric = ssim[:]
    elif sim_metric=='psnr': simirality_metric = psnr[:]
    elif sim_metric=='rmse': simirality_metric = rmse[:]

    simirality_metric.sort(key=lambda x:x[1], reverse=True) # 類似度でソート

    # 最初のマスクかけたやつ
    # マスクをかけたい画像(選択画像)とマスクに使う画像を読み込む
    masked_img = Image.open(DATABASE_PATH + f'satellite_images/{s_year}/7Wc/7Wc_{s_year}{s_month}{s_day}0000.png')
    # def decode2img(date):
    #     ''' FIXME: ここかなりget_img関数と被ってるのでうまく作り直せるはず (本当にそうか？うまくリファクタリングできそうではあるが...)
    #     '''FIXME: そもそもデコードして元の画像に戻せないなぜ？ 戻せればdatasetsの要領をだいぶ削れるんだけど...
    #     con = sqlite3.connect(DATABASE_PATH + 'encoded_img.db')
    #     c   = con.cursor()
    #     c.execute(f'SELECT encoded_img from encoded_imgs WHERE date=="{date}"')        
    #     img_b64data = [list(s)[0] for s in c]
    #     con.close()
        
    #     # とりあえず画像のデータがなかったときはNoneを返してみる
    #     if len(img_b64data)==0:
    #         print('対象画像のデータが見つかりませんでした') # ここすでにピックアップした画像を選んでるはずだから、ありえないと思うけど一応
    #         return None
    #     else:
    #         # image要素のsrc属性に埋め込めこむために、付与した付帯情報を消す
    #         img_b64data = img_b64data[0].replace('data:image/png;base64,', '').encode('utf-8')
    #         img_binary  = base64.b64decode(img_b64data)
    #         img         = np.frombuffer(img_binary, dtype=np.uint8)
    #         masked_img  = Image.fromarray(img)
    #     return masked_img        
    
    # masked_img = decode2img(f'{s_year}-{s_month}-{s_day}')
    # masked_img.show() # TODO: 開けないので結局うまくいってないぽい

    masked_img = masked_img.convert('RGBA')
    mask_img   = Image.open('./static/imgs/green30.png')
    # mask_img = mask_img.filter(ImageFilter.GaussianBlur(100)) # ガウシアンぼかしをかける つもりだけど効果ないっぽい？

    # 画像の部位がどこと対応するのかを見つける 256x256を縦横100ずつずらしていく
    masked_point = [] # 画像の部位とインデックスが対応したマスクする場所の左上のリスト
    for y in range(-(-(913-256)//100)+1):
        for x in range(-(-(700-256)//100)+1):
            masked_point.append([x*100, y*100])

    for i in range(10): # とりあえず全体的な類似度として10箇所マスクしてみる
        x, y = masked_point[simirality_metric[i][0]]
        masked_img.paste(mask_img, (x,y), mask=mask_img) # 謎仕様 Note that if you paste an “RGBA” image, the alpha band is ignored. You can work around this by using the same image as both source image and mask.

    buf = BytesIO()
    masked_img.save(buf, format='png')

    # バイナリデータをbase64でエンコードし、それをさらにutf-8でデコードしておく
    img_b64str = base64.b64encode(buf.getvalue()).decode("utf-8")
    # image要素のsrc属性に埋め込めこむために、適切に付帯情報を付与する
    simirality_masked_img = f'data:image/png;base64,{img_b64str}'

    def make_colorbar(c1,c2,num):
        '''
        類似度の色を見せるためのカラーバーを作成
        c1~c2までの色のグラデーションの要素num個のリストを作成する
        FIXME: このnew_colorsの長さは48を割り切れない数で分割しようとした時はエラー出る仕様です
        '''
        red = Color(c1)
        colors = list(red.range_to(Color(c2), num))
        colors = [c.rgb for c in colors] # カラーコードからRGBに変換
        new_colors = []
        for r,g,b in colors:
            for _ in range(48//num):
                new_colors.append((int(r*256),int(g*256),int(b*256),128))
        return new_colors
    colors = make_colorbar('red', 'blue', 3)

    # TODO: ポジティブかネガティブかで類似度の差をみてみる
    # HACK: ここら辺も関数化したい
    # ポジティブに対しての処理
    if len(positive_img_parts_simiralities)==0:
        positive_masked_img = None
    else:
        ssim, psnr, rmse, avg, avg_scaled = get_parts_simiralities(positive_img_parts_simiralities)
        # NOTE: ここで類似度に何を使うか選べる
        if sim_metric=='scaled_avg': simirality_metric = avg_scaled[:]
        elif sim_metric=='ssim': simirality_metric = ssim[:]
        elif sim_metric=='psnr': simirality_metric = psnr[:]
        elif sim_metric=='rmse': simirality_metric = rmse[:]
        simirality_metric.sort(key=lambda x:x[1], reverse=True) # 類似度でソート

        masked_img = Image.open(DATABASE_PATH + f'satellite_images/{s_year}/7Wc/7Wc_{s_year}{s_month}{s_day}0000.png')
        # masked_img = decode2img(f'{s_year}-{s_month}-{s_day}')
        masked_img = masked_img.convert('RGBA')
        # 矩形描画用の領域を別に作成し、そこに透明度を指定して描画
        sizex, sizey = 32, 32
        rect = Image.new('RGBA', masked_img.size)
        draw = ImageDraw.Draw(rect)
        for i,(idx,simirality) in enumerate(simirality_metric): # すでにソート済みの類似度順に点をつけてく
            # print(i,simirality)
            (x, y), color = masked_point[idx], colors[i]
            x,y = x+123, y+123
            draw.rectangle((x - sizex/2, y - sizey/2 , x + sizex/2, y + sizey/2), fill=color, outline=(0,0,0))
        positive_masked_img = Image.alpha_composite(masked_img, rect)

        buf = BytesIO()
        positive_masked_img.save(buf, format='png')

        img_b64str = base64.b64encode(buf.getvalue()).decode("utf-8")
        positive_masked_img = f'data:image/png;base64,{img_b64str}'

    # ネガティブに対しての処理
    if len(negative_img_parts_simiralities)==0:
        negative_masked_img = None
    else:
        ssim, psnr, rmse, avg, avg_scaled = get_parts_simiralities(negative_img_parts_simiralities)
        # NOTE: ここで類似度に何を使うか選べる
        if sim_metric=='scaled_avg': simirality_metric = avg_scaled[:]
        elif sim_metric=='ssim': simirality_metric = ssim[:]
        elif sim_metric=='psnr': simirality_metric = psnr[:]
        elif sim_metric=='rmse': simirality_metric = rmse[:]
        
        simirality_metric.sort(key=lambda x:x[1], reverse=True) # 類似度でソート

        masked_img = Image.open(DATABASE_PATH + f'satellite_images/{s_year}/7Wc/7Wc_{s_year}{s_month}{s_day}0000.png')
        # masked_img = decode2img(f'{s_year}-{s_month}-{s_day}')
        masked_img = masked_img.convert('RGBA')
        # 矩形描画用の領域を別に作成し、そこに透明度を指定して描画
        sizex, sizey = 32, 32
        rect = Image.new('RGBA', masked_img.size)
        draw = ImageDraw.Draw(rect)
        for i,(idx,simirality) in enumerate(simirality_metric): # すでにソート済みの類似度順に点をつけてく
            (x, y), color = masked_point[idx], colors[i]
            x,y = x+123, y+123
            draw.rectangle((x - sizex/2, y - sizey/2 , x + sizex/2, y + sizey/2), fill=color, outline=(0,0,0))
        negative_masked_img = Image.alpha_composite(masked_img, rect)

        buf = BytesIO()
        negative_masked_img.save(buf, format='png')

        img_b64str = base64.b64encode(buf.getvalue()).decode("utf-8")
        negative_masked_img = f'data:image/png;base64,{img_b64str}'

    # レイアウトの指定
    layout = go.Layout(xaxis=dict(title='日付', type='date', dtick='M6', tickformat='%Y-%m-%d'),
                    yaxis=dict(title='漁獲量(kg)'),
                    xaxis_rangeslider_visible=True,
                    width=1000, height=750,
                    clickmode='select+event',)
                    # yaxis_rangeslider_visible=True)

    fig = dict(data=traces, layout=layout)
    with open('./static/js/plotly_click.js') as f: plotly_click_js = f.read()
    plotly_div = plotly.io.to_html(
        fig,
        include_plotlyjs=True,
        # post_script=plotly_click_js,
        full_html=False, # 完全なHTMLファイルではなく、一つの<div>要素のみを含んだ文字列で埋め込むことにする
    )

    return render_template('img_details.html',
                        message=f'[{place}-{method}-{species}]選択 (類似度はユークリッド距離で計算)',
                        place=place, method=method, species=species, date=date,
                        threshold=str(threshold),
                        sim_metric=sim_metric,
                        num_nn=20, plotly_div=plotly_div,
                        seed_img=seed_img,
                        masked_img=simirality_masked_img,
                        positive_masked_img=positive_masked_img,
                        negative_masked_img=negative_masked_img,
                        # nn_upper_data=nn_upper_data,
                        # nn_middle_data=nn_middle_data,
                        # nn_lower_data=nn_lower_data
                        nn_positive_data=nn_positive_data,
                        nn_negative_data=nn_negative_data,
                        removed_data=removed_data
                        )


if __name__ == "__main__":
    # configparserの宣言とiniファイルの読み込み
    config_ini = configparser.ConfigParser()
    config_ini.read('../config.ini', encoding='utf-8')

    # 環境変数を設定
    DATABASE_PATH = config_ini['DEFAULT']['DATABASE_PATH']
    DB_PATH = config_ini['DEFAULT']['FISH_DB_PATH']

    # 漁港・漁業手法・魚種・画像の辞書形式のデータを読み込む
    with open(DATABASE_PATH+'method_dict.pickle', 'rb') as f: methods_dict = pickle.load(f)
    with open(DATABASE_PATH+'species_dict.pickle', 'rb') as f: species_dict = pickle.load(f)
    with open(DATABASE_PATH+'group_dict.pkl', 'rb') as f: img_group_dict = pickle.load(f)

    np.set_printoptions(suppress=True) # 指数表記にしない
    print('============== Initialized ==============')


    app.run(debug=True)