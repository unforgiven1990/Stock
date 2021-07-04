import operator
from multiprocessing import Process
import tushare as ts
from datetime import datetime
import pandas as pd
import numpy as np
import talib
import xlsxwriter
import Atest
import bottleneck as bd
import sys
import builtins
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt

from scipy.signal import argrelextrema
import threading
import smtplib
from email.message import EmailMessage
import math
import re
import os
import itertools
from win32com.client import Dispatch
import traceback
import _API_Tushare
import atexit
import Alpha
from time import time, strftime, localtime
import subprocess, os, platform
from datetime import timedelta
from playsound import playsound
from numba import jit
import numba
import enum
import pathlib
import time
import DB
import LB
import time
import Infographic
from scipy.stats import gmean
import random
import re
import numpy as np
import requests
import threadpool
from bs4 import BeautifulSoup

from selenium import webdriver



USER_AGENTS = [
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
    "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
    "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
    "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
    "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
    "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
    "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
]

driver = webdriver.Firefox(executable_path='geckodriver.exe')
driver.set_page_load_timeout(10)

def crawler(link, func,use_gecko=True):
    def create_headers():
        headers = dict()
        headers["User-Agent"] = random.choice(USER_AGENTS)
        #headers["Referer"] = "http://www.{0}.com".format(SPIDER_NAME)
        return headers
    def random_delay():
        randomdelay = random.randint(2, 10)
        time.sleep(randomdelay)

    headers = create_headers()
    random_delay()


    if use_gecko: #with selenium firefox
        driver.get(link)
        html = driver.page_source

    else:  #with natural html request
        pass
        response = requests.get(link, timeout=15, headers=headers)
        html = response.content


    #page is successfully loaded and not been blocked by beike
    soup = BeautifulSoup(html, "lxml")
    return func(soup)


def func_stock_holder_jrj(soup):
    a_trade_dates=[]
    a_gudong=[]
    a_shidagudong=[]
    a_qdii=[]
    a_average_holdnumber=[]
    try:
        tdbody = soup.find_all('tbody')[0]
        a_tr = tdbody.find_all('tr')
        for tr in a_tr:
            a_td = tr.find_all('td')
            a_trade_dates+=[a_td[0].text]
            a_gudong+=[a_td[1].text]
            a_average_holdnumber+=[a_td[5].text]
            a_shidagudong+=[a_td[7].text]
            a_qdii+=[a_td[11].text]
    except:
        return

    a_trade_dates=[int(str(x).replace("-","")) for x in a_trade_dates]
    df_result=pd.DataFrame(index=a_trade_dates)
    df_result["new holder"]=a_gudong
    df_result["top_10_ratio"]=a_shidagudong
    df_result["average_person_holds_count"]=a_average_holdnumber
    df_result["qdii_hold"]=a_qdii
    df_result=df_result.reindex(index=df_result.index[::-1])
    df_result.index.name="trade_date"
    return df_result


def main():

    df_ts_code=DB.get_ts_code()
    for ts_code in df_ts_code.index[::1]:

        for _ in range(3):
            try:
                link = f"http://stock.jrj.com.cn/share,{str(ts_code)[0:6]},gdhs.shtml"
                print(link)
                df_result = crawler(link=link, func=func_stock_holder_jrj, use_gecko=True)
                if not df_result.empty:
                    a_path=LB.a_path(f"Market/CN/Asset/E/sanhu_ratio/{ts_code}")
                    LB.to_csv_feather(df=df_result,a_path=a_path,skip_csv=True)
                break
            except :
                pass



def onetime_test():
    df_trade_date=DB.get_trade_date()
    df_trade_date=df_trade_date[df_trade_date["lastdayofseason"]==1]

    df_asset = DB.get_asset(ts_code="000004.SZ", freq="sanhu_ratio")
    df_result=pd.DataFrame(index=df_asset.index)
    df_average=pd.DataFrame()
    df_result["count"]=0
    df_result["qdii_hold"] =0
    df_result["top_10_ratio"] =0
    df_ts_code=DB.get_ts_code()

    for ts_code in df_ts_code.index:
        print(ts_code)
        try:
            df_asset=DB.get_asset(ts_code=ts_code,freq="sanhu_ratio")

            df_asset["qdii_hold"]=df_asset["qdii_hold"].str.replace("%","")
            df_asset["qdii_hold"]=df_asset["qdii_hold"].str.replace("--","0").astype(float)

            df_asset["top_10_ratio"] = df_asset["top_10_ratio"].str.replace("%", "")
            df_asset["top_10_ratio"] = df_asset["top_10_ratio"].str.replace("--", "0").astype(float)

            df_result["qdii_hold"]=df_result["qdii_hold"].add(df_asset["qdii_hold"],fill_value=0)
            df_result["top_10_ratio"]=df_result["top_10_ratio"].add(df_asset["top_10_ratio"],fill_value=0)

            df_asset["count"] = 1
            df_result["count"] = df_result["count"].add(df_asset["count"],fill_value=0)

            df_average.at[ts_code,"qdii_hold"]=df_asset["qdii_hold"].mean()
            df_average.at[ts_code,"top_10_ratio"]=df_asset["top_10_ratio"].mean()
        except:
            pass
    df_result["qdii_hold"] = df_result["qdii_hold"]/df_result["count"]
    df_result["top_10_ratio"] = df_result["top_10_ratio"]/df_result["count"]

    df_result.to_csv("test.csv",encoding="utf-8_sig")

    df_average = DB.add_static_data(df=df_average, asset=["E"])
    df_average.to_csv("df_average.csv",encoding="utf-8_sig")

