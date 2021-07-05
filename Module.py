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
import Crawler as cl
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

@LB.deco_print_name
def add_tp_copy(df, d_preload, trade_date, col):
    for ts_code,df_asset in d_preload.items():
        try:
            df.at[ts_code,col]=df_asset.at[trade_date,col]
        except:
            df.at[ts_code, col] =np.nan

@LB.deco_print_name
def add_abvma(df,d_preload,trade_date,freq=240):
    for ts_code,df_asset in d_preload.items():
        try:
            df.at[ts_code,f"abvma{freq}"]= (df_asset.at[trade_date,"close"] >= df_asset.at[trade_date,f"ma{freq}"]).astype(int)
        except:
            df.at[ts_code, f"abvma{freq}"] =np.nan

@LB.deco_print_name
def add_beta(df,d_preload,beta_ts_code="000001.SH",asset="I"):
    df_beta=DB.get_asset(ts_code=beta_ts_code,asset=asset)
    for ts_code,df_asset in d_preload.items():
        try:
            df.at[ts_code,f"beta_{beta_ts_code}"]= df_beta["close"].corr(df_asset["close"], method="pearson")
        except:
            df.at[ts_code, f"beta_{beta_ts_code}"] =np.nan

@LB.deco_print_name
def add_biggest_drawback(df,d_preload,freq=5):
    for ts_code, df_asset in d_preload.items():
        try:
            df.at[ts_code, f"drawback{freq}"] = LB.biggest_drawback(df_asset=df_asset,n=freq,drawback=True)
        except:
            df.at[ts_code, f"drawback{freq}"] = np.nan

@LB.deco_print_name
def add_offensive_rank(df,d_preload):
    for ts_code, df_asset in d_preload.items():
        df_asset["pct_change"]=1 + df_asset["close"].pct_change()
        try:
            df.at[ts_code, f"geomean_D"] = gmean(df_asset["pct_change"].dropna())
        except:
            df.at[ts_code, f"geomean_D"] = np.nan
            
    df["offensive_rank"] = df["geomean_D"].rank(ascending=False)


@LB.deco_print_name
def add_defensive_rank(df,d_preload):
    a_freqs = [20, 60, 240]
    a_freqs_db = [20, 60]
    for ts_code, df_asset in d_preload.items():
        #abv ma
        for freq in a_freqs:
            df_asset[f"abvma{freq}"]=(df_asset["close"]>=df_asset[f"ma{freq}"]).astype(int)
            df.at[ts_code, f"abvma{freq}[mean]"] = df_asset[f"abvma{freq}"].mean()

        #high and low
        for freq in a_freqs:
            df.at[ts_code, f"high{freq}[mean]"] = len(df_asset[df_asset["isminmax"]==freq])/len(df_asset)
            df.at[ts_code, f"low{freq}[mean]"] = len(df_asset[df_asset["isminmax"]==-freq])/len(df_asset)

        #worst drawback
        for freq in a_freqs_db:
            df.at[ts_code, f"drawback{freq}"] = LB.biggest_drawback(df_asset=df_asset, n=freq, drawback=True)


    df["defensive_rank"] = builtins.sum([df[f"high{freq}[mean]"].rank(ascending=False) for freq in a_freqs]) * (1/7)*1 \
                                 + builtins.sum([df[f"low{freq}[mean]"].rank(ascending=True) for freq in a_freqs]) * (1/7)*2 \
                                 + builtins.sum([df[f"abvma{freq}[mean]"].rank(ascending=False) for freq in a_freqs]) * (1/7)*3 \
                                 + builtins.sum([df[f"drawback{freq}"].rank(ascending=False) for freq in a_freqs_db]) * (1/7)*1
    df["defensive_rank"] = df["defensive_rank"].rank(ascending=True)


@LB.deco_print_name
def add_allround_rank(df,d_preload,oratio=4):
    df[f"allround_rank[{oratio}o{10-oratio}d]"] = df["offensive_rank"]*(oratio)*0.1+df["defensive_rank"]*(10-oratio)*0.1
    df[f"allround_rank[{oratio}o{10-oratio}d]"] = df[f"allround_rank[{oratio}o{10-oratio}d]"].rank(ascending=True)

@LB.deco_print_name
def add_ts_close(df,ts_code,asset):
    df_asset=DB.get_asset(ts_code=ts_code,asset=asset)
    df[f"close_{ts_code}"]=df_asset["close"]


@LB.deco_print_name
def add_ts_copy(df, group, d_preload, name,col,a_clip=[]):


    d_preload={key:value for key,value in d_preload.items() if key in group}

    df[f"{col}_{name}"] =0
    df[f"count"] =0
    for ts_code, df_asset in d_preload.items():
        df_asset[f"{col}"]=df_asset[f"{col}"]
        df_asset[f"count"] = 1

        try:
            if not a_clip:
                df[f"{col}_{name}"]=df[f"{col}_{name}"].add(df_asset[f"{col}"],fill_value=0)
            else:
                df[f"{col}_{name}"]=df[f"{col}_{name}"].add(df_asset[f"{col}"].clip(a_clip[0],a_clip[1]),fill_value=0)
            df[f"count"]=df[f"count"].add(df_asset['count'],fill_value=0)
        except:
            pass
    df[f"{col}_{name}"] = df[f"{col}_{name}"]/df[f"count"]


    del df["count"]



@LB.deco_print_name
def add_ts_abvma(df, group, d_preload, freq, name):
    d_preload={key:value for key,value in d_preload.items() if key in group}

    df[f"abvma{freq}_{name}"] =0
    df[f"count"] =0
    for ts_code, df_asset in d_preload.items():
        df_asset[f"abvma{freq}"]=(df_asset["close"]>=df_asset[f"ma{freq}"]).astype(int)
        df_asset[f"count"] = 1

        df[f"abvma{freq}_{name}"]=df[f"abvma{freq}_{name}"].add(df_asset[f"abvma{freq}"],fill_value=0)
        df[f"count"]=df[f"count"].add(df_asset['count'],fill_value=0)

    df[f"abvma{freq}_{name}"] = df[f"abvma{freq}_{name}"]/df[f"count"]
    del df["count"]




@LB.deco_print_name
def add_ts_macd(df, group, d_preload, name):
    d_preload = {key: value for key, value in d_preload.items() if key in group}

    df[f"macd_{name}"] = 0
    df[f"count"] = 0
    for ts_code, df_asset in d_preload.items():
        df[f"macd_{name}"] = df[f"macd_{name}"].add(df_asset[f"boll"], fill_value=0)
        df[f"count"] = df[f"count"].add(df_asset['count'], fill_value=0)

    df[f"macd_{name}"] = df[f"macd_{name}"] / df[f"count"]
    del df["count"]

@LB.deco_print_name
def add_ts_isminmax(df, group, d_preload, name,freq,ismax=True):
    d_preload = {key: value for key, value in d_preload.items() if key in group}

    if ismax == True:
        minmax_name=f"ismax{freq}"
        pole=1
    else:
        minmax_name=f"ismin{freq}"
        pole = -1
    df[f"{minmax_name}_{name}"] = 0
    for ts_code, df_asset in d_preload.items():
        df_asset_helper=df_asset[df_asset["isminmax"]==freq*pole]
        df[f"{minmax_name}_{name}"] = df[f"{minmax_name}_{name}"].add(df_asset_helper[f"isminmax"].clip(-1,1), fill_value=0)
    df[f"{minmax_name}_{name}"] = df[f"{minmax_name}_{name}"] / df[f"E_count"]


@LB.deco_print_name
def add_ts_closerank(df, group, d_preload, name,freq):
    def rollingRankArgsort(array):
        try:
            return  bd.rankdata(array)[-1]

        except:
            return np.nan

    d_preload = {key: value for key, value in d_preload.items() if key in group}

    df[f"closerank{freq}_{name}"] = 0
    df["count"]=0
    for ts_code, df_asset in d_preload.items():
        df_asset[f"closerank{freq}"]=df_asset["close"].rolling(freq).apply(rollingRankArgsort)
        df_asset["count"]=1

        df[f"closerank{freq}_{name}"] = df[f"closerank{freq}_{name}"].add(df_asset[f"closerank{freq}"], fill_value=0)
        df["count"]=df["count"].add(df_asset[f"count"],fill_value=0)

    df[f"closerank{freq}_{name}"] = df[f"closerank{freq}_{name}"] / df[f"count"]
    del df["count"]

@LB.deco_print_name
def add_general(df_ts, df_trade_date):
    df_ts["lastdayofmonth"]=df_trade_date["lastdayofmonth"].astype(int)
    df_ts["lastdayofseason"]=df_trade_date["lastdayofseason"].astype(int)
    df_ts["lastdayofseason"]=df_trade_date["lastdayofseason"].astype(int)
    df_ts["E_count"]=df_trade_date["E_count"].astype(int)
    df_ts["FD_count"]=df_trade_date["FD_count"].astype(int)


@LB.deco_print_name
def add_ts_holder_trade(df):
    """
            there are 3 ways to see what is the most relevant predictor

            1. the amount of holders sell and buy(=count)
            2. the amunt of values traded(vol X current price) does'nt work because price is not always public, not on tushare at least
            3. ratio of the stocks beeing traded of all this companys share
            """

    df["count"] = 0
    df["holder_trade"] = 0
    df["holder_trade_ratio"] = 0

    df_ts_code = DB.get_ts_code()
    for ts_code in df_ts_code.index:
        print(ts_code,"calc holder trade")
        df_holder_trade = DB.get_asset(ts_code=ts_code, freq="holder_trade")
        if df_holder_trade.empty:
            continue

        df_holder_trade.index = df_holder_trade.index.astype(int)
        df_holder_trade["count"] = 1
        df_holder_trade.index.name = "trade_date"

        df_holder_trade.loc[df_holder_trade["in_de"] == "IN", "helper"] = 1
        df_holder_trade.loc[df_holder_trade["in_de"] == "DE", "helper"] = -1
        df_holder_trade.loc[df_holder_trade["in_de"] == "DE", "change_ratio"] = df_holder_trade.loc[df_holder_trade["in_de"] == "DE", "change_ratio"]* (-1)
        df_holder_trade["index_helper"] = df_holder_trade.index

        df_grouped_mean = df_holder_trade.groupby("index_helper").mean()
        df_grouped_sum = df_holder_trade.groupby("index_helper").sum()

        df["holder_trade"] = df["holder_trade"].add(df_grouped_mean["helper"], fill_value=0)
        df["holder_trade_ratio"] = df["holder_trade_ratio"].add(df_grouped_sum["change_ratio"], fill_value=0)

        df["count"] = df["count"].add(df_grouped_mean["count"], fill_value=0)

    #df["holder_trade_ratio"] = df["holder_trade_ratio"]/df["count"]


    df["holder_trade"] = df["holder_trade"].rolling(10).mean()
    #df["holder_trade_ratio"] = df["holder_trade_ratio"].rolling(10).mean()
    del df["count"]



@LB.deco_print_name
def add_ts_repurchase(df_result,df_trade_date):


    df_append=pd.DataFrame()
    df_trade_date = df_trade_date[df_trade_date.index > 20050101]
    for trade_date in df_trade_date.index:
        df_date = DB.get_date(trade_date=trade_date, freq="repurchase")
        if df_date.empty:
            continue
        df_append = df_append.append(df_date)

    #add market cap
    df_group =df_append.groupby("ann_date").count()
    df_group.index = df_group.index.astype(int)
    df_group.index.name = "trade_date"

    df_group["E_count"] = df_trade_date["E_count"]
    df_group["proc"] = df_group["proc"] / df_group["E_count"]
    df_group["count10"] = df_group["proc"].rolling(10).mean()

    df_result["repurchase_count"]=df_group["count10"]


def create_group_by_market(market="主板"):
    df_ts_code=DB.get_ts_code(a_asset=["E"])
    df_ts_code=df_ts_code[df_ts_code["market"]==market]
    return df_ts_code.index.tolist()

def create_report( trade_date=20210525,step=1):
    url=fr"Market/CN/ATest/Module/report_{trade_date}.xlsx"

    d_report={}
    d_category={"E":["sw_industry2","concept"],
                "FD":[],
                }
    d_preload_cache={}

    # TIMEPOINT
    for asset in ["E","FD"]:
        d_preload = DB.preload(step=step,asset=asset)
        d_preload_cache[asset]=d_preload
        df_tp =pd.DataFrame()# index is ts_code

        #step 1: put all relevant attributes into one columns
        for cols in ["period","pct_chg","pe_ttm","vol","boll","e_pe_ttm_pct","isminmax","pgain5","pgain20","pgain60","pgain240","total_share","total_mv"]:
            for df in [df_tp]:
                add_tp_copy(df=df, d_preload=d_preload, trade_date=trade_date, col=cols)

        # step 2: calculate columns
        add_abvma(df=df_tp,d_preload=d_preload,trade_date=trade_date,freq=20)
        add_abvma(df=df_tp,d_preload=d_preload,trade_date=trade_date,freq=60)
        add_abvma(df=df_tp,d_preload=d_preload,trade_date=trade_date,freq=240)
        add_beta(df=df_tp,d_preload=d_preload,beta_ts_code="000001.SH",asset="I")
        add_beta(df=df_tp,d_preload=d_preload,beta_ts_code="399001.SZ",asset="I")
        add_beta(df=df_tp,d_preload=d_preload,beta_ts_code="399006.SZ",asset="I")
        add_defensive_rank(df=df_tp, d_preload=d_preload)
        add_offensive_rank(df=df_tp,d_preload=d_preload)
        add_allround_rank(df=df_tp,d_preload=d_preload)

        #step 3: add static data and add to report
        df_tp=DB.add_static_data(df=df_tp,asset=[asset])
        d_report[f"TP_{asset}"] = df_tp

        #step 4： group by industry
        for col in d_category[asset]:
            if col == "concept":
                # group by concept
                df_group_concept = pd.DataFrame()
                for group, a_instance in LB.c_d_groups(["E"], market="CN").items():
                    if group == "concept":
                        for instance in a_instance:
                            df_instance = df_tp[df_tp["concept"].str.contains(instance) == True]
                            s = df_instance.mean()
                            s["count"] = len(df_instance)
                            s.name = instance
                            df_group_concept = df_group_concept.append(s, sort=False)
                        df_group_concept.index.name = "concept"
                        df_group_concept = df_group_concept[["count"] + list(LB.df_to_numeric(df_tp).columns)]
                d_report[f"TP_{asset}_concept"] = df_group_concept
            else:
                df_tp_grouped = df_tp.groupby(col).mean()
                df_tp_grouped_helper= df_tp.groupby(col).count()
                df_tp_grouped["count"]=df_tp_grouped_helper["period"]
                d_report[f"TP_{asset}_{col}"] = df_tp_grouped




    # TIME SERIES
    df_trade_date=DB.get_trade_date(end_date=str(trade_date))
    for asset in ["E"]:
        df_ts_master = pd.DataFrame(index=df_trade_date.index.tolist())  # index is trade_date

        #add general E count
        add_general(df_ts_master, df_trade_date=df_trade_date)

        #aggregate by market
        add_ts_close(df_ts_master,ts_code='000001.SH',asset="I")
        add_ts_close(df_ts_master,ts_code='399001.SZ',asset="I")
        add_ts_close(df_ts_master,ts_code='399006.SZ',asset="I")

        # 大宗交易 / block trade
        #df= ak.stock_dzjy_sctj()
        df_dzjy=ak.stock_dzjy_sctj()
        df_dzjy["交易日期"]=df_dzjy["交易日期"].astype(str).str.replace("-","")
        df_dzjy["交易日期"] = df_dzjy["交易日期"].astype(int)
        df_dzjy=df_dzjy.set_index("交易日期",drop=True)
        df_ts_master["block_trade_premium"]=df_dzjy["溢价成交总额占比"]
        df_ts_master["block_trade_premium"]=df_ts_master["block_trade_premium"].rolling(10).mean()

        #margin trade / 融资融券
        stock_margin_sse_df = ak.stock_margin_sse(start_date="00000000", end_date=trade_date)
        stock_margin_sse_df["信用交易日期"] = stock_margin_sse_df["信用交易日期"].astype(int)
        stock_margin_sse_df = stock_margin_sse_df.set_index("信用交易日期", drop=True)
        df_ts_master["融资买入额"] = stock_margin_sse_df["融资买入额"].rolling(10).mean()

        #repurchase / 回购
        add_ts_repurchase(df_result=df_ts_master,df_trade_date=df_trade_date)

        # holder trade / 大股东增减持
        add_ts_holder_trade(df=df_ts_master)


        #http://data.eastmoney.com/zlsj/2021-03-31-2-2.html


        ##人
        #股东人数
        #开户数量
        #融资融券
        #机构研报
        #机构推荐
        #机构持股
        #北向资金
        #股票讨论
        #股票搜索
        #股票关注

        a_groups = [
            ["E", DB.get_ts_code(a_asset=["E"]).index.tolist()],
            ["主板", create_group_by_market("主板")],
            ["中小板", create_group_by_market("中小板")],
            ["创业板", create_group_by_market("创业板")],
                    ]


        optional = False

        # technical: Abv ma (ehlers ma)
        # valuation: Pe_ttm (PEG)
        # people: new registration
        # institution: stock holding
        # economy cycle: Time
        for ginfo in a_groups:
            df_ts=df_ts_master.copy()
            for freq in [20,60,240]:
                add_ts_copy(df=df_ts, group=ginfo[1], name=ginfo[0], d_preload=d_preload_cache["E"],col=f"pgain{freq}")
            add_ts_copy(df=df_ts, group=ginfo[1], name=ginfo[0], d_preload=d_preload_cache["E"], col="pe_ttm", a_clip=[0,200])

            add_ts_abvma(df=df_ts, group=ginfo[1], name=ginfo[0], d_preload=d_preload_cache["E"], freq=60)
            add_ts_abvma(df=df_ts, group=ginfo[1], name=ginfo[0], d_preload=d_preload_cache["E"], freq=240)
            #add_ts_abvma(df=df_ts, group=ginfo[1], name=ginfo[0], d_preload=d_preload_cache["E"], freq=500)


            # add_ts_macd(df=df_ts, group=a_group_all[1],name=groupinfo[0], d_preload=d_preload_cache["E"], freq=240)
            # todo find a way to use other freq to boll and macd
            # todo is bolline contracting or expanding
            if optional:
                add_ts_closerank(df=df_ts, group=ginfo[1], name=ginfo[0], d_preload=d_preload_cache["E"], freq=60)
                add_ts_closerank(df=df_ts, group=ginfo[1], name=ginfo[0], d_preload=d_preload_cache["E"], freq=240)
                add_ts_closerank(df=df_ts, group=ginfo[1], name=ginfo[0], d_preload=d_preload_cache["E"], freq=500)

                add_ts_copy(df=df_ts, group=ginfo[1], name=ginfo[0], d_preload=d_preload_cache["E"], col="boll")

                add_ts_isminmax(df=df_ts, group=ginfo[1],name=ginfo[0], d_preload=d_preload_cache["E"], ismax=True, freq=20)
                add_ts_isminmax(df=df_ts, group=ginfo[1],name=ginfo[0], d_preload=d_preload_cache["E"], ismax=True, freq=60)
                add_ts_isminmax(df=df_ts, group=ginfo[1],name=ginfo[0], d_preload=d_preload_cache["E"], ismax=True, freq=240)
                add_ts_isminmax(df=df_ts, group=ginfo[1], name=ginfo[0], d_preload=d_preload_cache["E"], ismax=False, freq=20)
                add_ts_isminmax(df=df_ts, group=ginfo[1], name=ginfo[0], d_preload=d_preload_cache["E"], ismax=False, freq=60)
                add_ts_isminmax(df=df_ts, group=ginfo[1], name=ginfo[0], d_preload=d_preload_cache["E"], ismax=False, freq=240)

            d_report[f"TS_{ginfo[0]}"] =df_ts


    LB.to_excel(path=url,d_df=d_report)







trade_date=LB.today()
#DB.update_all_in_one_cn(night_shift=True)
create_report(step=1,trade_date=trade_date)