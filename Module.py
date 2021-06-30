import operator
from multiprocessing import Process
import tushare as ts
from datetime import datetime
import pandas as pd
import numpy as np
import talib
import xlsxwriter
import Atest
import sys
import builtins
import numpy as np
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
def add_already_calclated(df,d_preload,trade_date,col):
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



def create_report( trade_date=20210525,step=2):

    url=fr"Market/CN/ATest/Module/report_{trade_date}.xlsx"
    d_preload_E=DB.preload(step=step)
    df_ts_E = pd.DataFrame()  # index is trade_date
    df_ts_FD = pd.DataFrame()  # index is trade_date
    df_tp_E = pd.DataFrame()  # index is ts_code
    df_tp_FD = pd.DataFrame()  # index is ts_code

    #step 1: put all relevant attributes into one columns
    for cols in ["period","pct_chg","pe_ttm","vol","boll","e_pe_ttm_pct","isminmax","pgain5","pgain20","pgain60","pgain240","total_share","total_mv"]:
        for df in [df_tp_E]:
            add_already_calclated(df=df,d_preload=d_preload_E,trade_date=trade_date,col=cols)

    # step 2: calculate columns
    add_abvma(df=df_tp_E,d_preload=d_preload_E,trade_date=trade_date)
    add_beta(df=df_tp_E,d_preload=d_preload_E,beta_ts_code="000001.SH",asset="I")
    add_beta(df=df_tp_E,d_preload=d_preload_E,beta_ts_code="399001.SZ",asset="I")
    add_beta(df=df_tp_E,d_preload=d_preload_E,beta_ts_code="399006.SZ",asset="I")
    add_offensive_rank(df=df_tp_E,d_preload=d_preload_E)
    add_defensive_rank(df=df_tp_E,d_preload=d_preload_E)

    #step 2: add static data and aggregate them
    df_tp_E=DB.add_static_data(df=df_tp_E,asset=["E"])


    #group by industry
    df_tp_E_grouped = df_tp_E.groupby("sw_industry2").mean()
    df_tp_E_grouped_helper= df_tp_E.groupby("sw_industry2").count()
    df_tp_E_grouped["count"]=df_tp_E_grouped_helper["period"]

    # group by concept
    df_group_concept = pd.DataFrame()
    for group, a_instance in LB.c_d_groups(["E"], market="CN").items():
        if group == "concept":
            for instance in a_instance:
                df_instance = df_tp_E[df_tp_E["concept"].str.contains(instance) == True]
                s = df_instance.mean()
                s["count"] = len(df_instance)
                s.name = instance
                df_group_concept = df_group_concept.append(s, sort=False)
            df_group_concept.index.name = "concept"
            df_group_concept = df_group_concept[["count"] + list(LB.df_to_numeric(df_tp_E).columns)]

    # group by groups





    LB.to_excel(path=url,d_df={"1":df_ts_E,"2":df_ts_FD,"3":df_tp_E,"4":df_tp_E_grouped,"5 concept":df_group_concept,"Time Point FD":df_tp_FD})



create_report(step=1)