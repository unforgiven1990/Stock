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
import pandas as pd
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



"""Report 2.0 is to create a new report based on Excel, Pivot table, Dashboard"""

a_ideal_cols = ["ts_code","asset", "period", "qdii_research/period","qdii_grade/period","pe_ttm", "pb", "total_mv", "pe_ttm_ALL", "pb_ALL", "close", "hk_hold", "hk_hold_ALL", "tech_off_rank", "tech_def_rank", "allround_rank_geo", "qdii_off_rank", "qdii_def_rank", "qdii_ratio_ls_mom", "qdii_ratio_ls", "M_std", "000001.SH_beta", "399001.SZ_beta", "399006.SZ_beta","head_sw_industry1","head_sw_industry2","head_sw_industry3"] + [f"pgain{freq}" for freq in [5, 20, 60, 240]] + [f"fgain{freq}" for freq in [5, 20, 60, 240]] + [f"qdii_research{freq}" for freq in [60]] + [f"qdii_grade{freq}" for freq in [60]]

a_freq = [20]

def atom_rank(df_asset):
    """calculate a part of the stock bullishness"""


    # INFO PCT_CHG std: (If two stock have same gmean, which one is more volatile?)

    df_asset["pct_change"] = 1 + df_asset["close"].pct_change()

    # RANK Geomean: implcitly reward stock with high monotony and punish stock with high volatilty.

    d_result={}
    d_result["geomean"]= gmean(df_asset["pct_change"].dropna())
    d_result["absgain"]= df_asset["close"].iat[-1]/df_asset["close"].iat[0]

    # RANK technical freqhigh = ability to create 20d,60d,120d,240high
    for freq in a_freq:
        df_asset[f"rolling_max{freq}"] = df_asset["close"].rolling(freq).max()
        df_helper = df_asset.loc[df_asset[f"rolling_max{freq}"] == df_asset["close"]]
        df_asset[f"{freq}high"] = df_helper[f"rolling_max{freq}"]
        d_result[f"{freq}high"] = df_asset[f"{freq}high"].clip(0, 1).sum() / len(df_asset)

    # RANK technical freqlow = ability to avoid 20d,60d,120d,240low
    for freq in a_freq:
        df_asset[f"rolling_min{freq}"] = df_asset["close"].rolling(freq).min()
        df_helper = df_asset.loc[df_asset[f"rolling_min{freq}"] == df_asset["close"]]
        df_asset[f"{freq}low"] = df_helper[f"rolling_min{freq}"]
        d_result[f"{freq}low"]= df_asset[f"{freq}low"].clip(0, 1).sum() / len(df_asset)

    # RANK check how long a stock is abv ma 20,60,120,240
    for freq in a_freq:
        abvma_name = Alpha.abv_ma(df=df_asset, abase="close", freq=freq, inplace=True)
        d_result[f"abv_ma{freq}"] = df_asset[abvma_name].mean()


    return d_result



def bullishness_by_freq(step=1):
    """
    Excel Table row: ts_code
    Excel Table column: Season:
    Excel Table content: Geo_rank, offensive Rank, Defensive Rank

    :return:
    """
    df_ts_code=DB.get_ts_code(a_asset=["FD"])
    d_results={"geomean":pd.DataFrame(),"absgain":pd.DataFrame()}
    for freq in a_freq:
        d_results[f"{freq}high"]=pd.DataFrame()
        d_results[f"{freq}low"]=pd.DataFrame()
        d_results[f"abv_ma{freq}"]=pd.DataFrame()

    df_trade_date=DB.get_trade_date()
    granularity="season"# month, season,year
    df_trade_date=df_trade_date[df_trade_date[f"lastdayof{granularity}"]==True]
    df_trade_date=df_trade_date[df_trade_date["year"]>2010]
    a_dates=[x for x in df_trade_date.index]

    #create dummy columns
    for min_date, max_date in LB.custom_pairwise_overlap(a_dates):
        for egal,df_result_part in d_results.items():
            df_result_part[f"{min_date}_{max_date}"]=np.nan

    for ts_code in df_ts_code.index[::step]:
        print("bullishness_by_freq ",ts_code)
        df_asset=DB.get_asset(ts_code=ts_code,asset="FD")
        if df_asset.empty:
            print(f"{ts_code} is all empty")
            continue
        df_asset["indexhelper"] = df_asset.index

        for min_date, max_date in LB.custom_pairwise_overlap(a_dates):

            df_asset_freq=df_asset[ (df_asset["indexhelper"]> min_date) & (df_asset["indexhelper"]<= max_date)]
            if df_asset_freq.empty:
                continue
            d_freq_result=atom_rank(df_asset_freq)
            for key, df_result_part in d_results.items():
                df_result_part.at[ts_code,f"{min_date}_{max_date}"]=d_freq_result[key]


    a_columns=[]
    for name in d_freq_result.keys():
        a_columns+=[f"{name}_mean"]
        a_columns+=[f"{name}_std"]

    df_overview=pd.DataFrame(columns=a_columns,index=d_results["geomean"].index)
    d_results_pct={}
    #rank stock by date
    for name, df in d_results.items():
        if "low" in name:
            ascending = False
        else:
            ascending = True
        df_name_pct = df.rank(axis=0, ascending=ascending, pct=True)
        df_name_pct["mean"]=df_overview[f"{name}_mean"]=df_name_pct.mean(axis=1)
        df_name_pct["std"]=df_overview[f"{name}_std"]=df_name_pct.std(axis=1)
        d_results_pct[name]=df_name_pct




    # add overview
    df_geomean=d_results["geomean"]
    s=df_geomean.notnull().astype(int).sum(axis=1)
    df_overview["period"]=s
    df_overview["link"]=df_overview.index
    df_overview["link"] =df_overview["link"].str.slice(0,6)
    df_overview["link"]=f"https://fund.eastmoney.com/"+df_overview["link"]+".html"
    df_overview=df_overview[["period","link"]+[x for x in df_overview if x not in ["period","link"] ]]
    df_overview["offensive"]= df_overview["absgain_mean"].rank(ascending=False)
    df_overview["defensive"]= builtins.sum([df_overview[f"{freq}high_mean"].rank(ascending=False) for freq in a_freq]) * 0.38 * 0.38 \
                            + builtins.sum([df_overview[f"{freq}low_mean"].rank(ascending=True) for freq in a_freq]) * 0.38 * 0.62 \
                            + builtins.sum([df_overview[f"abv_ma{freq}_mean"].rank(ascending=False) for freq in a_freq]) * 0.62
    df_overview["defensive"] =df_overview["defensive"].rank(ascending=True)
    df_overview["allround"]=df_overview["offensive"]+df_overview["defensive"]+df_overview["defensive"]

    df_overview=DB.add_static_data(df_overview,asset=["FD"])


    d_results_pct["overview"] = df_overview

    LB.to_excel(f"absolute{step}.xlsx",d_df=d_results)
    LB.to_excel(f"percent{step}.xlsx",d_df=d_results_pct)

    return







def create_dashboard():
    df_dashboard=pd.DataFrame()
    trade_date=LB.today()
    trade_date="20210618"
    market="CN"

    if not os.path.isfile(f"Market/{market}/ATest/bullishness/bullishness_{market}_0_{trade_date}.xlsx"):
        Atest.asset_bullishness(a_asset=["E", "FD", "I"], step=1, market=market, end_date=trade_date)
    xls = pd.ExcelFile(f"Market/{market}/ATest/bullishness/bullishness_{market}_0_{trade_date}.xlsx")
    df_bullishness_overview_original = pd.read_excel(xls, sheet_name="Overview")
    df_bullishness_overview_master = df_bullishness_overview_original.copy()
    df_bullishness_overview_master.set_index("ts_code", drop=True, inplace=True)

    df_dashboard=df_bullishness_overview_master
    excel_path = f"Market/{market}/Dashboard/Dashboard_{trade_date}.xlsx"
    LB.to_excel(path=excel_path,d_df={"Dashboard":df_dashboard})


def update_holder_number():
    """update holder number for each stock because tushare holder number is inaccurate"""

    pass


bullishness_by_freq(step=1)