import operator
from multiprocessing import Process
import tushare as ts
from datetime import datetime
import pandas as pd
import numpy as np
import talib
import xlsxwriter
import Atest
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

def create_daily_report(trade_date=-1,update_DB=False ,market="CN"):
    """
    this function creates an daily overview of the most important stocks and FD

    :param with_db_update:
    :return:
    """

    def helper_function(df_column,column,d_gain,df_gain_summary,p_setting=[0, 0.25, 0.5, 0.75, 1]):
        """
        this helper function takes a df, column , df_gain

        and returns how a group of stocks has performed
        it's complicated to explain

        :return:
        """
        # calculate the 4 quantile based on the column
        d_quantile = LB.custom_quantile(df_column, column, p_setting=p_setting, key_val=False)

        # calculate mean gain in last n ndays based on the quantile
        for key, df_quantile in d_quantile.items():
            for freq in a_gain_freq:
                df_gain = d_gain[freq]
                print(f"trade_date {trade_date},column {column}, key {key} , freq {freq}")

                # df_gain=df_gain.loc[df_quantile.index,list(df_gain.columns)]
                df_quantile[f"gain{freq}"] = df_gain[f"gain{freq}"]
                df_gain_summary.at[f"{column}_q{key}", f"gain{freq}"] =what= df_quantile[f"gain{freq}"].mean()


    # function start
    if trade_date==-1:
        trade_date = LB.latest_trade_date(market=market)
    if os.path.isfile(f"D:\Stock/Market/{market}/Report/report_{trade_date}.xlsx"):
        print(f"REPORT {trade_date} EXISTS!")
        return

    # init
    d_df = {}
    a_period={"Market":[(0,9999999)],"I":[(1200,9999999)],"FD":[(700,1200),(1200,9999999)],"E":[(700,1200),(1200,9999999)]}
    top={"I":0.10,"FD":0.12,"E":0.06}
    sortby={"I":"allround_rank_geo","FD":"allround_rank_geo","E":"allround_rank_geo"}

    #update if nessesary
    if update_DB:
        if market=="CN":
            DB.update_all_in_one_cn_v2()
        if market=="HK":
            DB.update_all_in_one_hk()

    if not os.path.isfile(f"Market/{market}/ATest/bullishness/bullishness_{market}_0_{trade_date}.xlsx"):
        Atest.asset_bullishness(a_asset=["E", "FD", "I"], step=1, market=market, end_date=trade_date)

    xls = pd.ExcelFile(f"Market/{market}/ATest/bullishness/bullishness_{market}_0_{trade_date}.xlsx")
    df_bullishness_master = pd.read_excel(xls, sheet_name="Overview")
    try:
        df_bullishness_master = df_bullishness_master[["ts_code", "period", "offensive_rank", "defensive_rank", "allround_rank_ari", "allround_rank_geo","qdii_research/period", "qdii_grade/period","M_std","000001.SH_beta","399001.SZ_beta","399006.SZ_beta","asset"]]
    except:
        df_bullishness_master = df_bullishness_master[["ts_code", "period", "offensive_rank", "defensive_rank", "allround_rank_ari", "allround_rank_geo","M_std","000001.SH_beta","399001.SZ_beta","399006.SZ_beta","asset"]]

    df_bullishness_master.set_index("ts_code", drop=True, inplace=True)

    #generate report
    #market overview
    a_gain_freq = [5,10,20,60,120,240]
    p_setting=[0,0.2,0.4,0.6,0.8, 1]
    df_trade_date=DB.get_trade_date()
    df_gain_summary=pd.DataFrame()

    #load all stocks and get their PE in table
    d_preload = DB.preload(asset="E",market=market)

    #create table of how each stock gained last n days
    d_gain = {}
    for freq in a_gain_freq:
        df_gain_freq=pd.DataFrame()
        for ts_code, df_asset in d_preload.items():
            try:
                df_gain_freq.at[ts_code,f"gain{freq}"]=df_asset["close"].iat[-freq]/df_asset["close"].iat[-1]
            except Exception as e :
                print(e)
        d_gain[freq]=df_gain_freq

    #group gain by pe, pb
    if market =="CN":
        for column in ["pe_ttm","pb"]:
            print(f"calculate {column}")
            df_column = pd.DataFrame()
            for ts_code, df_asset in d_preload.items():
                df_column.at[ts_code,column]=df_asset[column].iat[-1]
            helper_function(df_column=df_column, column=column, d_gain=d_gain, df_gain_summary=df_gain_summary, p_setting=p_setting)

    # group gain by
    # offensive or defensive
    # qdii grade and research
    # D_std, M_std, S_std,
    # Beta with 3 Index
    for column in ["allround_rank_geo","offensive_rank","defensive_rank","qdii_research/period", "qdii_grade/period","M_std","000001.SH_beta","399001.SZ_beta","399006.SZ_beta"]:
        print(f"calculate {column}")
        if column in df_bullishness_master.columns:
            df_column = df_bullishness_master[df_bullishness_master["asset"] == "E"]
            helper_function(df_column=df_column, column=column, d_gain=d_gain, df_gain_summary=df_gain_summary, p_setting=p_setting)


    # add the df to final report excel
    d_df[f"gain"] = df_gain_summary


    #stock in focus
    d_a_assets={"CN":["Market","I","FD","E"], "HK":["E"]}
    for asset in  d_a_assets[market]:
        for min_period, max_period in a_period[asset]:
            df_bullishness = df_bullishness_master if asset!="Market" else df_bullishness_master[["period" ]]

            # Select best stocks and by rank
            if asset =="Market":
                df_selected_assets = df_bullishness.loc[["000001.SH","399006.SZ","399001.SZ"]]
                #d_preload_filter = DB.preload(asset="I", d_queries_ts_code={"I": [f"ts_code in {df_selected_assets.index.to_list()}"]})
                d_preload_filter = {ts_code: df_asset for ts_code, df_asset in d_preload.items() if ts_code in df_selected_assets.index}

            elif asset in ["I","FD","E"]:
                df_selected_assets = df_bullishness[(df_bullishness["asset"] == asset) & (df_bullishness["period"] >= min_period) & (df_bullishness["period"] < max_period)].sort_values(by=sortby[asset])
                df_selected_assets = df_selected_assets.head(int(df_trade_date.at[trade_date,"E_count"] * top[asset]))
                #d_preload_filter=DB.preload(asset=asset,market=market,d_queries_ts_code={asset: [f"ts_code in {df_selected_assets.index.to_list()}"]})
                d_preload_filter= {ts_code:df_asset for ts_code,df_asset in d_preload.items() if ts_code in df_selected_assets.index}


            #load individual df_asset
            for ts_code,df_asset in d_preload_filter.items():

                #pct_chg STD => if two stock gmean are same(2*5 or 1*10), which one is more volatile?

                # NOW vs historic abs price
                for column in ["close"]:
                    for freq in [20,60,240]:
                        df_gain = df_asset.tail(freq)
                        df_selected_assets.at[ts_code, f"{column}_{freq}"] = (((1 - 0) * (df_gain[f"{column}"].iat[-1] - df_gain[f"{column}"].min())) / (df_gain[f"{column}"].max() - df_gain[f"{column}"].min())) + 0

                # Bollinger NOW on D and on W
                #calculate bollinger completely
                for freq in ["D","W"]:
                    # transform to D or W
                    df_asset_freq = LB.df_to_freq(df_asset, freq=freq)

                    # bollinger
                    df_asset_freq[f"boll_up"], df_asset_freq[f"boll_mid"], df_asset_freq[f"boll_low"] = talib.BBANDS(df_asset_freq["close"], 20, 2, 2)

                    # scale to between 0 and 1
                    df_asset_freq[f"boll_scale"] = (((1 - 0) * (df_asset_freq["close"] - df_asset_freq[f"boll_low"])) / (df_asset_freq[f"boll_up"] - df_asset_freq[f"boll_low"])) + 0

                    # take the last sample as NOW
                    df_selected_assets.at[ts_code, f"boll_NOW{freq}"] = df_asset_freq[f"boll_scale"].iat[-1]

                # NOW vs historic PE and PB
                if asset == "E" and market=="CN":
                    for column in ["pe_ttm", "pb"]:
                        df_selected_assets.at[ts_code, f"{column}_ALL"] = (df_asset[f"{column}"].iat[-1]>=df_asset[f"{column}"]).astype(int).mean()
                        df_selected_assets.at[ts_code, f"{column}_NOW"] = df_gain[f"{column}"].iat[-1]


            #after all individual assets are finished calculated
            # opportunity rank - short term rank - volatility
            if asset in ["E","FD","I"]:
                df_selected_assets["opportunity_rank"] = df_selected_assets["close_20"] * 0.38 * 0.33 \
                                                         + df_selected_assets["close_60"] * 0.38 * 0.33 \
                                                         + df_selected_assets["close_240"] * 0.38 * 0.33 \
                                                         + df_selected_assets["boll_NOWD"] * 0.62 * 0.38 \
                                                         + df_selected_assets["boll_NOWW"] * 0.62 * 0.62
                df_selected_assets["opportunity_rank"] = df_selected_assets["opportunity_rank"].rank(ascending=True)

                #long term rank adjusted
                #df_selected_assets["opportunity_rank"] = df_selected_assets["opportunity_rank"]*df_selected_assets["allround_rank_geo"]
                #df_selected_assets["opportunity_rank"] =df_selected_assets["opportunity_rank"] .rank(ascending=True)


            # investment rank - long term rank - value
            if asset in ["E"] and market in ["CN"]:
                df_selected_assets["investment_rank"] = df_selected_assets["pe_ttm_ALL"] * 0.62 \
                                                        + df_selected_assets["pb_ALL"] * 0.38
                df_selected_assets["investment_rank"] = df_selected_assets["investment_rank"].rank(ascending=True)

                # long term rank adjusted
                #df_selected_assets["investment_rank"] = df_selected_assets["investment_rank"] * df_selected_assets["allround_rank_geo"]
                #df_selected_assets["investment_rank"] = df_selected_assets["investment_rank"].rank(ascending=True)

            #add static data
            if asset in ["E","FD","I"]:
                df_selected_assets=df_selected_assets.loc[:, df_selected_assets.columns != 'asset']
                df_selected_assets = DB.add_static_data(df_selected_assets, asset=[asset],market=market)
            d_df[f"{asset}_{min_period}"] = df_selected_assets

    LB.to_csv_feather(pd.DataFrame(),a_path=LB.a_path(f"Market/{market}/Report/folder"))
    LB.to_excel(path=f"Market/{market}/Report/report_{trade_date}.xlsx",d_df=d_df,color=True)
    LB.file_open(f"D:\Stock/Market/{market}/Report/report_{trade_date}.xlsx")




if __name__ == '__main__':
    #TODO: when do stock revert? volume, time to previous date, ma, market, boll , supportresistance
    for market in ["HK"]:
        create_daily_report(update_DB=False,market=market)

    """df_trade_date=DB.get_trade_date()
    df_trade_date=df_trade_date[df_trade_date["lastdayofseason"]==True]
    for trade_date in df_trade_date.index[::-7]:
        print(f"CALCULATE REPORT for trade_date {trade_date}")
        create_daily_report(update_DB=False,trade_date=trade_date)
"""