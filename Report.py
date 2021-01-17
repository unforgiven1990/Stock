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

    def helper_function(df_column, column, d_pgain, df_pgain_summary,df_fgain_summary, p_setting=[0, 0.25, 0.5, 0.75, 1]):
        """
        this helper function takes a df, column , df_pgain

        and returns how a group of stocks has performed
        it's complicated to explain
        """
        # calculate the 4 quantile based on the column
        d_quantile = LB.custom_quantile(df_column, column, p_setting=p_setting, key_val=False)

        # calculate mean gain in last n ndays based on the quantile
        for key, df_quantile in d_quantile.items():
            for freq in a_gain_freq:
                df_pgain = d_pgain[freq]
                df_quantile[f"pgain{freq}"] = df_pgain[f"pgain{freq}"]
                df_pgain_summary.at[f"{column}_q{key}", f"pgain{freq}"] = df_quantile[f"pgain{freq}"].mean()

                df_fgain = d_fgain[freq]
                df_quantile[f"fgain{freq}"] = df_fgain[f"fgain{freq}"]
                df_fgain_summary.at[f"{column}_q{key}", f"fgain{freq}"] = df_quantile[f"fgain{freq}"].mean()



    # 1. update DB if nessesary
    if update_DB:
        if market == "CN":
            DB.update_all_in_one_cn_v2()
        if market == "HK":
            DB.update_all_in_one_hk()


    # 2. skip if report already exists
    if trade_date == -1:
        trade_date = LB.latest_trade_date(market=market)
    if os.path.isfile(f"D:\Stock/Market/{market}/Report/report_{trade_date}.xlsx"):
        print(f"REPORT {trade_date} EXISTS!")
        return





    # 3. get bullishness df, if not exist, create one
    if not os.path.isfile(f"Market/{market}/ATest/bullishness/bullishness_{market}_0_{trade_date}.xlsx"):
        Atest.asset_bullishness(a_asset=["E", "FD", "I"], step=1, market=market, end_date=trade_date)
    xls = pd.ExcelFile(f"Market/{market}/ATest/bullishness/bullishness_{market}_0_{trade_date}.xlsx")
    df_bullishness_overview_master = pd.read_excel(xls, sheet_name="Overview")
    try:
        df_bullishness_overview_master = df_bullishness_overview_master[["ts_code", "period", "pe_ttm","pb","total_mv","pe_ttm_ALL","pb_ALL","offensive_rank", "defensive_rank", "allround_rank_geo","qdii_rank","qdii_research_mom","qdii_grade_mom","qdii_research/period", "qdii_grade/period","M_std","000001.SH_beta","399001.SZ_beta","399006.SZ_beta","asset"]+[f"pgain{freq}" for freq in [5,20,60,240]]+[f"fgain{freq}" for freq in [5,20,60,240]]+[f"qdii_research{freq}" for freq in [20,60,240]]+[f"qdii_grade{freq}" for freq in [20,60,240]]]
    except:
        df_bullishness_overview_master = df_bullishness_overview_master[["ts_code", "period", "offensive_rank", "defensive_rank", "allround_rank_geo","M_std","000001.SH_beta","399001.SZ_beta","399006.SZ_beta","asset"]+[f"pgain{freq}" for freq in [5,20,60,240]]+[f"fgain{freq}" for freq in [5,20,60,240]]]
    df_bullishness_overview_master.set_index("ts_code", drop=True, inplace=True)


    # 4. settings
    d_df = {}
    a_period = {"Market": [(0, 9999999)], "I": [(1200, 9999999)], "FD": [(700, 1200), (1200, 9999999)], "E": [(700, 1200), (1200, 9999999)]}
    top = {"I": 0.10, "FD": 0.12, "E": 0.06}
    a_gain_freq = [5,20,60,240]
    p_setting=[0,0.2,0.4,0.6,0.8, 1]
    df_trade_date=DB.get_trade_date()
    df_pgain_summary=pd.DataFrame()
    df_fgain_summary=pd.DataFrame()
    d_pgain = {}
    d_fgain = {}
    #d_preload = DB.preload(asset="E",market=market,query_df=f"trade_date <= {trade_date}")


    #5. create table of how each stock gained last n days TODO REMOVE

    """for freq in a_gain_freq:
        df_pgain_freq=pd.DataFrame()
        for ts_code, df_asset in d_preload.items():
            try:
                df_pgain_freq.at[ts_code,f"gain{freq}"]=df_asset["close"].iat[-1]/df_asset["close"].iat[-freq]
            except Exception as e :
                print(e)
        d_pgain[freq]=df_pgain_freq"""

    for freq in a_gain_freq:
        df_pgain_freq = df_bullishness_overview_master[[f"pgain{freq}"]]
        d_pgain[freq] = df_pgain_freq

        df_fgain_freq = df_bullishness_overview_master[[f"fgain{freq}"]]
        d_fgain[freq] = df_fgain_freq

    #market overview

    # 6. group gain by
    for column in ["pe_ttm","pb","pe_ttm_ALL","pb_ALL","close","total_mv","qdii_rank","qdii_research_mom","qdii_grade_mom","qdii_research/period", "qdii_grade/period","M_std","000001.SH_beta","399001.SZ_beta","399006.SZ_beta""allround_rank_geo","offensive_rank","defensive_rank"]+[f"qdii_research{freq}" for freq in [20,60,240]]+[f"qdii_grade{freq}" for freq in [20,60,240]]:
        print(f"calculate {column}")
        if column in df_bullishness_overview_master.columns:
            df_column = df_bullishness_overview_master[df_bullishness_overview_master["asset"] == "E"]
            helper_function(df_column=df_column, column=column, d_pgain=d_pgain, df_pgain_summary=df_pgain_summary,df_fgain_summary=df_fgain_summary, p_setting=p_setting)

    # add the df to final report excel
    df_pgain_summary.index.name="group"
    df_fgain_summary.index.name="group"
    d_df[f"pgain"] = df_pgain_summary
    d_df[f"fgain"] = df_fgain_summary


    #7. stock in focus
    d_a_assets={"CN":["Market","I","FD","E"], "HK":["E"]}
    for asset in  d_a_assets[market]:
        print(market,asset)
        for min_period, max_period in a_period[asset]:
            df_bullishness_overview = df_bullishness_overview_master if asset!="Market" else df_bullishness_overview_master[["period" ]]

            # Select best stocks and by rank
            if asset =="Market":
                df_selected_assets = df_bullishness_overview.loc[["000001.SH","399006.SZ","399001.SZ"]]
                d_preload_filter = DB.preload(asset="I", d_queries_ts_code={"I": [f"ts_code in {df_selected_assets.index.to_list()}"]},query_df=f"trade_date <= {trade_date}")

            elif asset in ["I","FD","E"]:
                print("zes")
                df_selected_assets = df_bullishness_overview[(df_bullishness_overview["asset"] == asset) & (df_bullishness_overview["period"] >= min_period) & (df_bullishness_overview["period"] < max_period)].sort_values(by="allround_rank_geo")
                df_selected_assets = df_selected_assets.head(int(df_trade_date.at[trade_date,"E_count"] * top[asset]))
                d_preload_filter=DB.preload(asset=asset,market=market,d_queries_ts_code={asset: [f"ts_code in {df_selected_assets.index.to_list()}"]},query_df=f"trade_date <= {trade_date}")

            #load individual df_asset
            for ts_code,df_asset in d_preload_filter.items():
                if df_asset.empty:
                    continue

                # NOW vs historic abs price TODO REMOVE
                for column in ["close"]:
                    for freq in [20,60,240]:
                        df_gain = df_asset.tail(freq)
                        df_selected_assets.at[ts_code, f"{column}_{freq}"] = (((1 - 0) * (df_gain[f"{column}"].iat[-1] - df_gain[f"{column}"].min())) / (df_gain[f"{column}"].max() - df_gain[f"{column}"].min())) + 0

                # Bollinger NOW on D and on W
                #calculate bollinger completely TODO REMOVE
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
                        try:
                            df_selected_assets.at[ts_code, f"{column}_ALL"] = df_bullishness_overview.at[ts_code,f"{column}_ALL"]
                        except:
                            df_selected_assets.at[ts_code, f"{column}_ALL"] = np.nan
                        try:
                            df_selected_assets.at[ts_code, f"{column}_NOW"] = df_bullishness_overview.at[ts_code,f"{column}"]
                        except:
                            df_selected_assets.at[ts_code, f"{column}_NOW"] = np.nan


            #after all individual assets are finished calculated
            # opportunity rank - short term rank - volatility
            if asset in ["E","FD","I"]:
                df_selected_assets["opportunity_rank"] = df_selected_assets["close_20"] * 0.38 * 0.33 \
                                                         + df_selected_assets["close_60"] * 0.38 * 0.33 \
                                                         + df_selected_assets["close_240"] * 0.38 * 0.33 \
                                                         + df_selected_assets["boll_NOWD"] * 0.62 * 0.38 \
                                                         + df_selected_assets["boll_NOWW"] * 0.62 * 0.62
                df_selected_assets["opportunity_rank"] = df_selected_assets["opportunity_rank"].rank(ascending=True)

            # investment rank - long term rank - value
            if asset in ["E"] and market in ["CN"]:
                df_selected_assets["investment_rank"] = df_selected_assets["pe_ttm_ALL"] * 0.62 \
                                                        + df_selected_assets["pb_ALL"] * 0.38
                df_selected_assets["investment_rank"] = df_selected_assets["investment_rank"].rank(ascending=True)

            #add static data
            if asset in ["E","FD","I"]:
                df_selected_assets=df_selected_assets.loc[:, df_selected_assets.columns != 'asset']
                df_selected_assets = DB.add_static_data(df_selected_assets, asset=[asset],market=market)
            d_df[f"{asset}_{min_period}"] = df_selected_assets


    df_north=DB.get(a_path=("Market/CN/Asset/E/hsgt/hsgt"))
    if df_north.empty:
        df_north=DB.update_hk_hsgt()
    d_df[f"north"]=df_north

    LB.to_csv_feather(pd.DataFrame(),a_path=LB.a_path(f"Market/{market}/Report/folder"))
    LB.to_excel(path=f"Market/{market}/Report/report_{trade_date}.xlsx",d_df=d_df,color=True)
    LB.file_open(f"D:\Stock/Market/{market}/Report/report_{trade_date}.xlsx")




if __name__ == '__main__':
    #TODO: when do stock revert? volume, time to previous date, ma, market, boll , supportresistance
    import Alpha
    do=2


    if do==1:
        for market in ["CN"]:
            create_daily_report(update_DB=False,market=market)



    if do==2:
        for pf in ["p","f"]:
            df_sh = DB.get_asset("399006.SZ", asset="I")
            df_trade_date=DB.get_trade_date()
            df_trade_date=df_trade_date[df_trade_date["lastdayofseason"]==True]
            df_trade_date["399006.SZ"]=df_sh["close"]
            df_trade_date=df_trade_date[["399006.SZ"]]
            for trade_date in df_trade_date.index:
                try:
                    xls = pd.ExcelFile(f"Market/CN/Report/report_{trade_date}.xlsx")
                except:
                    continue
                df_pgain = pd.read_excel(xls, sheet_name=f"{pf}gain")
                try:
                    df_pgain=df_pgain.set_index("Unnamed: 0")
                except:
                    df_pgain=df_pgain.set_index("group")
                df_pgain.index.name="group"
                for freq in [60]:
                    for column in ["offensive_rank","defensive_rank","M_std","000001.SH_beta","399006.SZ_beta","399001.SZ_beta","pe_ttm","pb","pe_ttm_ALL","pb_ALL","total_mv","qdii_rank"]:
                        print("doin column",column,trade_date)
                        for q1,q2 in [(0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1)]:
                            try:
                                df_trade_date.at[trade_date,f"{column}_q{q1},{q2}"]=df_pgain.at[f"{column}_q{q1},{q2}",f"{pf}gain{freq}"]
                            except Exception as e:
                                df_trade_date.at[trade_date, f"{column}_q{q1},{q2}"] = np.nan
                                print(e)

            #convert to compgain
            df_copy=df_trade_date.copy()
            for column in df_copy.columns:
                if column !="399006.SZ":
                    df_copy[column]=Alpha.comp_chg2(df=df_trade_date,abase=column,inplace=False)


            df_copy.to_csv(f"{pf}gain_comp_chg.csv")



    if do==3:
        df_trade_date = DB.get_trade_date()
        df_trade_date = df_trade_date[df_trade_date["lastdayofseason"] == True]

        for trade_date in df_trade_date.index[::-1]:

            print(f"CALCULATE REPORT for trade_date {trade_date}")
            create_daily_report(update_DB=False,trade_date=trade_date)
