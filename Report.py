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
import Infographic

a_ideal_cols = ["ts_code","asset", "period", "qdii_research/period","qdii_grade/period","pe_ttm", "pb", "total_mv", "pe_ttm_ALL", "pb_ALL", "close", "hk_hold", "hk_hold_ALL", "tech_off_rank", "tech_def_rank", "allround_rank_geo", "qdii_off_rank", "qdii_def_rank", "qdii_ratio_ls_mom", "qdii_ratio_ls", "M_std", "000001.SH_beta", "399001.SZ_beta", "399006.SZ_beta","head_sw_industry1","head_sw_industry2","head_sw_industry3"] + [f"pgain{freq}" for freq in [5, 20, 60, 240]] + [f"fgain{freq}" for freq in [5, 20, 60, 240]] + [f"qdii_research{freq}" for freq in [60]] + [f"qdii_grade{freq}" for freq in [60]]
a_minus=["asset","name","ts_code","period"]
a_traceback=["trade_rank"]+[x for x in a_ideal_cols if x not in a_minus]
ideal_order_column = ["name", "period", "trade_rank", "qdii_off_rank", "qdii_def_rank", "qdii_research/period","qdii_grade/period","tech_off_rank", "tech_def_rank", "allround_rank_geo", "opportunity_rank", "investment_rank", "total_mv"]
p_setting=[0,0.05, 0.2,0.4,0.6,0.8, 0.95,1]

def create_daily_report(trade_date=-1, update_DB=False, market="CN", send_report=True, send_infographic=True):
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
            DB.update_all_in_one_cn()
        if market == "HK":
            DB.update_all_in_one_hk()


    # 2. skip if report already exists

    if trade_date == -1:
        trade_date = LB.latest_trade_date(market=market)

    excel_path = f"Market/{market}/Report/{market}_report_{trade_date}.xlsx"
    send_mail_title = f"{str(trade_date)}"

    files=[]
    if os.path.isfile(f"D:\Stock/{excel_path}"):
        print(f"REPORT {trade_date} EXISTS!")
        LB.file_open(f"D:\Stock/Market/{market}/Report/{market}_report_{trade_date}.xlsx")


        if send_infographic and market=="CN":
            infochart=Infographic.create_infographic(trade_date=str(trade_date))
            print(infochart)
            files += [infochart]

        #files += [excel_path]
        print(files)
        if send_report:
            LB.send_mail_report(trade_string=send_mail_title,files=files)


        return

    print(f"start report for {trade_date}")




    # 3. get bullishness df, if not exist, create one
    if not os.path.isfile(f"Market/{market}/ATest/bullishness/bullishness_{market}_0_{trade_date}.xlsx"):
        Atest.asset_bullishness(a_asset=["E", "FD", "I"], step=1, market=market, end_date=trade_date)
    xls = pd.ExcelFile(f"Market/{market}/ATest/bullishness/bullishness_{market}_0_{trade_date}.xlsx")
    df_bullishness_overview_original = pd.read_excel(xls, sheet_name="Overview")
    a_real_cols = [x for x in a_ideal_cols if x in df_bullishness_overview_original.columns]
    df_bullishness_overview_master=df_bullishness_overview_original[a_real_cols]
    df_bullishness_overview_master.set_index("ts_code", drop=True, inplace=True)


    # 4. settings
    d_df = {}
    a_period = {"Market": [(0, 9999999)], "I": [(1200, 9999999)], "FD": [(800,  9999999)], "E": [(500, 9999999)]}
    top = {"I": 0.10, "FD": 0.12, "E": 0.15}
    a_gain_freq = [5,20,60,240]

    df_trade_date=DB.get_trade_date()
    df_pgain_summary=pd.DataFrame()
    df_fgain_summary=pd.DataFrame()
    d_pgain = {}
    d_fgain = {}

    for freq in a_gain_freq:
        df_pgain_freq = df_bullishness_overview_master[[f"pgain{freq}"]]
        d_pgain[freq] = df_pgain_freq

        df_fgain_freq = df_bullishness_overview_master[[f"fgain{freq}"]]
        d_fgain[freq] = df_fgain_freq






    #7. stock in focus
    d_a_assets={"CN":["I","FD","E"], "HK":["E"]}
    for asset in  d_a_assets[market]:
        print(market,asset)
        for min_period, max_period in a_period[asset]:
            df_bullishness_overview = df_bullishness_overview_master if asset!="Market" else df_bullishness_overview_master[["period" ]]

            # Select best stocks and by rank

            if asset in ["I","FD","E"]:
                print("zes")
                df_selected_assets = df_bullishness_overview[(df_bullishness_overview["asset"] == asset) & (df_bullishness_overview["period"] >= min_period) & (df_bullishness_overview["period"] < max_period)].sort_values(by="allround_rank_geo")
                df_selected_assets = df_selected_assets.head(int(df_trade_date.at[trade_date,"E_count"] * 1))
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
                    try:
                        df_asset_freq = LB.df_to_freq(df_asset, freq=freq)

                        # bollinger
                        df_asset_freq[f"boll_up"], df_asset_freq[f"boll_mid"], df_asset_freq[f"boll_low"] = talib.BBANDS(df_asset_freq["close"], 20, 2, 2)

                        # scale to between 0 and 1
                        df_asset_freq[f"boll_scale"] = (((1 - 0) * (df_asset_freq["close"] - df_asset_freq[f"boll_low"])) / (df_asset_freq[f"boll_up"] - df_asset_freq[f"boll_low"])) + 0

                        # take the last sample as NOW
                        df_selected_assets.at[ts_code, f"boll_NOW{freq}"] = df_asset_freq[f"boll_scale"].iat[-1]
                    except:
                        pass

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
                try:
                    df_selected_assets["opportunity_rank"] = df_selected_assets["close_20"] * 0.38 * 0.33 \
                                                             + df_selected_assets["close_60"] * 0.38 * 0.33 \
                                                             + df_selected_assets["close_240"] * 0.38 * 0.33 \
                                                             + df_selected_assets["boll_NOWD"] * 0.62 * 0.38 \
                                                             + df_selected_assets["boll_NOWW"] * 0.62 * 0.62
                    df_selected_assets["opportunity_rank"] = df_selected_assets["opportunity_rank"].rank(ascending=True)
                except:
                    pass

            # investment rank - long term rank - value
            if asset in ["E"] and market in ["CN"]:
                df_selected_assets["investment_rank"] = df_selected_assets["pe_ttm_ALL"] * 0.62 \
                                                        + df_selected_assets["pb_ALL"] * 0.38
                df_selected_assets["investment_rank"] = df_selected_assets["investment_rank"].rank(ascending=True)


                #buy sell rank
                df_selected_assets["trade_rank"] =    df_selected_assets["qdii_off_rank"].rank(ascending=True)  * 0.8 *0.5*0.8\
                                                    + df_selected_assets["qdii_def_rank"].rank(ascending=True)  * 0.8 *0.5*0.2\
                                                    + df_selected_assets["tech_off_rank"].rank(ascending=True)  * 0.8 *0.4*0.3\
                                                    + df_selected_assets["tech_def_rank"].rank(ascending=True)  * 0.8 *0.4*0.7\
                                                    + df_selected_assets["hk_hold"].rank(ascending=False) *0.8*0.1*0.5 \
                                                    + df_selected_assets["hk_hold_ALL"].rank(ascending=False) * 0.8*0.1*0.5 \
                                                    + df_selected_assets["investment_rank"].rank(ascending=True)  * 0.20 *0.5 \
                                                    + df_selected_assets["opportunity_rank"].rank(ascending=True)  * 0.20 *0.5
                df_selected_assets["trade_rank"] = df_selected_assets["trade_rank"].rank(ascending=True)
                df_selected_assets=df_selected_assets.sort_values("trade_rank")
            #add static data
            if asset in ["E","FD","I"]:
                df_selected_assets=df_selected_assets.loc[:, df_selected_assets.columns != 'asset']
                df_selected_assets = DB.add_static_data(df_selected_assets, asset=[asset],market=market)

            # reorder columsn for better visibility
            df_selected_assets=LB.reorder_columns(df_selected_assets,ideal_order_column)

            #add to collection
            d_df[f"{asset}_{min_period}"] = df_selected_assets

    #north
    df_north=DB.get(a_path=("Market/CN/Asset/E/hsgt/hsgt"))
    if df_north.empty:
        df_north=DB.update_hk_hsgt()
    d_df[f"north"]=df_north



    #fixed summary

    df_simple_s = pd.DataFrame()
    df_simple_c = pd.DataFrame()
    df_ts_code = DB.get_ts_code(a_asset=["I", "E", "FD", "G"])

    def add_overview(df_simple,array_list):
        for asset, iarray in array_list.items():
            for ts_code in iarray:
                df = DB.get_asset(ts_code=ts_code, asset=asset)
                df = df[df.index <= trade_date]

                for freqn in ["D", "W","M"]:
                    df = LB.df_to_freq(df, freqn)

                    # add freq boll
                    boll, bolldown, bollup = Alpha.boll(df=df, abase="close", freq1=20, freq2=2, inplace=True)
                    df_simple.at[ts_code, "name"] = df_ts_code.at[ts_code, "name"]
                    df_simple.at[ts_code, f"boll_{freqn}"] = decide = df[boll].iat[-1]

                    if freqn in ["D","W"]:
                        if decide <= 0.2:
                            df_simple.at[ts_code, f"action_{freqn}"] = "buy"
                        elif decide > 0.2 and decide < 0.8:
                            df_simple.at[ts_code, f"action_{freqn}"] = "hold"
                        elif decide >= 0.8:
                            df_simple.at[ts_code, f"action_{freqn}"] = "sell"
                    elif freqn in ["M"]:
                        if decide <= 0.5:
                            df_simple.at[ts_code, f"action_{freqn}"] = "sell"
                        elif decide > 0.5:
                            df_simple.at[ts_code, f"action_{freqn}"] = "buy"


                    df["boll_dist_all"] = df[bollup] / df[bolldown]
                    df_simple.at[ts_code, f"boll_{freqn}_spread"] = df["boll_dist_all"].iat[-1]



    add_overview(df_simple_s, LB.c_imp_index())#all manual ts_code

    df_industry2 = DB.get_ts_code(a_asset=["G"])
    df_industry2["indexhelper"] = df_industry2.index
    df_industry2["what"] = ("sw_industry2" == df_industry2["indexhelper"].str.slice(0, 12))
    df_industry2 = df_industry2[df_industry2["what"] == True]
    d_industrz2 = {"G": [x for x in df_industry2.index]}
    print(d_industrz2)
    add_overview(df_simple_c, d_industrz2)#all industry lv1 tscode

    try:
        df_simple_s=df_simple_s.sort_values(by=["boll_W","boll_D","boll_M"])
        df_simple_c=df_simple_c.sort_values(by=["boll_W","boll_D","boll_M"])
    except:
        pass

    d_df[f"overview_s"] = df_simple_s
    d_df[f"overview_c"] = df_simple_c


    # 6. group gain by
    df_column=d_df["E_500"]
    for column in a_traceback :
        print(f"calculate {column} pfgain")
        if column in df_bullishness_overview_master.columns:
            helper_function(df_column=df_column, column=column, d_pgain=d_pgain, df_pgain_summary=df_pgain_summary, df_fgain_summary=df_fgain_summary, p_setting=p_setting)
    # add the df to final report excel
    df_pgain_summary.index.name = "group"
    df_fgain_summary.index.name = "group"
    d_df[f"pgain"] = df_pgain_summary
    d_df[f"fgain"] = df_fgain_summary

    #filter stock
    if market == "CN":
        df_concept=  pd.read_excel(xls, sheet_name="concept")
        df_concept=df_concept.sort_values("pgain20",ascending=False)
        d_df[f"concept"] =df_concept

    #龙头股
    if market == "CN":
        df_head=pd.DataFrame()
        for industry in df_bullishness_overview_original["sw_industry3"].unique():
            df_filter=df_bullishness_overview_original[df_bullishness_overview_original["sw_industry3"]==industry]
            df_filter=df_filter.sort_values("total_mv",ascending=False)
            df_head=df_head.append(df_filter.head(1))

        df_head=LB.reorder_columns(df_head,ideal_order_column)
        d_df[f"head"] = df_head


    #save
    LB.to_csv_feather(pd.DataFrame(),a_path=LB.a_path(f"Market/{market}/Report/folder"))
    LB.to_excel(path=excel_path,d_df=d_df,color=True)

    for i in range(10):
        try:
            LB.file_open(excel_path)
            break
        except:
            print("for some reason windows can't find file even if it is there")

    if send_infographic and market=="CN":
        infochart = Infographic.create_infographic(trade_date=str(trade_date))
        files += [infochart]

    files += [excel_path]
    if send_report:
        LB.send_mail_report(trade_string=send_mail_title, files=files)



def loop():

    print("loop start")
    while True:
        try:
            path =LB.c_root()+"Market/CN/update.csv"
            df =pd.read_csv(path)
        except:
            df=pd.DataFrame()

        onesec = 1
        onemin = 60*onesec
        onehour = 60 * onemin

        now_time = datetime.now()
        year = now_time.year
        month = now_time.month
        day = now_time.day
        hour = now_time.hour


        if hour==20:
            print("auto_report")
            #create report
            for market in ["CN"]:
                create_daily_report(update_DB=True, market=market, send_report=True)

            #note it in the update.csv
            index=len(df)
            df.at[index,"year"]=year
            df.at[index,"month"]=month
            df.at[index,"day"]=day
            df.at[index,"hour"]=hour
            df = df[["year", "month", "day", "hour", "minute"]]


            time.sleep(onehour)


        sys.stdout.write('\r' + str(now_time))
        sys.stdout.flush()
        time.sleep(onemin)



if __name__ == '__main__':
    """
    TODO AH股溢价，但是tushare没这个数据
    """
    import Alpha


    for do in [1]:

        #loop
        if do == 0:
            loop()

        # single report
        if do==1:
            for market in ["CN"]:
                create_daily_report(update_DB=True,market=market,send_report=True)


        # summary of report
        """
        important NOTE
        Offensive RANK: the better the past, the worse the future
        Defensive Rank: still holds. Past is better than future. So defensive > offensive
        STD: IF past STD is big, future STD will be small. In General, buy stock with small STD. This aligns with findings from defensive rank. 
        BETA with SH, SZ, index: smaller the better. Aligns with both pgain and fgain and with asset_bullishnes
        PE_ttm: no significant difference for stock. conclusion: Absolute PE is IRRELEVANT. pe 6 vs 200 stock both are same good/bad
        PB: small pb is significant better than others in long term. Buy stock with low ABS PB.
        PE_ttm_pct: normalized from 0-1: VERY SIGNIFICANT. USE IT.
        PB_pct: normalized from 0-1: VERY SIGNIFICANT. USE IT.
        Total_MV: Buy small stock is better in long term.
        QDII research/period. the least researched stock become the best instead
        small PB only until 2010. After that the small PB effect is gone
        HK Hold is significant. the higher the hk_hold, the better
        pgain: The worst in the past, the best in the future
        
        """
        market="CN"
        if do==2:
            for pf in ["p","f"]:
                df_sh = DB.get_asset("399006.SZ", asset="I")
                df_trade_date=DB.get_trade_date()
                df_trade_date=df_trade_date[df_trade_date["lastdayofseason"]==True]
                df_trade_date["399006.SZ"]=df_sh["close"]
                df_trade_date=df_trade_date[["399006.SZ"]]
                for trade_date in df_trade_date.index:
                    print("do ",trade_date)
                    try:
                        xls = pd.ExcelFile(f"Market/CN/Report/{market}_report_{trade_date}.xlsx")
                    except Exception as e:
                        continue

                    df_pgain = pd.read_excel(xls, sheet_name=f"{pf}gain")
                    try:
                        df_pgain=df_pgain.set_index("Unnamed: 0")
                    except:
                        df_pgain=df_pgain.set_index("group")
                    df_pgain.index.name="group"
                    for freq in [60]:
                        for column in a_traceback:
                            print("doin column",column,trade_date)
                            for q1,q2 in LB.custom_pairwise_overlap(p_setting):
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


        #loop and to all report
        if do==3:
            df_trade_date = DB.get_trade_date()
            df_trade_date = df_trade_date[df_trade_date["lastdayofseason"] == True]

            for trade_date in df_trade_date.index[::-1]:

                print(f"CALCULATE REPORT for trade_date {trade_date}")
                create_daily_report(update_DB=False,trade_date=trade_date,send_infographic=False,send_report=False)
