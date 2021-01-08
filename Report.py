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

def create_daily_report(with_db_update=False):
    """
    this function creates an daily overview of the most important stocks and FD

    :param with_db_update:
    :return:
    """

    # init
    d_df = {}
    trade_date = LB.latest_trade_date()
    min_period=1200
    top={"I":0.10,"FD":0.12,"E":0.05}
    tail = 500
    sortby={"I":"final_position","FD":"final_position","E":"defensive_rank"}

    #update if nessesary
    if with_db_update:
        #DB.update_all_in_one_cn_v2()
        Atest.asset_bullishness(a_asset=["E","FD","I"],step=1,market="CN")


    #read bullishness
    xls = pd.ExcelFile("Market/CN/ATest/bullishness/bullishness_CN_0_99999999.xlsx")
    df_bullishness_master= pd.read_excel(xls, sheet_name="Overview")

    #generate report
    for asset in ["Market","I","FD","E"]:
        df_bullishness = df_bullishness_master[["ts_code","period", "offensive_rank","defensive_rank","final_position", "asset", "name", "market"]] if asset!="Market" else df_bullishness_master[["ts_code","period",  "name", ]]
        df_bullishness.set_index("ts_code", drop=True, inplace=True)

        # Select best stocks and by rank
        if asset =="Market":
            df_selected_assets = df_bullishness.loc[["000001.SH","399006.SZ","399001.SZ"]]
            d_preload_E = DB.preload(asset="I", d_queries_ts_code={"I": [f"ts_code in {df_selected_assets.index.to_list()}"]})
        elif asset in ["I","FD","E"]:
            df_selected_assets = df_bullishness[(df_bullishness["asset"] == asset) & (df_bullishness["period"] > min_period)].sort_values(by=sortby[asset])
            df_selected_assets = df_selected_assets.head(int(len(df_selected_assets) * top[asset]))
            d_preload_E=DB.preload(asset=asset,d_queries_ts_code={asset: [f"ts_code in {df_selected_assets.index.to_list()}"]})

        #load individual df_asset
        for ts_code,df_asset in d_preload_E.items():
            # NOW vs historic abs price
            for column in ["close"]:
                for freq in [60,240,500]:
                    df_helper = df_asset.tail(freq)
                    df_selected_assets.at[ts_code, f"{column}_{freq}"] = (((1 - 0) * (df_helper[f"{column}"].iat[-1] - df_helper[f"{column}"].min())) / (df_helper[f"{column}"].max() - df_helper[f"{column}"].min())) + 0

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
            if asset == "E":
                for column in ["pe_ttm", "pb"]:
                    df_selected_assets.at[ts_code, f"{column}_ALL"] = (((1 - 0) * (df_asset[f"{column}"].iat[-1] - df_asset[f"{column}"].min())) / (df_asset[f"{column}"].max() - df_asset[f"{column}"].min())) + 0
                    df_helper = df_asset.tail(tail)
                    df_selected_assets.at[ts_code, f"{column}_{tail}"] = (((1 - 0) * (df_helper[f"{column}"].iat[-1] - df_helper[f"{column}"].min())) / (df_helper[f"{column}"].max() - df_helper[f"{column}"].min())) + 0

        d_df[asset]=df_selected_assets
    LB.to_csv_feather(pd.DataFrame(),a_path=LB.a_path(f"Market/CN/Report/folder"))
    LB.to_excel(path=f"Market/CN/Report/report_{trade_date}.xlsx",d_df=d_df)
    LB.file_open(f"D:\Stock/Market/CN/Report/report_{trade_date}.xlsx")




if __name__ == '__main__':
    #TODO: when do stock revert? volume, time to previous date, ma, market, boll , supportresistance
    create_daily_report(with_db_update=True)
