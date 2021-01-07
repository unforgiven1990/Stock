import operator
from multiprocessing import Process
import tushare as ts
from datetime import datetime
import pandas as pd
import numpy as np
import talib
import xlsxwriter
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
    trade_date = DB.get_asset("000001.SH", asset="I").index[-1]
    min_period=1200
    top=0.05
    tail = 500
    sortby={"I":"final_position","FD":"final_position","E":"defensive_rank"}

    #update if nessesary
    if with_db_update:
        DB.update_all_in_one_cn_v2()

    #read bullishness
    xls = pd.ExcelFile("Market/CN/ATest/bullishness/bullishness_CN_0_99999999.xlsx")
    df_bullishness_master= pd.read_excel(xls, sheet_name="Overview")


    #generate report
    for asset in ["I","FD","E"]:
        df_bullishness = df_bullishness_master[["ts_code","period", sortby[asset], "asset", "name", "market"]]
        df_bullishness.set_index("ts_code", drop=True, inplace=True)

        # Select best stocks and by rank
        df_selected_assets=df_bullishness[(df_bullishness["asset"]==asset)&(df_bullishness["period"]>min_period)].sort_values(by=sortby[asset])
        df_selected_assets=df_selected_assets.head(int(len(df_selected_assets)*top))
        d_preload_E=DB.preload(asset=asset,d_queries_ts_code={asset: [f"ts_code in {df_selected_assets.index.to_list()}"]})

        #load individual df_asset
        for ts_code,df_asset in d_preload_E.items():
            # NOW vs historic abs price
            for column in ["close"]:
                for freq in [60,240,500]:
                    df_helper = df_asset.tail(freq)
                    df_selected_assets.at[ts_code, f"{column}_{freq}"] = (((1 - 0) * (df_helper[f"{column}"].iat[-1] - df_helper[f"{column}"].min())) / (df_helper[f"{column}"].max() - df_helper[f"{column}"].min())) + 0

            # NOW vs historic PE and PB
            if asset == "E":
                for column in ["pe_ttm", "pb"]:
                    df_selected_assets.at[ts_code, f"{column}_ALL"] = (((1 - 0) * (df_asset[f"{column}"].iat[-1] - df_asset[f"{column}"].min())) / (df_asset[f"{column}"].max() - df_asset[f"{column}"].min())) + 0

                    df_helper = df_asset.tail(tail)
                    df_selected_assets.at[ts_code, f"{column}_{tail}"] = (((1 - 0) * (df_helper[f"{column}"].iat[-1] - df_helper[f"{column}"].min())) / (df_helper[f"{column}"].max() - df_helper[f"{column}"].min())) + 0

            # WHAT are their Bollinger?
            # WHAT are their 60 MA?
        d_df[asset]=df_selected_assets


    LB.to_csv_feather(pd.DataFrame(),a_path=LB.a_path(f"Market/CN/Report/folder"))
    LB.to_excel(path=f"Market/CN/Report/report_{trade_date}.xlsx",d_df=d_df)






if __name__ == '__main__':
    #TODO: is does top happen with no volume or WITH high volume?
    create_daily_report()
