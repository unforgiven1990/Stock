import DB
import LB
import pandas as pd
import numpy as np
import UI



def stock_market_abv_ma():
    """
    This strategy tests if using abv ma 5-240 is a good strategy

    Result:
    using abv_ma on any freq does not yield significant result
    all results are pretty the same. not very useful

    """
    df=DB.get_stock_market_all()
    df["tomorrow"]=df["open.fgain1"].shift(-1)

    for freq in [5,20,60,240]:
        df_filtered=df[df[f"abv_ma{freq}"]<=0.8]
        gain=df_filtered["tomorrow"].mean()
        print(freq,"und 0.8 :",gain)

    for freq in [5,20,60,240]:
        df_filtered=df[df[f"abv_ma{freq}"]>0.2]
        gain=df_filtered["tomorrow"].mean()
        print(freq,"abv 0.2 :",gain)


def pattern_bull():
    """
    this function tries to identify the pattern when a stock is retreated from a good long term bullishness trend
    1. Find long term bull trend
    2. find short term retreat

    optional:
    1. the asset is still abv ma
    2. last ext

    Basically:
    - Buy if 60 ma is close to 240 ma
    - And hope that it will bounce back


    Result:
    -This works ONLY if YOU are 100% SURE about the uptrend
    -
    -Sometimes it bounces back
    -Somestimes it doesnt
    -It totlly depends on stocks fundamental and self strength
    -This strategy itself is natural and only works on bullish stock retreat
    - This is a good example to show that technical analysis alone is not enough. Fundamental can compliment here a lot.
    - For 600519, in 3000 days, only 7 times bottom happens. Which means, 12 years, 7 times real bottom.
    """

    df= DB.get_asset(ts_code="600519.SH")

    df["signal"]=0
    print(df.abv_ma240)
    import Alpha
    df["lp"]=Alpha.lowpass(df=df,abase="close",freq=10)

    df[f"e_min60"]=df["lp"].rolling(60).min()
    df["ma120"]=df["lp"].rolling(60).mean()
    df["ma240"]=df["lp"].rolling(240).mean()
    df["abv_ma120"]= (df["lp"]>df["ma120"]).astype(int)

    df[f"e_gain60"]=df["pct_chg"].rolling(60).mean()

    df.loc[
            (df["ma240"] / df["ma120"]).between(0.95, 1.05)
            &(df["e_gain60"]>0)
             ,"signal" ]=1
    df["signal"]=df["signal"]*df["e_max"]

    UI.plot_chart(df, ["lp", "signal", "ma240", "ma60", "e_max"])






def after_big_down():
    #this tries to answer the question if stock has been down 30%, how much do they gain after that_

    """
    test to see if stock has lost n% in last 60 days, (except 2008,2015), who to they recover?
    Result:
    - the more they lose, the more they gain on average. But the std is very different. So which means I don't Like. I like things with small deviation.
    - Difference: after 20 days losing 40%, there is more bounce than 60 days losing 40%.
    - this finding matches the idea of mean reversal. the more gain the less gain in the future


    :return:
    """
    df_ts_code=DB.get_ts_code()
    df_result=pd.DataFrame()
    for ts_code in df_ts_code.index:
        print(ts_code)
        df_asset=DB.get_asset(ts_code)
        df_asset=LB.df_to_calender(df_asset)
        df_asset=df_asset[df_asset["year"] != 2008]
        df_asset=df_asset[df_asset["year"] != 2015]
        df_asset=df_asset[df_asset["year"] != "2008"]
        df_asset=df_asset[df_asset["year"] != "2015"]


        for gain in [0.9,0.8,0.7,0.6,0.5,0.4,0.3]:
            df_result.at[ts_code,f"mean_lose{gain}"]=df_asset.loc[df_asset["pgain20"]<gain,"fgain60"].mean()
        for gain in [1.1,1.2,1.3,1.4,1.5,1.6,1.7]:
            df_result.at[ts_code,f"mean_win{gain}"]=df_asset.loc[df_asset["pgain20"]>gain,"fgain60"].mean()

    df_result.to_csv("after 20daydown.csv")



def volatility():
    import random
    import Alpha
    #this test simulates stock volatility and random sample simulates user buy and sell point

    """
    Result:
    波动越小越好，股票涨幅约向上越好
    平衡这两种，就是最好的投资
    """
    repeat=1
    df_master=pd.DataFrame()
    for j in range(repeat):
        print("repeat", j)


        while True:
            df = pd.DataFrame()
            df["pct_chg"] = 0
            n = 500
            a_gain = [-10, -10, -7, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 7, 10, 10]
            a_gain = [-1, -0.5, 0, 0.5, 1]

            for i in range(n):
                pct_chg = random.choice(a_gain)
                df.at[i,"pct_chg"]=pct_chg

            df["close"]=Alpha.comp_chg(df=df,abase="pct_chg",inplace=False)
            if df["close"].iat[-1]>1.03 and df["close"].iat[-1]<1.06:
                break

        print("out")

        traden=10000
        holdp=20
        benjin=100
        df_trade=pd.DataFrame()
        for trade in range(traden):
            buy = random.choice([x for x in range(n)])
            sell = buy+holdp
            if sell>n-1:
                sell=n-1

            buyat=df.at[buy,"close"]
            sellat=df.at[sell,"close"]
            gain= sellat/buyat
            df_trade.at[trade,"buyat"]=buyat
            df_trade.at[trade,"sellat"]=sellat
            df_trade.at[trade,"return"]=gain
            df_trade.at[trade,"benjin"]=benjin=gain*benjin


        lastday=df['close'].iat[-1]
        print(f"the last day close is {lastday}")
        print(f"benjin is {benjin}")
        print()
        df_master.at[j,"lastday"]=lastday
        df_master.at[j,"benjin"]=benjin

        df_trade.to_csv(f"df_trade{j}.csv")
    df_master.to_csv("df_master.csv")

def qdii_count():
    """
    tries to see if the number of qdii research can predict the market hotness
    result: The effect is very minimal. Maybe qdii has to publish research on a given period. The results are published periodic
    :return:
    """
    df_trade_date=DB.get_trade_cal_D()
    df_trade_date["qdii_research"] = 0
    df_trade_date["qdii_grade"] = 0

    df_ts_code=DB.get_ts_code()

    for ts_code in df_ts_code.index:
        print(ts_code)
        for freq in ["qdii_research","qdii_grade"]:
            df_qdii=DB.get_asset(ts_code=ts_code,freq=freq)
            if df_qdii.empty:
                continue
            df_qdii=df_qdii.groupby("trade_date").count()
            df_qdii[freq]=1


            df_trade_date[freq] = df_trade_date[freq].add(df_qdii[freq],fill_value=0)

    df_sh=DB.get_asset(ts_code="000001.SH",asset="I")
    df_trade_date["sh"]=df_sh["close"]
    df_trade_date.to_csv("qdii_count_result.csv")

def technical_search():
    """
    tries to find stock on a given pattern, resistance, support and such


    :return:
    """
    df_ts_code=DB.get_ts_code()

    for ts_code in df_ts_code.index:
        df_asset=DB.get_asset(ts_code=ts_code,freq="D")


def qdii_ownage_ratio():
    """
    this function tries to see how much money institution has from all market capital
    :return:
    """

    df_market_cap=DB.get_trade_date()
    df_market_cap["E_cap"]=0
    df_market_cap["FD_cap"]=0

    #calculate total market cap
    df_ts_code=DB.get_ts_code()
    for ts_code in df_ts_code.index:
        print(ts_code)
        df_asset=DB.get_asset(ts_code=ts_code)
        if df_asset.empty:
            continue
        df_market_cap["E_cap"] = df_market_cap["E_cap"].add(df_asset["total_mv"],fill_value=0)

    # calculate how much money fund owns
    df_ts_code = DB.get_ts_code(a_asset=["FD"])
    df_ts_code=df_ts_code[df_ts_code["fund_type"].isin(["股票型","混合型"])]
    for ts_code in df_ts_code.index:
        print(ts_code)
        df_asset = DB.get_asset(ts_code=ts_code,asset="FD")
        if df_asset.empty:
            continue
        df_market_cap["FD_cap"] = df_market_cap["FD_cap"].add(df_asset["total_mv"], fill_value=0)

    df_market_cap["ratio"]=df_market_cap["FD_cap"] /df_market_cap["E_cap"]
    df_market_cap.to_csv("total market cap.csv")

def best_xinquan():

    """This function tests a strategy to produce the best outcome from all xinquan
    Result:风格切换相对比较慢，兴全模式如果好的话可以好100天，200天，差的话也可以差那么久
    买20天强势的好的，短期内买多涨的
    买240天趋势走的差的，长期内买少涨的
    """


    a_core=["163415.SZ","163402.SZ","163417.SZ","163412.SZ"]
    a_big=a_core+[]
    df_result=pd.DataFrame()

    """Strategy 1 buy the stock with 20 day best/worst gain"""
    for past in [60]:
        for ts_code in a_core:
            df_asset=DB.get_asset(ts_code=ts_code, asset="FD")
            df_result[f"{ts_code}_close"]=df_asset["close"]
            df_result[f"{ts_code}_pgain{past}"]=df_asset[f"pgain{past}"]

    """strategy excecute"""
    for ts_code in a_core:
        df_result[f"rank_{ts_code}"]=df_result[[f"{a}_pgain60" for a in a_core]].rank(axis=1)[f"{ts_code}_pgain60"]

    print("what")
    df_result.to_csv("what.csv")




    return

def qdii_size_return_relation():
    """do big sized fund perform better in the future?


    """

best_xinquan()