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







