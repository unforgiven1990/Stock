import cProfile
# set global variable flag
import Alpha
from Alpha import *
import numpy as np
import UI
from scipy.stats.mstats import gmean
from scipy.stats import gmean
import sys
import os
import talib
import matplotlib
import itertools
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import builtins
import _API_Tushare
# from Alpha import supersmoother_3p, highpass, cg_Oscillator, macd, ismax, ismin

sys.setrecursionlimit(1000000)

array = [2, 5, 10, 20, 40, 60, 120, 240]

"""
Atest (Assettest): 
= Test strategy on individual asset and then mean them 
= COMPARE past time to now (relative to past)(quantile to past)
= NOT COMPARE what other stocks do (NOT relative to time/market)(quantile to others)

Btest (Backtest):
= COMPARE past time to now (relative to past)(quantile to past)
= COMPARE assets with other (relative to other)(quantile to others)

"""


def asset_extrema_rdm_2(df, abase, a_n=[60]):
    """
    Second longer version of finding extrema and using it to calculate high and lows
    Strengh of resistance support are defined by:
    1. how long it remains a resistance or support (remains good for n = 20,60,120,240?)
    2. How often the price can not break through it. (occurence)
    """
    # Generate a noisy AR(1) sample
    a_bot_name = []
    a_peak_name = []
    np.random.seed(0)
    s = df[abase]
    xs = [0]
    for r in s:
        xs.append(xs[-1] * 0.9 + r)
    df = pd.DataFrame(xs, columns=[s.name])
    for n in a_n:
        # Find local peaks
        df[f'bot{n}'] = df.iloc[argrelextrema(df[s.name].values, np.less_equal, order=n)[0]][s.name]
        df[f'peak{n}'] = df.iloc[argrelextrema(df[s.name].values, np.greater_equal, order=n)[0]][s.name]
        a_bot_name.append(f'bot{n}')
        a_peak_name.append(f'peak{n}')

    # checks strenght of a maximum and miminum
    df["support_strengh"] = df[a_bot_name].count(axis=1)
    df["resistance_strengh"] = df[a_peak_name].count(axis=1)

    # puts all maxima and minimaxs togehter
    d_all_rs = {}
    for n in a_n:
        d_value_index_pairs = df.loc[df[f'bot{n}'].notna(), f'bot{n}'].to_dict()
        d_all_rs.update(d_value_index_pairs)

        d_value_index_pairs = df.loc[df[f'peak{n}'].notna(), f'peak{n}'].to_dict()
        d_all_rs.update(d_value_index_pairs)

    d_final_rs = {}
    for index_1, value_1 in d_all_rs.items():
        keep = True
        for index_2, value_2 in d_final_rs.items():
            closeness = value_2 / value_1
            if 0.95 < closeness < 1.05:
                keep = False
        if keep:
            d_final_rs[index_1] = value_1

    # count how many rs we have. How many support is under price, how many support is over price
    df["total_support_resistance"] = 0
    df["abv_support"] = 0
    df["und_resistance"] = 0
    a_rs_names = []
    for counter, (resistance_index, resistance_val) in enumerate(d_final_rs.items()):
        print("unique", resistance_index, resistance_val)
        df.loc[df.index >= resistance_index, f"rs{counter}"] = resistance_val
        a_rs_names.append(f"rs{counter}")
        if counter == 23:
            df.loc[(df["close"] / df[f"rs{counter}"]).between(0.98, 1.02), f"rs{counter}_challenge"] = 100
            df[f"rs{counter}_challenge"] = df[f"rs{counter}_challenge"].fillna(0)

        df["total_support_resistance"] = df["total_support_resistance"] + (df[f"rs{counter}"].notna()).astype(int)
        df["abv_support"] = df["abv_support"] + (df["close"] > df[f"rs{counter}"]).astype(int)
        df["und_resistance"] = df["und_resistance"] + (df["close"] < df[f"rs{counter}"]).astype(int)

    a_trend_name = []
    for index1, index2 in LB.custom_pairwise_overlap([*d_final_rs]):
        print(f"pair {index1, index2}")
        value1 = d_final_rs[index1]
        value2 = d_final_rs[index2]
        df.loc[df.index == index1, f"{value1, value2}"] = value1
        df.loc[df.index == index2, f"{value1, value2}"] = value2
        df[f"{value1, value2}"] = df[f"{value1, value2}"].interpolate()
        a_trend_name.append(f"{value1, value2}")

    # Create trend support resistance lines using two max and two mins
    df.to_csv("test.csv")

    # Plot results
    # plt.scatter(df.index, df['min'], c='r')
    # plt.scatter(df.index, df['max'], c='g')
    df[["close"] + a_rs_names + a_trend_name + ["total_support_resistance", "abv_support", "und_resistance", "rs23_challenge"]].plot(legend=True)
    plt.show()


def asset_extrema():
    """
    This test tries to combine two rules.
    1. The long term trend longer the better
    2. The price the lower the better

    This seems to be contradicting at first. But the idea is to buy stock have keep their current rend as low as possible.

    If trend goes down, loss limit. Else always go in and buy stocks with good trend.

    More details:
    1. How long is the trend in the past and how long for future?
    2. How strong is the trend?
    3.

    procedure:
    1. first find the last significant turning point(High/low) for that period
    2. check if close is in up or downtrend since then
    3. check from that turning point, smaller freq high/low.
    4. if highs are higher, lows are higher, then in uptrend.
    5. Slope of trend

    #Result: MACD is quicker and a bit quicker and less noise than using extrema to define trade signals

    1. Calculate reverse from today on all high /lows and make regression on then. if the regression does not fit anymore. then the trend is the last strongest trend.
    A: Done that. the problem is tat trend exhange happens too often. So by default, it rarely fits perfect, even if it fits perfect, it changes very quickly.
    A: sometimes you have to skip last couple high/lows because they are a new trend
    """

    df = DB.get_asset(ts_code="000001.SH", asset="I")
    df = LB.df_ohlcpp(df)
    df = df.reset_index()

    df["hp"] = highpass(df=df, abase="close", freq=20, inplace=False)
    df["lp"] = lowpass(df=df, abase="close", freq=20, inplace=False)

    order = 20
    signal = 100
    distance = 1
    abase = "close"

    from scipy.signal import argrelmin, argrelmax, peak_prominences

    def outlier_remove(array, n_neighbors=20, match=1):
        from sklearn.neighbors import LocalOutlierFactor

        X = [[x] for x in array]
        clf = LocalOutlierFactor(n_neighbors=n_neighbors)
        a_predict = clf.fit_predict(X)

        a_result = []
        for predict, value in zip(a_predict, array):
            print(f"predict", predict, value)
            if predict == match:
                a_result.append(value)
        return a_result

    """
    rules
    0. Basically: Starting a new trend requires BOTH high and low to be consistent
    1. If only one deviates, it continues the previous trend.
    2. the result is a signal that is very safe and does not take risk.


    1. only one outlier allowed. if second time the low is not strictly higher than last one, it a downtrend.
    2. if A extrema has no confirmation, and the B has 2. B dominates
    3. Extrema with the most recent information dominates

    """

    # init
    x = df[abase]
    lp = df["lp"]

    # peaks, _ = find_peaks(x,prominence=0,width=60)
    bottom = argrelmin(x.to_numpy(), order=order)[0]
    peaks = argrelmax(x.to_numpy(), order=order)[0]

    # data cleaning
    bottom_noutlier = outlier_remove(bottom, n_neighbors=20, match=1)
    bottom_outlier = outlier_remove(bottom, n_neighbors=2, match=-1)

    # prominence in case needed
    prominences = peak_prominences(x, peaks)[0]
    contour_heights = x[peaks] - prominences

    # 1. iteration assign value/pct_chg of extrema
    for counter, (label, extrema) in enumerate({"bott": bottom, "peakk": peaks}.items()):
        df[f"{label}_pvalue"] = df.loc[extrema, "close"]
        df[f"{label}_fvalue"] = df[f"{label}_pvalue"].fillna(method="ffill")
        df[f"{label}_value_pct_chg"] = df[f"{label}_fvalue"].pct_change()

    h_peak = df[f"peakk_value_pct_chg"]
    h_bott = df[f"bott_value_pct_chg"]

    # 2. iteration assign signal
    for counter, (label, extrema) in enumerate({"bott": bottom, "peakk": peaks}.items()):
        df[f"{label}_diff"] = 0

        # for now the peak and bott are actually the SAME
        if label == "peakk":
            df.loc[(h_peak > 0.05) | (df["close"] / df[f"peakk_fvalue"] > 1.05), f"{label}_diff"] = signal  # df["bott_diff"]=df["bott_ffill"].diff()*500
            df.loc[(h_bott < -0.05) | (df["close"] / df[f"bott_fvalue"] < 0.95), f"{label}_diff"] = -signal  # df["bott_diff"]=df["bott_ffill"].diff()*500
        elif label == "bott":
            df.loc[(h_peak > 0.05) | (df["close"] / df[f"peakk_fvalue"] > 1.05), f"{label}_diff"] = signal  # df["bott_diff"]=df["bott_ffill"].diff()*500
            df.loc[(h_bott < -0.05) | (df["close"] / df[f"bott_fvalue"] < 0.95), f"{label}_diff"] = -signal  # df["bott_diff"]=df["bott_ffill"].diff()*500

        df[f"{label}_diff"] = df[f"{label}_diff"].replace(0, np.nan).fillna(method="ffill")
        df[f"{label}_diff"] = df[f"{label}_diff"] * 40

    # This is actually a second function to generate PLOT
    # simualte past iteration
    if False:
        matplotlib.use("TkAgg")
        for counter, (index, df) in enumerate(LB.custom_expand(df, 1000).items()):
            if counter % 20 != 0:
                continue

            print(counter, index)

            # array of extrema without nan = shrink close only to extrema
            s_bott_pvalue = df["bott_pvalue"].dropna()
            s_peakk_pvalue = df["peakk_pvalue"].dropna()

            dict_residuals_bott = {}
            dict_residuals_peakk = {}
            dict_regression_bott = {}
            dict_regression_peakk = {}

            # do regression with extrema with all past extrema. The regression with lowest residual wins
            for counter, _ in enumerate(s_bott_pvalue):
                if counter > 3:
                    # bott
                    s_part_pvalue = s_bott_pvalue.tail(counter)
                    distance = index - s_part_pvalue.index[0]
                    # s_part_pvalue[index]=df.at[index,"close"]
                    s_bott_lg, residual = LB.polyfit_full(s_part_pvalue.index, s_part_pvalue)
                    dict_residuals_bott[counter] = residual / distance ** 2
                    dict_regression_bott[counter] = (s_part_pvalue, s_bott_lg)

                    # peak
                    s_part_pvalue = s_peakk_pvalue.tail(counter)
                    distance = index - s_part_pvalue.index[0]
                    # s_part_pvalue[index] = df.at[index, "close"]
                    s_bott_lg, residual = LB.polyfit_full(s_part_pvalue.index, s_part_pvalue)
                    dict_residuals_peakk[counter] = residual / distance ** 2
                    dict_regression_peakk[counter] = (s_part_pvalue, s_bott_lg)

            # find the regression with least residual
            from operator import itemgetter
            n = 1
            dict_min_residuals_bott = dict(sorted(dict_residuals_bott.items(), key=itemgetter(1), reverse=True)[-n:])
            dict_min_residuals_peakk = dict(sorted(dict_residuals_bott.items(), key=itemgetter(1), reverse=True)[-n:])

            # plot them
            for key, residual in dict_min_residuals_bott.items():
                _, s_bott_lg = dict_regression_bott[key]
                plt.plot(s_bott_lg)

            for key, residual in dict_min_residuals_peakk.items():
                _, s_peakk_lg = dict_regression_peakk[key]
                plt.plot(s_peakk_lg)

            # plot chart
            plt.plot(df["close"])
            # plt.show()
            plt.savefig(f"Plot/extrema/{index}.jpg")
            plt.clf()
            plt.close()

    # add macd signal for comparison
    label = macd(df=df, freq=360, freq2=500, abase="close", type=4, score=df["close"].max(), inplace=True)
    label = label[0]

    plt.plot(x)
    plt.plot(df[label])

    plt.plot(df["bott_diff"])
    # plt.plot(df["peakk_diff"])

    plt.plot(df["bott_pvalue"], "1")
    plt.plot(df["peakk_pvalue"], "1")

    df.to_csv("test.csv")
    plt.show()



def generic_comparison(df, abase):
    """
    go thorugh ALL possible indicators and COMBINE them together to an index that defines up, down trend or no trend.

    cast all indicator to 3 values: -1, 0, 1 for down trend, no trend, uptrend.

    """
    a_freq = [240]
    df["ma20"] = df[abase].rolling(20).mean()
    a_stable = []

    # unstable period
    # momentum
    df[f"bop"] = talib.BOP(df["open"], df["high"], df["low"], df["close"])  # -1 to 1

    # volume
    df[f"ad"] = talib.AD(df["high"], df["low"], df["close"], df["vol"])
    df[f"obv"] = talib.OBV(df["close"], df["vol"])
    a_unstable = ["bop", "ad", "obv"]
    # stable period
    for freq in a_freq:
        df[f"rsi{freq}"] = talib.RSI(df[abase], timeperiod=freq)

        """ idea: 
        1.average direction of past freq up must almost be same as average direction of past freq down. AND price should stay somewhat the same.
        2. count the peak and bot value of an OSCILATOR. IF last peak and bot are very close, then probably flat time
        """

        # MOMENTUM INDICATORS
        """ok 0 to 100, rarely over 60 https://www.fmlabs.com/reference/ADX.htm similar to dx"""
        df[f"adx{freq}"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=freq)
        df.loc[df[f"adx{freq}"] < 6, f"adx{freq}_trend"] = 0
        df.loc[df[f"adx{freq}"] > 6, f"adx{freq}_trend"] = 10

        """osci too hard difference between two ma, not normalized. need to normalize first https://www.fmlabs.com/reference/default.htm?url=PriceOscillator.htm"""
        df[f"apo{freq}"] = talib.APO(df["close"], freq, freq * 2)
        df[f"apo{freq}_trend"] = 10
        df.loc[(df[f"apo{freq}"].between(-1, 1)) & (df[f"apo{freq}"].shift(int(freq / 2)).between(-2, 2)), f"apo{freq}_trend"] = 0

        """osci too hard -100 to 100 https://www.fmlabs.com/reference/default.htm?url=AroonOscillator.htm"""
        df[f"aroonosc{freq}"] = talib.AROONOSC(df["high"], df["low"], freq)
        df[f"aroonosc{freq}_trend"] = 10
        df.loc[df[f"aroonosc{freq}"].between(-70, 70), f"aroonosc{freq}_trend"] = 0

        """osci too hard -100 to 100 https://www.fmlabs.com/reference/default.htm?url=CCI.htm"""
        df[f"cci{freq}"] = talib.CCI(df["high"], df["low"], df["close"], freq)  # basically modified rsi
        df[f"cci{freq}_trend"] = 10
        df.loc[df[f"cci{freq}"].between(-70, 70), f"cci{freq}_trend"] = 0

        """osci too hard 0 to 100 but rarely over 60 https://www.fmlabs.com/reference/default.htm?url=CMO.htm"""
        df[f"cmo{freq}"] = talib.CMO(df["close"], freq)  # -100 to 100
        df[f"cmo{freq}_trend"] = 10
        df.loc[df[f"cmo{freq}"].between(-70, 70), f"cmo{freq}_trend"] = 0

        """osci too hard 0 to 100 https://www.fmlabs.com/reference/default.htm?url=DX.htm"""
        df[f"dx{freq}"] = talib.DX(df["high"], df["low"], df["close"], timeperiod=freq)
        df[f"dx{freq}_trend"] = 10
        df.loc[df[f"dx{freq}"].between(-70, 70), f"dx{freq}_trend"] = 0

        """0 to 100 https://www.fmlabs.com/reference/default.htm?url=MFI.htm"""
        df[f"mfi{freq}"] = talib.MFI(df["high"], df["low"], df["close"], df["vol"], timeperiod=freq)
        df[f"mfi{freq}_trend"] = 10
        df.loc[df[f"mfi{freq}"].between(-70, 70), f"mfi{freq}_trend"] = 0

        """https://www.fmlabs.com/reference/default.htm?url=DI.htm"""
        df[f"minus_di{freq}"] = talib.MINUS_DI(df["high"], df["low"], df["close"], timeperiod=freq)
        df[f"plus_di{freq}"] = talib.PLUS_DI(df["high"], df["low"], df["close"], timeperiod=freq)

        """no doc"""
        df[f"minus_dm{freq}"] = talib.MINUS_DM(df["high"], df["low"], timeperiod=freq)
        df[f"plus_dm{freq}"] = talib.PLUS_DM(df["high"], df["low"], timeperiod=freq)

        """abs value, not relative http://www.fmlabs.com/reference/Momentum.htm"""
        df[f"mom{freq}"] = talib.MOM(df["close"], timeperiod=freq)

        """http://www.fmlabs.com/reference/PriceOscillatorPct.htm"""
        df[f"ppo{freq}"] = talib.PPO(df["close"], freq, freq * 2)
        # df["test"] = find_peaks_array(df[f"ppo{freq}"], freq)
        # df["test"] = df["test"].fillna(method="ffill")
        # print(df["test"].notna())

        """http://www.fmlabs.com/reference/PriceOscillatorPct.htm"""
        df[f"roc{freq}"] = talib.ROC(df["close"], freq)

        """0 to 100 rsi"""
        df[f"rsi{freq}"] = talib.RSI(df["close"], freq)
        df[f"rsi{freq}"] = ss3(df[f"rsi{freq}"], int(freq / 4))
        df[f"rsi{freq}_diff"] = df[f"rsi{freq}"].diff()

        df[f"rsi{freq}_trend"] = 10
        df.loc[(df[f"rsi{freq}"].between(-48, 52)) & (df[f"rsi{freq}_diff"].between(-0.3, 0.3)), f"rsi{freq}_trend"] = 0

        """ """
        df[f"stochrsi_fastk{freq}"], df[f"stochrsi_fastd{freq}"] = talib.STOCHRSI(df["close"], freq, int(freq / 2), int(freq / 3))

        """http://www.fmlabs.com/reference/PriceOscillatorPct.htm"""
        df[f"ultiosc{freq}"] = talib.ULTOSC(df["high"], df["low"], df["close"], int(freq / 2), freq, int(freq * 2))

        """http://www.fmlabs.com/reference/PriceOscillatorPct.htm"""
        df[f"willr{freq}"] = talib.WILLR(df["high"], df["low"], df["close"], freq)

        # VOLUME INDICATORS
        df[f"adosc{freq}"] = talib.ADOSC(df["high"], df["low"], df["close"], df["vol"], freq, freq * 3)

        # volatility indicators
        df[f"atr{freq}"] = talib.ATR(df["high"], df["low"], df["close"], freq)
        df[f"atr_helper{freq}"] = df[f"atr{freq}"].rolling(60).mean()

        df[f"rsi{freq}_true"] = (df[f"rsi{freq}"].between(45, 55)).astype(int)
        df[f"adx{freq}_true"] = (df[f"adx{freq}"] < 10).astype(int)
        df[f"atr{freq}_true"] = (df[f"atr{freq}"] < df[f"atr_helper{freq}"]).astype(int)

    df[f"flat{freq}"] = (df[f"rsi{freq}_true"] + df[f"adx{freq}_true"] + df[f"atr{freq}_true"]) * 10

    a_stable = a_stable + [f"adx{freq}", f"adx{freq}_trend"]

    df[["close"] + a_stable].plot(legend=True)
    plt.show()


# generate test for all fund stock index and for all strategy and variables.
# a_freqs=[5, 10, 20, 40, 60, 80, 120, 160, 200, 240, 360, 500, 750],
# kwargs= {"func": Alpha.macd, "fname": "macd_for_all", "a_kwargs": [{}, {}, {}, {}]}
def atest(asset="E", step=1, d_queries={}, kwargs={}, tomorrow=1):
    """
    This is a general statistic test creator
    1. provide all cases
    2. The little difference between this and brute force: bruteforce only creates indicator, but not assign buy/sell signals with 10 or -10
    Variables on how to loop over are in the function. apply function variables are in the dict kwargs

    Tomorrow
    tomorrow is a bit tricky here: it describes how many days to look forward
    It should be paired using past days

    So it is better to be modeled here in atest rather than in auto
    """
    d_preload = DB.preload(asset=asset, step=step, period_abv=240, d_queries_ts_code=d_queries)

    for counter_outer, one_kwarg in enumerate(kwargs["a_kwargs"]):
        param_string = '_'.join([f'{key}{value}' for key, value in one_kwarg.items()])
        a_path = LB.a_path(f"Market/CN/Atest/{kwargs['fname']}/{one_kwarg['abase']}/{asset}_step{step}_{kwargs['fname']}_tomorrow{tomorrow}_{param_string}")
        if os.path.exists(a_path[0]):
            print(f"path exists: {a_path[0]}")
            continue

        df_result = pd.DataFrame()
        for counter, (ts_code, df_asset) in enumerate(d_preload.items()):
            print(f"{counter}: asset:{asset}, {ts_code}, step:{step}, {kwargs['fname']}, {one_kwarg}")

            try:
                func_return_column = kwargs["func"](df=df_asset, **one_kwarg)[0]

                """
                -CAN NOT use sharp here Because len are different. smaller period will always have different std and hence different sharp
                -could also add pearson, but since outcome is binary, no need for pearson"""

                # init: calculate future price in theory by using tomorrow variable
                # Very important: shift always -1 because wait for night to see the signal. fgain choices are limited by creation
                df_asset[f"tomorrow{tomorrow}"] = df_asset[f"open.fgain(freq={tomorrow})"].shift(-1)  # one day delayed signal. today signal, tomorrow buy, atomorrow sell

                # general
                df_result.at[ts_code, "period"] = len(df_asset)
                # df_result.at[ts_code, "sharp"] = asset_sharp = (df_asset[f"tomorrow{tomorrow}"]).mean()/(df_asset[f"tomorrow{tomorrow}"]).std()
                df_result.at[ts_code, "gmean"] = asset_gmean = gmean(df_asset[f"tomorrow{tomorrow}"].dropna())
                df_result.at[ts_code, "daily_winrate"] = ((df_asset[f"tomorrow{tomorrow}"] > 1).astype(int)).mean()

                # if strategy signals buy
                df_long = df_asset[df_asset[func_return_column] == one_kwarg["score"]]
                # df_result.at[ts_code, "long_sharp_"] = (df_long[f"tomorrow{tomorrow}"].mean()) / (df_long[f"tomorrow{tomorrow}"].std()) /asset_sharp
                df_result.at[ts_code, "long_gmean_"] = gmean(df_long[f"tomorrow{tomorrow}"].dropna()) / asset_gmean
                df_result.at[ts_code, "long_daily_winrate"] = ((df_long[f"tomorrow{tomorrow}"] > 1).astype(int)).mean()
                df_result.at[ts_code, "long_occ"] = len(df_long) / len(df_asset)

                # if strategy signals sell
                df_short = df_asset[df_asset[func_return_column] == -one_kwarg["score"]]
                # df_result.at[ts_code, "short_sharp_"] = (df_short[f"tomorrow{tomorrow}"].mean()) / (df_short[f"tomorrow{tomorrow}"].std()) / asset_sharp
                df_result.at[ts_code, "short_gmean_"] = gmean(df_short[f"tomorrow{tomorrow}"].dropna()) / asset_gmean
                df_result.at[ts_code, "short_daily_winrate"] = ((df_short[f"tomorrow{tomorrow}"] > 1).astype(int)).mean()
                df_result.at[ts_code, "short_occ"] = len(df_short) / len(df_asset)


            except Exception as e:
                print("exception at func execute", e)
                continue

        # create sample
        if counter_outer == len(kwargs["a_kwargs"]) - 1:
            key = [x for x in d_preload.keys()]
            df_sample = d_preload[key[0]]
            a_path_sample = LB.a_path(f"Market/CN/Atest/{kwargs['fname']}/{one_kwarg['abase']}/SAMPLE_{asset}_step{step}_{kwargs['fname']}_tomorrow{tomorrow}_{param_string}")
            LB.to_csv_feather(df=df_sample, a_path=a_path_sample, skip_feather=True)

        # important check only if up/downtrend_gmean are not nan. Which means they actually exist for this strategy.
        for one in ["long", "short"]:
            for two in ["gmean"]:  # "sharp"
                df_result.loc[df_result[f"{one}_{two}_"].notna(), f"{one}_{two}_better"] = (df_result.loc[df_result[f"{one}_{two}_"].notna(), f"{one}_{two}_"] > 1).astype(int)
                df_result[f"{one}_{two}_std"] = df_result[f"{one}_{two}_"].std()

        # very slow witgh DB.to_excel_with_static_data(df_ts_code=df_result, path=path, sort=[])
        LB.to_csv_feather(df=df_result, a_path=a_path, skip_feather=True)

    # create summary for all

    d_summary = {"summary": pd.DataFrame()}
    for one, two, three in itertools.product(["long", "short"], ["gmean", ], ["", "better", "std"]):  # sharp
        name = f"{one}_{two}_{three}"
        print(name)
        d_summary[name] = pd.DataFrame()

    abase = one_kwarg['abase']  # abase should not change during iteration.otherwise unstable
    for one_kwarg in kwargs["a_kwargs"]:
        param_string = '_'.join([f'{key}{value}' for key, value in one_kwarg.items()])
        a_path = LB.a_path(f"Market/CN/Atest/{kwargs['fname']}/{one_kwarg['abase']}/{asset}_step{step}_{kwargs['fname']}_tomorrow{tomorrow}_{param_string}")

        print(f"summarizing {a_path[0]}")
        df_saved = pd.read_csv(a_path[0])
        d_summary["summary"].at[a_path[0], "daily_winrate"] = df_saved["daily_winrate"].mean()

        d_summary["summary"].at[a_path[0], "long_occ"] = df_saved["long_occ"].mean()
        d_summary["summary"].at[a_path[0], "short_occ"] = df_saved["short_occ"].mean()
        d_summary["summary"].at[a_path[0], f"long_daily_winrate"] = df_saved["long_daily_winrate"].mean()
        d_summary["summary"].at[a_path[0], f"short_daily_winrate"] = df_saved["short_daily_winrate"].mean()
        # d_summary["summary"].at[a_path[0], "sharp"] = df_saved["sharp"].mean()
        d_summary["summary"].at[a_path[0], "gmean"] = df_saved["gmean"].mean()

        d_helper = {}
        for one in ["long", "short"]:
            for two in ["gmean"]:  # sharp
                for three in ["", "better", "std"]:
                    d_summary["summary"].at[a_path[0], f"{one}_{two}_{three}"] = d_helper[f"{one}_{two}_{three}"] = df_saved[f"{one}_{two}_{three}"].mean()

        # create heatmap only if two frequencies are involved in creation
        # if up/downtrend exists, is it better than mean?
        # if "sfreq" in one_kwarg and "bfreq" in one_kwarg:
        #     df_long_sharp.at[f"{one_kwarg['sfreq']}_abv", one_kwarg["bfreq"]] = long_sharp
        #     df_short_sharp.at[f"{one_kwarg['sfreq']}_abv", one_kwarg["bfreq"]] = short_sharp
        #     df_long_sharp_better.at[f"{one_kwarg['sfreq']}_abv", one_kwarg["bfreq"]] = long_sharp_better
        #     df_short_sharp_better.at[f"{one_kwarg['sfreq']}_abv", one_kwarg["bfreq"]] = short_sharp_better

        for one in ["long", "short"]:
            for two in ["gmean"]:  # sharp
                for three in ["", "better", "std"]:
                    lol = d_helper[f"{one}_{two}_{three}"]
                    d_summary[f"{one}_{two}_{three}"].at[f"norm_freq{one_kwarg['norm_freq']}", f"{one_kwarg['q_low'], one_kwarg['q_high']}"] = lol

        # d_summary["short_sharp"].at[f"norm_freq{one_kwarg['norm_freq']}", f"{one_kwarg['q_low'],one_kwarg['q_high']}"] = short_sharp
        # d_summary["long_sharp_better"].at[f"norm_freq{one_kwarg['norm_freq']}", f"{one_kwarg['q_low'],one_kwarg['q_high']}"] = long_sharp_better
        # d_summary["short_sharp_better"].at[f"norm_freq{one_kwarg['norm_freq']}", f"{one_kwarg['q_low'],one_kwarg['q_high']}"] = short_sharp_better
        # d_summary["long_sharp_std"].at[f"norm_freq{one_kwarg['norm_freq']}", f"{one_kwarg['q_low'], one_kwarg['q_high']}"] = long_sharp_std
        # d_summary["short_sharp_std"].at[f"norm_freq{one_kwarg['norm_freq']}", f"{one_kwarg['q_low'], one_kwarg['q_high']}"] = short_sharp_std

    LB.to_excel(path=f"Market/CN/Atest/{kwargs['fname']}/{abase}/SUMMARY_{asset}_step{step}_{kwargs['fname']}_tomorrow{tomorrow}_{param_string}.xlsx", d_df=d_summary)


def atest_manu(fname="macd", a_abase=["close"]):
    for abase in a_abase:

        # setting generation
        a_kwargs = []
        if fname == "macd":
            func = macd
            d_steps = {"F": 1, "FD": 2, "G": 1, "I": 2, "E": 6}
            for sfreq, bfreq in LB.custom_pairwise_combination([5, 10, 20, 40, 60, 80, 120, 180, 240, 320, 400, 480], 2):
                if sfreq < bfreq:
                    a_kwargs.append({"abase": abase, "sfreq": sfreq, "bfreq": bfreq, "type": 1, "score": 1})
        elif fname == "is_max":
            func = ismax
            d_steps = {"F": 1, "FD": 1, "G": 1, "I": 1, "E": 1}
            for q in np.linspace(0, 1, 6):
                a_kwargs.append({"abase": abase, "q": q, "score": 1})
        elif fname == "is_min":
            func = ismin
            d_steps = {"F": 1, "FD": 1, "G": 1, "I": 1, "E": 1}
            for q in np.linspace(0, 1, 6):
                a_kwargs.append({"abase": abase, "q": q, "score": 1})

        # run atest
        LB.print_iterables(a_kwargs)
        for asset in ["F", "FD", "G", "I", "E"]:
            atest(asset=asset, step=d_steps[asset], kwargs={"func": func, "fname": fname, "a_kwargs": a_kwargs}, d_queries=LB.c_G_queries() if asset == "G" else {})


def atest_auto(type=4):
    def auto(df, abase, q_low=0.2, q_high=0.4, norm_freq=240, type=1, score=10):
        """can be used on any indicator
        gq=Generic quantile
        0. create an oscilator of that indicator
        1. create expanding mean of that indicator
        2. create percent=today_indicator/e_indicator
        3. assign rolling quantile quantile of percent

        This appoach needs to be mean stationary !!!! otherwise quantile makes nosense
        Also: This approach only makes sense on non stationary data!!!
        On columns like pct_chg it doesnt make any sense

        """
        # init
        name = f"{abase}.auto{norm_freq}.type{type}"
        if f"norm_{name}" not in df.columns:
            # 3 choices. cg_oscilator, rsi, (today-yesterday)/today
            if type == 1:
                df[f"norm_{name}"] = cg(df[abase], norm_freq)
            elif type == 2:
                df[f"norm_{name}"] = talib.RSI(df[abase], norm_freq)
            elif type == 3:
                # this is the same as ROCP rate of change percent
                df[f"norm_{name}"] = (df[abase] - df[abase].shift(1)) / df[abase].shift(1)
            elif type == 4:
                # DONT ADD 1+ here
                df[f"norm_{name}"] = 1 + df[abase].pct_change(norm_freq)

        # create expanding quantile
        for q in [q_low, q_high]:
            if f"q{q}_{name}" not in df.columns:
                df[f"q{q}_{name}"] = df[f"norm_{name}"].expanding(240).quantile(q)

        # assign todays value to a quantile
        df[f"in_q{q_low, q_high}_{name}"] = ((df[f"q{q_low}_{name}"] <= df[f"norm_{name}"]) & (df[f"norm_{name}"] <= df[f"q{q_high}_{name}"])).astype(int) * score
        df[f"in_q{q_low, q_high}_{name}"] = df[f"in_q{q_low, q_high}_{name}"].replace(to_replace=0, value=-score)
        return [f"in_q{q_low, q_high}_{name}", ]

    # atest_auto starts here
    for asset in ["E", "I", "FD"]:  # ,"FD","G","I","E"
        # get example column of this asset
        a_example_column = DB.get_example_column(asset=asset, numeric_only=True)
        # remove unessesary columns:
        a_columns = []
        for column in a_example_column:
            for exclude_column in ["fgain"]:
                if exclude_column not in column:
                    a_columns.append(column)

        for col in a_columns:
            # setting generation
            a_kwargs = []
            func = auto
            fname = func.__name__
            tomorrow = 1  # how many days to forward predict. ideally [1,5,10,20,60,240]
            d_steps = {"F": 1, "FD": 1, "G": 1, "I": 1, "E": 4}
            for norm_freq in [5, 10, 20, 60, 120, 240, 500]:
                for q_low, q_high in LB.custom_pairwise_overlap(LB.drange(0, 101, 10)):
                    a_kwargs.append({"abase": col, "q_low": q_low, "q_high": q_high, "norm_freq": norm_freq, "score": 1, "type": type})

            # run atest
            LB.print_iterables(a_kwargs)
            atest(asset=asset, tomorrow=tomorrow, step=d_steps[asset], kwargs={"func": func, "fname": fname, "a_kwargs": a_kwargs}, d_queries=LB.c_G_queries() if asset == "G" else {})


def asset_start_season(df, n=1, type="year"):
    """
    Hypothesis Question: if first n month return is positive, how likely is the whole year positive?

    1. convert day format to month format
    2. create month and year df
    3. merge together
    4. analyze pct_chg

    True_True, True_False,False_True, False_False are to determine the correct prediction
    Pearson, spearman are to predict the strengh of prediction
    """

    if type == "monthofyear":  # 1-12
        suffix1 = "_y"
        suffix2 = "_m"
        df_year = LB.df_to_freq(df, "Y")
        df_month = LB.df_to_freq(df, "W")

        df_year["index_copy"] = df_year.index
        df_year["year"] = df_year["index_copy"].apply(lambda x: LB.trade_date_to_year(x))  # can be way more efficient

        df_month["index_copy"] = df_month.index
        df_month["year"] = df_month["index_copy"].apply(lambda x: LB.trade_date_to_year(x))  # can be way more efficient
        df_month["month"] = df_month["index_copy"].apply(lambda x: LB.trade_date_to_month(x))  # can be way more efficient
        df_month = df_month[df_month["month"] == n]

        df_combined = pd.merge(df_year, df_month, on="year", how="left", suffixes=[suffix1, suffix2], sort=False)
    elif type == "seasonofyear":  # 1-4
        suffix1 = "_y"
        suffix2 = "_s"
        df_year = LB.df_to_freq(df, "Y")
        df_season = LB.df_to_freq(df, "S")

        df_year["index_copy"] = df_year.index
        df_year["year"] = df_year["index_copy"].apply(lambda x: LB.trade_date_to_year(x))  # can be way more efficient

        df_season["index_copy"] = df_season.index
        df_season["year"] = df_season["index_copy"].apply(lambda x: LB.trade_date_to_year(x))  # can be way more efficient
        df_season["season"] = df_season["index_copy"].apply(lambda x: LB.trade_date_to_season(x))  # can be way more efficient
        df_season = df_season[df_season["season"] == n]

        df_combined = pd.merge(df_year, df_season, on="year", how="left", suffixes=[suffix1, suffix2], sort=False)

        pass
    elif type == "weekofmonth":  # 1-6
        suffix1 = "_m"
        suffix2 = "_w"
        df_month = LB.df_to_freq(df, "M")
        df_week = LB.df_to_freq(df, "W")

        df_month["index_copy"] = df_month.index
        df_month["year"] = df_month["index_copy"].apply(lambda x: LB.trade_date_to_year(x))  # can be way more efficient
        df_month["month"] = df_month["index_copy"].apply(lambda x: LB.trade_date_to_month(x))  # can be way more efficient

        df_week["index_copy"] = df_week.index
        df_week["year"] = df_week["index_copy"].apply(lambda x: LB.trade_date_to_year(x))  # can be way more efficient
        df_week["month"] = df_week["index_copy"].apply(lambda x: LB.trade_date_to_month(x))  # can be way more efficient
        df_week["weekofmonth"] = df_week["index_copy"].apply(lambda x: LB.trade_date_to_weekofmonth(x))  # can be way more efficient
        df_week = df_week[df_week["weekofmonth"] == n]

        df_combined = pd.merge(df_month, df_week, on=["year", "month"], how="left", suffixes=[suffix1, suffix2], sort=False)


    elif type == "dayofweek":  # 1-5
        pass

    # many ways to determine that
    periods = len(df_combined)
    TT = len(df_combined[(df_combined[f"pct_chg{suffix2}"] > 0) & (df_combined[f"pct_chg{suffix1}"] > 0)]) / periods
    TF = len(df_combined[(df_combined[f"pct_chg{suffix2}"] > 0) & (df_combined[f"pct_chg{suffix1}"] < 0)]) / periods
    FT = len(df_combined[(df_combined[f"pct_chg{suffix2}"] < 0) & (df_combined[f"pct_chg{suffix1}"] > 0)]) / periods
    FF = len(df_combined[(df_combined[f"pct_chg{suffix2}"] < 0) & (df_combined[f"pct_chg{suffix1}"] < 0)]) / periods
    pearson = df_combined[f"pct_chg{suffix2}"].corr(df_combined[f"pct_chg{suffix1}"])
    spearman = df_combined[f"pct_chg{suffix2}"].corr(df_combined[f"pct_chg{suffix1}"], method="spearman")
    return pd.Series({"periods": periods, "TT": TT, "TF": TF, "FT": FT, "FF": FF, "pearson": pearson, "spearman": spearman})


def asset_start_season_initiator(asset="I", a_n=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], type="monthofyear"):
    if asset == "I":
        d_queries_ts_code = {"I": ["category != '债券指数' "]}
    elif asset == "G":
        d_queries_ts_code = LB.c_G_queries()
    else:
        d_queries_ts_code = {}

    d_preload = DB.preload(asset=asset, step=1, d_queries_ts_code=d_queries_ts_code)
    for n in a_n:
        a_path = LB.a_path(f"Market/CN/ATest/start_season/{type}/{asset}/n{n}")
        if not os.path.isfile(a_path[0]):
            a_result = []

            for ts_code, df_asset in d_preload.items():
                print(f"start_tester {ts_code} {n}")
                s = asset_start_season(df=df_asset, n=n, type=type)
                s["ts_code"] = ts_code
                a_result.append(s)
            df_result = pd.DataFrame(a_result)
            LB.to_csv_feather(df=df_result, a_path=a_path)

    # summarizing summary
    a_result = []
    for n in a_n:
        a_path = LB.a_path(f"Market/CN/ATest/start_season/{type}/{asset}/n{n}")
        df = DB.get(a_path, set_index="index")
        df = df.mean()
        a_result.append(df)
        print("load", a_path[0])
    df_result = pd.DataFrame(a_result)

    a_path = LB.a_path(f"Market/CN/ATest/start_season/{type}/{asset}/summary_{type}_{asset}")
    LB.to_csv_feather(df=df_result, a_path=a_path, skip_feather=True)



def asset_seasonal_first_week_year():

    """
    Result:
    年底效应要比年初效应明显
    可以比较的是每年过年5五天+后5天

    :return:
    """
    df_trade_date = DB.get_trade_date()
    df_trade_date = df_trade_date[df_trade_date.index >= 20110101]
    df_trade_date["counter"] = range(1, len(df_trade_date) + 1)

    # find all week before and after ny
    a_year_end = []
    a_year_begin = []
    for year in df_trade_date["year"].unique():
        df_last_jan_master = df_trade_date[df_trade_date["year"] == year]
        df_last_jan = df_last_jan_master.tail(5)

        for index in df_last_jan.index:
            a_year_end += [index]

        df_last_jan = df_last_jan_master.head(5)
        for index in df_last_jan.index:
            a_year_begin += [index]

    for index, asset in zip(["000001.SH", "399001.SZ", "399006.SZ", "asset_E","IXIC"], ["I", "I", "I", "G","Egal"]):
        if index != "IXIC":
            df = DB.get_asset(index, asset=asset)
        else:
            pass
        end = df.loc[a_year_end, "pct_chg"].mean()
        begin = df.loc[a_year_begin, "pct_chg"].mean()
        normalmean = df["pct_chg"].mean()
        print(index, "last 5 day of year", end / normalmean)
        print(index, "first 5 day of year", begin / normalmean)


def asset_seasonal_last_jan_week():
    """index 1 week before and after ny

    result:
    -的确每年一月份最后一周都跌的厉害
    """

    df_trade_date=DB.get_trade_date()
    df_trade_date=df_trade_date[df_trade_date.index >= 20110101]
    df_trade_date["counter"]=range(1, len(df_trade_date) + 1)

    #find all week before and after ny
    a_jan=[]
    for year in df_trade_date["year"].unique():
        df_last_jan=df_trade_date[df_trade_date["year"]==year]
        df_last_jan=df_last_jan[df_last_jan["month"]==1]
        df_last_jan=df_last_jan.tail(5)

        for index in df_last_jan.index:
            a_jan+=[index]


    for index,asset in zip(["000001.SH","399001.SZ","399006.SZ","asset_E","IXIC"],["I","I","I","G","egal"]):
        if index != "IXIC":
            df = DB.get_asset(index, asset=asset)
        else:
            pass
        jan = df.loc[a_jan,"pct_chg"].mean()
        normalmean=df["pct_chg"].mean()
        print(index,"last 5 days in jan",jan,normalmean)


def asset_seasonal_before_ny():
    """index 1 week before and after ny

    result:
    - after ny is better than before new year
    - before new year is best when 5 days before
    -Summary: buy 5 days before ny and hold 30 days after ny.

    """

    df_trade_date=DB.get_trade_date()
    df_trade_date=df_trade_date[df_trade_date.index >= 20110101]
    df_trade_date["counter"]=range(1, len(df_trade_date) + 1)

    for days in [5,10,15,20, 30]:


        #find all week before and after ny
        df_ny=df_trade_date[df_trade_date["new_year"]==1]
        a_trade_dates_before=[]
        a_trade_dates_after=[]
        for trade_date in df_ny.index:
            c=df_trade_date.at[trade_date,"counter"]

            for i in range(0,days):
                df_filter=df_trade_date[df_trade_date["counter"]==c-i]
                try:
                    a_trade_dates_before+=[int(df_filter.index[0])]
                except:
                    pass

            for i in range(0,days):
                df_filter=df_trade_date[df_trade_date["counter"]==c+i+1]
                try:
                    a_trade_dates_after+=[int(df_filter.index[0])]
                except:
                    pass



        for index,asset in zip(["000001.SH","399001.SZ","399006.SZ","asset_E","IXIC"],["I","I","I","G","egal"]):
            if index != "IXIC":
                df = DB.get_asset(index, asset=asset)
            else:
                pass
            mean_before = df.loc[a_trade_dates_before,"pct_chg"].mean()
            mean_after = df.loc[a_trade_dates_after,"pct_chg"].mean()
            normalmean=df["pct_chg"].mean()
            print(days,index,"before",mean_before/normalmean)
            print(days,index,"after",mean_after/normalmean)


def asset_seasonal_statistic_sh():
    #use sh index as source
    df_trade_date=DB.get_trade_date()
    df_sh=DB.get_asset(ts_code="000001.SH",asset="I")
    df_sh["vol_pct_chg"]=df_sh["vol"].pct_change()
    df_sh=df_sh[["pct_chg","vol_pct_chg"]]

    for division in ["month","day","weekofyear","dayofweek"]:
        df_sh[division]=df_trade_date[division]
        df_result = df_sh.groupby([division]).agg(["mean","std"])
        a_path = LB.a_path(f"Market/CN/ATest/seasonal_sh/{division}")
        LB.to_csv_feather(df=df_result, a_path=a_path, skip_feather=True)
        del df_sh[division]


    #add new year data (this method only works for SH. it is a lazy method)
    df_new_year=pd.DataFrame()
    df_trade_date= df_trade_date.reset_index()
    df_sh= df_sh.reset_index()

    for freq in [3,5,10,15,20,30,40]:
        #mark all dates that are within the range of new year
        #get all start of ny
        df_trade_date[f"new_year_marker{freq}"]=0
        df_helper= df_trade_date[df_trade_date["new_year"]==1]

        for start in df_helper.index:

            #n days AFTER START of ny
            for marker in range(start-freq+1,start+1):
                df_trade_date.at[marker,f"new_year_marker{freq}"]=1

            # n days AFTER END of ny
            for marker in range(start , start + freq+1):
                df_trade_date.at[marker, f"new_year_marker{freq}"] = 1

        df_sh[f"new_year_marker{freq}"]=df_trade_date[f"new_year_marker{freq}"]
        df_sh_helper=df_sh[df_sh[f"new_year_marker{freq}"]==1]
        df_new_year.at[f"freq{freq}","pct_chg"]=df_sh_helper["pct_chg"].mean()

    a_path = LB.a_path(f"Market/CN/ATest/seasonal_sh/new_year")
    LB.to_csv_feather(df=df_new_year, a_path=a_path, skip_feather=True)





def asset_seasonal_statistic_stock():
    #use all stocks as source
    df_trade_date=DB.get_trade_date()
    d_preload=DB.preload(asset="E", freq="D", on_asset=True, step=1, market="CN")

    for division in ["month","day","weekofyear","dayofweek"]:
        a_division_result=[]
        divideby=0

        for ts_code, df_asset in d_preload.items():
            print(f"{ts_code}, {division}")
            divideby+=1
            df_asset["vol_pct_chg"] = df_asset["vol"].pct_change()
            df_asset = df_asset[["pct_chg", "vol_pct_chg"]]
            df_asset[division]=df_trade_date[division]
            df_asset_result = df_asset.groupby([division]).agg(["mean"])
            a_division_result+=[df_asset_result]


        #choose first df as base and then set al values to 0
        df_division_result=a_division_result[0]
        for column in df_division_result.columns:
            df_division_result[column]=0.0

        #add all df together
        for df in a_division_result:
            df_division_result=df_division_result.add(df, fill_value=0)

        df_division_result=df_division_result/divideby
        a_path = LB.a_path(f"Market/CN/ATest/seasonal_stock/{division}")
        LB.to_csv_feather(df=df_division_result, a_path=a_path, skip_feather=True)




def asset_prob_gain_asset(asset="E"):
    """
    Answers this question:
    1. If % of previous n days is up/down, what is probability for next n day to be up/down

    The goal is to predict the direction of the movement.
    A: the more previous days are down, the more likely future days are up
    """
    d_preload = DB.preload(asset=asset, step=10)
    for n in [3, 4, 5, 6, 10, 20, 40, 60]:
        df_result = pd.DataFrame()
        df_heat = pd.DataFrame()
        path = f"Market/CN/Atest/prob_gain/{n}_summary.xlsx"

        if not os.path.isfile(path):
            for ts_code, df_asset in d_preload.items():
                print(f"prob_gain {n}: {ts_code}")

                # reset index to later easier calcualte days
                df_asset = df_asset.reset_index()

                df_asset["probgaingeneric"] = (df_asset["pct_chg"] > 0).astype(int)
                df_asset[f"probggain_init{n}"] = df_asset["probgaingeneric"].rolling(n).sum()

                for subn in range(0, n + 1):
                    # Mark day where % of past n days are positive = meet the criteria
                    df_asset[f"probggain_marker{n, subn}"] = 0
                    df_asset.loc[df_asset[f"probggain_init{n}"] == subn, f"probggain_marker{n, subn}"] = 1

                    df_filter = df_asset[df_asset[f"probggain_marker{n, subn}"] == 1]
                    occurence = len(df_filter) / len(df_asset)
                    sharp_fgain5 = df_filter["close.fgain5"].mean() / df_filter["close.fgain5"].std() if df_filter["close.fgain5"].std() != 0 else np.nan

                    # for each match day in df_filter, check out their next 5 days
                    a_pct_positive = []
                    for index in df_filter.index:
                        df_part = df_asset.loc[index + 1:index + 6]
                        print(f"prob_gain {n}: {ts_code} {len(df_part)}")
                        if not df_part.empty:
                            pct_positive = len(df_part[df_part["pct_chg"] > 0]) / len(df_part)
                            a_pct_positive.append(pct_positive)

                    s_result = pd.Series(a_pct_positive)
                    positive = s_result.mean()

                    df_result.at[ts_code, f"{n, subn}_occ"] = occurence
                    df_result.at[ts_code, f"{n, subn}_pct_positive"] = positive
                    df_result.at[ts_code, f"{n, subn}_sharp_gain5"] = sharp_fgain5

            LB.to_excel(path=path, d_df={"Overview": df_result, "Heat": df_heat})





def asset_intraday_analysis():
    """
    The result of intraday analysis:
    -Highest deviation at first 15 min of trading day
    -Lowest deviation at last 15 min
    -if first 15 min are positive, there are 65% the whole day is positive
    """
    var = 15
    asset = "I"
    for ts_code in ["000001.SH", "399006.SZ", "399001.SZ"]:
        df = pd.read_csv(f"D:\Stock\Market\CN\Asset\{asset}\{var}m/{ts_code}.csv")

        df["pct_chg"] = df["close"].pct_change()

        df["day"] = df["date"].str.slice(0, 10)
        df["intraday"] = df["date"].str.slice(11, 22)
        df["h"] = df["intraday"].str.slice(0, 2)
        df["m"] = df["intraday"].str.slice(3, 5)
        df["s"] = df["intraday"].str.slice(6, 8)

        df_result = pd.DataFrame()
        a_intraday = list(df["intraday"].unique())
        # 1.part stats about mean and volatility
        for intraday in a_intraday:
            df_filter = df[df["intraday"] == intraday]
            mean = df_filter["pct_chg"].mean()
            pct_chg_pos = len(df_filter[df_filter["pct_chg"] > 0]) / len(df_filter)
            pct_chg_neg = len(df_filter[df_filter["pct_chg"] < 0]) / len(df_filter)
            std = df_filter["pct_chg"].std()
            sharp = mean / std
            df_result.at[intraday, "mean"] = mean
            df_result.at[intraday, "pos"] = pct_chg_pos
            df_result.at[intraday, "neg"] = pct_chg_neg
            df_result.at[intraday, "std"] = std
            df_result.at[intraday, "sharp"] = sharp
        df_result.to_csv(f"intraday{ts_code}.csv")

        # 2.part:prediction. first 15 min predict today
        a_results = []
        for intraday in a_intraday:
            df_day = DB.get_asset(ts_code=ts_code, asset=asset)
            df_filter = df[df["intraday"] == intraday]
            df_filter["trade_date"] = df_filter["day"].apply(LB.df_switch_trade_date)
            df_filter["trade_date"] = df_filter["trade_date"].astype(int)
            df_final = pd.merge(LB.df_ohlcpp(df=df_day), df_filter, on="trade_date", suffixes=["_d", "_15m"], sort=False)

            df_final["pct_chg_d"] = df_final["pct_chg_d"].shift(-1)
            df_final.to_csv(f"intraday_prediction_{ts_code}.csv")

            len_df = len(df_final)

            TT = len(df_final[(df_final["pct_chg_15m"] > 0) & (df_final["pct_chg_d"] > 0)]) / len_df
            TF = len(df_final[(df_final["pct_chg_15m"] > 0) & (df_final["pct_chg_d"] < 0)]) / len_df
            FT = len(df_final[(df_final["pct_chg_15m"] < 0) & (df_final["pct_chg_d"] > 0)]) / len_df
            FF = len(df_final[(df_final["pct_chg_15m"] < 0) & (df_final["pct_chg_d"] < 0)]) / len_df

            # rolling version
            rolling = 5
            df_final[f"pct_chg_15m_r{rolling}"] = df_final[f"pct_chg_15m"].rolling(rolling).mean()
            pearson = df_final[f"pct_chg_15m_r{rolling}"].corr(df_final["pct_chg_d"])
            spearman = df_final[f"pct_chg_15m_r{rolling}"].corr(df_final["pct_chg_d"], method="spearman")

            s = pd.Series({"intraday": intraday, "TT": TT, "TF": TF, "FT": FT, "FF": FF, "pearson": pearson, "sparman": spearman})
            a_results.append(s)
        df_predict_result = pd.DataFrame(a_results)
        df_predict_result.to_csv(f"intraday_prediction_result_{ts_code}.csv")


# WHO WAS GOOD DURING THAT TIME PERIOD
# ASSET INFORMATION
# measures the fundamentals aspect
"""does not work in general. past can not predict future here"""


def asset_fundamental(start_date, end_date, freq, assets=["E"]):
    asset = assets[0]
    ts_codes = DB.get_ts_code(a_asset=[asset])
    a_result_mean = []
    a_result_std = []

    ts_codes = ts_codes[::-1]
    small = ts_codes[(ts_codes["exchange"] == "创业板") | (ts_codes["exchange"] == "中小板")]
    big = ts_codes[(ts_codes["exchange"] == "主板")]

    print("small size", len(small))
    print("big size", len(big))

    ts_codes = ts_codes

    for ts_code in ts_codes.ts_code:
        print("start appending to asset_fundamental", ts_code)

        # get asset
        df_asset = DB.get_asset(ts_code, asset, freq)
        if df_asset.empty:
            continue
        df_asset = df_asset[df_asset["trade_date"].between(int(start_date), int(end_date))]

        # get all label
        fun_balancesheet_label_list = ["pe_ttm", "ps_ttm", "pb", "total_mv", "profit_dedt", "total_cur_assets", "total_nca", "total_assets", "total_cur_liab", "total_ncl", "total_liab"]
        fun_cashflow_label_list = ["n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act"]
        fun_indicator_label_list = ["netprofit_yoy", "or_yoy", "grossprofit_margin", "netprofit_margin", "debt_to_assets", "turn_days"]
        fun_pledge_label_list = ["pledge_ratio"]
        fun_label_list = fun_balancesheet_label_list + fun_cashflow_label_list + fun_indicator_label_list + fun_pledge_label_list
        df_asset = df_asset[["ts_code", "period"] + fun_label_list]

        # calc reduced result
        ts_code = df_asset.at[0, "ts_code"]
        period = df_asset.at[len(df_asset) - 1, "period"]

        # calc result
        fun_result_mean_list = [df_asset[label].mean() for label in fun_label_list]
        fun_result_std_list = [df_asset[label].std() for label in fun_label_list]

        a_result_mean.append(list([asset, ts_code, period] + fun_result_mean_list))
        a_result_std.append(list([asset, ts_code, period] + fun_result_std_list))

    # create tab Asset View
    df_result_mean = pd.DataFrame(a_result_mean, columns=["asset"] + list(df_asset.columns))
    df_result_std = pd.DataFrame(a_result_std, columns=["asset"] + list(df_asset.columns))

    # create std rank
    # THE LESS STD THE BETTER
    df_result_mean["std_growth_rank"] = df_result_std["netprofit_yoy"] + df_result_std["or_yoy"]
    df_result_mean["std_margin_rank"] = df_result_std["grossprofit_margin"] + df_result_std["netprofit_margin"]
    df_result_mean["std_cashflow_op_rank"] = df_result_std["n_cashflow_act"]
    df_result_mean["std_cashflow_inv_rank"] = df_result_std["n_cashflow_inv_act"]
    df_result_mean["std_cur_asset_rank"] = df_result_std["total_cur_assets"]
    df_result_mean["std_cur_liab_rank"] = df_result_std["total_cur_liab"]
    df_result_mean["std_plus_rank"] = df_result_mean["std_growth_rank"] + df_result_mean["std_margin_rank"] + df_result_mean["std_cashflow_op_rank"] * 2 + df_result_mean["std_cashflow_inv_rank"] + df_result_mean["std_cur_asset_rank"] * 3

    df_result_mean["std_growth_rank"] = df_result_mean["std_growth_rank"].rank(ascending=False)
    df_result_mean["std_margin_rank"] = df_result_mean["std_margin_rank"].rank(ascending=False)
    df_result_mean["std_cashflow_op_rank"] = df_result_mean["std_cashflow_op_rank"].rank(ascending=False)
    df_result_mean["std_cashflow_inv_rank"] = df_result_mean["std_cashflow_inv_rank"].rank(ascending=False)
    df_result_mean["std_cur_asset_rank"] = df_result_mean["std_cur_asset_rank"].rank(ascending=False)
    df_result_mean["std_cur_liab_rank"] = df_result_mean["std_cur_liab_rank"].rank(ascending=False)
    df_result_mean["std_plus_rank"] = df_result_mean["std_plus_rank"].rank(ascending=False)

    # create mean rank

    # 7  asset rank
    # SMALLER BETTER, rank LOWER BETTER
    # the bigger the company, the harder to get good asset ratio
    df_result_mean["asset_score"] = (df_result_mean["debt_to_assets"] + df_result_mean["pledge_ratio"] * 3) * np.sqrt(df_result_mean["total_mv"])
    df_result_mean["asset_rank"] = df_result_mean["asset_score"].rank(ascending=True)

    # 0 mv score
    # Higher BETTER, the bigger the company the better return
    # implies that value stock are better than value stock
    df_result_mean["mv_score"] = df_result_mean["total_mv"]
    df_result_mean["mv_rank"] = df_result_mean["mv_score"].rank(ascending=False)

    # 6 cashflow rank
    # SMALLER BETTER, rank LOWER BETTER
    # cashflow the closer to profit the better
    df_result_mean["cashflow_o_rank"] = 1 - abs(df_result_mean["n_cashflow_act"] / df_result_mean["profit_dedt"])
    df_result_mean["cashflow_o_rank"] = df_result_mean["cashflow_o_rank"].rank(ascending=True)

    # higher the better
    df_result_mean["cashflow_netsum_rank"] = (df_result_mean["n_cashflow_act"] + df_result_mean["n_cashflow_inv_act"] + df_result_mean["n_cash_flows_fnc_act"]) / df_result_mean["total_mv"]
    df_result_mean["cashflow_netsum_rank"] = df_result_mean["cashflow_netsum_rank"].rank(ascending=False)

    df_result_mean["non_current_asset_ratio"] = df_result_mean["total_nca"] / df_result_mean["total_assets"]
    df_result_mean["non_current_liability_ratio"] = df_result_mean["total_ncl"] / df_result_mean["total_liab"]
    df_result_mean["current_liability_to_mv"] = df_result_mean["total_cur_assets"] / df_result_mean["total_mv"]

    # 8 other rank
    # SMALLER BETTER, rank LOWER BETTER
    df_result_mean["other_rank"] = df_result_mean["turn_days"]
    df_result_mean["other_rank"] = df_result_mean["other_rank"].rank(ascending=True)

    # 1 margin score
    # HIGHER BETTER, rank LOWER BETTER
    # the bigger and longer a company, the harder to get high margin
    df_result_mean["margin_score"] = (df_result_mean["grossprofit_margin"] * 0.5 + df_result_mean["netprofit_margin"] * 0.5) * (np.sqrt(df_result_mean["total_mv"])) * (df_result_mean["period"])
    df_result_mean["margin_rank"] = df_result_mean["margin_score"].rank(ascending=False)

    # 2 growth rank
    # the longer a firm exists, the bigger a company, the harder to keep growth rate
    # the higher the margin, the higher the growthrate, the faster it grow
    # HIGHER BETTER, rank LOWER BETTER
    df_result_mean["average_growth"] = df_result_mean["netprofit_yoy"] * 0.2 + df_result_mean["or_yoy"] * 0.8
    df_result_mean["period_growth_score"] = ((df_result_mean["average_growth"]) * (df_result_mean["margin_score"]))
    df_result_mean["period_growth_rank"] = df_result_mean["period_growth_score"].rank(ascending=False)

    # the bigger the better
    df_result_mean["test_score"] = df_result_mean["average_growth"] * (df_result_mean["grossprofit_margin"] * 0.5 + df_result_mean["netprofit_margin"] * 0.5) * np.sqrt(np.sqrt(np.sqrt(df_result_mean["total_mv"]))) * np.sqrt(df_result_mean["period"]) * (100 - df_result_mean["pledge_ratio"])
    df_result_mean["test_rank"] = df_result_mean["test_score"].rank(ascending=False)

    # 3 PEG
    # SMALLER BETTER, rank LOWER BETTER
    df_result_mean["peg_rank"] = df_result_mean["pe_ttm"] / df_result_mean["average_growth"]
    df_result_mean["peg_rank"] = df_result_mean["peg_rank"].rank(ascending=True)

    # 4 PSG
    # SMALLER BETTER, rank LOWER BETTER
    df_result_mean["psg_rank"] = df_result_mean["ps_ttm"] / df_result_mean["average_growth"]
    df_result_mean["psg_rank"] = df_result_mean["psg_rank"].rank(ascending=True)

    # 5 PBG
    # SMALLER BETTER, rank LOWER BETTER
    df_result_mean["pbg_rank"] = df_result_mean["pb"] / df_result_mean["average_growth"]
    df_result_mean["pbg_rank"] = df_result_mean["pbg_rank"].rank(ascending=True)

    # final rank
    df_result_mean["final_fundamental_rank"] = df_result_mean["margin_rank"] * 0.40 + \
                                               df_result_mean["period_growth_rank"] * 0.2 + \
                                               df_result_mean["peg_rank"] * 0.0 + \
                                               df_result_mean["psg_rank"] * 0.0 + \
                                               df_result_mean["pbg_rank"] * 0.0 + \
                                               df_result_mean["cashflow_o_rank"] * 0.0 + \
                                               df_result_mean["cashflow_netsum_rank"] * 0.1 + \
                                               df_result_mean["asset_rank"] * 0.05 + \
                                               df_result_mean["other_rank"] * 0.05 + \
                                               df_result_mean["std_plus_rank"] * 0.2
    df_result_mean["final_fundamental_rank"] = df_result_mean["final_fundamental_rank"].rank(ascending=True)

    # add static data and sort by final rank
    df_result_mean = DB.add_static_data(df_result_mean, assets=assets)
    df_result_mean.sort_values(by=["final_fundamental_rank"], ascending=True, inplace=True)

    path = "Market/" + "CN" + "/Atest/" + "fundamental" + "/" + ''.join(assets) + "_" + freq + "_" + start_date + "_" + end_date + ".xlsx"
    DB.to_excel_with_static_data(df_result_mean, path=path, sort=["final_fundamental_rank", True], a_assets=assets)


# measures the volatility aspect
def asset_volatility(start_date, end_date, assets, freq):
    a_result = []
    for asset in assets:
        ts_codes = DB.get_ts_code(a_asset=[asset])
        for ts_code in ts_codes.ts_code:
            print("start appending to asset_volatility", ts_code)

            # get asset
            df_asset = DB.get_asset(ts_code, asset, freq)
            if df_asset.empty:
                continue
            df_asset = df_asset[df_asset["trade_date"].between(int(start_date), int(end_date))]

            # get all label
            close_std_label_list = [s for s in df_asset.columns if "close_std" in s]
            ivola_std_label_list = [s for s in df_asset.columns if "ivola_std" in s]
            turnover_rate_std_label_list = [s for s in df_asset.columns if "turnover_rate_std" in s]
            beta_list = [s for s in df_asset.columns if "beta" in s]

            std_label_list = close_std_label_list + ivola_std_label_list + turnover_rate_std_label_list + beta_list
            df_asset = df_asset[["ts_code", "period"] + std_label_list]

            # calc reduced result
            ts_code = df_asset.at[0, "ts_code"]
            period = df_asset.at[len(df_asset) - 1, "period"]

            # calc result
            std_result_list = [df_asset[label].mean() for label in std_label_list]

            df_asset_reduced = [asset, ts_code, period] + std_result_list
            a_result.append(list(df_asset_reduced))

    # create tab Asset View
    df_result = pd.DataFrame(a_result, columns=["asset"] + list(df_asset.columns))

    # ranking
    # price: the higher the volatility between close prices each D the better
    # interday: the higher interday volatility the better
    # volume: the lower tor the better
    # beta: the lower the beta the better

    # calculate score
    df_result["close_score"] = sum([df_result[label] for label in close_std_label_list]) / len(close_std_label_list)
    df_result["ivola_score"] = sum([df_result[label] for label in ivola_std_label_list]) / len(ivola_std_label_list)
    if (asset == "E"):
        df_result["turnover_rate_score"] = sum([df_result[label] for label in turnover_rate_std_label_list]) / len(turnover_rate_std_label_list)
    # df_result["beta_score"]=sum([df_result[label] for label in beta_list])

    # rank them
    df_result["close_rank"] = df_result["close_score"].rank(ascending=False)
    df_result["ivola_rank"] = df_result["ivola_score"].rank(ascending=False)
    if (asset == "E"):  # TODO add turnover_rate for I ,FD
        df_result["turnover_rate_rank"] = df_result["turnover_rate_score"].rank(ascending=True)
    # df_result["beta_rank"] = df_result["beta_score"].rank(ascending=True)

    # final rank
    if (asset == "E"):
        df_result["final_volatility_rank"] = df_result["close_rank"] + df_result["ivola_rank"] + df_result["turnover_rate_rank"]
    else:
        df_result["final_volatility_rank"] = df_result["close_rank"] + df_result["ivola_rank"]
    df_result["final_volatility_rank"] = df_result["final_volatility_rank"].rank(ascending=True)

    # add static data and sort by final rank
    df_result = DB.add_static_data(df_result, assets)
    df_result.sort_values(by=["final_volatility_rank"], ascending=True, inplace=True)

    path = "Market/" + "CN" + "/Atest/" + "volatility" + "/" + ''.join(assets) + "_" + freq + "_" + start_date + "_" + end_date + ".xlsx"
    DB.to_excel_with_static_data(df_result, path=path, sort=["final_volatility_rank", True], a_assets=assets)



"""very slow, this can be done AFTER report or in midnight"""
def asset_fund_portfolio():
    """
    this function creates a statistic about the most hold stock by fund

    1. at all time
    2. at current time
    TODO check if during the 3 month time the stock has 增发，减发

    count: how many qdii has it in portfolio
    total_share: amount of shares available of one stock
    amount: amount of shares all qdii together of one stock
    qdii_ratio: qdii holding ratio from all available shares. e.g. 60% of all shares are hold by qdii
    qdii_mom: if qdii_ratio has gained or lost in last season. e.g. 15% to 35% are institutional holders
    """

    #INIT
    df_ts_code=DB.get_ts_code(a_asset=["FD"])
    df_result = DB.get_ts_code(a_asset=["E"])
    df_trade_date = DB.get_trade_date()

    range_obj=range(1990,LB.get_today_year()+1)
    range_obj=range(2005,LB.get_today_year()+1)

    #first fund starts in 2005
    for year in range_obj:
        for season in [1,2,3,4]:
            df_result[f"{year}_{season}_count"] = 0
            df_result[f"{year}_{season}_amount"] = 0
            df_result[f"{year}_{season}_rank"] = 0


    #loop over each fund
    for ts_code in df_ts_code.index[::1]:
        try:
            df_asset=DB.get_asset(ts_code=ts_code,asset="FD",freq="fund_portfolio")
        except:
            continue


        if df_asset.empty:
            continue
        else:
            print("asset_fund_portfolio counting from FD port folio ",ts_code)


        df_asset["count"]=1
        #df_asset=df_asset[df_asset["stk_mkv_ratio"]>0]#remove IPO stocks that are not on market yet. I don't know why and funds buy them.

        df_asset["helper"]=df_asset.index
        df_asset["year"]=df_asset["helper"].str.slice(0,4)
        df_asset["year"] =df_asset["year"].astype(int)
        df_asset["season"]=df_asset["helper"].str.slice(4,6)
        df_asset["season"]=df_asset["season"].astype(int)
        df_asset["season"]=df_asset["season"].map({3:1,6:2,9:3,12:4})


        df_result.at[ts_code,"period"]=len(df_asset)

        for year in range_obj:
            df_asset_filtered=df_asset[df_asset["year"]==year]
            for season in [1,2,3,4]:
                df_asset_filtered2=df_asset_filtered[df_asset_filtered["season"]==season ]
                df_asset_count=df_asset_filtered2.groupby("symbol").sum()
                try:
                    df_result[f"{year}_{season}_count"]=df_result[f"{year}_{season}_count"].add(df_asset_count["count"],fill_value=0)
                    df_result[f"{year}_{season}_amount"]=df_result[f"{year}_{season}_amount"].add(df_asset_count["amount"],fill_value=0)
                except Exception as e:
                    print(e)
                #calculate how much a stock has gained compared to last season

                #add total share of that stock at the end of the season



    #a very inefficient way to calculate pct of qddi buying the stock share

    #preparation, otherwise will be very slow
    d_preload = DB.preload(asset="E")
    df_trade_date=DB.get_trade_date()
    for ts_code, df_asset in d_preload.items():
        df_asset["year"]=df_trade_date["year"]
        df_asset["season"]=df_trade_date["season"]
        d_preload[ts_code]=df_asset

    a_last_season = []
    for year in range_obj:
        for season in [1, 2, 3, 4]:
            print(f"add total share {year} {season}")
            df_result[f"{year}_{season}_total_share"]=np.nan

            for ts_code,df_asset in d_preload.items():
                if df_asset.empty:
                    continue

                df_asset_filtered = df_asset[(df_asset["year"] == year)&(df_asset["season"] == season)]
                if df_asset_filtered.empty:
                    continue
                try:
                    total_share=df_asset_filtered.at[df_asset_filtered["total_share"].last_valid_index(), "total_share"]
                except:
                    continue

                df_result.at[ts_code, f"{year}_{season}_total_share"] = total_share

            df_result[f"{year}_{season}_qdii_ratio"] = df_result[f"{year}_{season}_amount"] /df_result[f"{year}_{season}_total_share"]
            a_last_season += [f"{year}_{season}_qdii_ratio"]


    for firsts,seconds in LB.custom_pairwise_overlap(a_last_season):
        try:
            df_result[f"{seconds}_qdii_mom"]=df_result[f"{seconds}"].sub(df_result[f"{firsts}"])
        except Exception as e:
            print(e)


    #rank these results
    for year in range_obj:
        for season in [1,2,3,4]:
            column_total=df_result[f"{year}_{season}_count"].sum()
            if column_total>0:#this removes future seasons that have not published any report yet
                df_result[f"{year}_{season}_rank"]=df_result[f"{year}_{season}_count"].rank(ascending=False)
            else:
                del df_result[f"{year}_{season}_rank"]
            #del df_result[f"{year}_{season}_count"]

    #save
    a_path = LB.a_path(f"Market/CN/ATest/fund_portfolio/all_time_statistic")
    LB.to_csv_feather(df=df_result, a_path=a_path)

    return df_result



# measures the overall bullishness of an asset using GEOMEAN. replaces bullishness
def asset_bullishness(df_ts_code=pd.DataFrame(), start_date=00000000,end_date = LB.latest_trade_date(),market="CN", step=1, a_asset=["E", "I", "FD"]):

    # init
    if df_ts_code.empty:
        print("is empty")
        df_ts_code = DB.get_ts_code(a_asset=a_asset, market=market)[::1]

    # init
    df_result = pd.DataFrame()
    extrem_pct=0.06
    a_freq=[20,60,240]
    a_freqd = []
    a_freq_s=["D", "W", "M"] #this is better than D,W,M,S because year is important, it guarantees stock with high long term trend to be higher
    df_trade_date=DB.get_trade_date()
    #get the last n days of trade date from end_date
    df_trade_date=df_trade_date[(df_trade_date.index > start_date)&(df_trade_date.index <= end_date)]
    for freq in [20,60,240]:
        a_freqd+=[df_trade_date.index[-freq]]

    print(a_freqd)
    #preload 3 main index
    #TODO for now just use cn index for all, later use nasdaq and Hang seng
    d_preload_index = DB.preload_index(market="CN")

    # loop
    for ts_code, asset in zip(df_ts_code.index[::step], df_ts_code["asset"][::step]):
        print("calc bullishness",start_date,end_date, market, asset, ts_code)

        try:
            df_asset = DB.get_asset(ts_code=ts_code, asset=asset, market=market)
            df_asset=df_asset[(df_asset.index > start_date)&(df_asset.index <= end_date)&(df_asset["period"] > 40)]
            df_result.at[ts_code, "period"] = lenofdf = len(df_asset)
        except:
            continue

        if lenofdf > 240 and not df_asset.empty:

            # scale close to all start at 1
            df_asset["close"] = df_asset["close"] / df_asset.at[df_asset.index[0], "close"]

            #create W,M,S,Y Chart
            d_asset_freq = {freq: LB.df_to_freq(df_asset, freq) for freq in a_freq_s}

            #modify W,M,S,Y Chart to be without top n% gain, used later for offensive rank
            """
            for freq, df_asset_freq in d_asset_freq.items():
                low, high = df_asset_freq["pct_chg"].quantile([0, 1 - extrem_pct])
                d_asset_freq[freq] = df_asset_freq[df_asset_freq["pct_chg"].between(low, high)]
"""

            """INFO are just for information, not used for ranking"""
            # INFO pgain, fgain
            for column in ["pgain","fgain"]:
                for freq in [5,20,60,240]:
                    try:
                        df_result.at[ts_code, f"{column}{freq}"]=df_asset[f"{column}{freq}"].iat[-1]
                    except:
                        df_result.at[ts_code, f"{column}{freq}"]= np.nan

            # INFO PE, PB, Close ASP
            if market =="CN":
                for column in ["pe_ttm","pb","close","total_mv"]:
                    try:
                        df_result.at[ts_code, f"{column}"] = df_asset[column].iat[-1]
                    except:
                        df_result.at[ts_code, f"{column}"] =np.nan
                # INFO PE, PB, vs past

                for column in ["pe_ttm","pb"]:
                    try:
                        df_result.at[ts_code, f"{column}_ALL"] = (df_asset[f"{column}"].iat[-1] >= df_asset[f"{column}"]).astype(int).mean()
                    except:
                        df_result.at[ts_code, f"{column}_ALL"] =np.nan


            # INFO beta, lower the better TODO make beta for HK stock align with hk index
            for ts_code_index, df_index in d_preload_index.items():
                df_index[ts_code] = df_asset["close"]
                df_index_notna = df_index
                # df_index_notna = df_index[df_index[ts_code].notna()]
                df_result.at[ts_code, f"{ts_code_index}_beta"] = df_index_notna["close"].corr(df_index_notna[ts_code])

            # INFO PCT_CHG std: (If two stock have same gmean, which one is more volatile?)
            for freq, df_asset_freq in d_asset_freq.items():
                df_asset_freq["pct_change"] = 1 + df_asset_freq["close"].pct_change()
                df_result.at[ts_code, f"{freq}_std"] = df_asset_freq["pct_change"].std()


            """RANK are used for ranking"""
            # RANK Geomean: implcitly reward stock with high monotony and punish stock with high volatilty.
            for freq, df_asset_freq in d_asset_freq.items():
                df_result.at[ts_code, f"{freq}_geomean"] = gmean(df_asset_freq["pct_change"].dropna())

            # Boll upper abd lower band distance, the smaller, the cycle mode, the bigger the trend mode
            for freq, df_asset_freq in d_asset_freq.items():
                try:
                    boll, bolldown, bollup = Alpha.boll(df=df_asset_freq, abase="close", freq1=20, freq2=2, inplace=True)
                    df_asset_freq["boll_dist_all"]=df_asset_freq[bollup]/df_asset_freq[bolldown]

                    #general distance between up and low describes the risk of a stock
                    df_result.at[ts_code, f"{freq}_boll_dist_all"] = temp= df_asset_freq["boll_dist_all"].mean()

                    #last day avg dis/last day dis describes if the stock is currently in trend or cycle mode
                    df_result.at[ts_code, f"{freq}_boll_dist_nowpt"] = df_asset_freq["boll_dist_all"].iat[-1]
                    df_result.at[ts_code, f"{freq}_boll_dist_nowrel"] = df_asset_freq["boll_dist_all"].iat[-1]/temp

                    #last day avg dist vs velocity/tiltness of the distance. If the distance is getting smaller or bigger
                except:
                    pass

            # RANK technical freqhigh = ability to create 20d,60d,120d,240high
            for freq in a_freq:
                df_asset[f"rolling_max{freq}"] = df_asset["close"].rolling(freq).max()
                df_helper = df_asset.loc[df_asset[f"rolling_max{freq}"] == df_asset["close"]]
                df_asset[f"{freq}high"] = df_helper[f"rolling_max{freq}"]
                df_result.at[ts_code, f"{freq}high"] = df_asset[f"{freq}high"].clip(0, 1).sum() / len(df_asset)

            # RANK technical freqlow = ability to avoid 20d,60d,120d,240low
            for freq in a_freq:
                df_asset[f"rolling_min{freq}"] = df_asset["close"].rolling(freq).min()
                df_helper = df_asset.loc[df_asset[f"rolling_min{freq}"] == df_asset["close"]]
                df_asset[f"{freq}low"] = df_helper[f"rolling_min{freq}"]
                df_result.at[ts_code, f"{freq}low"] = df_asset[f"{freq}low"].clip(0, 1).sum() / len(df_asset)


            # RANK check how long a stock is abv ma 20,60,120,240
            for freq in a_freq:
                abvma_name=Alpha.abv_ma(df=df_asset,abase="close",freq=freq,inplace=True)
                df_result.at[ts_code, f"abv_ma{freq}"]=df_asset[abvma_name].mean()


            # sharp/sortino ratio: Note my sharp ratio is not anuallized but period adjusted
            """
            for freq, df_asset_freq in d_asset_freq.items():
                s = df_asset_freq["pct_change"]
                df_result.at[ts_code, f"{freq}_sharp"] = (s.mean() / s.std()) * np.sqrt(len(s))
                df_result.at[ts_code, "sortino"] = (s.mean()/s[s<0].std())*np.sqrt(len(s))
            """

            # trend swap. how long a trend average lasts
            """
            for freq in [240]:
                df_result.at[ts_code, f"abv_ma_days{freq}"] = LB.trend_swap(df_asset, f"abv_ma{freq}", 1)
            """

            # volatility of the high pass, the smaller the better
            """
            highpass_mean = 0
            for freq in [240]:
                highpass_mean = highpass_mean + df_asset[f"highpass{freq}"].mean()
            df_result.at[ts_code, "highpass_mean"] = highpass_mean
            """



            # dividend
            if asset == "E" and market == "CN":

                # qdii research and grade
                """qddi grae starts from 2018, research starts around 2008?"""
                for qdii in ["research","grade"]:
                    df_qdii_rg = DB.get_asset(ts_code=ts_code, freq=f"qdii_{qdii}", market="CN")
                    df_qdii_rg=df_qdii_rg[(df_qdii_rg.index > start_date)&(df_qdii_rg.index <= end_date)]

                    a_freqs=[60]
                    if not df_qdii_rg.empty:
                        df_result.at[ts_code, f"qdii_{qdii}/period"] = len(df_qdii_rg) / len(df_asset)
                        for freqd_trade_date,freqd in zip(a_freqd,a_freqs):
                            df_qdii_rd_freqd=df_qdii_rg[df_qdii_rg.index >= freqd_trade_date]
                            df_result.at[ts_code, f"qdii_{qdii}{freqd}"] = len(df_qdii_rd_freqd) / freqd
                    else:
                        df_result.at[ts_code, f"qdii_{qdii}/period"] =0
                        for freqd_trade_date,freqd in zip(a_freqd,a_freqs):
                            df_result.at[ts_code, f"qdii_{qdii}{freqd}"] = 0


                        # hk hold, 北上资金
                df_hk = DB.get_asset(ts_code=ts_code, freq=f"hk_hold", market="CN")
                df_hk.index=df_hk.index.astype(int)
                df_hk = df_hk[(df_hk.index > start_date) & (df_hk.index <= end_date)]
                if not df_hk.empty:
                    df_result.at[ts_code, f"hk_hold"] = df_hk["ratio"].iat[-1]
                    df_result.at[ts_code, f"hk_hold_ALL"] = df_hk["ratio"].mean()
                else:
                    df_result.at[ts_code, f"hk_hold"] = 0
                    df_result.at[ts_code, f"hk_hold_ALL"] = 0


    #add if they are the head of industry
    df_result_static =   DB.add_static_data(df_result, asset=a_asset, market=market)
    try:
        if market=="CN":
            for counter in [1,2,3]:
                df_head = pd.DataFrame()
                for industry in df_result_static[f"sw_industry{counter}"].unique():
                    if industry is None:
                        continue
                    df_filter = df_result_static[df_result_static[f"sw_industry{counter}"] == industry]
                    df_filter = df_filter.sort_values("total_mv", ascending=False)
                    df_head = df_head.append(df_filter.head(1))

                df_result[f"head_sw_industry{counter}"]=0
                df_result.loc[df_head.index,f"head_sw_industry{counter}"]=1
    except:
        pass

    #add qdii hold after looping individual stocks
    try:
        qdii_path = "Market/CN/ATest/fund_portfolio/all_time_statistic.feather"
        df_qdii = pd.read_feather(qdii_path)
        df_qdii = df_qdii.set_index("ts_code", drop=True)

        """NOTE we always check for LAST season, not this season"""
        trade_date_str = str(end_date)
        year = int(trade_date_str[0:4])
        month = int(trade_date_str[4:6])
        if month <= 3:
            season = 1-1
        elif month <= 6:
            season = 2-1
        elif month <= 9:
            season = 3-1
        elif month <= 12:
            season = 4-1

        if season==0:
            year=year-1
            season=4
        qdii_ratio_column = f"{year}_{season}_qdii_ratio"
        qdii_ratio_qdii_mom_column = f"{year}_{season}_qdii_ratio_qdii_mom"
        df_result["qdii_ratio_ls"]=df_qdii[qdii_ratio_column]
        df_result["qdii_ratio_ls_mom"]=df_qdii[qdii_ratio_qdii_mom_column]
    except Exception as e:
        print(e)


    """RANK ALL STUFF"""
    """QDII Rank"""
    # qdii rank = all time rank. how much all time qdii attention does this stock get
    try:
        """focus on long term"""
        df_result["qdii_def_rank"] = df_result[f"qdii_ratio_ls"].rank(ascending=False) *0.62 + \
                                df_result[f"qdii_research/period"].rank(ascending=False) *0.38*0.5+ \
                                df_result[f"qdii_grade/period"].rank(ascending=False)*0.38*0.5
        df_result["qdii_def_rank"] = df_result["qdii_def_rank"].rank(ascending=True)

    except:
        df_result["qdii_def_rank"] = np.nan

    try:
        """focus on short term = offensive"""
        df_result["qdii_off_rank"] = df_result[f"qdii_ratio_ls_mom"].rank(ascending=False) * 0.62 + \
                                      df_result[f"qdii_research60"].rank(ascending=False) * 0.38 * 0.5 + \
                                      df_result[f"qdii_grade60"].rank(ascending=False) * 0.38 * 0.5
        df_result["qdii_off_rank"] = df_result["qdii_off_rank"].rank(ascending=True)
    except:
        df_result["qdii_off_rank"] = np.nan


    """Technical Rank"""
    # offensive rank = ability to gain high return, no mater how the path is. Monotony and std is implicitly ranked here.
    df_result["tech_off_rank"] = + builtins.sum([df_result[f"{freq}_geomean"].rank(ascending=False) for freq in a_freq_s])
    df_result["tech_off_rank"] = df_result["tech_off_rank"].rank(ascending=True)

    # defensive rank = ability not to fall down no matter what. Accounts also ability to make new high.
    df_result["tech_def_rank"] = builtins.sum([df_result[f"{freq}high"].rank(ascending=False) for freq in a_freq]) * 0.38 * 0.38 \
                                + builtins.sum([df_result[f"{freq}low"].rank(ascending=True) for freq in a_freq]) * 0.38 * 0.62 \
                                + builtins.sum([df_result[f"abv_ma{freq}"].rank(ascending=False) for freq in a_freq]) * 0.62
    df_result["tech_def_rank"] =df_result["tech_def_rank"].rank(ascending=True)


    # combine using Arithmetic mean
    """df_result["allround_rank_ari"] =  df_result["tech_off_rank"]*0.38 \
                                    + df_result["tech_def_rank"]*0.62
    df_result["allround_rank_ari"] = df_result["allround_rank_ari"].rank(ascending=True)
    """

    # combine using Geometric mean
    df_result["allround_rank_geo"] =   (len(df_result) - df_result["tech_off_rank"]) \
                                     * (len(df_result) - df_result["tech_def_rank"]) \
                                     * (len(df_result) - df_result["tech_def_rank"])
    df_result["allround_rank_geo"] = df_result["allround_rank_geo"].rank(ascending=False)


    df_result.to_csv(f"Market/{market}/Atest/bullishness/bullishness_{market}_{start_date}_{end_date}_{a_asset}.csv", encoding='utf-8_sig')
    DB.to_excel_with_static_data(df_ts_code=df_result, sort=["final_rank", True], path=f"Market/{market}/Atest/bullishness/bullishness_{market}_{start_date}_{end_date}.xlsx", group_result=True, market=market)
    return df_result


def asset_bollinger(asset="E" , freq="D"):
    """
    Input: a group of good stocks
    Output: on what scale are their bollinger bands


    Idea: use bollinger to see if all stocks are too high or not
    Idea: use bollinger to detect stocks that is low even if all stocks are high

    :return:
    """

    #1 define input of stocks. Not more than 50
    a_ts_code= [
        '002821.SZ',
        '603338.SH',
        '603288.SH',
        '603899.SH',
        '601100.SH',
        '000333.SZ',
        '300529.SZ',
        '002507.SZ',
        '601012.SH',
        '300357.SZ',
        '002475.SZ',
        '300347.SZ',
        '300015.SZ',
        '300122.SZ',
        '002714.SZ',
        '603939.SH',
        '603737.SH',
        '601888.SH',
        '300124.SZ',
        '002810.SZ',
        '601799.SH',
        '300308.SZ',
        '300316.SZ',
        '300225.SZ',
        '603658.SH',
        '603027.SH',
        '002415.SZ',
        '300285.SZ',
        '002607.SZ',
        '002311.SZ',
        '600436.SH',
        '603369.SH',
        '600276.SH',
        '300413.SZ',
        '603520.SH',
        '300142.SZ',
        '002410.SZ',
        '603806.SH',
        '300136.SZ',
        '002271.SZ',
        '002597.SZ',
        '002179.SZ',
        '600519.SH',
        '300496.SZ',
        '002372.SZ',
        '002791.SZ',
        '002511.SZ',
        '002508.SZ',
        '002677.SZ',
        '300551.SZ',
        '300450.SZ',
        '601155.SH',
        '002032.SZ',
        '300223.SZ',
        '002690.SZ',
        '300207.SZ',
        '300395.SZ',
        '300014.SZ',
        '002371.SZ',
        '002626.SZ',
        '603866.SH',
        '300012.SZ',
        '601636.SH',
        '600570.SH',
        '603606.SH',
        '002049.SZ',
        '002236.SZ',
        '600309.SH',
        '300037.SZ',
        '002007.SZ',
        '600887.SH',
        '002241.SZ',
        '002409.SZ',
        '601966.SH',
        '002601.SZ',
        '002262.SZ',
        '600406.SH',
        '300552.SZ',
        '002142.SZ',
        '300271.SZ',
        '002460.SZ',
        '002493.SZ',
        '002001.SZ',
        '002557.SZ',
        '000538.SZ',
        '603060.SH',
        '300327.SZ',
        '600340.SH',
        '002439.SZ',
        '300059.SZ',
        '600426.SH',
        '002138.SZ',
        '300188.SZ',
        '002456.SZ',
        '603799.SH',
        '603816.SH',
        '000661.SZ',

    ]

    df_ts_code=DB.get_ts_code(a_asset=["E","I","FD"])
    a_freqs=["D","W"]
    d_result_freq={}

    for freq in a_freqs:
        #2. setup alginment with 3 index
        df_sh=LB.df_to_freq(DB.get_asset(ts_code="000001.SH", asset="I"), freq=freq)
        df_sz=LB.df_to_freq(DB.get_asset(ts_code="399001.SZ", asset="I"), freq=freq)
        df_cy=LB.df_to_freq(DB.get_asset(ts_code="399006.SZ", asset="I"), freq=freq)
        df_sh["000001.SH"]=df_sh["close"]
        df_sh["399001.SZ"]=df_sz["close"]
        df_sh["399006.SZ"]=df_cy["close"]

        df_sh = df_sh[["period", "000001.SH","399001.SZ","399006.SZ"]]
        d_preload = DB.preload(asset=asset, market="CN", d_queries_ts_code={asset: [f"ts_code in {a_ts_code}"]})


        #1. align all ts code with sh index
        for ts_code in a_ts_code:
            print(f"bollinger for {freq} {ts_code}")
            df_asset=LB.df_to_freq(d_preload[ts_code], freq=freq)
            #close
            df_sh[f"{ts_code}"]=df_asset["close"]

            # bollinger
            df_sh[f"{ts_code}_up"],df_sh[f"{ts_code}_mid"],df_sh[f"{ts_code}_low"]=talib.BBANDS(df_asset["close"], 20, 2, 2)

            # scale to between 0 and 1
            df_sh[f"{ts_code}_scale"]=(((1 - 0) * (df_sh[f"{ts_code}"] - df_sh[f"{ts_code}_low"])) / (df_sh[f"{ts_code}_up"] - df_sh[f"{ts_code}_low"])) + 0



        df_stats = pd.DataFrame()
        for ts_code in a_ts_code:
            df_stats.at[ts_code,"name"]=df_ts_code.at[ts_code,"name"]
            df_stats.at[ts_code,"period"]=len(d_preload[ts_code])

            df_stats.at[ts_code, f"bollingerNOW"]=df_sh[f"{ts_code}_scale"].iat[-1]
            for last_days in [20,60,240,500, "ALL"]:
                df_sh_freq = df_sh.tail(last_days) if last_days != "ALL" else df_sh.copy()
                df_stats.at[ts_code,f"bollinger{last_days}"]=df_sh_freq[f"{ts_code}_scale"].mean()
                #df_stats.at[ts_code,f"geomean{last_days}"]=gmean(1 + df_sh_freq[ts_code].pct_change().dropna())

        d_result_freq[freq]=df_stats
        LB.to_csv_feather(df=df_stats, a_path=LB.a_path(f"Market/CN/ATest/bollinger/boll_stats_{freq}"))

    df_final= pd.merge(d_result_freq["D"], d_result_freq["W"], how="left", left_on=["index","name","period"], right_on=["index","name","period"], suffixes=["_D", "_W"], sort=False)
    LB.to_csv_feather(df=df_final, a_path=LB.a_path(f"Market/CN/ATest/bollinger/boll_stats_final"))


def asset_portfolio_correlation():
    """
    Input: a group of good stocks
    Output: Their pearson with 3 Index
    Output: Matrix showing their correlation with each other, what is the best pair

    Idea: use that to switch portfolio, when market is high, switch to less high funds.
    REALITY: all stock have high market correlation. Past can not predict future, law of small number, even knowing past corr number, will not guarantee future correlation. So it is useless.


    For now only support FD
    :return:
    """

    #1 define input of stocks. Not more than 50
    a_ts_code= [
        "512600.SH",
        "159928.SZ",
        "163415.SZ",
        "163402.SZ",
        "163412.SZ",
        "160133.SZ",
        "169101.SZ",
        "165516.SZ",
        "169104.SZ",
        "160916.SZ",
    ]


    #2. setup alginment with 3 index
    df_sh=DB.get_asset(ts_code="000001.SH",asset="I")
    df_sz=DB.get_asset(ts_code="399001.SZ",asset="I")
    df_cy=DB.get_asset(ts_code="399006.SZ",asset="I")
    df_sh["000001.SH"]=df_sh["close"]
    df_sh["399001.SZ"]=df_sz["close"]
    df_sh["399006.SZ"]=df_cy["close"]


    df_sh_master = df_sh[["period", "000001.SH","399001.SZ","399006.SZ"]]
    d_preload = DB.preload(asset="FD", market="CN", d_queries_ts_code={"FD": [f"ts_code in {a_ts_code}"]})


    #calculate beta for last n days
    for last_days in [20,60,240,500,"ALL"]:
        df_corr = pd.DataFrame(index=["000001.SH", "399001.SZ", "399006.SZ"] + a_ts_code, columns=a_ts_code)
        df_sh=df_sh_master.tail(last_days) if last_days!="ALL" else df_sh_master.copy()

        #1. align all ts code with sh index
        for ts_code in a_ts_code:
            df_sh[f"{ts_code}"]=d_preload[ts_code]["close"]

        #2. calculate beta between all stocks and index
        for ts_code in a_ts_code:
            for index_code in ["000001.SH","399001.SZ","399006.SZ"]:
                #df_corr.at[index_code, ts_code] = LB.calculate_beta(df_sh[f"{index_code}_close"], df_sh[f"{ts_code}_close"])
                df_corr.at[index_code, ts_code] = df_sh[index_code].corr(df_sh[ts_code], method="pearson")

        #3. calculate beta between all stocks and all stocks
        for ts_code1 in a_ts_code:
            for ts_code2 in a_ts_code:
                df_corr.at[ts_code1, ts_code2] = df_sh[ts_code1].corr(df_sh[ts_code2], method="pearson")

        LB.to_csv_feather(df=df_corr,a_path=LB.a_path(f"Market/CN/ATest/portfolio_correlation/corr{last_days}"))
    LB.to_csv_feather(df=df_sh, a_path=LB.a_path(f"Market/CN/ATest/portfolio_correlation/chart"))


def asset_candlestick_analysis_once(ts_code, pattern, func):
    df_asset = DB.get_asset(ts_code)
    rolling_freqs = [2, 5, 10, 20, 60, 240]
    # labels
    candle_1 = ["future_gain" + str(i) + "_1" for i in rolling_freqs] + ["future_gain" + str(i) + "_std_1" for i in rolling_freqs]
    candle_0 = ["future_gain" + str(i) + "_0" for i in rolling_freqs] + ["future_gain" + str(i) + "_std_0" for i in rolling_freqs]

    try:
        df_asset = df_asset[df_asset["period"] > 240]
        df_asset[pattern] = func(open=df_asset["open"], high=df_asset["high"], low=df_asset["low"], close=df_asset["close"])
    except:
        s_interim = pd.Series(index=["candle", "ts_code", "occurence_1", "occurence_0"] + candle_1 + candle_0)
        s_interim["ts_code"] = ts_code
        s_interim["candle"] = pattern
        return s_interim

    occurence_1 = len(df_asset[df_asset[pattern] == 100]) / len(df_asset)
    occurence_0 = len(df_asset[df_asset[pattern] == -100]) / len(df_asset)

    a_future_gain_1_mean = []
    a_future_gain_1_std = []
    a_future_gain_0_mean = []
    a_future_gain_0_std = []

    for freq in rolling_freqs:
        a_future_gain_1_mean.append(df_asset.loc[df_asset[pattern] == 100, "future_gain" + str(freq)].mean())
        a_future_gain_1_std.append(df_asset.loc[df_asset[pattern] == 100, "future_gain" + str(freq)].std())
        a_future_gain_0_mean.append(df_asset.loc[df_asset[pattern] == -100, "future_gain" + str(freq)].mean())
        a_future_gain_0_std.append(df_asset.loc[df_asset[pattern] == -100, "future_gain" + str(freq)].std())

    data = [pattern, ts_code, occurence_1, occurence_0] + a_future_gain_1_mean + a_future_gain_1_std + a_future_gain_0_mean + a_future_gain_0_std
    s_result = pd.Series(data=data, index=["candle", "ts_code", "occurence_1", "occurence_0"] + candle_1 + candle_0)
    return s_result


def asset_candlestick_analysis_multiple():
    d_pattern = LB.c_candle()
    df_all_ts_code = DB.get_ts_code(a_asset=["E"])

    for key, array in d_pattern.items():
        function = array[0]
        a_result = []
        for ts_code in df_all_ts_code.ts_code:
            print("start candlestick with", key, ts_code)
            a_result.append(asset_candlestick_analysis_once(ts_code=ts_code, pattern=key, func=function))

            df_result = pd.DataFrame(data=a_result)
            path = "Market/CN/Atest/candlestick/" + key + ".csv"
            df_result.to_csv(path, index=False)
            print("SAVED candlestick", key, ts_code)

    a_all_results = []
    for key, array in d_pattern.items():
        path = "Market/CN/Atest/candlestick/" + key + ".csv"
        df_pattern = pd.read_csv(path)
        df_pattern = df_pattern.mean()
        df_pattern["candle"] = key
        a_all_results.append(df_pattern)
    df_all_result = pd.DataFrame(data=a_all_results)
    path = "Market/CN/Atest/candlestick/summary.csv"
    df_all_result.to_csv(path, index=True)

def asset_beat_index_loop():
    a_years=["20000101","20050101","20100101","20150101","20200101"]
    for end_date in a_years:
         asset_beat_index("0000101",end_date)

def asset_distribution(asset="I", column="close", bins=10):
    d_preload = DB.preload(asset=asset, step=5)

    for freq in [10, 20, 40, 60, 120, 240, 500]:
        a_path = LB.a_path(f"Market/CN/Atest/distribution/{asset}/{column}/{column}_freq{freq}_bin{bins}")
        df_result = pd.DataFrame()
        if not os.path.isfile(a_path[0]):
            for ts_code, df in d_preload.items():
                print(f"{asset}, freq{freq}, bins{bins}, {ts_code}, {column}")

                # normalize past n values to be between 0 and 1. 0 is lowest and 1 is highest.
                df["norm"] = df["close"].rolling(freq).apply(Alpha.normalize_apply, raw=False)

                # count cut as result
                df_result.at[ts_code, "len"] = len(df)
                for c1, c2 in LB.custom_pairwise_overlap(LB.drange(0, 101, bins)):
                    df_result.at[ts_code, f"c{c1, c2}"] = len(df[df["norm"].between(c1, c2)])

            LB.to_csv_feather(df_result, a_path=a_path, skip_feather=True)

def asset_beat_index(start_date="00000000",end_date=LB.today()):
    """checks how many stock beats their index
    1. normalize all index to the day certain asset is IPOd
    2. Check if index or asset is better until today

    Amazing Result:

    53%主板beat index
    60%中小板beat index
    30%创业板beat index

    only 30% beat industry1
    only 30% beat industry2
    only 30% beat industry3

    many stock who beat the index, in the past n years, dont beat index in the next future years
    in earlier years like 2000 to 2005, 2/3 of stock who were good in the past stay good.
    now like 2015 to 2020, half of them become bad. This means it is hard now to pickup stock that stays good

    The pct% of stocks beating index,industry1,industry2,industry3 is increasing over time
    in 2000-2005 it was 7%
    in 2005-2010 it was 17%
    in 2010-2015 it was 29%
    in 2015-2020 it was 22%
    of course, this number is distored by new IPOS and crazy period timing

    TAKEAWAY:
    Stocks which perform better than index,industry1-3 might be a good stock. some of them are temporal good.
    But a good stock, always perform better than index.
    =shrinks down the pool
    =step 2: from this pool. take the best industry
    BUT in theory, past does not predict future. Past good stock does not remain good, but is only more likely to remain good.

    """

    #init
    df_ts_code=DB.get_ts_code()
    df_industry1_code=DB.get_ts_code(a_asset=[f"industry1"])
    df_industry2_code=DB.get_ts_code(a_asset=[f"industry2"])
    df_industry3_code=DB.get_ts_code(a_asset=[f"industry3"])

    #preload
    d_index=DB.preload(asset="I", step=1, d_queries_ts_code=LB.c_index_queries())
    d_e=DB.preload(step=1)
    df_result=pd.DataFrame()

    for ts_code, df_asset in d_e.items():

        print(ts_code)
        #compare against exchange
        exchange=df_ts_code.at[ts_code,"exchange"]
        if exchange=="创业板":
            compare="399006.SZ"
        elif exchange=="中小板":
            compare ="399001.SZ"
        elif exchange=="主板":
            compare="000001.SH"
        df_exchange = d_index[compare].copy()
        df_exchange=LB.df_between(df_exchange,start_date,end_date)

        #compare against industry
        try:
            industry1 = df_industry1_code.at[ts_code, "industry1"]
            industry2 = df_industry2_code.at[ts_code, "industry2"]
            industry3 = df_industry3_code.at[ts_code, "industry3"]
        except Exception as e:
            print(ts_code,"skipped",e)
            continue

        df_industry1= DB.get_asset(ts_code=f"industry1_{industry1}", asset="G")
        df_industry2= DB.get_asset(ts_code=f"industry2_{industry2}", asset="G")
        df_industry3= DB.get_asset(ts_code=f"industry3_{industry3}", asset="G")
        df_industry1 = LB.df_between(df_industry1, start_date, end_date)
        df_industry2 = LB.df_between(df_industry2, start_date, end_date)
        df_industry3 = LB.df_between(df_industry3, start_date, end_date)

        df_result.at[ts_code,"in1"]=industry1
        df_result.at[ts_code,"in2"]=industry2
        df_result.at[ts_code,"in3"]=industry3
        df_asset["gmean_norm"]=df_asset["close"].pct_change()
        df_result.at[ts_code,"gmean"]=df_asset["gmean_norm"].mean()/df_asset["gmean_norm"].std()

        #run and evaluate
        for key,df_compare in {"index":df_exchange, "industry1":df_industry1,"industry2":df_industry2,"industry3":df_industry3}.items():
            df_asset_slim=LB.df_ohlcpp(df_asset).reset_index()
            df_index_slim=LB.df_ohlcpp(df_compare).reset_index()

            df_asset_slim["trade_date"]=df_asset_slim["trade_date"].astype(int)
            df_index_slim["trade_date"]=df_index_slim["trade_date"].astype(int)

            df_slim=pd.merge(df_asset_slim,df_index_slim,on="trade_date",how="inner",suffixes=[f"_{ts_code}",f"_{compare}"],sort=False)

            if df_slim.empty:
                continue

            for code in [ts_code,compare]:
                df_slim[f"norm_{code}"]=df_slim[f"close_{code}"]/df_slim.at[0,f"close_{code}"]
                df_slim[f"norm_pct_{code}"]=df_slim[f"norm_{code}"].pct_change()
            #result=norm_ts_code/norm_compare

            df_result.at[ts_code,f"{key}_period"]=period=len(df_slim)-1
            df_result.at[ts_code,f"{key}_asset_vs_index_gain"]= df_slim.at[period, f"norm_{ts_code}"] / df_slim.at[period, f"norm_{compare}"]
            df_result.at[ts_code,f"{key}_asset_vs_index_sharp"]= (df_slim[f"norm_pct_{ts_code}"].mean() / df_slim[f"norm_pct_{ts_code}"].std()) / df_slim[f"norm_pct_{compare}"].mean() / df_slim[f"norm_pct_{compare}"].std()
            df_result.at[ts_code,f"{key}_asset_vs_index_gmean"]= (df_slim[f"norm_pct_{ts_code}"].mean() / df_slim[f"norm_pct_{ts_code}"].std())

            #TODO  beat industry, concept
            df_result.at[ts_code,"index"]=compare


    for key in ["index", "industry1", "industry2", "industry3"]:
        df_result[f"beat_{key}"]=(df_result[f"{key}_asset_vs_index_gain"]>1).astype(int)
    df_result.loc[ (df_result[f"beat_index"]==1) & (df_result[f"beat_industry1"]==1) & (df_result[f"beat_industry2"]==1) & (df_result[f"beat_industry3"]==1), "beat_all"]=1
    df_result.index.name="ts_code"

    DB.to_excel_with_static_data(df_ts_code=df_result,path=f"Market/CN/ATest/Beat_Index/result_{start_date}_{end_date}.xlsx")
    # a_path=LB.a_path("Market/CN/ATest/Beat_Index/result")
    # LB.to_csv_feather(df=df_result,a_path=a_path)

def asset_beat_index_freq(step=1):
    """
    compares index,industry1-3 vs asset on a yearly basis
    it seems that only 2007-2009, 2015 more than 50% stock beat index
    This means, small stocks are only good at crazy time
    Over all years. Only 30 to 40% beat the index
    Mostly when time is crazy, bad stock gain more, and start to beat index in that wmsy.
    But this happens very rarely. So beating index is highly correlated with crazy time.
    The % of stocks beating index is not directly predictive as policical directions are disturbing this indicator

    The mean reverse effect in this is not that strong. Too much noise to be added to as a signal.

    """

    # preload index and asset
    d_index = DB.preload(asset="I", step=1, d_queries_ts_code=LB.c_index_queries())
    d_e = DB.preload(step=step)
    d_result={}

    for wmsy in ["W","M","S","Y"]:
        # transform index
        d_i_wmsy={}
        for key,df in d_index.items():
            d_i_wmsy[key]=LB.df_to_freq(df,freq=wmsy)

        #transform asset
        d_e_wmsy={}
        for ts_code, df_asset in d_e.items():
            d_e_wmsy[ts_code] = LB.df_to_freq(df_asset,freq=wmsy)

        #compare each asset with index:
        df_result_freq=pd.DataFrame()
        for ts_compare,df_compare in d_i_wmsy.items():
            for trade_date in df_compare.index:
                exists_counter=0
                better_than_index_counter=0
                compare_pct_chg=df_compare.at[trade_date,"pct_chg"]

                for ts_code, df_asset in d_e_wmsy.items():
                    print(ts_compare, wmsy,trade_date,ts_code)
                    if trade_date in df_asset.index:
                        exists_counter+=1

                        if df_asset.at[trade_date,"pct_chg"]>compare_pct_chg:
                            better_than_index_counter+=1
                try:
                    df_compare.at[trade_date,f"better_than_{ts_compare}"]=better_than_index_counter/exists_counter
                except:
                    pass

            df_result_freq[f"close_{ts_compare}"] = df_compare["close"]
            df_result_freq[f"better_than_{ts_compare}"] = df_compare[f"better_than_{ts_compare}"]
        d_result[wmsy]=df_result_freq
    LB.to_excel(path=f"Market/CN/Atest/Beat_Index/beat_index_freq.xlsx",d_df=d_result)










def date_daily_stocks_abve():
    """this tells me how difficult my goal is to select the stocks > certain pct_chg every day
                get 1% everyday, 33% of stocks
                2% everyday 25% stocks
                3% everyda 19% stocks
                4% everyday 12% stocks
                5% everyday 7% stocks
                6% everyday 5%stocks
                7% everyday 3% stocks
                8% everday  2.5% Stocks
                9% everday  2% stocks
                10% everday, 1,5% Stocks  """

    df_asset = DB.preload("E", step=2)
    df_result = pd.DataFrame()
    for ts_code, df in df_asset.items():
        print("ts_code", ts_code)
        for pct in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            df_copy = df[(100 * (df["pct_chg_open"] - 1) > pct)]
            df_result.at[ts_code, f"pct_chg_open > {pct} pct"] = len(df_copy) / len(df)

            df_copy = df[(100 * (df["pct_chg_close"] - 1) > pct)]
            df_result.at[ts_code, f"pct_chg_close > {pct} pct"] = len(df_copy) / len(df)

            df_copy = df[(((df["close"] / df["open"]) - 1) * 100 > pct)]  # trade
            df_result.at[ts_code, f"trade > {pct} pct"] = len(df_copy) / len(df)

            df_copy = df[((df["co_pct_chg"] - 1) * 100 > pct)]  # today open and yester day close
            df_result.at[ts_code, f"non trade > {pct} pct"] = len(df_copy) / len(df)
    df_result.to_csv("test.csv")


def date_volatility():
    """one day pct_chg std
    Hypothesis: if one day pct_chg.std of all stocks is small market is stable
    result:generally, all stats are only high at crazy time. Not very usefule or predictive.
    """
    d_date = DB.preload(asset='E', on_asset=False)
    df_result = pd.DataFrame()
    for trade_date, df_date in d_date.items():
        print(trade_date)
        df_result.at[trade_date, "close"] = df_date["close"].mean()
        df_result.at[trade_date, "mean"] = df_date["pct_chg"].mean()
        df_result.at[trade_date, "std"] = df_date["pct_chg"].std()
        df_result.at[trade_date, "sharp"] = df_date["pct_chg"].mean() / df_date["pct_chg"].std()
    for i in [5, 10, 20, 60, 240]:
        df_result[f"std{i}"] = df_result["std"].rolling(i).mean()
    df_result.to_csv("volatilty.csv")


def stop_rule():
    """this one time function tries to find the best time to buy or sell using 37% rule, stop rule.


    short term distribution: random, rule of small number



     variation 1: Look back freq days, see the highest


     result by logic:
     this is not possible because the rule of small number says that if the number is small ,the distribution is normal.
     Only if the distribution is big enough, the distribution is equal distributed.

     That's why even if you count the short term results, it will not predict the future result because short term is normal distributed and random.

     """



    pass



def short_vs_long_stg():
    """
    premise:
    - Dimension 1: trend = There are 3 types of stock, one only goes up, one only goes down, one goes in cycle mode sometimes up, sometimes down
    - for the first stock, you should only buy and never sell, for second you should never buy, for third one you should sometimes buy, and sometimes sell.
    - So predicting the stock is more important than your overall strategy.

    - Dimension 2: volatility = Even if a stock goes up all the time, but if it goes up too slowly, then it is still not better than a stock in cycle mode.
    - so the final scope of a stock = General trend(up, cycle, down) and abs. volatility.

    - Ideal stock = stock that always goes up and has infinite volatility.

    - following the historic value of a stock, we create the strategy.
    - Problem: It is a very long term strategy.
    - Idea: is there a short term strategy that is better than long?

    :return:
    """

    df_result=pd.DataFrame()
    onestock=True
    for asset in ["FD"]:


        for freq in ["D"]:
            if onestock:
                df_one = DB.get_asset(ts_code="163402.SZ", asset="FD")
                if freq!="D":
                    df=LB.df_to_freq(df_one,freq)
                    df[f"boll_up"], egal, df[f"boll_low"] = talib.BBANDS(df["close"], 20, 2, 2)
                    df[f"boll"] = (((1 - 0) * (df["close"] - df[f"boll_low"])) / (df[f"boll_up"] - df[f"boll_low"])) + 0
                    d_preload = {"163402.SZ": df}
                else:
                    d_preload = {"163402.SZ": df_one}
            else:
                d_preload = DB.preload(asset=asset,step=1)

            """
            for each stock, simulate using Expanding
            -define the stocks past trend
            -define the stocks past volatility
            -use boll
            """

            for ts_code, df_asset in d_preload.items():
                print("start", ts_code)

                if "OF" in ts_code:
                    continue

                if df_asset.empty:
                    continue



                "chg pct_chg into 1.1 form instead of 10% form"
                df_asset["pct_chg"]=1+(df_asset["pct_chg"]/100)

                """go through each day and simulate which day would be bought and sold"""

                """calculate the portfolio by using comp_gain"""
                try:

                    # df_port["comp_chg"]=Alpha.comp_chg2(df=df_port,inplace=False,abase="pct_chg")
                    df_asset["nat_comp_chg"] = Alpha.comp_chg2(df=df_asset, inplace=False, abase="pct_chg")

                    """compare result"""
                    df_result.at[ts_code, "nat_comp_chg"] = df_asset["nat_comp_chg"].iat[-1]

                except Exception as e:
                    print("error", e)
                    continue

                #boll
                if False:
                    for lower, upper in [(0.1,0.9)]:

                        df_asset[f"port{lower,upper}"]=np.nan
                        trend="up"
                        for trade_date in df_asset.index:

                            pct_chg=df_asset.at[trade_date, "pct_chg"]

                            if df_asset.at[trade_date,"boll"]>upper and trend=="up":
                                df_asset.at[trade_date, f"port{lower,upper}"]=1
                                trend="down"
                            elif df_asset.at[trade_date, "boll"] < lower and trend=="down":
                                df_asset.at[trade_date, f"port{lower,upper}"] = pct_chg
                                trend = "up"
                            else:
                                if trend== "up":
                                    df_asset.at[trade_date, f"port{lower,upper}"] = pct_chg
                                elif trend=="down":
                                    df_asset.at[trade_date, f"port{lower,upper}"] = 1

                            df_asset[f"stg_comp_chg{lower,upper}"] = Alpha.comp_chg2(df=df_asset, inplace=False, abase=f"port{lower,upper}")

                            df_result.at[ts_code, f"stg_comp_chg{lower,upper}"] = df_asset[f"stg_comp_chg{lower,upper}"].iat[-1]
                df_result.at[ts_code, f"period"] = len(df_asset)
                df_result.at[ts_code, f"start close"] = df_asset["close"].iat[0]
                #macd
                for freq1, freq2 in [(20,60),(20,120),(20,240),(60,240)]:
                #for freq1, freq2 in [(10,20),(10,60),(20,60),(20,120),(60,120),(20,240),(60,240),(120,240)]:
                    macdlabel=Alpha.macd(df=df_asset,abase="close",freq=freq1,freq2=freq2 ,inplace=True)

                    for counter,label in enumerate(macdlabel):
                        if counter !=0:
                            del df_asset[label]

                    for trade_date in df_asset.index:
                        if df_asset.at[trade_date,macdlabel[0]]==10:
                            df_asset.at[trade_date, f"macd_port{freq1,freq2}"] = df_asset.at[trade_date, "pct_chg"]
                        else:
                            df_asset.at[trade_date, f"macd_port{freq1,freq2}"] = 1
                    df_asset[f"macd_comp{freq1,freq2}"] = Alpha.comp_chg2(df=df_asset, inplace=False, abase=f"macd_port{freq1,freq2}")
                    df_result.at[ts_code, f"macd_comp{freq1,freq2}"] = df_asset[f"macd_comp{freq1,freq2}"].iat[-1]

                    df_asset.to_csv(f"temp/{ts_code}.csv")

            if not onestock:
                df_result.to_csv(f"test_{freq}2.csv")
            else:
                df_asset.to_csv(f"onestock{ts_code}.csv")


def asset_bullishness2():
    """this bullishness tries to find the stock by comparing their return each defined period.
    1. For each period, we rank the stocks
    2. In the end, we get the mean rank of all stocks


    """
    for freq in ["D","W"]:
        for asset in ["I","FD","E"]:
            # 2. setup alginment with 3 index
            df_ts_code=DB.get_ts_code(a_asset=[asset])
            df_sh = DB.get_asset(ts_code="000001.SH", asset="I")
            df_sz = DB.get_asset(ts_code="399001.SZ", asset="I")
            df_cy = DB.get_asset(ts_code="399006.SZ", asset="I")
            df_sh["000001.SH"] = df_sh["close"]
            df_sh["399001.SZ"] = df_sz["close"]
            df_sh["399006.SZ"] = df_cy["close"]

            d_preload = DB.preload(asset=asset,step=1)

            df_sh_Y = LB.df_to_freq(df_sh, freq)
            df_sh["000001.SH"]=df_sh_Y["close"]
            df_sh_Y = LB.df_to_freq(df_sz, freq)
            df_sh["399001.SZ"] = df_sh_Y["close"]
            df_sh_Y = LB.df_to_freq(df_cy, freq)
            df_sh["399006.SZ"] = df_sh_Y["close"]
            df_sh = df_sh[["000001.SH", "399001.SZ", "399006.SZ"]]
            df_sh = df_sh[df_sh["000001.SH"].notna()]

            for ts_code,df_asset in d_preload.items():
                print(f"calculate for {ts_code}")
                df_asset_Y= LB.df_to_freq(df_asset, freq)
                df_sh[ts_code] = df_asset_Y["close"]


            df_sh=df_sh.pct_change()


            df_sh=df_sh.rank(axis=1,na_option="keep",pct=True,ascending=True)

            #calculate gmean of each rank each day for each stock
            #punish inconsistency, reward consistency. rank(0.9 * 0.1) < rank(0.5 * 0.5)

            df_summary=pd.Series()
            for ts_code,df_asset in d_preload.items():

                s=df_sh[ts_code][df_sh[ts_code].notna()]
                df_summary.at[ts_code] = gmean(s)

            #gmean is better df_summary = df_sh.mean(axis=0)
            df_summary = df_summary.to_frame()
            df_summary.columns=[f"mean gain RANK per {freq}"]
            for ts_code in df_summary.index:
                if ts_code in ["000001.SH", "399001.SZ", "399006.SZ"]:
                    continue
                df_summary.at[ts_code,"period"]=d_preload[ts_code]["period"].iat[-1]
                df_summary.at[ts_code,"name"]=df_ts_code.at[ts_code,"name"]

            LB.to_csv_feather(df=df_sh, a_path=LB.a_path(f"Market/CN/ATest/bullishness2/data_{asset}_{freq}"),skip_feather=True)
            LB.to_csv_feather(df=df_summary, a_path=LB.a_path(f"Market/CN/ATest/bullishness2/df_summary_{asset}_{freq}"),skip_feather=True)


def random_dist():
    import random
    """
    Is long time holding periods return proportinoally better than random holding period on:
    rising stock
    cyclic stock
    falling stock

    
    
    result:
    the longer holding period, the more certain the daily return
    the less holding period, the more volatile, the more gamble
    the holding period return is the same for all periods 
    so the only difference is certainty


    :return:
    """
    df_final_result=pd.DataFrame()
    asset="I"
    df_ts_code=DB.get_ts_code(a_asset=[asset])
    #sample_ts_code=random.sample(range(1, len(df_ts_code)),3000)
    sample_ts_code=["399006.SZ"]
    for n in sample_ts_code:
        #ts_code=df_ts_code.index[n]
        ts_code="399006.SZ"
        df_asset= DB.get_asset(ts_code=ts_code,asset=asset)

        if False:
            df_asset["close"]=1.01**df_asset["period"]
            df_asset["sinus"]=df_asset["period"].apply(lambda x: math.sin(x) )
            df_asset["close"] =df_asset["close"]+df_asset["sinus"]*10


        df_result=pd.DataFrame()
        for trade in range(100000):

            sample_output = random.sample(range(1,len(df_asset)), 2)
            start_trade=builtins.min(sample_output)
            end_trade=builtins.max(sample_output)

            start_trade_date=df_asset.index[start_trade]
            end_trade_date=df_asset.index[end_trade]

            start_price=df_asset["close"].iat[start_trade]
            end_price=df_asset["close"].iat[end_trade]
            period_return=end_price/start_price
            period=end_trade-start_trade
            averager_ari=(period_return-1)/period
            averager_log=period_return**(1/period)
            print(trade,ts_code,start_trade,end_trade,period,period_return)

            df_result.at[trade,"start_trade_date"]=start_trade_date
            df_result.at[trade,"end_trade_date"]=end_trade_date
            df_result.at[trade,"start_price"]=start_price
            df_result.at[trade,"end_price"]=end_price
            df_result.at[trade,"period"]=period
            df_result.at[trade,"period_return"]=period_return
            df_result.at[trade,"averager_log"]=averager_log
            df_result.at[trade,"averager_ari"]=averager_ari

        for lower, upper in LB.custom_pairwise_overlap([5,10,20,60,250,500,1000,2000,4000,8000]):
            print(lower,upper)
            df_filter=df_result[df_result["period"].between(lower,upper)]
            df_final_result.at[ts_code,f"averager_log{lower,upper}"]=df_filter["averager_log"].mean()
            #df_final_result.at[ts_code,f"averager_ari{lower,upper}"]=df_filter["averager_ari"].mean()

        df_result.to_csv(f'result{ts_code}.csv')
        df_final_result.to_csv(f'final{asset}.csv')


if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()


    if False:
        df=pd.read_csv("result600519.SH.csv")
        df_res=pd.DataFrame()
        for period in range(1,5000):
            print(period)
            df_filter=df[df["period"] == period]
            df_res.at[period,"mean"]=df_filter["averager"].mean()
        df_res.to_csv("distri.csv")

    #asset_bullishness2()
    #asset_bollinger()
    #short_vs_long_stg()
    short_vs_long_stg()
    #asset_bullishness(start_date=20180401, end_date=LB.latest_trade_date(), market="CN", step=1, a_asset=["FD"])





