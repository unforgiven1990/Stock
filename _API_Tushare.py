import tushare as ts
import pandas as pd
import time
import LB
import traceback

pro = ts.pro_api('c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32')
ts.set_token("c473f86ae2f5703f58eecf9864fa9ec91d67edbc01e3294f6a4f9c32")


def get(func, fname, kwargs, df_fallback=pd.DataFrame()):
    for _ in range(200):
        try:
            df = func(**kwargs)
            if df is None:
                print("Tushare", fname, "NONE", kwargs)
                return df_fallback
            elif df.empty:
                print("Tushare", fname, "EMPTY", kwargs)
                return df_fallback
            else:
                print("Tushare", fname, "SUCCESS", kwargs)
                return df
        except Exception as e:
            print("Tushare", fname, "ERROR", kwargs)
            traceback.print_exc()
            time.sleep(10)


def my_pro_bar(asset: str, ts_code: str, freq: str, start_date: str, end_date: str, adj="qfq", factors=[]):
    """for FD limited to 1000 rows. For E, limited to 4000 Rows"""
    return get(func=ts.pro_bar, fname="ts.pro_bar", kwargs=locals(), df_fallback=LB.empty_df("pro_bar"))

def my_hk_daily( ts_code, start_date="00000000", end_date="30000000"):
    """for FD limited to 1000 rows. For E, limited to 4000 Rows"""
    return get(func=pro.hk_daily, fname="pro.hk_daily", kwargs=locals(), df_fallback=LB.empty_df("pro_bar"))


def my_query(api_name="", ts_code="000001.SZ", start_date="00000000", end_date="00000000"):
    return get(func=pro.query, fname="pro.query", kwargs=locals())


def my_stockbasic(is_hs="", list_status="L", exchange="", fields='ts_code,name,area,list_date,is_hs,market'):
    return get(func=pro.stock_basic, fname="pro.stock_basic", kwargs=locals())

def my_hk_basic(list_status="L"):
    return get(func=pro.hk_basic, fname="pro.hk_basic", kwargs=locals())


def my_pro_daily(trade_date="00000000"):
    return get(func=pro.daily, fname="pro.daily", kwargs=locals())


def my_trade_cal(api_name="trade_cal", start_date="00000000", end_date="00000000", exchange="SSE"):
    return get(func=pro.query, fname="pro.query", kwargs=locals())


def my_holder_trade(ts_code): #IMPORTANT! holder_trade instead of holdertrade in my version!
    return get(func=pro.stk_holdertrade, fname="pro.stk_holdertrade", kwargs=locals(), df_fallback=LB.empty_df("holder_trade"))


def my_pledge_stat(ts_code):
    return get(func=pro.pledge_stat, fname="pro.pledge_stat", kwargs=locals(), df_fallback=LB.empty_df("pledge_stat"))


def my_cashflow(ts_code, start_date="00000000", end_date="30000000"):
    return get(func=pro.cashflow, fname="pro.cashflow", kwargs=locals(), df_fallback=LB.empty_df("cashflow"))


def my_fina_indicator(ts_code, start_date="00000000", end_date="30000000"):
    return get(func=pro.fina_indicator, fname="pro.fina_indicator", kwargs=locals(), df_fallback=LB.empty_df("fina_indicator"))


def my_balancesheet(ts_code, start_date="00000000", end_date="30000000"):
    return get(func=pro.balancesheet, fname="pro.balancesheet", kwargs=locals(), df_fallback=LB.empty_df("balancesheet"))


def my_income(ts_code, start_date="00000000", end_date="30000000"):
    return get(func=pro.income, fname="pro.income", kwargs=locals(), df_fallback=LB.empty_df("income"))


def my_block_trade(ts_code):
    return get(func=pro.block_trade, fname="pro.block_trade", kwargs=locals(), df_fallback=LB.empty_df("block_trade"))


def my_share_float(ts_code):
    return get(func=pro.share_float, fname="pro.share_float", kwargs=locals(), df_fallback=LB.empty_df("share_float"))


def my_repurchase(ann_date):
    return get(func=pro.repurchase, fname="pro.repurchase", kwargs=locals())


def my_index_classify(level="L1", src="SW"):
    return get(func=pro.index_classify, fname="pro.index_classify", kwargs=locals())


def my_index_basic(market="SSE"):
    return get(func=pro.index_basic, fname="pro.index_basic", kwargs=locals())


def my_index_member(index_code):
    return get(func=pro.index_member, fname="pro.index_member", kwargs=locals())


def my_fund_basic(market="E", fields="ts_code,name,fund_type,list_date,delist_date,issue_amount,m_fee,c_fee,benchmark,invest_type,type,market,custodian,management"):
    return get(func=pro.fund_basic, fname="pro.fund_basic", kwargs=locals())

def my_fund_daily(ts_code,start_date="00000000", end_date="30000000"):
    return get(func=pro.fund_daily, fname="pro.fund_daily", kwargs=locals())

def my_fund_nav(ts_code,market="O"):
    return get(func=pro.fund_nav, fname="pro.fund_nav", kwargs=locals())

def my_fund_portfolo(ts_code=""):
    return get(func=pro.fund_portfolio, fname="pro.fund_portfolio", kwargs=locals())


def my_fx_daily( trade_date="", start_date="", end_date=""):
    """limited to 1000 rows"""
    return get(func=pro.fx_daily, fname="pro.fx_daily", kwargs=locals())


def my_cb_basic(ts_code="", list_date="", exchange=""):
    return get(func=pro.cb_basic, fname="pro.cb_basic", kwargs=locals())

def my_cb_daily(ts_code="", trade_date="", start_date="00000000", end_date="30000000"):
    return get(func=pro.cb_daily, fname="pro.cb_daily", kwargs=locals())

def my_yc_cb(ts_code="", curve_type="", trade_date="",start_date="00000000", end_date="30000000"):
    return get(func=pro.yc_cb, fname="pro.yc_cb", kwargs=locals())


def my_margin_detail(ts_code):
    return get(func=pro.margin_detail, fname="pro.margin_detail", kwargs=locals(),df_fallback=LB.empty_df("margin_detail"))

def my_top10_holders(ts_code,start_date="00000000",end_date="30000000"):
    return get(func=pro.top10_holders, fname="pro.top10_holders", kwargs=locals(), df_fallback=LB.empty_df("top10_holders"))

def my_suspended(ts_code):
    return get(func=pro.suspend_d, fname="pro.suspend_d", kwargs=locals(),df_fallback=LB.empty_df("suspended"))

def my_dividend(ts_code):
    return get(func=pro.dividend, fname="pro.dividend", kwargs=locals(),df_fallback=LB.empty_df("dividend"))

def my_forecast(ts_code):
    return get(func=pro.forecast, fname="pro.forecast", kwargs=locals(),df_fallback=LB.empty_df("forecast"))


def my_major_news(start_date="",end_date="",fields="title,content,pub_time,src"):
    return get(func=pro.major_news, fname="pro.major_news", kwargs=locals())

def my_cctv_news(date=""):
    return get(func=pro.cctv_news, fname="pro.cctv_news", kwargs=locals())

def my_concept(src="ts"):
    return get(func=pro.concept, fname="pro.concept", kwargs=locals())

def my_concept_detail(id="",ts_code="",fields="id,concept_name,ts_code,name,in_date,out_date"):
    return get(func=pro.concept_detail, fname="pro.concept_detail", kwargs=locals())

if __name__ == '__main__':

    df = ts.pro_bar(ts_code='000001.SZ', asset='E', start_date='20180101', end_date='20181011')

    df.to_csv("test.csv",encoding="utf-8_sig")