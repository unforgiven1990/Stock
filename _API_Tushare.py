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
    return get(func=ts.pro_bar, fname="ts.pro_bar", kwargs=locals(), df_fallback=LB.df_empty("pro_bar"))


def my_hk_daily(ts_code, start_date="00000000", end_date="30000000"):
    """for FD limited to 1000 rows. For E, limited to 4000 Rows"""
    return get(func=pro.hk_daily, fname="pro.hk_daily", kwargs=locals(), df_fallback=LB.df_empty("pro_bar"))


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


def my_holder_trade(ts_code):  # IMPORTANT! holder_trade instead of holdertrade in my version!
    return get(func=pro.stk_holdertrade, fname="pro.stk_holdertrade", kwargs=locals(), df_fallback=LB.df_empty("holder_trade"))


def my_pledge_stat(ts_code):
    return get(func=pro.pledge_stat, fname="pro.pledge_stat", kwargs=locals(), df_fallback=LB.df_empty("pledge_stat"))


def my_cashflow(ts_code, start_date="00000000", end_date="30000000"):
    return get(func=pro.cashflow, fname="pro.cashflow", kwargs=locals(), df_fallback=LB.df_empty("cashflow"))


def my_fina_indicator(ts_code, start_date="00000000", end_date="30000000"):
    fields = [x for x in LB.c_indicator_rank().keys()]
    fields = ["ts_code", "ann_date", "end_date"] + fields  # add these, otherwise it is not there lol
    fields = ",".join(fields)
    return get(func=pro.fina_indicator, fname="pro.fina_indicator", kwargs=locals(), df_fallback=LB.df_empty("fina_indicator"))


def my_balancesheet(ts_code, start_date="00000000", end_date="30000000"):
    return get(func=pro.balancesheet, fname="pro.balancesheet", kwargs=locals(), df_fallback=LB.df_empty("balancesheet"))


def my_income(ts_code, start_date="00000000", end_date="30000000"):
    return get(func=pro.income, fname="pro.income", kwargs=locals(), df_fallback=LB.df_empty("income"))


def my_block_trade(ts_code):
    return get(func=pro.block_trade, fname="pro.block_trade", kwargs=locals(), df_fallback=LB.df_empty("block_trade"))


def my_share_float(ts_code):
    return get(func=pro.share_float, fname="pro.share_float", kwargs=locals(), df_fallback=LB.df_empty("share_float"))


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


def my_fund_daily(ts_code, start_date="00000000", end_date="30000000"):
    return get(func=pro.fund_daily, fname="pro.fund_daily", kwargs=locals())


def my_fund_nav(ts_code, market="O"):
    return get(func=pro.fund_nav, fname="pro.fund_nav", kwargs=locals())


def my_fund_portfolio(ts_code=""):
    return get(func=pro.fund_portfolio, fname="pro.fund_portfolio", kwargs=locals())


def my_fx_daily(trade_date="", start_date="", end_date=""):
    """limited to 1000 rows"""
    return get(func=pro.fx_daily, fname="pro.fx_daily", kwargs=locals())


def my_cb_basic(ts_code="", list_date="", exchange=""):
    return get(func=pro.cb_basic, fname="pro.cb_basic", kwargs=locals())


def my_cb_daily(ts_code="", trade_date="", start_date="00000000", end_date="30000000"):
    return get(func=pro.cb_daily, fname="pro.cb_daily", kwargs=locals())


def my_yc_cb(ts_code="", curve_type="", trade_date="", start_date="00000000", end_date="30000000"):
    return get(func=pro.yc_cb, fname="pro.yc_cb", kwargs=locals())


def my_margin_detail(ts_code):
    return get(func=pro.margin_detail, fname="pro.margin_detail", kwargs=locals(), df_fallback=LB.df_empty("margin_detail"))


def my_top10_holders(ts_code, start_date="00000000", end_date="30000000"):
    return get(func=pro.top10_holders, fname="pro.top10_holders", kwargs=locals(), df_fallback=LB.df_empty("top10_holders"))


def my_suspended(ts_code):
    return get(func=pro.suspend_d, fname="pro.suspend_d", kwargs=locals(), df_fallback=LB.df_empty("suspended"))


def my_dividend(ts_code):
    return get(func=pro.dividend, fname="pro.dividend", kwargs=locals(), df_fallback=LB.df_empty("dividend"))

def my_us_basic(ts_code=None,classify=None):
    return get(func=pro.us_basic, fname="pro.us_basic", kwargs=locals())


def my_forecast(ts_code):
    return get(func=pro.forecast, fname="pro.forecast", kwargs=locals(), df_fallback=LB.df_empty("forecast"))


def my_major_news(start_date="", end_date="", fields="title,content,pub_time,src"):
    return get(func=pro.major_news, fname="pro.major_news", kwargs=locals())


def my_cctv_news(date=""):
    return get(func=pro.cctv_news, fname="pro.cctv_news", kwargs=locals())


def my_concept(src="ts"):
    return get(func=pro.concept, fname="pro.concept", kwargs=locals())


def my_concept_detail(id="", ts_code="", fields="id,concept_name,ts_code,name,in_date,out_date"):
    return get(func=pro.concept_detail, fname="pro.concept_detail", kwargs=locals())


def my_cn_m(start_m="00000000", end_m="99999999", fields='month,m0,m1,m2,m0_yoy,m1_yoy,m2_yoy', ):
    return get(func=pro.cn_m, fname="pro.cn_m", kwargs=locals())

def my_hsgt(start_date="00000000",end_date="99999999"):
    return get(func=pro.moneyflow_hsgt, fname="pro.moneyflow_hsgt", kwargs=locals())

def my_hk_hold(ts_code):
    return get(func=pro.hk_hold, fname="pro.hk_hold", kwargs=locals())


if __name__ == '__main__':
    import DB
    ts_code = "600887.SH"
    df=DB.tushare_limit_breaker(pro.hk_hold, {"ts_code":ts_code,"start_date":"20180101"}, limit=1000)
    df=LB.df_reverse_reindex(df)
    df.to_csv(f"{ts_code}.csv", encoding="utf-8_sig")
