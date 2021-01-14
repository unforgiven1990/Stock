import scrapy
import requests
import DB
import pandas as pd
import LB
import os
import numpy as np

def qdii_research():
    """券商研报 from sina finance
    scrapy method
    """
    def scrape(url):
        sel = scrapy.Selector(text=requests.get(url).text)
        s_titles = sel.xpath('//td[@class="tal f14"]/a/@title').extract()
        s_date = sel.xpath('//td[@class="tal f14"]/following-sibling::td[2]/text()').extract()
        s_qdii = sel.xpath('//td[@class="tal f14"]/following-sibling::td[3]/a/div/span/text()').extract()
        s_person = sel.xpath('//td[@class="tal f14"]/following-sibling::td[4]/div/span/text()').extract()

        a_result = []
        for counter, (title, date, qdii, person) in enumerate(zip(s_titles, s_date, s_qdii, s_person)):
            a_result.append([title, date, qdii, person])
        return a_result


    #1. Create a list of all urls
    df_ts_code=DB.get_ts_code(a_asset=["E"])
    for ts_code in df_ts_code.index:
        ts_code_sina=ts_code[-2:].lower()+ts_code[0:6]

        ts_result=[]
        page_count = 1
        while True:
            print("scrap qdii research",ts_code,page_count)
            url = f"http://stock.finance.sina.com.cn/stock/go.php/vReport_List/kind/search/index.phtml?symbol={ts_code_sina}&t1=all&p={page_count}"
            a_result = scrape(url=url)
            if a_result:
                ts_result+=a_result
                page_count +=  1
            else:
                break

        #post processing
        df_result=pd.DataFrame(data=ts_result,columns=["title","date","qdii","person"])
        df_result["date"]=df_result["date"].str.replace("-","")
        df_result=LB.df_reverse_reindex(df_result)
        df_result["ts_code"]=ts_code
        df_result=df_result.set_index("date",drop=True)
        a_path=LB.a_path(f"Market/CN/Asset/E/qdii_research/{ts_code}")
        LB.to_csv_feather(df_result,a_path=a_path,skip_csv=True)



def qdii_grade(offset=0,step=1, qdii="grade"):
    """券商评估 from sina finance
    direct pandas method
    """
    def scrape(url):
        sel = scrapy.Selector(text=requests.get(url).text)
        s_titles = sel.xpath('//td[@class="tal f14"]/a/@title').extract()
        s_date = sel.xpath('//td[@class="tal f14"]/following-sibling::td[2]/text()').extract()
        s_qdii = sel.xpath('//td[@class="tal f14"]/following-sibling::td[3]/a/div/span/text()').extract()
        s_person = sel.xpath('//td[@class="tal f14"]/following-sibling::td[4]/div/span/text()').extract()

        a_result = []
        for counter, (title, date, qdii, person) in enumerate(zip(s_titles, s_date, s_qdii, s_person)):
            a_result.append([title, date, qdii, person])
        return a_result


    df_ts_code=DB.get_ts_code(a_asset=["E"])
    qdii_grade_columns = ['ts_code', 'name', "target", "grade", "qdii", "person", "industry", "date","summary", "latest_close", "latest_pct_chg", "favorite", "thread"]

    for ts_code in df_ts_code.index[offset::step]:

        #1. check if existing file exists:
        a_path=LB.a_path(f"Market/CN/Asset/E/qdii_{qdii}/{ts_code}")
        if os.path.isfile(a_path[0]):
            df_saved = DB.get(a_path=a_path,set_index="trade_date")
            try:
                from_date = df_saved.index[-1]
                print("from date is ",from_date)
                if from_date == LB.today():
                    continue
                if np.isnan(from_date):
                    from_date = 00000000
            except Exception as e:
                from_date = 00000000
                print(e)
        else:
            df_saved = pd.DataFrame(columns=qdii_grade_columns)
            from_date = 00000000


        ts_code_sina = ts_code[-2:].lower() + ts_code[0:6]
        a_df_result=[]
        page_count = 1
        looping=True
        while looping:

                try:
                    if qdii == "grade":
                        url = f"http://stock.finance.sina.com.cn/stock/go.php/vIR_StockSearch/key/{ts_code_sina}.phtml?p={page_count}"
                        #df = pd.read_html(url,flavor=['bs4'])
                        print(f"{offset} scrap qdii grade", url)

                        df = pd.read_html(url)
                        df = df[0]
                        df.columns = df.iloc[0]
                        df = df.drop(df.index[0])
                        df.columns = qdii_grade_columns
                    elif qdii=="research":
                        url = f"http://stock.finance.sina.com.cn/stock/go.php/vReport_List/kind/search/index.phtml?symbol={ts_code_sina}&t1=all&p={page_count}"
                        print(f"{offset} scrap qdii research", url)
                        a_result = scrape(url=url)
                        df=pd.DataFrame(a_result,columns=["title","date","qdii","person"])


                    if df.empty:
                        looping = False
                    else:
                        df["date"] = df["date"].str.replace("-", "")
                        last_day=df["date"].iat[0]

                    if len(df)>1:
                        a_df_result.append(df)
                        page_count += 1
                    else:
                        print("finished",ts_code)
                        looping=False

                    if int(last_day)<=int(from_date):
                        print(f"{ts_code} is up-to-date qdii_{qdii}")
                        break

                except:
                    break



        #post processing
        if a_df_result:
            df_result=pd.concat(a_df_result,sort=False,ignore_index=True)
            df_result = LB.df_reverse_reindex(df_result)
        else:
            df_result=pd.DataFrame(columns=qdii_grade_columns)

        df_result["ts_code"] = ts_code
        df_result = df_result.set_index("date", drop=True)
        df_result.index.name="trade_date"

        #use previously saved result

        if df_result.empty and df_saved.empty:
            #both empty, dont change the data at all
            pass
        if df_result.empty and not df_saved.empty:
            #leave df_saved as it is. nothing to change
            pass
        if not df_result.empty and not df_saved.empty:
            #both not empty, merge and save
            df_result = df_saved.append(df_result, ignore_index=False, sort=False)

            #remove duplicated
            df_result=df_result.drop_duplicates()
            LB.to_csv_feather(df_result,a_path=a_path,skip_feather=True,index_relevant=True)
        if not df_result.empty and df_saved.empty:
            #save df_result
            LB.to_csv_feather(df_result, a_path=a_path, skip_feather=True, index_relevant=True)





if __name__ == "__main__":
    #df=pd.read_html("https://finviz.com/screener.ashx?v=111&o=ticker&r=0",flavor=['bs4'])
    df=pd.read_csv("egal.csv")
    print(df)
    # df=df[0]
    # df.columns = df.iloc[0]
    # df=df.drop(df.index[0])
    # df["股票代码"]=df["股票代码"].astype(str)
    # df.to_csv("test.csv",encoding="utf-8_sig")
