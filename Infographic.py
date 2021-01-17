import pandas as pd
import numpy as np
import _API_Tushare
import _API_JQ
import _API_Investpy
import _API_Scrapy
import LB
import os.path
import cProfile
from tqdm import tqdm
import traceback
import Alpha
import datetime
import talib
import DB
import UI
import builtins
import matplotlib.pyplot as plt


from PIL import Image, ImageDraw, ImageFont

# init
width = 2000
height = int(width * 0.62)
backupfontpath = r'c:\windows\fonts\msyh.ttc'
mega1 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width * 0.15))
mega2 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width * 0.12))
mega3 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width * 0.08))
h1 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width * 0.05))
h2 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width * 0.04))
text = ImageFont.truetype(r'c:\windows\fonts\msyh.ttc', size=int(width * 0.02))

red="#3a9e57"
green='#bd613c'
white='#ffffff'

title_padding=(100,100)

def section_gain(trade_date, df_date):
    def get_histo(s, smallbins=True):

        # init
        step = 2
        a_boundary = [(0, 0)]
        for lower, upper in LB.custom_pairwise_overlap([x for x in range(0, 11, step)]):
            if bin != 0:
                a_boundary = [(-upper, -lower)] + a_boundary + [(lower, upper)]

        y = []
        for lower, upper in a_boundary:
            if lower == 0 and upper == 0:
                counter = len(s[s == 0])
            elif lower == -10:
                # lso count values smaller than -10
                counter = len(s[(s < upper)])
            elif upper == 10:
                counter = len(s[(s >= lower)])
            elif lower == 0 and upper !=0:
                counter = len(s[(s > lower) & (s < upper)])
            else:
                counter = len(s[(s >= lower) & (s < upper)])
            y += [counter]

        # transform data into bar chart
        # create chart
        # n, bins, patches = plt.hist(s, rwidth=0.9, bins=bins, log=False)  # use this to draw histogram of your data

        x = [x for x in range(0, len(y))]
        line = plt.bar(x, y)

        # draw values O
        for enum, i in enumerate(range(len(y))):
            # OVER each bar
            plt.annotate(str(y[i]), xy=(x[i], y[i] + 30), ha='center', va='bottom', color="white", size=8)
            # UNDER each bar
            lower = abs(a_boundary[i][0])
            upper = abs(a_boundary[i][1])
            if lower == upper:
                text = 0
            elif enum == 0:
                text = f"{upper}<"
            elif enum == len(y) - 1:
                text = f">{lower}"
            else:
                text = f"{lower}-{upper}"
            plt.annotate(text, xy=(x[i], -180), ha='center', va='bottom', color="white", size=9)

        # use this to draw histogram of your data
        # scale down the chart by making the y axis seem taller
        axes = plt.gca()
        padding = 300
        axes.set_ylim([-padding, 2500])
        plt.axis('off')

        # add color
        gradient = False
        if gradient == False:
            for i in x:
                if i < int(len(y) / 2):
                    line[i].set_color(red)
                elif i > int(len(y) / 2):
                    line[i].set_color(green)
            line[int(len(y) / 2)].set_color('w')
        else:
            """
                    https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
                    """
            cmgreen = plt.cm.Greens
            cmred = plt.cm.Reds
            cmtest = plt.cm.RdYlGn
            cmtest = plt.cm.plasma

            """
            for i, p in enumerate(patches):
                print("path",i)
                if False:
                    if i > 9:
                        plt.setp(p, 'facecolor', cmred(i / 12))
                    else:
                        plt.setp(p, 'facecolor', cmgreen(1-(i / 12)))
                else:
                    plt.setp(p, 'facecolor', cmtest(i / len(patches)))
    """
            pass

        # save png
        path = f'Plot/report_D/{trade_date}/histo.png'
        dpi=130
        try:
            plt.savefig(path, transparent=True, dpi=dpi * 3)
        except:
            os.makedirs(f'Plot/report_D/{trade_date}')
            plt.savefig(path, transparent=True, dpi=dpi * 3)
        return path

    histo = Image.open(get_histo(df_date["pct_chg"], smallbins=False))
    # histo=histo.resize((2000, 800), Image.ANTIALIAS)
    hw, hh = histo.size
    section1 = Image.new('RGBA', (width, 2200), '#222222')

    # add title
    idraw = ImageDraw.Draw(section1)
    idraw.text(title_padding, f"涨跌分布", font=h1)

    #paste all stuff into the section
    offsetfrom_hiddenaxis=30
    section1.paste(histo, (int((width - hw) / 2)-offsetfrom_hiddenaxis, -500), mask=histo)


    #涨跌幅统计
    die = len(df_date[df_date["pct_chg"] < 0])
    zhang=len(df_date[df_date["pct_chg"]>0])
    diepct = round(len(df_date[df_date["pct_chg"] < 0]) / len(df_date) * 100, 0)
    zhangpct = round(len(df_date[df_date["pct_chg"] > 0]) / len(df_date) * 100, 0)
    offset=-600
    for i,tuple in enumerate([["跌",die,diepct],["涨",zhang,zhangpct]]):
        text=tuple[0]
        absnum = f"{int(tuple[1])}家"
        counter=f"{int(tuple[2])}%"
        color= red if i==0 else green
        norm=-1 if i==0 else 1
        distance=350

        #pct of stocks 涨跌
        w, height1 = idraw.textsize(f"{counter}", font=mega2)
        idraw.text(( int(((width - w) / 2)) + norm*distance, hh+offset+0), f"{counter}", font=mega2, fill=color)

        # abs of stocks 涨跌
        w, height2 = idraw.textsize(absnum, font=h2)
        idraw.text((int(((width - w) / 2)) + norm*distance, hh + height1 + offset + 20), absnum, font=h2, fill=color)

        # text 涨跌
        w, height3 = idraw.textsize(text, font=mega1)
        idraw.text( (  int(((width - w) / 2)) + norm*distance, hh +height1 + height2+ offset-50), text, font=mega1, fill=color)

    return section1

def section_index(trade_date, df_date):
    section = Image.new('RGBA', (width, 2200), '#333')

    # add title
    idraw = ImageDraw.Draw(section)
    idraw.text(title_padding, f"指数涨幅", font=h1)

    a_index_data=[]
    for ts_index,name in zip(["000001.SH","399006.SZ","399001.SZ"],["上证指数","深成指数","创业板指"]):

        #create miniature view of todays chart
        df_freq = DB.get_asset(ts_code="000001.SH",asset="I",freq="5min")
        if df_freq.empty:
            df_freq = _API_JQ.my_get_price_5min(security=LB.df_switch_ts_code("000001.SH"), start_date=trade_date, end_date=trade_date)
            LB.to_csv_feather(df_freq,a_path=LB.a_path(f"Market/CN/Asset/I/5min/{ts_index}/{trade_date}"),skip_feather=True)


        d_data={}
        d_data[f"name"]=name
        d_data[f"df"]=df=DB.get_asset(ts_index,asset="I")
        d_data[f"close"] = df.at[int(trade_date), "close"]
        d_data[f"pct_chg"]=df.at[int(trade_date),"pct_chg"]
        a_index_data+=[d_data]

    offset = 400
    for i, d_data in enumerate(a_index_data):

        if d_data['pct_chg']> 0 :
            color = red
        elif d_data['pct_chg']==0:
            color = white
        elif d_data['pct_chg'] < 0:
            color = green

        text = d_data["name"]
        absclose = f"{int(d_data['close'])}"
        pct_chg = f"{round(d_data['pct_chg'],2)}%"
        norm = {0:-1,1:0,2:1}
        distance = 650

        # 标题
        w, height1 = idraw.textsize(f"{text}", font=h2)
        idraw.text((int(((width - w) / 2)) + norm[i] * distance,   offset + 0), text, font=h2, fill=color)

        # text 涨跌
        w, height2 = idraw.textsize(absclose, font=h1)
        idraw.text((int(((width - w) / 2)) + norm[i] * distance, height1  + offset + 10), absclose, font=h1, fill=color)

        # abs of stocks 涨跌
        w, height3 = idraw.textsize(pct_chg, font=mega3)
        idraw.text((int(((width - w) / 2)) + norm[i] * distance,  height1 + height2 + offset + 20), pct_chg, font=mega3, fill=color)


    return section


def create_infographic(trade_date=LB.latest_trade_date()):

    # init
    df_date=DB.get_date(trade_date=trade_date)
    if df_date.empty:
        raise  AssertionError
    df_date = df_date[df_date["pct_chg"].notna()]

    #add sections
    a_sections = []
    a_sections += [section_index(trade_date=trade_date, df_date=df_date)]
    a_sections+=[section_gain(trade_date=trade_date,df_date=df_date)]

    #put all sections together
    offset = 300
    autoheight=builtins.sum([offset]+[section.size[1] for section in a_sections])
    infographic = Image.new('RGBA', (width, autoheight), "#999999")
    y_helper=0
    for section in a_sections:
        infographic.paste(section,( 0, offset+y_helper),mask=section)
        y_helper += section.size[1]

    #draw title, later to be moved to anoter section
    idraw = ImageDraw.Draw(infographic)
    title=f"A股盘后总结 {trade_date[0:4]}.{trade_date[4:6]}.{trade_date[6:8]}. "
    w, h = idraw.textsize(title,font=h1)
    idraw.text(  ( int((width - w) / 2), 100), title,font=h1)




    infographic.show()
    infographic.save("cool.png")




if __name__ == '__main__':
    trade_date = "20200424"

    create_infographic(trade_date="20200424")