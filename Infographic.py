import pandas as pd
import numpy as np
import _API_Tushare

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
import matplotlib
import squarify


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
h3 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width * 0.03))
text = ImageFont.truetype(r'c:\windows\fonts\msyh.ttc', size=int(width * 0.02))

red="#3a9e57"
green='#bd613c'
white='#ffffff'
gray="#333333"

treemapred1="#FF8A80"
treemapred2="#FF5252"
treemapred3="#FF1744"
treemapred4="#D50000"

treemapgreen1="#B9F6CA"
treemapgreen2="#69F0AE"
treemapgreen3="#00E676"
treemapgreen4="#00C853"

title_padding=(100,100)
matplotlib.rc('font', family='Microsoft Yahei')
#https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

def img_saver(path,dpi=130):
    try:
        plt.savefig(path, transparent=True, dpi=dpi ,bbox_inches='tight')
    except:
        folder = "/".join(path.rsplit("/")[:-1])
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(path, transparent=True, dpi=dpi ,bbox_inches='tight')
    plt.clf()
    return path

def draw_text(section, text, fontsize=h1, left=True):
    idraw = ImageDraw.Draw(section)
    if left:
        idraw.text(title_padding, text, font=fontsize)
    else:
        w, h = idraw.textsize(text, font=fontsize)
        idraw.text((int((width - w) / 2), title_padding[1]), text, font=fontsize)
    return idraw

def get_bar(s,path,y_limit=[-300, 2500]):

    # init: create boundary
    step = 2
    a_boundary = [(0, 0)]
    for lower, upper in LB.custom_pairwise_overlap([x for x in range(0, 11, step)]):
        if bin != 0:
            a_boundary = [(-upper, -lower)] + a_boundary + [(lower, upper)]

    #y axis
    y = []
    for lower, upper in a_boundary:
        if lower == 0 and upper == 0:
            counter = len(s[s == 0])
        elif lower == -10:
            # ALSO count values smaller than -10
            counter = len(s[(s < upper)])
        elif upper == 10:
            # ALSO count values bigger than 10
            counter = len(s[(s >= lower)])
        elif lower == 0 and upper !=0:
            counter = len(s[(s > lower) & (s < upper)])
        else:
            counter = len(s[(s >= lower) & (s < upper)])
        y += [counter]

    # x axis: transform data into bar chart
    x = [x for x in range(0, len(y))]
    line = plt.bar(x, y)

    # draw values as text
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
    axes.set_ylim(y_limit)
    plt.axis('off')

    # add color
    for i in x:
        if i < int(len(y) / 2):
            line[i].set_color(red)
        elif i > int(len(y) / 2):
            line[i].set_color(green)
    line[int(len(y) / 2)].set_color(f'#ffffff')

    # save png
    return img_saver(path=path,dpi=390)



def section_stock_gain(trade_date, df_date):


    # add init
    section = Image.new('RGBA', (width, 2200), '#222222')
    idraw = draw_text(section, f"涨跌分布")

    # add histogram
    histo = Image.open(get_bar(s=df_date["pct_chg"],path = f'Plot/report_D/{trade_date}/histo.png'))
    hw, hh = histo.size

    #paste all stuff into the section
    offsetfrom_hiddenaxis=0
    histooffset=300
    section.paste(histo, (int((width - hw) / 2)-offsetfrom_hiddenaxis, -histooffset), mask=histo)

    #涨跌幅统计
    die = len(df_date[df_date["pct_chg"] < 0])
    zhang = len(df_date[df_date["pct_chg"]>0])
    diepct = round(die / len(df_date) * 100, 0)
    zhangpct = round(zhang / len(df_date) * 100, 0)
    offset = -histooffset-100
    for i,tuple in enumerate([["跌",die,diepct,-1],["涨",zhang,zhangpct,1]]):
        abs_count = f"{int(tuple[1])}家"
        pct_count=f"{int(tuple[2])}%"
        text = tuple[0]
        color= red if i==0 else green
        norm=tuple[3]
        hdistance=350

        #pct of stocks 涨跌
        w, height1 = idraw.textsize(f"{pct_count}", font=mega2)
        idraw.text(( int(((width - w) / 2)) + norm*hdistance, hh+offset+0), f"{pct_count}", font=mega2, fill=color)

        # abs of stocks 涨跌
        w, height2 = idraw.textsize(abs_count, font=h2)
        idraw.text((int(((width - w) / 2)) + norm*hdistance, hh + height1 + offset + 20), abs_count, font=h2, fill=color)

        # text 涨跌
        w, height3 = idraw.textsize(text, font=mega1)
        idraw.text( (  int(((width - w) / 2)) + norm*hdistance, hh +height1 + height2+ offset-50), text, font=mega1, fill=color)

    return section

def section_index(trade_date, df_date):

    # add init
    section = Image.new('RGBA', (width, 2600), '#111111')
    idraw = draw_text(section, f"指数涨幅")

    #calculate data for 3 index
    a_pct_chg=[]
    vertical_offset = 400
    horizontal_offset = 650
    for ts_index,name,i in zip(["000001.SH","399006.SZ","399001.SZ"],["上证指数","深成指数","创业板指"],[-1,0,1]):

        #create miniature view of todays chart
        df_freq = DB.get_asset(ts_code=f"{ts_index}_{trade_date}",asset="I",freq="10min")
        if df_freq.empty:
            import _API_JQ
            df_freq = _API_JQ.my_get_price_10min(security=LB.df_switch_ts_code(ts_index), start_date=trade_date, end_date=trade_date)
            df_freq.index.name="trade_date"
            LB.to_csv_feather(df_freq,a_path=LB.a_path(f"Market/CN/Asset/I/10min/{ts_index}_{trade_date}"),skip_feather=True)

        #save chart and read chart into pillow
        df_freq["close"].plot(color='#ffffff',legend=False,linewidth=8)
        plt.axis("off")
        path = f'Plot/report_D/{trade_date}/{ts_index}.png'
        img_saver(path=path,dpi=80)

        #create data pack of index
        chart=Image.open(path)
        df=DB.get_asset(ts_index,asset="I")
        close= int(df.at[int(trade_date), "close"])
        pct_chg=df.at[int(trade_date),"pct_chg"]
        a_pct_chg+=[pct_chg]

        if pct_chg> 0 :
            color = red
        elif pct_chg==0:
            color = white
        elif pct_chg < 0:
            color = green

        # display 标题
        w, height1 = idraw.textsize(f"{name}", font=h2)
        idraw.text((int(((width - w) / 2)) + i * horizontal_offset,   vertical_offset + 0), name, font=h2, fill=white)

        # display close
        w, height2 = idraw.textsize(f"{close}", font=h1)
        idraw.text((int(((width - w) / 2)) + i * horizontal_offset, height1  + vertical_offset + 0), f"{close}", font=h1, fill=white)

        # display pct_chg 涨跌
        w, height3 = idraw.textsize(f"{round(pct_chg,2)}%", font=mega3)
        idraw.text((int(((width - w) / 2)) + i * horizontal_offset,  height1 + height2 + vertical_offset + 40), f"{round(pct_chg,2)}%", font=mega3, fill=color)

        # display mini chart
        section.paste(chart, (int(((width - w) / 2))+70 + i * horizontal_offset, height1 + height2 +height3 + vertical_offset + 50), mask=chart)



    #STOCK DATA
    #beat index E data
    abs_beat_index = len(df_date[df_date["pct_chg"] > builtins.max(a_pct_chg)])
    pct_beat_index = int((abs_beat_index / len(df_date)) * 100)

    #beat index E pct
    baseheight=1300
    w, height4 = idraw.textsize(f"{pct_beat_index}%", font=mega1)
    idraw.text((int(((width-w) / 2)) , baseheight), f"{pct_beat_index}%", font=mega1, fill=white)

    #beat index E text
    textmsg=f"{abs_beat_index}只股票跑赢三指"
    w, height5 = idraw.textsize(textmsg, font=h2)
    idraw.text((int(((width - w) / 2)), height4+ baseheight), textmsg, font=h2, fill=white)

    textmsg = f"(上市40交易日以上)"
    w, height55 = idraw.textsize(textmsg, font=text)
    idraw.text((int(((width - w) / 2)), height4 + height5 + baseheight), textmsg, font=text, fill=white)

    #FD DATA
    df_date_FD=DB.get_date(trade_date=trade_date,a_asset=["FD"])
    df_date_FD=df_date_FD[ (df_date_FD["fund_type"]=="股票型") | (df_date_FD["fund_type"]=="混合型")]
    baseheight2=100

    # beat index E data
    abs_beat_index = len(df_date_FD[df_date_FD["pct_chg"] > builtins.max(a_pct_chg)])
    pct_beat_index = int((abs_beat_index / len(df_date_FD)) * 100)

    # beat index FD pct
    w, height6 = idraw.textsize(f"{pct_beat_index}%", font=mega1)
    idraw.text((int(((width - w) / 2)), height4+height5+ baseheight2+baseheight), f"{pct_beat_index}%", font=mega1, fill=white)

    # beat index FD text基金
    textmsg = f"{abs_beat_index}只基金跑赢三指"
    w, height7 = idraw.textsize(textmsg, font=h2)
    idraw.text((int(((width - w) / 2)), height4+height5+height6+ baseheight2+baseheight), textmsg, font=h2, fill=white)

    # beat index FD text基金
    textmsg = f"(股票型+混合型场内基金)"
    w, height8 = idraw.textsize(textmsg, font=text)
    idraw.text((int(((width - w) / 2)), height4 + height5 + height6 +height7 + baseheight2 + baseheight), textmsg, font=text, fill=white)

    return section

def section_ma(trade_date, df_date):

    # add init
    section = Image.new('RGBA', (width, 2200), '#111111')
    idraw = draw_text(section, f"均线统计")

    #calculate data
    d_ma={}
    a_y_abs=[]
    for freqn in [5,20,60,240]:
        y_abs=len(df_date[df_date["close"]>=df_date[f"ma{freqn}"]])
        a_y_abs += [y_abs]
        pct=y_abs/len(df_date)
        d_ma[f"{freqn}日 均线上"]=int(pct*100)

    #display data
    x=[x for x in d_ma.keys()]
    y=[x for x in d_ma.values()]
    y_fake=[100 for x in d_ma.values()]
    plt.bar(x, y_fake)#background
    line=plt.bar(x, y)
    for i in range(0,len(line)):
        line[i].set_color(white)
    axes = plt.gca()
    axes.set_ylim([-20, 110])
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(6, 4)

    # draw values as text
    for enum, i in enumerate(range(len(y))):
        # OVER each bar
        if y[i]<=80:
            add=(5,1)
            color="white"
        else:
            add = (-16,-21)
            color="black"

        plt.annotate(f"{y[i]}%", xy=(i, y[i]+add[0]), ha='center', va='bottom', color=color, size=24)
        plt.annotate(f"{a_y_abs[i]}家收盘", xy=(i, y[i]+add[1]), ha='center', va='bottom', color=color, size=8)

        # UNDER each bar
        plt.annotate(x[i], xy=(i, -10), ha='center', va='bottom', color="white", size=11)

    # use this to draw histogram of your data
    path = f'Plot/report_D/{trade_date}/ma.png'
    img_saver(path=path,dpi=350)
    chart = Image.open(path)

    cw,ch=chart.size
    section.paste(chart, (int((width-cw)/2), 300), mask=chart)
    return section


def section_rsi(trade_date, df_date):

    # add init
    section = Image.new('RGBA', (width, 2200), '#222222')
    idraw = draw_text(section, f"RSI值统计")

    #calculate data
    d_ma={}
    for freqn in [5,20,60,240]:
        y_abs=df_date[f"rsi{freqn}"].mean()
        d_ma[f"{freqn}日 RSI"]=round(y_abs,1)

    #display data
    x=[x for x in d_ma.keys()]
    y=[x for x in d_ma.values()]
    y_fake = [100 for x in d_ma.values()]
    plt.bar(x, y_fake)  # background
    line=plt.bar(x, y)
    for i in range(0,len(line)):
        line[i].set_color(white)
    axes = plt.gca()
    axes.set_ylim([-20, 110])
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(6, 4)

    # draw values as text
    for enum, i in enumerate(range(len(y))):
        # OVER each bar
        if y[i]<=80:
            add=(5,1)
            color="white"
        else:
            add = (-12,-16)
            color="black"

        plt.annotate(f"{y[i]}", xy=(i, y[i]+add[0]), ha='center', va='bottom', color=color, size=24)
        plt.annotate(f"所有股票平均", xy=(i, y[i]+add[1]), ha='center', va='bottom', color=color, size=8)

        # UNDER each bar
        plt.annotate(x[i], xy=(i, -10), ha='center', va='bottom', color="white", size=11)

    # use this to draw histogram of your data
    path = f'Plot/report_D/{trade_date}/rsi.png'
    img_saver(path=path,dpi=350)
    chart = Image.open(path)

    cw,ch=chart.size
    section.paste(chart, (int((width-cw)/2), 300), mask=chart)
    return section



def section_nsmoney(trade_date, df_date):

    # add init
    section = Image.new('RGBA', (width, 4200), 'red')
    idraw = draw_text(section, f"南北向资金")

    #calculate data
    df_nsmoney=DB.get(a_path=LB.a_path("Market/CN/Asset/E/hsgt/hsgt"))
    df_nsmoney=df_nsmoney[df_nsmoney.index <= int(trade_date)]
    df_nsmoney=df_nsmoney.tail(120)#only check last 120 days as half year data
    df_nsmoney["zero"]=0
    df_nsmoney.index = df_nsmoney.index.astype(str)
    df_nsmoney["date"]=df_nsmoney.index
    df_nsmoney["date"]=df_nsmoney["date"].str.slice(4,6) + f"月"+df_nsmoney["date"].str.slice(6,8) +f"日"
    df_nsmoney.index = df_nsmoney["date"]
    df_nsmoney.index.name=""

    #display data
    for i, name in enumerate(["north","south"]):
        ax=df_nsmoney[f"{name}_money"].plot(color='#ffffff', legend=False, linewidth=3)
        df_nsmoney["zero"].plot(color='#ffffff', legend=False, linewidth=1)
        plt.axis("on")

        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_label("万元")
        [t.set_color('white') for t in ax.xaxis.get_ticklabels()]
        [t.set_color('white') for t in ax.yaxis.get_ticklabels()]
        ax.tick_params(axis="x",color="white")
        ax.tick_params(axis="y",color="white")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45)

        fig=plt.gcf()
        fig.set_size_inches(6, 2)

        path = f'Plot/report_D/{trade_date}/{name}.png'
        img_saver(path=path, dpi=350)
        chart = Image.open(path)
        cw,ch=chart.size
        section.paste(chart, (int((width-cw)/2), 300+i*1000), mask=chart)

    return section

def section_industry1(trade_date, df_date):

    # add init
    section = Image.new('RGBA', (width, 2200), '#111111')
    idraw = draw_text(section, f"申万一级行业热力图")

    #treemap data
    df_group_sum=df_date.groupby(by="sw_industry1", sort=True).sum()
    df_group_mean=df_date.groupby(by="sw_industry1", sort=True).mean()
    df_group_sum=df_group_sum.sort_values(by="total_mv",ascending=False)

    #cheat size by make the last n smallest industry bigger. Otherwise too small to display
    last=15
    for i in range(1,last):
        df_group_sum["total_mv"].iloc[-i]=df_group_sum["total_mv"].iat[-last]

    sizes = df_group_sum["total_mv"]
    label=[]
    for name,pct_chg in zip(df_group_sum.index,df_group_mean["pct_chg"]):
        if pct_chg>0:
            label+=[f"{name}\n+{round(pct_chg,2)}%"]
        else:
            label+=[f"{name}\n{round(pct_chg,2)}%"]

    """
    use gradient color map
    cmap = matplotlib.cm.Wistia
    mini =df_group_mean["pct_chg"].min()
    maxi =df_group_mean["pct_chg"].max()
    norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
    colors = [cmap(norm(value)) for value in df_group_mean["pct_chg"]]
    """

    colors= []
    for name, pct_chg in zip(df_group_sum.index, df_group_mean["pct_chg"]):
        if pct_chg==0:
            colors+=[gray]

        elif pct_chg>0 and pct_chg<=1:
            colors += [treemapred1]
        elif pct_chg>1 and pct_chg<=2:
            colors += [treemapred2]
        elif pct_chg > 2 and pct_chg <= 3:
            colors += [treemapred3]
        elif pct_chg > 3:
            colors += [treemapred4]

        elif pct_chg>=-1 and pct_chg<0:
            colors += [treemapgreen1]
        elif pct_chg>=-2 and pct_chg<-1:
            colors += [treemapgreen2]
        elif pct_chg>=-3 and pct_chg<-2:
            colors += [treemapgreen3]
        elif pct_chg<-3 :
            colors += [treemapgreen4]

    squarify.plot(sizes=sizes, label=label, color=colors, alpha=1,text_kwargs={'fontsize':9, 'fontname':"Microsoft Yahei"},bar_kwargs=dict(linewidth=1, edgecolor="#222222"))
    fig = plt.gcf()
    fig.set_size_inches(6,6)
    plt.axis('off')

    path = f'Plot/report_D/{trade_date}/tree.png'
    img_saver(path=path,dpi=390)

    chart = Image.open(path)
    cw,cw=chart.size
    section.paste(chart, (int((width-cw)/2), 300 ), mask=chart)
    return section


def create_infographic(trade_date=LB.latest_trade_date()):

    # init
    df_date=DB.get_date(trade_date=trade_date)
    if df_date.empty:
        raise  AssertionError
    df_date = df_date[df_date["pct_chg"].notna()]

    #add sections
    a_func=[section_index,section_stock_gain,section_industry1,section_nsmoney,section_ma,section_rsi]
    a_sections = [func(trade_date=trade_date, df_date=df_date) for func in a_func]

    #put all sections together
    offset = 300
    autoheight=builtins.sum([offset]+[section.size[1] for section in a_sections])
    infographic = Image.new('RGBA', (width, autoheight), "#999999")
    y_helper=0
    for section in a_sections:
        infographic.paste(section,( 0, offset+y_helper),mask=section)
        y_helper += section.size[1]

    #draw title, later to be moved to anoter section
    title=f"A股盘后总结 {trade_date[0:4]}.{trade_date[4:6]}.{trade_date[6:8]}. "
    draw_text(section=infographic, text=title, fontsize=h1, left=False)

    #show and save
    infographic.show()
    infographic.save(f'Plot/report_D/{trade_date}/report.png')



if __name__ == '__main__':
    trade_date = "20200115"
    create_infographic(trade_date=trade_date)

