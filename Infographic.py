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
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import squarify
from scipy.stats import gmean


from PIL import Image, ImageDraw, ImageFont

# init
width = 2000
height = int(width * 0.62)
backupfontpath = r'c:\windows\fonts\msyh.ttc'
mega1 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width * 0.15))
mega2 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width * 0.11))
mega3 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width * 0.07))
mega4 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width * 0.06))
h1 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width * 0.05))
h2 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width * 0.04))
h3 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width * 0.03))
text = ImageFont.truetype(r'c:\windows\fonts\msyh.ttc', size=int(width * 0.02))


red="#FF5959"
green="#05CA00"
white='#ffffff'
gray="#333333"
yellow='#F6B900'

treemapred1="#e53935"
treemapred2="#d32f2f"
treemapred3="#c62828"
treemapred4="#b71c1c"

treemapgreen1="#43A047"
treemapgreen2="#388E3C"
treemapgreen3="#2E7D32"
treemapgreen4="#1B5E20"

title_padding=(100,100)
matplotlib.rc('font', family='Microsoft Yahei')
#https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html


def add_corners(im, rad):
    circle = Image.new('L', (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2, rad * 2), fill=255)
    alpha = Image.new('L', im.size, 255)
    w, h = im.size
    alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
    alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
    im.putalpha(alpha)
    return im

def img_saver(path,dpi=130):
    try:
        plt.savefig(path, transparent=True, dpi=dpi ,bbox_inches="tight", pad_inches=0)
    except:
        folder = "/".join(path.rsplit("/")[:-1])
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(path, transparent=True, dpi=dpi ,bbox_inches="tight", pad_inches=0)
    plt.clf()
    return path

def draw_text(section, text, fontsize=h1, left=True,fill="#ffffff",yoffset=0):
    idraw = ImageDraw.Draw(section)
    if left:
        idraw.text((title_padding[0],title_padding[1]+yoffset), text, font=fontsize,fill=fill)
    else:
        w, h = idraw.textsize(text, font=fontsize)
        idraw.text((int((width - w) / 2), title_padding[1]+yoffset), text, font=fontsize,fill=fill)
    return idraw


def get_bar2(s,path,y_limit=[-300, 250]):

    #y axis transform
    y = []
    x=[]
    for index,item in s.iteritems():
        if int(index)<0:
            if int(index)==-999:
                x += [f"历史新低"]
            else:
                x+=[f"{int(abs(index))}日新低"]
            y+=[item]
        elif int(index)==0:
            continue
        elif int(index)>0:
            if int(index)==999:
                x += [f"历史新高"]
            else:
                x += [f"{int(index)}日新高"]
            y += [item]

    line = plt.bar(x, y)


    # draw values as text
    for enum, i in enumerate(range(len(y))):
        # OVER each bar
        plt.annotate(f"{y[i]}家", xy=(x[i], y[i]+15), ha='center', va='bottom', color="white", size=12)

        # UNDER each bar
        plt.annotate(x[i][:-2], xy=(x[i], -60), ha='center', va='bottom', color="white", size=12)

    plt.annotate("新低", xy=(1.5, -200), ha='center', va='bottom', color="white", size=40)
    plt.annotate("新高", xy=(5.5, -200), ha='center', va='bottom', color="white", size=40)

    # use this to draw histogram of your data
    # scale down the chart by making the y axis seem taller
    axes = plt.gca()
    axes.set_ylim(y_limit)
    plt.axis('off')

    # add color
    for i in range(len(x)):
        if i >= len(x)/2:
            line[i].set_color(red)
        else:
            line[i].set_color(green)

    # save png
    return img_saver(path=path,dpi=390)


def get_bar(s,path,y_limit=[-300, 2300]):

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
        plt.annotate(str(y[i]), xy=(x[i], y[i] + 30), ha='center', va='bottom', color="white", size=12)

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
            line[i].set_color(green)
        elif i > int(len(y) / 2):
            line[i].set_color(red)
    line[int(len(y) / 2)].set_color(f'#ffffff')

    # save png
    return img_saver(path=path,dpi=390)


def section_title(trade_date, df_date,bcolor):
    # add init
    section = Image.new('RGBA', (width, 650), bcolor)
    path = f'Plot/static/title.png'
    chart = Image.open(path)
    chart = chart.resize((1700, 350))
    cw, ch = chart.size
    yoffset=120
    section.paste(chart, (int((width - cw) / 2) - 25, yoffset), mask=chart)

    # draw title, later to be moved to anoter section
    title = f"今日收盘 {trade_date[0:4]}.{trade_date[4:6]}.{trade_date[6:8]}. "
    draw_text(section=section, text=title, fontsize=h1, left=False, fill="#343434", yoffset=yoffset-20)

    title = f"来自于 Pollipoll.com"
    draw_text(section=section, text=title, fontsize=h2, left=False, fill="#ffffff",yoffset=yoffset+230)

    return section


def section_end(trade_date, df_date,bcolor):
    # add init
    section = Image.new('RGBA', (width, 1000), bcolor)

    title = f"更多数据就在 Pollipoll.com"
    draw_text(section=section, text=title, fontsize=h1, left=False, fill="#ffffff")

    path = f'Plot/static/title.png'
    chart = Image.open(path)
    chart = chart.resize((1700, 350))
    yoffset=400
    cw, ch = chart.size
    section.paste(chart, (int((width - cw) / 2) - 25, yoffset+20), mask=chart)
    title = f"今日收盘 {trade_date[0:4]}.{trade_date[4:6]}.{trade_date[6:8]}. "
    draw_text(section=section, text=title, fontsize=h1, left=False, fill="#343434",yoffset=yoffset)

    return section


def section_pe(trade_date, df_date, bcolor):
    # add init
    section = Image.new('RGBA', (width, 6800), bcolor)
    idraw = draw_text(section, f"申万一级行业市盈率")
    countfrom=20050101

    df_sort=pd.DataFrame()
    for industry in df_date["sw_industry1"].unique():
        if industry is None:
            continue
        df_asset_g=DB.get_asset(ts_code=f"sw_industry1_{industry}",freq="D",asset="G")
        df_asset_g=df_asset_g[(df_asset_g.index>=countfrom)& (df_asset_g.index<= int(trade_date))]
        df_sort.at[industry,"min"]=df_asset_g["pe_ttm"].min()
        df_sort.at[industry,"max"]=df_asset_g["pe_ttm"].max()
        df_sort.at[industry,"cur"]=df_asset_g["pe_ttm"].iat[-1]
        cur2=df_asset_g["pe_ttm"].rank(pct=True)
        df_sort.at[industry,"cur2"]=cur2.iat[-1]

    #add asset_E as industry
    for index in ["创业板","主板","中小板"]:
        df_asset_g = DB.get_asset(ts_code=f"market_{index}", freq="D", asset="G")
        df_asset_g=df_asset_g[(df_asset_g.index>=countfrom)& (df_asset_g.index<= int(trade_date))]
        df_sort.at[f"{index}","min"]=df_asset_g["pe_ttm"].min()
        df_sort.at[f"{index}","max"]=df_asset_g["pe_ttm"].max()
        df_sort.at[f"{index}","cur"]=df_asset_g["pe_ttm"].iat[-1]
        cur2 = df_asset_g["pe_ttm"].rank(pct=True)
        df_sort.at[f"{index}", "cur2"] = cur2.iat[-1]


    df_sort["norm"]= (((1 - 0) * (df_sort["cur"] - df_sort["min"])) / (df_sort["max"] - df_sort["min"])) + 0
    df_sort["norm"] =df_sort["norm"]*100
    df_sort["cur2"]=df_sort["cur2"]*100
    df_sort=df_sort.sort_values(by="cur2",ascending=True)
    df_sort["fakex"]=100

    # display data
    x_fake  = [x for x in df_sort["fakex"]]
    x       = [x for x in df_sort["cur2"]]
    y       = [x for x in df_sort.index]

    # fake
    line = plt.barh(y, x_fake)  # background
    for i in range(0, len(line)):
        line[i].set_color("#bbbbbb")

    # real
    lines = plt.barh(y, x)
    for i in range(0, len(lines)):
        if df_sort.index[i] in ["创业板","主板","中小板"]:
            lines[i].set_color(yellow)
        else:
            lines[i].set_color(white)

    ax = plt.gca()
    ax.set_xlim([-70, 130])
    ax.set_ylim([-1, len(x_fake)+3])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.yticks(color="#ffffff")
    manu=-0.47


    #draw label
    plt.annotate("历史最低PE", xy=(-10, len(x) ), ha='center', va='bottom', color="white", size=12)
    plt.annotate("今日PE", xy=(50, len(x) ), ha='center', va='bottom', color="white", size=12)
    plt.annotate("历史最高PE", xy=(105, len(x)), ha='center', va='bottom', color="white", size=12)

    # draw values as text
    for enum, i in enumerate(range(len(x))):
        # LEFT LABEL

        plt.annotate(df_sort.index[i], xy=(-70, -0.42+i*1), ha='left', va='bottom', color=white, size=12)

        # LEFT min
        plt.annotate(f"{int(df_sort['min'].iat[i])}", xy=(-10, manu + i * 1), ha='center', va='bottom', color="white", size=12)

        # RIGHT max
        plt.annotate(f"{int(df_sort['max'].iat[i])}", xy=(102, manu + i * 1), ha='left', va='bottom', color="white", size=12)

        # CENTER curr
        plt.annotate(f"{int(df_sort['cur'].iat[i])}", xy=(int(df_sort['cur2'].iat[i]+5), manu + i * 1), ha='left', va='bottom', color="white", size=12)


    fig = plt.gcf()
    fig.set_size_inches(4, 11)

    # use this to draw histogram of your data
    path = f'Plot/report_D/{trade_date}/pe_ttm.png'
    img_saver(path=path, dpi=560)
    chart = Image.open(path)
    cw, ch = chart.size
    section.paste(chart, (int((width - cw) / 2)+20, 200), mask=chart)


    #创业板chart
    df_asset_G = DB.get_asset(ts_code=f"market_创业板", freq="D", asset="G")
    df_asset_G = df_asset_G[(df_asset_G.index >= countfrom) ]
    LB.trade_date_to_vieable(df_asset_G)
    df_asset_G.index.name=""
    ax=df_asset_G["pe_ttm"].plot(color='#ffffff',linewidth=1)
    ax.set_ylim([df_asset_G["pe_ttm"].min(), df_asset_G["pe_ttm"].max()+20])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(white)
    ax.spines['bottom'].set_color(white)
    plt.yticks(color="#ffffff")
    plt.xticks(color="#ffffff")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45)
    ax.tick_params(axis='x', colors="#ffffff")
    ax.tick_params(axis='y', colors="#ffffff")
    plt.ylabel('市盈率 PE', color="white")
    plt.annotate('今天', xy=(len(df_asset_G), df_asset_G["pe_ttm"].iat[-1]+15), xytext=(len(df_asset_G), df_asset_G["pe_ttm"].iat[-1]+50),ha='center', color=white, arrowprops=dict(facecolor=yellow,color=yellow, shrink=0.05))
    plt.annotate('创业板历史市盈率', xy=(int(len(df_asset_G)/2), df_asset_G["pe_ttm"].max()+15),ha='center', color=white)


    fig.set_size_inches(4, 2)

    # use this to draw histogram of your data
    path = f'Plot/report_D/{trade_date}/cy.png'
    img_saver(path=path, dpi=470)
    chart = Image.open(path)
    cw2, ch2 = chart.size
    section.paste(chart, (int((width - cw) / 2) + 20, ch+200+100), mask=chart)



    return section


def section_abs_pe_backup(trade_date, df_date,bcolor):

    # add init
    section = Image.new('RGBA', (width, 2300), bcolor)
    idraw = draw_text(section, f"申万一级行业市盈率")

    #treemap data
    df_date_filtered=df_date[df_date["pe_ttm"].between(0,500)]
    df_group_mean=df_date_filtered.groupby(by="sw_industry1", sort=True).mean()
    df_group_mean=df_group_mean.sort_values(by="pe_ttm",ascending=False)

    #cheat size by make the last n smallest industry bigger. Otherwise too small to display
    last=3
    df_group_mean["pe_ttm_size"]=df_group_mean["pe_ttm"]
    for i in range(1,last):
        df_group_mean["pe_ttm_size"].iloc[-i]=df_group_mean["pe_ttm_size"].iat[-last]

    sizes = df_group_mean["pe_ttm_size"]
    label=[]
    for name,pe_ttm in zip(df_group_mean.index,df_group_mean["pe_ttm"]):
            label+=[f"{name}\n{round(pe_ttm,1)}PE"]




    colors= []
    for name, pe_ttm in zip(df_group_mean.index, df_group_mean["pe_ttm"]):
        if pe_ttm==0:
            colors+=[gray]

        elif pe_ttm>0 and pe_ttm<=1:
            colors += [treemapred1]
        elif pe_ttm>1 and pe_ttm<=2:
            colors += [treemapred2]
        elif pe_ttm > 2 and pe_ttm <= 3:
            colors += [treemapred3]
        elif pe_ttm > 3:
            colors += [treemapred4]

        elif pe_ttm>=-1 and pe_ttm<0:
            colors += [treemapgreen1]
        elif pe_ttm>=-2 and pe_ttm<-1:
            colors += [treemapgreen2]
        elif pe_ttm>=-3 and pe_ttm<-2:
            colors += [treemapgreen3]
        elif pe_ttm<-3 :
            colors += [treemapgreen4]



    cmap = matplotlib.cm.Wistia
    mini =df_group_mean["pe_ttm"].min()
    maxi =df_group_mean["pe_ttm"].max()
    norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
    colors = [cmap(norm(value)) for value in df_group_mean["pe_ttm"]]


    squarify.plot(sizes=sizes, label=label, color=colors, alpha=1,text_kwargs={'fontsize':9, 'fontname':"Microsoft Yahei","color":"#ffffff"},bar_kwargs=dict(linewidth=1, edgecolor="#ffffff"))
    fig = plt.gcf()
    fig.set_size_inches(6,6)
    plt.axis('off')
    path = f'Plot/report_D/{trade_date}/pe_ttm.png'
    img_saver(path=path,dpi=400)
    chart = Image.open(path)
    cw,ch=chart.size
    offset=350
    cw,cw=chart.size

    section.paste(chart, (int((width-cw)/2)-10, offset), mask=chart)
    idraw.text((title_padding[0], ch + offset), "市盈率越大面积越大", font=h3, fill="#ffffff")

    return section

def section_divergence(trade_date, df_date, bcolor):
    # add init
    section = Image.new('RGBA', (width, 2700), bcolor)
    idraw = draw_text(section, f"市场分化特征")

    s_pct_chg=df_date["pct_chg"].sort_values(ascending=True)
    top20=int(len(s_pct_chg)*0.2)
    s_top=s_pct_chg.nlargest(top20)
    s_bot=s_pct_chg.nsmallest(top20)

    top_gain=s_top.mean()
    s_bot=s_bot.mean()
    today_divergence=top_gain-s_bot

    """
        NOTE the axis is hard coded for now.
        I found out that 4=min, 6=max is a good scale that most of the time remains reasonable
        """

    minium=4
    maximium=8.1
    if today_divergence <=minium:
        today_divergence=minium
    elif today_divergence>=maximium:
        today_divergence=maximium

    #conver the 4-6 scale to 0-1 scale
    today_divergence=(( (today_divergence- minium)) / (maximium - minium))

    # display data
    x_fake = [x for x in [1]]
    x = [x for x in [today_divergence]]
    y = [x for x in ["egal"]]

    # fake
    line = plt.barh(y, x_fake)  # background
    for i in range(0, len(line)):
        line[i].set_color("#bbbbbb")

    # real
    lines = plt.barh(y, x)
    for i in range(0, len(lines)):
        lines[i].set_color(white)

    ax = plt.gca()
    ax.set_xlim([-1, 2])
    ax.set_ylim([0, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.annotate("小", xy=(-0.15, 0), ha='center', va='bottom', color="white", size=28)
    plt.annotate("大", xy=(1.15, 0), ha='center', va='bottom', color="white", size=28)
    plt.yticks(color="#ffffff")
    fig = plt.gcf()
    fig.set_size_inches(14, 2)

    # draw arrow
    path = f'Plot/static/arrow.png'
    chart = Image.open(path)
    chart = chart.resize((150, 150))
    cw, ch = chart.size
    hnorm = 106
    diff=today_divergence - 0.5
    distance=diff*hnorm*10

    section.paste(chart, (int(((width - cw) / 2)+distance), 500), mask=chart)

    if today_divergence<=0.2:
        summary_text="分化极小"
    elif today_divergence>0.2 and today_divergence<=0.35:
        summary_text="分化偏小"
    elif today_divergence>0.35 and today_divergence<=0.65:
        summary_text="分化中等"
    elif today_divergence>0.65 and today_divergence<=0.8:
        summary_text="分化偏大"
    elif today_divergence>0.8:
        summary_text="分化极大"

    w, h = idraw.textsize(summary_text, font=mega3)
    idraw.text((int(((width - w) / 2)+distance), title_padding[1] + 500-300), summary_text, font=mega3, fill=white)

    # draw label
    path = f'Plot/report_D/{trade_date}/divergence.png'
    img_saver(path=path, dpi=300)
    chart = Image.open(path)
    cw, ch = chart.size
    section.paste(chart, (int((width-cw)/2), 400), mask=chart)



    #gain by groups data
    a_cont_cols=["turnover_rate","vol","amount","total_mv","close","pb","pe_ttm"]+[f"pgain{x}" for x in [5,20,60,240]]


    df_group_result=pd.DataFrame()
    for group in a_cont_cols:
        d_quantile = LB.custom_quantile(df=df_date,p_setting=[0,0.1,0.9,1],key_val=False,column=group)
        for key, df_quantile in d_quantile.items():
            if key==f"0,0.1":
                df_group_result.at[f"{group}_top","gain"]=df_quantile["pct_chg"].mean()
            elif key == f"0.9,1":
                df_group_result.at[f"{group}_bot", "gain"] = df_quantile["pct_chg"].mean()

    if False:
        a_disc_cols = ["sw_industry2"]
        for col in a_disc_cols:
            for instance in df_date[col].unique():
                df_date_group=df_date[df_date[col]==instance]
                if len(df_date_group)<=8:#disregard very small groups
                    df_group_result.at[f"{col}_{instance}","gain"] =np.nan
                else:
                    df_group_result.at[f"{col}_{instance}","gain"]=df_date_group["pct_chg"].mean()

    # display data
    displayn=5
    df_group_result=df_group_result.sort_values("gain",ascending=True)

    xpos = [1 for x in range(displayn)]
    xneg = [-1 for x in range(displayn)]
    y = [x for x in range(displayn)]

    #negative
    line = plt.barh(y, xpos)  # background
    for i in range(0, len(line)):
        line[i].set_color(red)

    # positive
    lines = plt.barh(y, xneg)
    for i in range(0, len(lines)):
        lines[i].set_color(green)

    ax = plt.gca()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-1, len(y)])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    for posneg in [1,-1]:
        for i,(index,gain) in enumerate(zip(df_group_result.index[::posneg],df_group_result["gain"][::posneg])):
            if i>=displayn:
                continue

            array=LB.col_translate()[index[:-4]]
            chinesename=array[0]
            looup= LB.col_gd() if array[1] == True else LB.col_gd()
            value=looup[0] if index[-3:]=="top" else looup[1]


            plt.annotate(f"{chinesename}{value}{gain}", xy=(0.5*posneg*(-1), displayn - i - 1), ha='center', va='bottom', color="white", size=18)


    plt.yticks(color="#ffffff")
    fig = plt.gcf()
    fig.set_size_inches(15, 8)

    path = f'Plot/report_D/{trade_date}/zhuli.png'
    img_saver(path=path, dpi=280)
    chart = Image.open(path)
    cw2, ch2 = chart.size
    section.paste(chart, (int((width - cw2) / 2), 400+ch+0), mask=chart)

    return section

def section_stock_gain(trade_date, df_date,bcolor):
    # add init
    section = Image.new('RGBA', (width, 2700), bcolor)
    idraw = draw_text(section, f"涨跌分布")

    #add average gain
    avg_gain=df_date["pct_chg"].mean()
    avg_gain = round(avg_gain, 2)
    avg_gain= f"平均涨跌\n+{avg_gain}%" if avg_gain>0 else f"平均涨跌\n+{avg_gain}%"
    idraw.text((title_padding[0], 450), avg_gain, font=mega4, fill="white")

    zd_ratio=len(df_date[df_date["pct_chg"]>9.5])/len(df_date[df_date["pct_chg"]<-9.5])
    zd_ratio=int(zd_ratio)
    idraw.text((1400, 450), f"涨跌停比\n≈{zd_ratio}:1", font=mega4, fill="white",align="right")


    # add histogram
    histo = Image.open(get_bar(s=df_date["pct_chg"],path = f'Plot/report_D/{trade_date}/histo.png'))
    hw, hh = histo.size

    #paste all stuff into the section
    offsetfrom_hiddenaxis=0
    histooffset=0
    section.paste(histo, (int((width - hw) / 2)-offsetfrom_hiddenaxis, -histooffset), mask=histo)

    #涨跌幅统计
    die = len(df_date[df_date["pct_chg"] < 0])
    zhang = len(df_date[df_date["pct_chg"]>0])
    diepct = round(die / len(df_date) * 100, 0)
    zhangpct = round(zhang / len(df_date) * 100, 0)
    offset = -histooffset+200
    for i,tuple in enumerate([["跌",die,diepct,-1],["涨",zhang,zhangpct,1]]):
        abs_count = f"{int(tuple[1])}家"
        pct_count=f"{int(tuple[2])}%"
        text = tuple[0]
        color= red if i==0 else green
        color=white
        norm=tuple[3]
        hdistance=400

        # load image and put it as background
        bpath = f'Plot/static/greenarrow.png' if i ==0 else f'Plot/static/redarrow.png'
        bground = Image.open(bpath)
        bground = bground.resize((600, 800))
        w, height1 = bground.size
        if text=="涨":
            minus_height=-80
        else:
            minus_height = 0
        section.paste(bground, (int(((width - w) / 2)) + norm * hdistance, hh+offset+minus_height-60), mask=bground)

        #pct of stocks 涨跌
        w, height1 = idraw.textsize(f"{pct_count}", font=mega3)
        idraw.text(( int(((width - w) / 2)) + norm*hdistance, hh+offset+0), f"{pct_count}", font=mega3, fill=color)

        # abs of stocks 涨跌
        w, height2 = idraw.textsize(abs_count, font=h2)
        idraw.text((int(((width - w) / 2)) + norm*hdistance, hh + height1 + offset + 0), abs_count, font=h2, fill=color)

        # text 涨跌
        w, height3 = idraw.textsize(text, font=mega2)
        idraw.text( (  int(((width - w) / 2)) + norm*hdistance, hh +height1 + height2+ offset+50), text, font=mega2, fill=color)

    return section

def section_index(trade_date, df_date,bcolor):

    # add init
    section = Image.new('RGBA', (width, 2400), bcolor)
    idraw = draw_text(section, f"三指涨幅")

    #calculate data for 3 index
    a_pct_chg=[]
    vertical_offset = 400
    horizontal_offset = 650

    for ts_index,name,i in zip(["000001.SH","399001.SZ","399006.SZ"],["上证指数","深成指数","创业板指"],[-1,0,1]):

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
        pct_chg_text=f"+{round(pct_chg,2)}%" if pct_chg>=0 else f"{round(pct_chg,2)}%"
        a_pct_chg+=[pct_chg]

        if pct_chg> 0 :
            color = red
            bpath = f'Plot/static/red.png'
        elif pct_chg==0:
            color = white
            bpath = f'Plot/static/gray.png'
        elif pct_chg < 0:
            color = green
            bpath = f'Plot/static/green.png'

        color=white

        #load image and put it as background
        bground = Image.open(bpath)
        bground = bground.resize((600, 850))
        bground =add_corners(bground, 70)
        w, height1 = bground.size
        section.paste(bground, (int(((width - w) / 2))+ i * horizontal_offset,  350),mask=bground)

        # display 标题
        w, height1 = idraw.textsize(f"{name}", font=h2)
        idraw.text((int(((width - w) / 2)) + i * horizontal_offset, vertical_offset + 0), name, font=h2, fill=white)

        # display close
        w, height2 = idraw.textsize(f"{close}", font=h1)
        idraw.text((int(((width - w) / 2)) + i * horizontal_offset, height1 + vertical_offset + 0), f"{close}", font=h1, fill=white)

        # display pct_chg 涨跌
        w, height3 = idraw.textsize(pct_chg_text, font=mega3)
        idraw.text((int(((width - w) / 2)) + i * horizontal_offset, height1 + height2 + vertical_offset + 40), pct_chg_text, font=mega3, fill=color)

        # display mini chart
        section.paste(chart, (int(((width - w) / 2)) + 70 + i * horizontal_offset, height1 + height2 + height3 + vertical_offset + 60), mask=chart)



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


    #deterine the bull or bear image depending on todays market
    pos_index=0
    for pct_chg in a_pct_chg:
        if pct_chg>=0:
            pos_index+=1

    if pos_index in [1,2]:
        a_image = ["bear", "bull"]
    elif pos_index in [3]:
        a_image = ["bull", "bull"]
    elif pos_index in [0]:
        a_image = ["bear", "bear"]

    for bb,lr in zip(a_image,["l","r"]):
        bb_image = Image.open(f"Plot/static/{bb}{lr}.png")
        bb_image = bb_image.resize((570, 570))
        w, height1 = bb_image.size
        if lr == "l":
            section.paste(bb_image, (0, height1 + height2 + height3 +height4 + vertical_offset -50), mask=bb_image)
        elif lr=="r":
            section.paste(bb_image, (width-w, height1 + height2 + height3 +height4 +vertical_offset -50), mask=bb_image)

    return section


def section_alltimehighlow(trade_date, df_date,bcolor):

    # add init
    section = Image.new('RGBA', (width, 3500), bcolor)
    idraw = draw_text(section, f"历史新高低价")

    df_date["isminmax"]=df_date["isminmax"].replace(np.nan,0)
    df_date_groupby=df_date.groupby("isminmax").count()

    # use this to draw histogram of your data
    path = f'Plot/report_D/{trade_date}/allimeghighlow.png'
    chart = Image.open(get_bar2(s=df_date_groupby["pct_chg"],path=path))
    cw,ch=chart.size
    section.paste(chart, (int((width-cw)/2), 300), mask=chart)
    return section


def section_ma(trade_date, df_date,bcolor):

    # add init
    section = Image.new('RGBA', (width, 1800), bcolor)
    idraw = draw_text(section, f"均线统计")

    #calculate data
    a_y_abs=[]
    a_colors=[treemapred1,treemapred2,treemapred3,treemapred4]
    for enum,freqn in enumerate([5,20,60,240]):
        y_abs=len(df_date[df_date["close"]>=df_date[f"ma{freqn}"]])
        a_y_abs += [y_abs]
        pct=y_abs/len(df_date)

        yes_name=f"{freqn}日 均线上"
        no_name=f"{freqn}日 均线下"
        yes=int(pct*100)
        no=100-yes
        s=pd.Series()
        s.at[yes_name]=yes
        s.at[no_name]=no
        ax=s.plot(kind='pie',labels=["",""],colors=[white,a_colors[enum]],startangle=90,wedgeprops={"edgecolor":"white",'linewidth': 1})
        plt.axis('off')



        radius=200
        space=600
        offset=700
        path = f'Plot/report_D/{trade_date}/ma{enum}.png'
        img_saver(path=path, dpi=900-enum*radius)
        chart = Image.open(path)

        cw, ch = chart.size
        section.paste(chart, (int((width - cw) / 2), int((space-ch)/2)+offset), mask=chart)


    if False:
        #display data
        x=[x for x in d_ma.keys()]
        y=[x for x in d_ma.values()]
        y_fake=[100 for x in d_ma.values()]

        line=plt.bar(x, y_fake)#background
        for i in range(0,len(line)):
            line[i].set_color(yellow)

        line=plt.bar(x, y)
        for i in range(0,len(line)):
            line[i].set_color(green)
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


def section_rsi(trade_date, df_date,bcolor):

    # add init
    section = Image.new('RGBA', (width, 1500), bcolor)
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

    #fake
    line=plt.bar(x, y_fake)  # background
    for i in range(0,len(line)):
        line[i].set_color(yellow)

    #real
    line=plt.bar(x, y)
    for i in range(0,len(line)):
        line[i].set_color(green)
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



def section_nsmoney(trade_date, df_date,bcolor):

    # add init
    section = Image.new('RGBA', (width, 2300), bcolor)
    idraw = draw_text(section, f"南北资金")

    #calculate data
    df_nsmoney=DB.get(a_path=LB.a_path("Market/CN/Asset/E/hsgt/hsgt"))
    df_nsmoney=df_nsmoney[df_nsmoney.index <= int(trade_date)]
    df_nsmoney=df_nsmoney.tail(120)#only check last 120 days as half year data
    df_nsmoney["zero"]=0
    df_nsmoney.index = df_nsmoney.index.astype(str)
    df_nsmoney["date"]=df_nsmoney.index
    df_nsmoney["date"]=df_nsmoney["date"].str.slice(4,6) + f"月"+df_nsmoney["date"].str.slice(6,8) +f"日"
    df_nsmoney.index = df_nsmoney["date"]
    df_nsmoney.index.name =""



    #display data
    for i, (name,nb) in enumerate(zip(["north","south"],["北","南"])):

        use_yi = True
        if use_yi:
            df_nsmoney[f"{name}_money"] = df_nsmoney[f"{name}_money"] / 100
            plt.ylabel('亿元', color="white")
        else:
            plt.ylabel('百万元', color="white")

        ax=df_nsmoney[f"{name}_money"].plot(color='#ffffff', legend=False, linewidth=2)
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

        label = f"{nb}向资金"
        plt.annotate(label , xy=(120,df_nsmoney[f"{name}_money"].max()) , ha='center', va='bottom', color="white", size=12)

        fig=plt.gcf()
        fig.set_size_inches(6, 2)

        path = f'Plot/report_D/{trade_date}/{name}.png'
        img_saver(path=path, dpi=350)
        chart = Image.open(path)
        cw,ch=chart.size
        section.paste(chart, (int((width-cw)/2), 350+i*1000), mask=chart)

    return section

def section_industry1_gain(trade_date, df_date, bcolor):

    # add init
    section = Image.new('RGBA', (width, 2500), bcolor)
    idraw = draw_text(section, f"申万一级行业热力图")
    d_quantile=LB.custom_quantile(df=df_date,column="pct_chg",key_val=False,p_setting=[0,1])

    for enum, (key,df_date_cut) in enumerate(d_quantile.items()):
        #treemap data

        df_group_sum=df_date_cut.groupby(by="sw_industry1", sort=True).sum()
        df_group_mean=df_date_cut.groupby(by="sw_industry1", sort=True).mean()
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


        squarify.plot(sizes=sizes, label=label, color=colors, alpha=1,text_kwargs={'fontsize':9, 'fontname':"Microsoft Yahei","color":"#ffffff"},bar_kwargs=dict(linewidth=1, edgecolor="#ffffff"))
        fig = plt.gcf()
        fig.set_size_inches(6,6)
        plt.axis('off')
        path = f'Plot/report_D/{trade_date}/tree{enum}.png'
        print(path)
        img_saver(path=path,dpi=400)
        chart = Image.open(path)
        cw,ch=chart.size
        offset=350
        section.paste(chart, (int((width-cw)/2)-10, offset+enum*ch), mask=chart)

        idraw.text((title_padding[0], ch + offset), "市值越大面积越大", font=h3, fill="#ffffff")

    return section


def create_infographic(trade_date=LB.latest_trade_date()):

    # init
    print("start")
    df_date=DB.get_date(trade_date=trade_date)
    if df_date.empty:
        raise  AssertionError
    df_date = df_date[df_date["pct_chg"].notna()]

    #add sections
    a_func=[section_title, section_index, section_stock_gain,section_divergence, section_industry1_gain,section_pe, section_nsmoney, section_ma,section_alltimehighlow, section_rsi, section_end]
    a_bcolor=["#1D37FF","#1D37FF","#1D37FF","#1D37FF","#1D37FF","#1D37FF","#1D37FF","#1D37FF","#1D37FF","#1D37FF","#1D37FF"]
    a_bcolor=[None]*len(a_func)
    a_sections = [func(trade_date=trade_date, df_date=df_date,bcolor=bcolor) for func,bcolor in zip(a_func,a_bcolor)]

    #put all sections together
    autoheight=builtins.sum([section.size[1] for section in a_sections])
    infographic = Image.new('RGBA', (width, autoheight), "#1D37FF")

    # add background
    path = f'Plot/static/gradient.png'
    chart = Image.open(path)
    chart = chart.resize((2200, autoheight))
    infographic.paste(chart, (0, 0), )

    #add sections
    y_helper=0
    for section in a_sections:
        infographic.paste(section,( 0, y_helper),mask=section)
        y_helper += section.size[1]



    #show and save
    infographic.show()
    infographic.save(f'Plot/report_D/{trade_date}/report_{trade_date}.png')



if __name__ == '__main__':

    if False:
        df_trade_date=DB.get_trade_date()
        df_result=pd.DataFrame()
        for trade_date in df_trade_date.index:
            if trade_date>=20050101:
                print("yes",trade_date)
                df_date=DB.get_date(trade_date=trade_date,a_asset=["E"])
                df_date["pct_chg"]=df_date["pct_chg"].clip(-10,10)
                s_divergence=df_date["pct_chg"].sort_values(ascending=False)

                df_result.at[trade_date,"skew"]=s_divergence.skew()
                df_result.at[trade_date,"std"]=s_divergence.std()
                df_result.at[trade_date,"kurt"]=s_divergence.kurt()

                all_gain=s_divergence.mean()
                for top in [0.05,0.1,0.2,0.3,0.4]:
                    top20 = int(len(s_divergence) * top)
                    bot20 = int(len(s_divergence) * top)
                    s_divergence_top = s_divergence.nlargest(top20)
                    s_divergence_bot = s_divergence.nsmallest(bot20)
                    top_gain = s_divergence_top.mean()
                    bot_gain = s_divergence_bot.mean()
                    df_result.at[trade_date,f"top{top}"]=top_gain-bot_gain

                #arith - geomean
                arith=df_date["pct_chg"].mean()
                geo=gmean(df_date["pct_chg"])
                df_result.at[trade_date, "arithgeo"]=arith-geo


                market_mean=df_date["pct_chg"].mean()
                df_result.at[trade_date, "marketMean"] = len(df_date[df_date["pct_chg"]>market_mean]) / len(df_date)


        egal=DB.get_asset(ts_code="000001.SH",asset="I")
        df_result["sh"]=egal["close"]
        df_result.to_csv("test.csv")

    if True:
        trade_date = fr"20210118"
        create_infographic(trade_date=trade_date)

