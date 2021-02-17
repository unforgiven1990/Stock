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
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import matplotlib
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import squarify
from scipy.stats import gmean
import pyexiv2
import piexif
import gc
import requests
import json
import ast
import imageio

from PIL import Image, ImageDraw, ImageFont

# init
width = 2000
height = int(width * 0.62)
backupfontpath = r'c:\windows\fonts\msyh.ttc'
mega1 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width * 0.15))
mega2 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width * 0.11))
mega3 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width * 0.07))
h1 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width * 0.05))
h2 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width * 0.04))
h3 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width * 0.02))
text = ImageFont.truetype(r'c:\windows\fonts\msyh.ttc', size=int(width * 0.02))


red="#FF5959"
green="#05bf00"
white='#ffffff'
gray="#333333"
lightgray="#bbbbbb"

orange= '#F6B900'
yellow= '#e8eb34'
violet="#150e9c"

treemapred1="#e53935"
treemapred2="#d32f2f"
treemapred3="#c62828"
treemapred4="#b71c1c"

treemapgreen1="#43A047"
treemapgreen2="#388E3C"
treemapgreen3="#2E7D32"
treemapgreen4="#1B5E20"

a_line_plot_color=[orange, white]

title_padding=(100,100)
matplotlib.rc('font', family='Microsoft Yahei')

short_maxiium = 3
short_minimum = -short_maxiium
countfrom=20050101
#https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html



def axes_to_round(ax,mutation_aspect=300,mutation_scale=0.5):
    new_patches = []

    for i, patch in enumerate(reversed(ax.patches)):
        bb = patch.get_bbox()
        color = patch.get_facecolor()

        p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
                                abs(bb.width), abs(bb.height),
                                boxstyle="round",
                                ec="none", fc=color,
                                mutation_aspect=mutation_aspect,
                                mutation_scale=mutation_scale
                                )




        patch.remove()
        # line.patches[i]=p_bbox
        new_patches.append(p_bbox)


    for patch in reversed(new_patches):
        ax.add_patch(patch)


def add_corners(im, rad,a_corners=[True]*4):
    circle = Image.new('L', (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2, rad * 2), fill=255)
    alpha = Image.new('L', im.size, 255)
    w, h = im.size
    if a_corners[0]:
        alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    if a_corners[1]:
        alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
    if a_corners[2]:
        alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    if a_corners[3]:
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
    plt.close()
    plt.close('all')
    gc.collect()
    return path

def draw_text(section, text, fontsize=h1, left=True,fill="#ffffff",yoffset=0,background=False,align="left"):
    idraw = ImageDraw.Draw(section)
    if background:
        # background
        idraw = ImageDraw.Draw(section)
        w, h = idraw.textsize(text, font=fontsize)
        path = f'Plot/static/yellow.png'
        chart = Image.open(path)
        chart = chart.resize((w + title_padding[0] + 40, h + 40))
        chart = add_corners(chart, 40,a_corners=[False,False,True,True])
        section.paste(chart, (0, title_padding[1]-10), mask=chart)
        fill=gray

    if left:
        idraw.text((title_padding[0],title_padding[1]+yoffset), text, font=fontsize,fill=fill,align=align)
    else:
        w, h = idraw.textsize(text, font=fontsize)
        idraw.text((int((width - w) / 2), title_padding[1]+yoffset), text, font=fontsize,fill=fill,align=align)
    return idraw


def get_bar2(s,path,y_limit=[-600, 600]):

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



    #line = plt.bar(x, y)
    df_plot=pd.DataFrame(data=y,index=[i for i in range(len(x))])
    ax=df_plot.plot.bar(legend=False)

    axes_to_round(ax,mutation_aspect=80,mutation_scale=0.5)

    axes = plt.gca()
    print(s.max())
    axes.set_ylim(y_limit)
    plt.axis('off')

    fig = plt.gcf()
    fig.set_size_inches(6, 40)

    # draw values as text
    for enum, i in enumerate(range(len(y))):
        # OVER each bar
        plt.annotate(f"{y[i]}家", xy=(i, y[i]+20), ha='center', va='bottom', color="white", size=12)

        # UNDER each bar
        plt.annotate(x[i][:-2], xy=(i, -30), ha='center', va='top', color="white", size=12)


    #plt.annotate("新低", xy=(1.5, -120), ha='center', va='bottom', color="white", size=40)
    #plt.annotate("新高", xy=(5.5, -120), ha='center', va='bottom', color="white", size=40)

    # add color
    for i in range(len(x)):
        if i >= len(x)/2:
            ax.patches[i].set_color(red)
        else:
            ax.patches[i].set_color(green)

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
    #line = plt.bar(x, y)
    df=pd.DataFrame(data=y,index=x)
    ax=df.plot.bar(legend=False)


    # draw values as text
    for enum, i in enumerate(range(len(y))):
        # OVER each bar
        plt.annotate(str(y[i]), xy=(x[i], y[i] + 50), ha='center', va='bottom', color="white", size=12)

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

    #make corners round
    axes_to_round(ax)

    # add color
    for i in x:
        if i < 5:
            ax.patches[i].set_color(green)
        elif i > 5:
            ax.patches[i].set_color(red)
    ax.patches[5].set_color(f'#ffffff')

    # save png
    return img_saver(path=path,dpi=390)


def helper_social(section,yoffset):
    # website link
    if False:
        title = f"www.5849cai.com"
        draw_text(section=section, text=title, fontsize=h1, left=False, fill="#ffffff", yoffset=yoffset + 200, align="center")

        title = f"我不是韭菜"
        draw_text(section=section, text=title, fontsize=h2, left=False, fill="#ffffff", yoffset=yoffset + 330, align="center")
    else:
        path = f'Plot/static/5849cai.png'
        chart = Image.open(path)
        chart = chart.resize((800, 200))
        section.paste(chart, (600, yoffset + 350), mask=chart)

    # add hand pointer
    path = f'Plot/static/pointer.png'
    chart = Image.open(path)
    chart = chart.resize((140, 180))
    section.paste(chart, (1370, yoffset + 420), mask=chart)


def section_title(trade_date, df_date,bcolor,pos):
    # add init
    section = Image.new('RGBA', (width, 750), bcolor)
    path = f'Plot/static/title.png'
    chart = Image.open(path)
    chart = chart.resize((1700, 350))
    cw, ch = chart.size
    yoffset=120
    section.paste(chart, (int((width - cw) / 2) - 25, yoffset), mask=chart)

    # draw title, later to be moved to anoter section
    title = f"每日收盘 {trade_date[0:4]}.{trade_date[4:6]}.{trade_date[6:8]}. "
    draw_text(section=section, text=title, fontsize=h1, left=False, fill="#343434", yoffset=yoffset-20)

    helper_social(section=section,yoffset=yoffset)
    return section


def section_end(trade_date, df_date,bcolor,pos):
    # add init
    section = Image.new('RGBA', (width, 3600), bcolor)

    if False:
        title = f"www.5849cai.com"
        draw_text(section=section, text=title, fontsize=h1, left=False, fill="#ffffff",yoffset=200)
    else:
        helper_social(section=section, yoffset=700)

    path = f'Plot/static/title.png'
    chart = Image.open(path)
    chart = chart.resize((1700, 350))
    yoffset=700
    cw, ch = chart.size
    section.paste(chart, (int((width - cw) / 2) - 25, yoffset+20), mask=chart)
    title = f"每日收盘 {trade_date[0:4]}.{trade_date[4:6]}.{trade_date[6:8]}. "
    draw_text(section=section, text="投资有风险\n入市需谨慎", fontsize=mega2, left=False, fill="#ffffff",yoffset=yoffset-700)
    draw_text(section=section, text=title, fontsize=h1, left=False, fill="#343434", yoffset=yoffset)

    path = f'Plot/static/bottom.png'
    chart = Image.open(path)
    chart = chart.resize((2000, yoffset+1600))
    cw, ch = chart.size
    section.paste(chart, (int((width - cw) / 2), yoffset + 700), mask=chart)

    return section


def linechart_helper(section,idraw,name,trade_date,a_index,a_summary):

    try:
        df_ts_code=DB.get_ts_code(a_asset=["E","I","FD"])
        fig = plt.gcf()

        a_legend=[]
        a_minmax=[]
        a_pct_chg=[]
        def round_down(num, divisor):
            return num - (num % divisor)
        for i,index in enumerate(a_index):
            df_asset = DB.get_asset(ts_code=index, freq="D", asset="I")

            df_asset = df_asset[(df_asset.index >= int(trade_date)-10000)]
            df_asset = df_asset[(df_asset.index <= int(trade_date))]
            LB.trade_date_to_vieable(df_asset)


            df_asset.index.name = ""

            df_asset["close"]=df_asset["close"]/df_asset.at[df_asset["close"].first_valid_index(),"close"]
            df_asset["close"] =df_asset["close"]*100-100
            index_name=df_ts_code.at[index,"name"]
            df_asset[f"{index_name}"]=df_asset["close"]

            a_pct_chg+=[df_asset["close"].iat[-1]]

            ax = df_asset[index_name].plot(color=a_line_plot_color[i], linewidth=2.5,legend=True)
            a_minmax+=[round_down(df_asset[index_name].min(),10),df_asset[index_name].max()]

            label = f'{index_name}'
            a_legend+=[mpatches.Patch(color=a_line_plot_color[i], label=label)]


        #ax.set_ylim([df_asset["close"].min(), df_asset["close"].max() + 20])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(white)
        ax.spines['bottom'].set_color(white)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

        if (max(a_minmax)-min(a_minmax)) > 90 :
            step=20
        else:
            step=10
        plt.yticks(np.arange(min(a_minmax), max(a_minmax), step))

        plt.yticks(color="#ffffff")
        plt.xticks(color="#ffffff")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45,fontsize=8)
        ax.tick_params(axis='x', colors="#ffffff")
        ax.tick_params(axis='y', colors="#ffffff")
        #plt.annotate('今天', xy=(len(df_asset), df_asset["close"].iat[-1] + 15), xytext=(len(df_asset), df_asset["close"].iat[-1] + 50), ha='center', color=white, arrowprops=dict(facecolor=yellow, color=yellow, shrink=0.05))
        fig.set_size_inches(4, 2)

        leg = plt.legend(handles=a_legend,mode="expand",bbox_to_anchor = (0.5, 1.25),ncol=len(a_index))
        leg.get_frame().set_alpha(0)
        for text in leg.get_texts():
            text.set_color(white)

        #plt.fill_between(df_asset["close"].index, df_asset["close"], 0, color=yellow)

        #plot text which index is stronger
        if abs((a_pct_chg[0]-a_pct_chg[1]))< 5:
            text = f"{a_summary[0]} ≈ {a_summary[1]}"
        elif a_pct_chg[0]>a_pct_chg[1]:
            text=f"{a_summary[0]} > {a_summary[1]}"
        else:
            text = f"{a_summary[1]} > {a_summary[0]}"

        w, h = idraw.textsize(text, font=mega3)
        idraw.text((int(((width - w) / 2)), 1650), text, font=mega3, fill=white)

        # use this to draw histogram of your data
        path = f'Plot/report_D_material/{trade_date}/{name}.png'
        img_saver(path=path, dpi=500)
        plt.clf()

        return path
    except:
        return -1


def section_industry_pe(trade_date, df_date, bcolor, pos):
    # add init
    section = Image.new('RGBA', (width, 5000), bcolor)
    idraw = draw_text(section, f"{pos}. 行业市盈率",background=True)


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
        try:
            df_asset_g = DB.get_asset(ts_code=f"market_{index}", freq="D", asset="G")
            df_asset_g=df_asset_g[(df_asset_g.index>=countfrom)& (df_asset_g.index<= int(trade_date))]
            if df_asset_g.empty:
                continue

            df_sort.at[f"{index}","min"]=df_asset_g["pe_ttm"].min()
            df_sort.at[f"{index}","max"]=df_asset_g["pe_ttm"].max()
            df_sort.at[f"{index}","cur"]=df_asset_g["pe_ttm"].iat[-1]
            cur2 = df_asset_g["pe_ttm"].rank(pct=True)
            df_sort.at[f"{index}", "cur2"] = cur2.iat[-1]
        except:
            print(f"{index} probably doesnt exist yet")


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
        line[i].set_color(white)

    # real
    lines = plt.barh(y, x)
    for i in range(0, len(lines)):
        if df_sort.index[i] in ["创业板","主板","中小板"]:
            lines[i].set_color(red)
        else:
            lines[i].set_color(orange)

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
        try:
            # LEFT LABEL
            plt.annotate(df_sort.index[i], xy=(-70, -0.42+i*1), ha='left', va='bottom', color=white, size=12)

            # LEFT min
            plt.annotate(f"{int(df_sort['min'].iat[i])}", xy=(-10, manu + i * 1), ha='center', va='bottom', color=white, size=12)

            # RIGHT max
            plt.annotate(f"{int(df_sort['max'].iat[i])}", xy=(105, manu + i * 1), ha='left', va='bottom', color=white, size=12)

            # CENTER curr
            percent=int(df_sort['cur2'].iat[i])
            if percent>70:
                adjust_dist = -5
                ha="right"
                color=white
            else:
                adjust_dist = +5
                ha="left"
                color = gray
            plt.annotate(f"{int(df_sort['cur'].iat[i])}", xy=(int(df_sort['cur2'].iat[i]+adjust_dist), manu + i * 1), ha=ha, va='bottom', color=color, size=12)
        except:
            pass

    fig = plt.gcf()
    fig.set_size_inches(4, 11)

    # use this to draw histogram of your data
    path = f'Plot/report_D_material/{trade_date}/pe_ttm.png'
    img_saver(path=path, dpi=560)
    chart = Image.open(path)
    cw, ch = chart.size
    section.paste(chart, (int((width - cw) / 2)+20, 200), mask=chart)
    return section


def pe_helper(section,bcolor,index_G,pos):

    fig = plt.gcf()
    for index in index_G:
        try:
            df_asset_G = DB.get_asset(ts_code=f"market_{index}", freq="D", asset="G")
            df_asset_G = df_asset_G[(df_asset_G.index >= countfrom) ]
            df_asset_G = df_asset_G[(df_asset_G.index <= int(trade_date)) ]
            LB.trade_date_to_vieable(df_asset_G)
            df_asset_G.index.name=""
            ax=df_asset_G["pe_ttm"].plot(color=orange, linewidth=1)
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
            #plt.annotate('今天', xy=(len(df_asset_G), df_asset_G["pe_ttm"].iat[-1]+15), xytext=(len(df_asset_G), df_asset_G["pe_ttm"].iat[-1]+20), ha='center', color=white, arrowprops=dict(facecolor=orange, color=orange, shrink=0.05))
            #plt.annotate('创业板历史市盈率', xy=(int(len(df_asset_G)/2), df_asset_G["pe_ttm"].max()+15),ha='center', color=white)
            fig.set_size_inches(4, 2)

            #write summary text
            rank=df_asset_G["pe_ttm"].rank(pct=True)
            today_pe=rank.iat[-1]
            if today_pe > 0.8 and today_pe <= 1:
                text=f"{index}市盈率极高"
            elif today_pe > 0.6 and today_pe < 0.8:
                text=f"{index}市盈率偏高"
            elif today_pe > 0.4 and today_pe < 0.6:
                text=f"{index}市盈率中位"
            elif today_pe > 0.2 and today_pe < 0.4:
                text=f"{index}市盈率偏低"
            elif today_pe >= 0 and today_pe < 0.2:
                text=f"{index}市盈率极低"

            idraw = draw_text(section, f"{pos}. {index}市盈率", background=True)
            w, h = idraw.textsize(text, font=mega3)
            idraw.text((int((width - w) / 2), 1550), text, font=mega3, fill=white)

            #egal, (ax1) = plt.subplots(1, 1, sharex=True)
            plt.fill_between(df_asset_G["pe_ttm"].index, df_asset_G["pe_ttm"], 0, color=orange)


            # use this to draw histogram of your data
            path = f'Plot/report_D_material/{trade_date}/{index}.png'
            img_saver(path=path, dpi=470)
            chart = Image.open(path)
            cw2, ch2 = chart.size
            section.paste(chart, (int((width - cw2) / 2) + 20, 400), mask=chart)

            return section
        except Exception as e:
            plt.clf()
            print(f"{index} probably doesnt exist yet")
            print(e)
    else:
        return Image.new('RGBA', (width, 0), bcolor)



def section_sh_pe(trade_date, df_date, bcolor, pos):
    # add init
    section = Image.new('RGBA', (width, 1850), bcolor)
    return pe_helper(section=section, bcolor=bcolor,index_G=["主板"],pos=pos)


def section_cy_pe(trade_date, df_date, bcolor, pos):
    # add init
    section = Image.new('RGBA', (width, 1850), bcolor)
    return pe_helper(section=section,bcolor=bcolor,index_G=["创业板","中小板"],pos=pos)






def section_abs_pe_backup(trade_date, df_date,bcolor,pos):

    # add init
    section = Image.new('RGBA', (width, 2300), bcolor)
    idraw = draw_text(section, f"{pos}. 申万一级行业市盈率",background=True)

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
    path = f'Plot/report_D_material/{trade_date}/pe_ttm.png'
    img_saver(path=path,dpi=400)
    chart = Image.open(path)
    cw,ch=chart.size
    offset=350
    cw,cw=chart.size

    section.paste(chart, (int((width-cw)/2)-10, offset), mask=chart)
    idraw.text((title_padding[0], ch + offset), "市盈率越大面积越大", font=h3, fill="#ffffff")

    return section

def section_divergence(trade_date, df_date, bcolor,pos):
    # add init
    section = Image.new('RGBA', (width, 1000), bcolor)
    idraw = draw_text(section, f"{pos}. 市场分化特征",background=True)

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
        line[i].set_color(violet)

    # real
    lines = plt.barh(y, x)
    for i in range(0, len(lines)):
        lines[i].set_color(orange)

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
    path = f'Plot/report_D_material/{trade_date}/divergence.png'
    img_saver(path=path, dpi=300)
    chart = Image.open(path)
    cw, ch = chart.size
    section.paste(chart, (int((width-cw)/2), 400), mask=chart)


    if False:
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


                plt.annotate(f"{chinesename}{value}{round(gain,1)}", xy=(0.5*posneg*(-1), displayn - i - 1), ha='center', va='bottom', color="white", size=18)


        plt.yticks(color="#ffffff")
        fig = plt.gcf()
        fig.set_size_inches(15, 8)

        path = f'Plot/report_D_material/{trade_date}/zhuli.png'
        img_saver(path=path, dpi=280)
        chart = Image.open(path)
        cw2, ch2 = chart.size
        section.paste(chart, (int((width - cw2) / 2), 400+ch+0), mask=chart)

    return section

def helper_leftorright(section,idraw, divergence,a_leftright_label,a_description,a_gain,name,a_index,y_offset=0):
    """
        NOTE the axis is hard coded for now.
        I found out that 4=min, 6=max is a good scale that most of the time remains reasonable
    """
    #cast divergence to be between minimum and maximum
    if False:
        if divergence <= short_minimum:
            divergence = short_minimum
        elif divergence >= short_maxiium:
            divergence = short_maxiium

        # convert the 4-8 scale to 0-1 scale
        divergence = (((divergence - short_minimum)) / (short_maxiium - short_minimum))
        divergence=1-divergence

        # display data
        x_fake = [x for x in [1]]
        x = [x for x in [divergence]]
        y = [x for x in ["egal"]]

        # fake
        line = plt.barh(y, x_fake)  # background
        for i in range(0, len(line)):
            line[i].set_color(white)

        # real
        lines = plt.barh(y, x)
        for i in range(0, len(lines)):
            lines[i].set_color(orange)

        #make it round
        #axes_to_round(ax)

        #standard visual setting
        ax = plt.gca()
        ax.set_xlim([-1, 2])
        ax.set_ylim([0, 1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.annotate(a_leftright_label[0], xy=(-0.20, 0.08), ha='center', va='bottom', color="white", size=28)
        plt.annotate(a_leftright_label[1], xy=(1.20, 0.08), ha='center', va='bottom', color="white", size=28)

        if a_gain[0]>=0:
            text1=f"+{round(a_gain[0],1)}%"
        else:
            text1 = f"{round(a_gain[0], 1)}%"

        if a_gain[1]>=0:
            text2=f"+{round(a_gain[1],1)}%"
        else:
            text2 = f"{round(a_gain[1], 1)}%"
        plt.annotate(text1, xy=(-0.20, 0.35), ha='center', va='bottom', color="white", size=28)
        plt.annotate(text2, xy=(1.20, 0.35), ha='center', va='bottom', color="white", size=28)

        plt.yticks(color="#ffffff")
        fig = plt.gcf()
        fig.set_size_inches(14, 1.8)

        # draw arrow
        path = f'Plot/static/arrow.png'
        chart = Image.open(path)
        chart = chart.resize((150, 150))
        cw, ch = chart.size
        hnorm = 106
        diff = divergence - 0.5
        distance = diff * hnorm * 10

        section.paste(chart, (int(((width - cw) / 2) + distance), 560+y_offset), mask=chart)

        if divergence <= 0.20:
            summary_text = a_description[0]
        elif divergence > 0.20 and divergence <= 0.40:
            summary_text = a_description[1]
        elif divergence > 0.40 and divergence <= 0.60:
            summary_text = a_description[2]
        elif divergence > 0.60 and divergence <= 0.80:
            summary_text = a_description[3]
        elif divergence > 0.80:
            summary_text = a_description[4]


        w, h = idraw.textsize(summary_text, font=mega3)
        #idraw.text((int(((width - w) / 2) + distance), title_padding[1] + 560 - 300+y_offset), summary_text, font=mega3, fill=white)
        idraw.text((int(((width - w) / 2) ), title_padding[1] + 560 +250+y_offset), summary_text, font=mega3, fill=white)
    else:
        for counter, (hdistance,label,stkgain,index) in enumerate(zip([-450,450],a_leftright_label,a_gain,a_index)):

            # best should be at left
            if stkgain == max(a_gain):
                hdistance=-abs(hdistance)
            else:
                hdistance = abs(hdistance)


            #background image
            if hdistance<0:
                bpath = f'Plot/static/yellow.png'
            else:
                bpath = f'Plot/static/white.png'
            # load image and put it as background
            bground = Image.open(bpath)
            bground = bground.resize((650, 430))
            bground = add_corners(bground, 70)
            w, height1 = bground.size
            section.paste(bground, (int(((width - w) / 2)) + hdistance, 410+y_offset), mask=bground)



            #pct
            stkgain = round(stkgain, 1)
            if stkgain > 0:
                stkgain = f"+{stkgain}%"
            else:
                stkgain = f"{stkgain}%"
            w, h = idraw.textsize(stkgain, font=mega3)
            idraw.text((int(((width - w) / 2)) + hdistance, 630 + y_offset), stkgain, font=mega3, fill=gray)

            #label
            w, h = idraw.textsize(label, font=mega3)
            idraw.text((int(((width - w) / 2)) + hdistance, 480+y_offset), label, font=mega3, fill=gray)

            #index ts_code
            index=index[:-3]
            w, h = idraw.textsize(index, font=h3)
            idraw.text((int(((width - w) / 2)) + hdistance, 450 + y_offset), index, font=h3, fill=gray)

            idraw.text((int(((width) / 2))-120 , 400-30+y_offset ), ">", font=mega1, fill="white",align="center")

    # draw label
    if False:
        path = f'Plot/report_D_material/{trade_date}/{name}.png'
        return img_saver(path=path, dpi=300)



def section_style(trade_date, df_date, bcolor, pos):
    # add init
    section_height=4250
    section = Image.new('RGBA', (width, section_height), bcolor)
    idraw = draw_text(section, f"{pos}. 市场风格",background=True)

    d_size={
        "index":["000903.SH","000852.SH"],
        "a_leftright_label" : ["大市值", "小市值"],
        "a_description" : ["大市值股极强", "大市值股偏强", "平衡", "小市值股偏强", "小市值股极强"],
    }

    d_style = {
        "index": ["000117.SH", "000118.SH"],
        "a_leftright_label": ["成长股", "价值股"],
        "a_description": ["成长股极强", "成长股偏强", "平衡", "价值股偏强", "价值股极强"],
    }


    d_beta = {
        "index": ["399407.SZ", "399406.SZ"],
        "a_leftright_label": ["高波动", "低波动"],
        "a_description": ["高波动股极强", "高波动股偏强", "平衡", "低波动股偏强", "低波动股偏强"],
    }

    d_up_down = {
        "index": ["399704.SZ", "399706.SZ"],
        "a_leftright_label": ["上游股", "下游股"],
        "a_description": ["上游股极强", "上游股偏强", "平衡", "下游股偏强", "下游股极强"],
    }

    d_head = {
        "index": ["399653.SZ", "000300.SH"],
        "a_leftright_label": ["龙头股", "非龙头"],
        "a_description": ["龙头股极强", "龙头股偏强", "平衡", "非龙头股偏强", "非龙头股极强"],
    }


    d_new = {
        "index": ["399678.SZ", "000300.SH"],
        "a_leftright_label": ["次新股", "非次新"],
        "a_description": ["次新股极强", "次新股偏强", "平衡", "非次新偏强", "非次新极强"],
    }

    d_cyclic = {
        "index": ["000063.SH", "000064.SH"],
        "a_leftright_label": ["周期股", "非周期"],
        "a_description": ["周期股极强", "周期股偏强", "平衡", "非周期偏强", "非周期极强"],
    }

    real_counter=0
    for counter,d_ in enumerate([d_size,d_style,d_beta,d_up_down,d_head,d_cyclic,d_new]):
        try:
            df_asset_big=DB.get_asset(ts_code=d_["index"][0],asset="I")
            df_asset_small=DB.get_asset(ts_code=d_["index"][1],asset="I")
            biggain = df_asset_big.at[int(trade_date),"pct_chg"]
            smallgain= df_asset_small.at[int(trade_date),"pct_chg"]
            today_divergence = biggain-smallgain

            #y_offset=counter*800
            y_offset=real_counter*550
            helper_leftorright(section=section,idraw=idraw,divergence=today_divergence,a_leftright_label=d_["a_leftright_label"],a_description=d_["a_description"],y_offset=y_offset,name=f"shor_size{counter}",a_gain=[biggain,smallgain],a_index=d_["index"])
            real_counter+=1
        except:
            print(d_["a_leftright_label"],"there is an error here")
            continue

    if real_counter==-1:
        return  Image.new('RGBA', (width, 0), bcolor)
    else:
        #crop image
        missing=counter - real_counter +1
        cut=missing*550
        section = section.crop((0, 0, width, section_height-cut))
        return section



def section_mid_size(trade_date, df_date, bcolor, pos):
    # add init
    section = Image.new('RGBA', (width, 1950), bcolor)
    idraw = draw_text(section, f"{pos}. 中长风格: 大市值vs小市值",background=True)

    path=linechart_helper(section=section,idraw=idraw,name=f"size{pos}",trade_date=trade_date,a_index=[f"000903.SH","000852.SH"],a_summary=["大市值","小市值"])
    if path==-1:
        return Image.new('RGBA', (width, 0), bcolor)

    chart = Image.open(path)
    cw, ch = chart.size
    section.paste(chart, (int((width - cw) / 2), 350), mask=chart)
    return section

def section_mid_valuegrowth(trade_date, df_date, bcolor, pos):
    # add init
    section = Image.new('RGBA', (width, 1950), bcolor)
    idraw = draw_text(section, f"{pos}. 中长风格: 成长股vs价值股",background=True)

    path=linechart_helper(section=section,idraw=idraw,name=f"size{pos}",trade_date=trade_date,a_index=["000117.SH", "000118.SH"],a_summary=["成长股","价值股"])
    if path==-1:
        return Image.new('RGBA', (width, 0), bcolor)

    chart = Image.open(path)
    cw, ch = chart.size
    section.paste(chart, (int((width - cw) / 2), 350), mask=chart)
    return section

def section_mid_beta(trade_date, df_date, bcolor, pos):
    # add init
    section = Image.new('RGBA', (width, 1950), bcolor)
    idraw = draw_text(section, f"{pos}. 中长风格: 高波动vs低波动",background=True)

    path=linechart_helper(section=section,idraw=idraw,name="size",trade_date=trade_date,a_index=["399406.SZ", "399407.SZ"],a_summary=["高波动","低波动"])
    if path==-1:
        return Image.new('RGBA', (width, 0), bcolor)

    chart = Image.open(path)
    cw, ch = chart.size
    section.paste(chart, (int((width - cw) / 2), 350), mask=chart)
    return section

def section_mid_head(trade_date, df_date, bcolor, pos):
    # add init
    section = Image.new('RGBA', (width, 1950), bcolor)
    idraw = draw_text(section, f"{pos}. 中长风格: 龙头股vs非龙头",background=True)

    path=linechart_helper(section=section,idraw=idraw,name=f"size{pos}",trade_date=trade_date,a_index=["399653.SZ", "000300.SH"],a_summary=["龙头股","非龙头"])
    if path==-1:
        return Image.new('RGBA', (width, 0), bcolor)

    chart = Image.open(path)
    cw, ch = chart.size
    section.paste(chart, (int((width - cw) / 2), 350), mask=chart)
    return section

def section_mid_supplier(trade_date, df_date, bcolor, pos):
    # add init
    section = Image.new('RGBA', (width, 1950), bcolor)
    idraw = draw_text(section, f"{pos}. 中长风格: 上游股vs下游股",background=True)

    path=linechart_helper(section=section,idraw=idraw,name=f"size{pos}",trade_date=trade_date,a_index=["399704.SZ", "399706.SZ"],a_summary=["上游股","下游股"])
    if path==-1:
        return Image.new('RGBA', (width, 0), bcolor)

    chart = Image.open(path)
    cw, ch = chart.size
    section.paste(chart, (int((width - cw) / 2), 350), mask=chart)
    return section

def section_mid_cyclic(trade_date, df_date, bcolor, pos):
    # add init
    section = Image.new('RGBA', (width, 1950), bcolor)
    idraw = draw_text(section, f"{pos}. 中长风格: 周期股vs非周期",background=True)

    path=linechart_helper(section=section,idraw=idraw,name=f"size{pos}",trade_date=trade_date,a_index=["000063.SH", "000064.SH"],a_summary=["周期股","非周期"])
    if path==-1:
        return Image.new('RGBA', (width, 0), bcolor)

    chart = Image.open(path)
    cw, ch = chart.size
    section.paste(chart, (int((width - cw) / 2), 350), mask=chart)
    return section

def section_mid_new(trade_date, df_date, bcolor, pos):
    # add init
    section = Image.new('RGBA', (width, 1950), bcolor)
    idraw = draw_text(section, f"{pos}. 中长风格: 次新股vs非次新",background=True)

    path=linechart_helper(section=section,idraw=idraw,name=f"size{pos}",trade_date=trade_date,a_index=["399678.SZ", "000300.SH"],a_summary=["次新股","非次新"])
    if path==-1:
        return Image.new('RGBA', (width, 0), bcolor)

    chart = Image.open(path)
    cw, ch = chart.size
    section.paste(chart, (int((width - cw) / 2), 350), mask=chart)
    return section

def section_mid_fund(trade_date, df_date, bcolor, pos):
    # add init
    section = Image.new('RGBA', (width, 1950), bcolor)
    idraw = draw_text(section, f"{pos}. 中长风格: 基金vs沪深300",background=True)

    path=linechart_helper(section=section,idraw=idraw,name=f"size{pos}",trade_date=trade_date,a_index=["399379.SZ", "399300.SZ"],a_labelhelper=["",""],a_summary=["基金","沪深300"])
    if path==-1:
        return Image.new('RGBA', (width, 0), bcolor)

    chart = Image.open(path)
    cw, ch = chart.size
    section.paste(chart, (int((width - cw) / 2), 350), mask=chart)
    return section

def section_distribution(trade_date, df_date, bcolor, pos):
    # add init
    section = Image.new('RGBA', (width, 2500), bcolor)
    idraw = draw_text(section, f"{pos}. 涨跌分布",background=True)


    # add histogram
    histo = Image.open(get_bar(s=df_date["pct_chg"],path = f'Plot/report_D_material/{trade_date}/histo.png'))
    hw, hh = histo.size

    #paste all stuff into the section
    offsetfrom_hiddenaxis=0
    histooffset=0
    section.paste(histo, (int((width - hw) / 2)-offsetfrom_hiddenaxis, -histooffset), mask=histo)

    # add average gain
    avg_gain = df_date["pct_chg"].mean()
    avg_gain = round(avg_gain, 2)
    avg_gain = f"平均涨跌\n+{avg_gain}%" if avg_gain > 0 else f"平均涨跌\n{avg_gain}%"
    idraw.text((title_padding[0], 400), avg_gain, font=mega3, fill="white")

    try:
        zd_ratio = len(df_date[df_date["pct_chg"] > 9.5]) / len(df_date[df_date["pct_chg"] < -9.5])
        zd_ratio = int(round(zd_ratio))
        idraw.text((1350, 400), f"涨跌停比\n≈{zd_ratio}:1", font=mega3, fill="white", align="right")
    except:
        pass

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
        bground = bground.resize((500, 800))
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

def section_index(trade_date, df_date,bcolor,pos):

    # add init
    section = Image.new('RGBA', (width, 2400), bcolor)
    idraw = draw_text(section, f"{pos}. 三指涨幅",fill=gray,background=True)



    #calculate data for 3 index
    a_pct_chg=[]
    vertical_offset = 400
    horizontal_offset = 650

    for ts_index,name,i in zip(["000001.SH","399001.SZ","399006.SZ"],["上证指数","深成指数","创业板指"],[-1,0,1]):

        #redundant way to determine if stock exist on that day
        try:

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
            path = f'Plot/report_D_material/{trade_date}/{ts_index}.png'
            img_saver(path=path,dpi=80)
            chart = Image.open(path)
            chart=chart.resize((480, 300))


            #create data pack of index
            df = DB.get_asset(ts_index, asset="I")
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
            w,h=chart.size
            section.paste(chart, (int(((width - w) / 2) )+ 0 + i * horizontal_offset, height1 + height2 + height3 + vertical_offset + 90), mask=chart)

        except:

            #if stock doesnt exist back then.create dummy


            # create data pack of index
            pct_chg_text = f"+0.0%"



            bpath = f'Plot/static/gray.png'

            color = white

            # load image and put it as background
            bground = Image.open(bpath)
            bground = bground.resize((600, 850))
            bground = add_corners(bground, 70)
            w, height1 = bground.size
            section.paste(bground, (int(((width - w) / 2)) + i * horizontal_offset, 350), mask=bground)

            # display 标题
            w, height1 = idraw.textsize(f"{name}", font=h2)
            idraw.text((int(((width - w) / 2)) + i * horizontal_offset, vertical_offset + 0), name, font=h2, fill=white)

            # display close
            w, height2 = idraw.textsize(f"未上市", font=h1)
            idraw.text((int(((width - w) / 2)) + i * horizontal_offset, height1 + vertical_offset + 0), f"未上市", font=h1, fill=white)

            # display pct_chg 涨跌
            w, height3 = idraw.textsize(pct_chg_text, font=mega3)
            idraw.text((int(((width - w) / 2)) + i * horizontal_offset, height1 + height2 + vertical_offset + 40), pct_chg_text, font=mega3, fill=color)

            # display mini chart
            #section.paste(chart, (int(((width - w) / 2)) + 70 + i * horizontal_offset, height1 + height2 + height3 + vertical_offset + 60), mask=chart)

    #STOCK DATA
    #beat index E data
    abs_beat_index = len(df_date[df_date["pct_chg"] > builtins.max(a_pct_chg)])
    pct_beat_index = int((abs_beat_index / len(df_date)) * 100)

    #beat index E pct
    baseheight=1300
    w, height4 = idraw.textsize(f"{pct_beat_index}%", font=mega1)
    idraw.text((int(((width-w) / 2)) , baseheight), f"{pct_beat_index}%", font=mega1, fill=white)

    #beat index E text
    biggest_pct=max(a_pct_chg)
    if a_pct_chg.index(biggest_pct)==0:
        sanzhiname="上证指数"
    elif a_pct_chg.index(biggest_pct)==1:
        sanzhiname="深证指数"
    elif a_pct_chg.index(biggest_pct)==2:
        sanzhiname="创业板指"

    textmsg=f"{abs_beat_index}只股票今跑赢{sanzhiname}"
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
    textmsg = f"{abs_beat_index}只基金今跑赢{sanzhiname}"
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


def section_mid_highlow(trade_date, df_date, bcolor, pos):

    # add init


    # add cliff hanger illustration
    if False:
        path = f'Plot/static/cliff.png'
        chart = Image.open(path)
        chart = chart.resize((1900, 1100))
        section.paste(chart, (0, 0), mask=chart)

    #add data
    df_date["isminmax"]=df_date["isminmax"].replace(np.nan,0)

    df_date["isminmax"] = df_date["isminmax"].replace(120, 60)
    df_date["isminmax"] = df_date["isminmax"].replace(-120, -60)
    df_date["isminmax"] = df_date["isminmax"].replace(500, 240)
    df_date["isminmax"] = df_date["isminmax"].replace(-500, -240)

    df_date_groupby=df_date.groupby("isminmax").count()

    # use this to draw histogram of your data
    path = f'Plot/report_D_material/{trade_date}/allimeghighlow.png'

    #create statistic
    s=df_date_groupby["pct_chg"]



    #if 60 day high is zero, fill zero
    for number in [20,60,240,999]:
        for i in [number,-number]:
            if i not in s.index:
                s.at[i]=0



    s=s.sort_index()
    largest=s.nlargest(2)#because 0 is also there and is always big
    bigest=largest.iat[-1]
    bigest=bigest+100


    #create iamge intermediately
    setting=len(df_date)
    chart = Image.open(get_bar2(s=s,path=path,y_limit=[-setting,setting]))
    cw,ch=chart.size

    #cut cupper and lower part #cropchart
    cropheight=(ch/2)/setting*(setting-bigest)
    chart=chart.crop((0,cropheight,cw,(ch/2)+130))
    chart.save(path)
    cw, ch = chart.size

    #create section based on image size
    section_height = ch+700
    section = Image.new('RGBA', (width, section_height), bcolor)
    idraw = draw_text(section, f"{pos}. 新高统计", background=True)

    #paste image to section
    section.paste(chart, (int((width - cw) / 2), 300), mask=chart)

    #add新高 新低两字
    for hdistance, text in [(450,"新高"),(-450,"新低")]:
        w, height2 = idraw.textsize(text, font=mega2)
        idraw.text((int(((width - w) / 2)) + hdistance, 300+ch+0), text, font=mega2, fill=white)

    #section=section.crop((0, 0, width, section_height-ch-300-200))
    return section


def bar_helper(x,y,trade_date,y_label="",name="rsima"):
    # calculate data
    y_fake = [100 for x in y]

    # fake
    line = plt.bar(x, y_fake)  # background
    for i in range(0, len(line)):
        line[i].set_color(white)

    # real
    line = plt.bar(x, y)
    for i in range(0, len(line)):
        line[i].set_color(orange)
    axes = plt.gca()
    axes.set_ylim([-20, 110])
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(6, 4)

    # draw values as text
    for enum, i in enumerate(range(len(y))):
        # OVER each bar
        if y[i] <= 75:
            add = 5
            color = gray
        else:
            add = -20
            color = gray

        plt.annotate(f"{y[i]}{y_label}", xy=(i, y[i] + add), ha='center', va='bottom', color=color, size=24)

        # UNDER each bar
        plt.annotate(x[i], xy=(i, -10), ha='center', va='bottom', color="white", size=13)

    # use this to draw histogram of your data
    path = f'Plot/report_D_material/{trade_date}/{name}.png'
    img_saver(path=path, dpi=350)
    return path

def section_ma(trade_date, df_date,bcolor,pos):

    # add init
    section = Image.new('RGBA', (width, 1900), bcolor)
    idraw = draw_text(section, f"{pos}. 均线统计",background=True)


    # calculate data
    if False:

        a_y_abs = []
        a_colors = [treemapred1, treemapred2, treemapred3, treemapred4]
        for enum, freqn in enumerate([5, 20, 60, 240]):
            y_abs = len(df_date[df_date["close"] >= df_date[f"ma{freqn}"]])
            a_y_abs += [y_abs]
            pct = y_abs / len(df_date)

            yes_name = f"{freqn}日 均线上"
            no_name = f"{freqn}日 均线下"
            yes = int(pct * 100)
            no = 100 - yes
            s = pd.Series()
            s.at[yes_name] = yes
            s.at[no_name] = no
            ax = s.plot(kind='pie', labels=["", ""], colors=[white, a_colors[enum]], startangle=90, wedgeprops={"edgecolor": "white", 'linewidth': 1})
            plt.axis('off')

        radius=200
        space=600
        offset=700
        path = f'Plot/report_D_material/{trade_date}/ma{enum}.png'
        img_saver(path=path, dpi=900-enum*radius)
        chart = Image.open(path)

        cw, ch = chart.size
        section.paste(chart, (int((width - cw) / 2), int((space-ch)/2)+offset), mask=chart)

    else:
        #display data
        y = []
        a_y_abs=[]
        for freqn in [5, 20, 60, 240]:
            df_filter=df_date[df_date["close"]>df_date[f"ma{freqn}"]]
            ma_pct=len(df_filter)/len(df_date)
            y+=[int(ma_pct*100)]
            a_y_abs+=[len(df_filter)]
            x=[f"{x}日均线上" for x in [5, 20, 60, 240]]

        chart = Image.open(bar_helper(x,y,trade_date,"%",name="ma"))
        cw,ch=chart.size
        section.paste(chart, (int((width-cw)/2), 400), mask=chart)

    explain=f"{y[0]}% 的股票\n收盘在5日均线上"
    w, h = idraw.textsize(explain, font=mega3)
    idraw.text((int((width - w) / 2), 1450), explain, font=mega3, fill=white,align="center")

    return section


def section_rsi(trade_date, df_date,bcolor,pos):

    # add init
    section = Image.new('RGBA', (width, 1900), bcolor)
    idraw = draw_text(section, f"{pos}. RSI统计",background=True)

    # display data
    x=[]
    y=[]
    for freqn in [5, 20, 60, 240]:
        y+= [round(df_date[f"rsi{freqn}"].mean(), 1)]
        x+=[f"{freqn}日RSI"]
    chart = Image.open(bar_helper(x,y,trade_date,"",name="rsi"))

    cw,ch=chart.size
    section.paste(chart, (int((width-cw)/2), 400), mask=chart)

    explain=f"所有股票5日RSI\n的均值为{y[0]}"
    w, h = idraw.textsize(explain, font=mega3)
    idraw.text((int((width - w) / 2), 1450), explain, font=mega3, fill=white,align="center")

    return section

def northsouth_helper(section,df_nsmoney,pos, i, name,nb,trade_date):


        use_yi = True
        if use_yi:
            df_nsmoney[f"{name}_money"] = df_nsmoney[f"{name}_money"] / 100
            df_nsmoney[f"{name}_money"] = df_nsmoney[f"{name}_money"].cumsum()
            plt.ylabel('亿元', color="white")
        else:
            plt.ylabel('百万元', color="white")

        ax=df_nsmoney[f"{name}_money"].plot(color=orange, legend=False, linewidth=4)
        #df_nsmoney["zero"].plot(color=white, legend=False, linewidth=1)
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

        path = f'Plot/report_D_material/{trade_date}/{name}.png'
        img_saver(path=path, dpi=350)
        chart = Image.open(path)
        cw,ch=chart.size
        section.paste(chart, (int((width-cw)/2), 400), mask=chart)



def section_northmoney(trade_date, df_date, bcolor, pos):

    # add init
    section = Image.new('RGBA', (width, 1300), bcolor)
    idraw = draw_text(section, f"{pos}. 北向资金累计", background=True)

    #calculate data
    df_nsmoney=DB.get(a_path=LB.a_path("Market/CN/Asset/E/hsgt/hsgt"))
    df_nsmoney=df_nsmoney[df_nsmoney.index <= int(trade_date)]
    df_nsmoney=df_nsmoney.tail(120)#only check last 120 days as half year data
    if df_nsmoney.empty:
        return Image.new('RGBA', (width, 0), bcolor)

    df_nsmoney["zero"]=0

    LB.trade_date_to_vieable(df_nsmoney)
    df_nsmoney.index.name =""

    #display data
    northsouth_helper(section=section,df_nsmoney=df_nsmoney,pos=pos,i=0,name="north",nb="北",trade_date=trade_date)
    return section

def section_southmoney(trade_date, df_date, bcolor, pos):

    # add init
    section = Image.new('RGBA', (width, 1300), bcolor)
    idraw = draw_text(section, f"{pos}. 南向资金累计", background=True)

    #calculate data
    df_nsmoney=DB.get(a_path=LB.a_path("Market/CN/Asset/E/hsgt/hsgt"))
    df_nsmoney=df_nsmoney[df_nsmoney.index <= int(trade_date)]
    df_nsmoney=df_nsmoney.tail(120)#only check last 120 days as half year data
    if df_nsmoney.empty:
        return Image.new('RGBA', (width, 0), bcolor)

    df_nsmoney["zero"]=0

    LB.trade_date_to_vieable(df_nsmoney)
    df_nsmoney.index.name =""

    #display data
    northsouth_helper(section=section, df_nsmoney=df_nsmoney, pos=pos, i=1, name="south", nb="南",trade_date=trade_date)

    return section


def section_mid_industry1(trade_date, df_date, bcolor, pos):

    # add init
    section = Image.new('RGBA', (width, 2400), bcolor)
    idraw = draw_text(section, f"{pos}. 行业涨幅",background=True)
    d_quantile=LB.custom_quantile(df=df_date,column="pct_chg",key_val=False,p_setting=[0,1])

    for enum, (key,df_date_cut) in enumerate(d_quantile.items()):
        #treemap data

        df_group_sum=df_date_cut.groupby(by="sw_industry1", sort=True).sum()
        df_group_mean=df_date_cut.groupby(by="sw_industry1", sort=True).mean()
        df_group_sum=df_group_sum.sort_values(by="total_mv",ascending=False)

        #cheat size by make the last n smallest industry bigger. Otherwise too small to display
        last=len(df_group_sum)
        for i in range(1,last):
            df_group_sum["total_mv"].iloc[-i]=df_group_sum["total_mv"].iat[-last]

        #sort the box by who gained most

        df_group_sum["pct_chg_mean"]=df_group_mean["pct_chg"]
        #df_group_sum=df_group_sum.sort_values("pct_chg_mean")


        sizes = df_group_sum["total_mv"]
        label=[]
        for name,pct_chg in zip(df_group_sum.index,df_group_sum["pct_chg_mean"]):
            if pct_chg>0:
                label+=[f"{name}\n+{round(pct_chg,2)}%"]
            else:
                label+=[f"{name}\n{round(pct_chg,2)}%"]


        colors= []
        for name, pct_chg in zip(df_group_sum.index, df_group_sum["pct_chg_mean"]):
            if pct_chg==0:
                colors+=[gray]

            elif pct_chg>0 and pct_chg<=1:
                colors += [treemapred1]
            elif pct_chg>1 and pct_chg<=2:
                colors += [treemapred1]
            elif pct_chg > 2 and pct_chg <= 3:
                colors += [treemapred1]
            elif pct_chg > 3:
                colors += [treemapred1]

            elif pct_chg>=-1 and pct_chg<0:
                colors += [treemapgreen1]
            elif pct_chg>=-2 and pct_chg<-1:
                colors += [treemapgreen1]
            elif pct_chg>=-3 and pct_chg<-2:
                colors += [treemapgreen1]
            elif pct_chg<-3 :
                colors += [treemapgreen1]


        squarify.plot(sizes=sizes, label=label, color=colors, alpha=1,text_kwargs={'fontsize':10, 'fontname':"Microsoft Yahei","color":"#ffffff"},bar_kwargs=dict(linewidth=1, edgecolor="#ffffff"))
        fig = plt.gcf()
        fig.set_size_inches(6,6)
        plt.axis('off')
        path = f'Plot/report_D_material/{trade_date}/tree{enum}.png'

        img_saver(path=path,dpi=400)
        chart = Image.open(path)
        cw,ch=chart.size
        offset=400
        section.paste(chart, (int((width-cw)/2)-10, offset+enum*ch), mask=chart)

        if False:
            explain = f"市值越大面积越大"
            w, h = idraw.textsize(explain, font=h2)
            idraw.text((int((width - w) / 2), ch + offset), explain, font=h2, fill=white)


    return section


def create_infographic(trade_date=LB.latest_trade_date(),showimg=True,png=False,send=False,delete_asset=True,save_section=True):



    path = f'Plot/report_D_result_hq/每日收盘_{trade_date}.png'
    if os.path.isfile(path):
        print(f"{trade_date} infochart exists")
        if send:
            #format month-day-year

            trade_date_web=f"{int(trade_date[4:6])}/{trade_date[6:8]}/{trade_date[2:4]}"
            print(trade_date_web)
            files=[path]
            LB.send_mail_report(trade_string=trade_date_web, files=files,receiver="dailystockinfo.uqhpqf@zapiermail.com")
        return

    # init
    print("start")
    df_date=DB.get_date(trade_date=trade_date)
    if df_date.empty:
        raise  AssertionError
    df_date = df_date[df_date["pct_chg"].notna()]

    #add sections, size, value vs growth, industry, jijing vs husheng300, zhuti, beta
    a_func=[section_title, section_index, section_distribution, section_style, section_mid_industry1, section_mid_highlow, section_ma, section_northmoney, section_southmoney, section_industry_pe, section_sh_pe, section_cy_pe, section_mid_size, section_mid_valuegrowth, section_mid_beta, section_mid_supplier, section_mid_head, section_mid_cyclic, section_mid_new, section_end]
    a_bcolor=["#1D37FF","#1D37FF","#1D37FF","#1D37FF","#1D37FF","#1D37FF","#1D37FF","#1D37FF","#1D37FF","#1D37FF","#1D37FF"]
    a_bcolor=[None]*len(a_func)
    a_sections_name=[name.__name__ for name in a_func]
    a_sections = [func(trade_date=trade_date, df_date=df_date,bcolor=bcolor,pos=pos) for pos,(func,bcolor) in enumerate(zip(a_func,a_bcolor))]

    #put all sections together
    autoheight=builtins.sum([section.size[1] for section in a_sections])
    infographic = Image.new('RGBA', (width, autoheight), "#1D37FF")

    # add background
    path = f'Plot/static/gradient5.png'
    chart = Image.open(path)
    chart = chart.resize((2200, autoheight))
    infographic.paste(chart, (0, 0), )


    #save sections
    if save_section:
        for section,name in zip(a_sections,a_sections_name):
            for n in range(10):
                try:
                    #png first
                    path = f'Plot/report_D_section/{trade_date}/{name}.png'
                    section.save(path,optimize=True)

                    # jpeg second
                    if False:
                        section = section.convert('RGB')
                        path = f'Plot/report_D_section/{trade_date}/{name}.jpeg'
                        section.save(path, 'jpeg', quality=10,optimize=True,progressive=True)

                    break
                except:
                    folder = "/".join(path.rsplit("/")[:-1])
                    if not os.path.exists(folder):
                        os.makedirs(folder)


    #add sections
    y_helper=0
    for section in a_sections:
        infographic.paste(section,( 0, y_helper),mask=section)
        y_helper += section.size[1]

    #show and save
    if showimg:
        infographic.show()


    path=f'Plot/report_D_result_hq/每日收盘{trade_date}.png'
    infographic.save(path)

    infographic = infographic.convert('RGB')
    path = f'Plot/report_D_result_lq/每日收盘_{trade_date}.jpeg'
    infographic.save(path, 'jpeg', quality=10, optimize=True, progressive=True)

    #delete generated assets and only keep the result
    if delete_asset:
        path = f'Plot/report_D_material/{trade_date}'
        if os.path.isdir(path):
            os.remove(path)

    return path


def get_request(url):
    r= requests.get(url)
    return r


def server_existing_infocharts(version_test="version-test"):
    # make request to see what the server already has
    url = f"https://www.5849cai.com/{version_test}/api/1.1/obj/infochart"
    r = requests.get(url)
    decode_r = r.content.decode("UTF-8")
    dict_str = ast.literal_eval(decode_r)
    a_result = dict_str["response"]["results"]
    for dict_enty in a_result:
        print()
        for key,data in dict_enty.items():
            print(key,":",data)


def server_post_infochart(version_test="version-test"):
    # make request to see what the server already has
    url = f"https://www.5849cai.com/{version_test}/api/1.1/obj/infochart"
    test = {
        "Image": "",
        "Created Date": "",
        "Created By": "",
        "Modified Date": "",
        "Tradedate": "egal",
        "_id": "egalasadasdsad",
        "_type": "custom.infochart"
    }
    test_json=json.dumps(test)
    r=requests.post(url, json=test_json,data=test_json)
    print(r)

def server_delete_infochart(version_test="version-test",id=""):
    # make request to see what the server already has
    url = f"https://www.5849cai.com/{version_test}/api/1.1/obj/infochart/{id}"
    r=requests.delete(url)
    print(r)

def server_patch_infochart(version_test="version-test",id=""):
    # make request to see what the server already has
    url = f"https://www.5849cai.com/{version_test}/api/1.1/obj/infochart/{id}"
    test={
        "uploadImage": "https://www.baidu.com/img/flexible/logo/pc/result.png",
        "uploadTradedate": "1/10/21",
        "uploadSections": ["https://www.baidu.com/img/flexible/logo/pc/result.png","https://www.baidu.com/img/flexible/logo/pc/result.png","https://www.baidu.com/img/flexible/logo/pc/result.png"],
          }

    test_json=json.dumps(test,indent=4)
    r=requests.post(url, json=test_json)
    print(r)

def workflow_api():
    url = f"https://www.5849cai.com/version-test/api/1.1/wf/uploadchart"
    test = {
        "Image": "",
        "Created Date": "",
        "Created By": "",
        "Modified Date": "",
        "Tradedate": "egal",
        "_id": "egalasadasdsad",
        "_type": "custom.infochart"
    }
    test_json = json.dumps(test)
    r = requests.post(url, json=test_json, data=test_json)
    print(r)

if __name__ == '__main__':

    do=3

    if do==0:
        workflow_api()



    elif do==1:
        trade_date = fr"20210118"
        trade_date = fr"20210119"
        trade_date = fr"20080630"
        trade_date = fr"20100331"
        create_infographic(trade_date=trade_date)
    elif do==2:
        df_trade_date = DB.get_trade_date()
        df_trade_date = df_trade_date[df_trade_date["lastdayofseason"] == True]

        for trade_date in df_trade_date.index[::-1]:
            print(f"Infochart for trade_date {trade_date}")
            try:
                create_infographic(trade_date=str(trade_date),showimg=False)
            except Exception as e:
                print(e)

    elif do==3:
        df_trade_date = DB.get_trade_date()
        for trade_date in df_trade_date.index[::-3]:
            print(f"Infochart for trade_date {trade_date}")
            try:
                create_infographic(trade_date=str(trade_date), showimg=False)
                gc.collect()
            except Exception as e:
                print(e)
    elif do==4:
        #send data to website
        df_trade_date = DB.get_trade_date()
        for trade_date in df_trade_date.index[:-10:-1]:
            print(f"Infochart for trade_date {trade_date}")
            try:
                create_infographic(trade_date=str(trade_date), showimg=False,send=False)
                gc.collect()
            except Exception as e:
                print(e)

    elif do==5:
        from webptools import cwebp
        #imageio.plugins.freeimage.download()
        sourcepath = f"Plot/plot.jpg"
        destipath = f"Plot/plot.webp"
        print(cwebp(sourcepath,destipath,"-q 80"))