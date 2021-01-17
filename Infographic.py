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
import matplotlib.pyplot as plt


from PIL import Image, ImageDraw, ImageFont

def section1_img():
    pass

def section2_img():
    pass

def create_infographic(trade_date="20200427"):
    dpi=80
    def get_histo(series, smallbins=True):
        if smallbins== True:
            n, bins, patches =plt.hist(series, rwidth=0.9,bins= [-20,-8,-5,-2, 0, 2,5,8, 20] )  # use this to draw histogram of your data
        else:
            n, bins, patches =plt.hist(series, rwidth=0.9,bins= [x for x in range(-10,10,1)] )  # use this to draw histogram of your data
        plt.axis('off')
        cm = plt.cm.RdBu_r
        for i, p in enumerate(patches):
            plt.setp(p, 'facecolor', cm(i / 19))  # notice the i/25
        path=f'Plot/report_D/{trade_date}/histo.png'
        try:
            plt.savefig(path,transparent=True,dpi=dpi*3)
        except:
            os.makedirs(f'Plot/report_D/{trade_date}')
            plt.savefig(path, transparent=True,dpi=dpi*3)
        return path



    df_date=DB.get_date(trade_date=trade_date)
    if df_date.empty:
        raise  AssertionError


    #init
    width=2000
    height=int(width*0.62)
    backupfontpath=r'c:\windows\fonts\msyh.ttc'
    h1 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width*0.05))
    h2 = ImageFont.truetype(r'c:\windows\fonts\msyhbd.ttc', size=int(width*0.03))
    text = ImageFont.truetype(r'c:\windows\fonts\msyh.ttc', size=int(width*0.02))
    infographic = Image.new('RGBA', (width, height*3), (0,1,60))
    idraw = ImageDraw.Draw(infographic)

    #section 1

    histo= Image.open(get_histo(df_date["pct_chg"], smallbins=False))
    hw, hh = histo.size
    section1 = Image.new('RGBA', (width, height), 'red')
    infographic.paste(histo, ( int((width - hw) / 2), 0),mask=histo)



    # section 2
    """section2 = Image.new('RGBA', (width, height), 'blue')
    infographic.paste(section2, (0, height))"""



    #test
    title=f"{trade_date[0:4]}年{trade_date[4:6]}月{trade_date[6:8]}日 A股总结报告"
    w, h = idraw.textsize(title,font=h1)
    idraw.text(  ( int((width - w) / 2), 80), title,font=h1)




    infographic.show()
    infographic.save("cool.png")




if __name__ == '__main__':
    create_infographic()