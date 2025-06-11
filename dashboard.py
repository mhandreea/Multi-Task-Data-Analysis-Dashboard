# -*- coding: utf-8 -*-
""""
!pip install reportlab
!pip install gradio
!pip instal IPython.display
# Install dependencies (run once)
!pip install gradio pandas numpy matplotlib reportlab

Please Run on local URL:  http://127.0.0.1:7860
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
import gradio as gr

# --- Task 1: Agricultural Yield Analysis ---
def task1(
    output_csv="sample_yield_12.csv",
    output_pdf="agri_yield_report.pdf",
    seed:int=0
):
    np.random.seed(seed)
    # 1) Generate 12-row CSV
    farm_ids    = [f"F{str(i).zfill(3)}" for i in range(1,13)]
    dates       = pd.date_range("2025-01-01", periods=12)
    crops       = ["Wheat","Corn","Soybean","Barley","Rice"]*3
    df = pd.DataFrame({
        "FarmID": farm_ids,
        "Date": dates,
        "CropType": crops[:12],
        "YieldTons": [10.5,8.0,7.2,9.1,6.5,11.0,8.5,7.8,9.3,6.8,10.2,8.3],
        "FertilizerCost": [200,150,180,170,160,210,155,185,175,165,205,158],
        "Rainfall": [50,40,45,55,60,52,42,47,57,62,51,43],
        "Temperature": [15,18,20,17,16,14,19,21,18,17,15,18]
    })
    df.to_csv(output_csv, index=False)

    # 2) Read & Clean
    df = pd.read_csv(output_csv, parse_dates=['Date'])
    df['CropType'] = df['CropType'].str.strip().str.title()
    num = ['YieldTons','FertilizerCost','Rainfall','Temperature']
    df[num] = df[num].fillna(df[num].mean())
    for c in num:
        df = df[df[c]>=0]

    # 3) Metrics
    df['NetYieldValue'] = df['YieldTons']*500 - df['FertilizerCost']
    total_yield = df.groupby('CropType')['YieldTons'].sum()
    farm_avg    = df.groupby(['CropType','FarmID'])['YieldTons'].mean().reset_index()
    most_prod   = farm_avg.groupby('CropType')['YieldTons'].mean().idxmax()
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    monthly = df.groupby('Month').agg(
        total_yield=('YieldTons','sum'),
        total_cost=('FertilizerCost','sum'),
        avg_rain=('Rainfall','mean'),
        avg_temp=('Temperature','mean')
    )
    df['YieldEff'] = df['YieldTons']/df['FertilizerCost']
    weather_corr  = np.corrcoef(df['Rainfall'],df['YieldTons'])[0,1]
    above = df[df['Temperature']>25]['YieldTons'].mean()
    below = df[df['Temperature']<=25]['YieldTons'].mean()
    temp_eff = ((above-below)/below*100) if below and not np.isnan(above) else None

    # 4) farm dict
    farm_dict = {
        row.FarmID:(row.CropType,row.NetYieldValue,row.YieldEff)
        for row in df.itertuples()
    }

    # 5) Charts
    imgs1=[]
    # Line: monthly total yield
    fig,ax=plt.subplots(figsize=(6,4))
    monthly['total_yield'].plot(marker='o',ax=ax)
    ax.set_title("Monthly Total Yield"); ax.set_xlabel("Month"); ax.set_ylabel("Tons")
    plt.xticks(rotation=45); plt.tight_layout()
    imgs1.append("t1_line.png"); fig.savefig(imgs1[-1]); plt.close(fig)
    # Bar: avg by crop
    fig,ax=plt.subplots(figsize=(6,4))
    total_yield.plot(kind='bar',ax=ax)
    ax.set_title("Average Yield by Crop"); ax.set_xlabel("Crop"); ax.set_ylabel("Tons")
    plt.xticks(rotation=45); plt.tight_layout()
    imgs1.append("t1_bar.png"); fig.savefig(imgs1[-1]); plt.close(fig)
    # Scatter
    x,y=df['Rainfall'],df['YieldTons']; m,b=np.polyfit(x,y,1); p=np.poly1d((m,b))
    fig,ax=plt.subplots(figsize=(6,4))
    ax.scatter(x,y); ax.plot(x,p(x),'--')
    ax.set_title("Rainfall vs Yield"); ax.set_xlabel("Rainfall"); ax.set_ylabel("Tons")
    plt.tight_layout()
    imgs1.append("t1_scatter.png"); fig.savefig(imgs1[-1]); plt.close(fig)

    # 6) PDF
    doc=SimpleDocTemplate(output_pdf,pagesize=letter)
    st=getSampleStyleSheet()
    elems=[Paragraph("Agricultural Yield Analysis",st['Title']),Spacer(1,12),
           Paragraph("Key Metrics",st['Heading2'])]
    tbl=[["Metric","Value"]]
    for crop,tot in total_yield.items():
        tbl.append([f"Total Yield – {crop}",f"{tot:.1f}"])
    tbl+=[["Most Productive Crop",most_prod],
          ["Rainfall–Yield Corr",f"{weather_corr:.3f}"],
          [">25°C Effect (%)",f"{temp_eff:.2f}" if temp_eff is not None else "N/A"]]
    t=Table(tbl,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightblue),
                           ('GRID',(0,0),(-1,-1),0.5,colors.grey),
                           ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                           ('ALIGN',(0,0),(-1,-1),'CENTER')]))
    elems+=[t,Spacer(1,12),Paragraph("Visualizations",st['Heading2'])]
    for img in imgs1:
        elems+=[Image(img,400,300),Spacer(1,12)]
    analysis=(
        "Over January 2025, Wheat led total yield followed by Corn. The "
        "rainfall-yield correlation is slightly negative, indicating "
        "rainier days didn’t boost output. No days exceeded 25°C, so "
        "heat stress is inconclusive. Yield efficiency (tons per $ fert.) "
        "was highest in Barley and Wheat. Monthly trends were stable. "
        "These insights can guide fertilizer allocation and planting schedules."
    )
    elems+=[Paragraph("Analysis",st['Heading2']),Paragraph(analysis,st['Normal'])]
    doc.build(elems)

    return {
        'csv':output_csv,'pdf':output_pdf,
        'images':imgs1
    }

# --- Task 2: E-commerce Behavior (v2) ---
def task2_v2(
    output_csv="sample_orders_13.csv",
    output_pdf="ecom_behavior_v2.pdf",
    seed:int=1
):
    np.random.seed(seed)
    n=13
    cust=[f"C{str(i).zfill(3)}" for i in range(1,n+1)]
    dates=pd.date_range("2025-01-01",periods=n)
    cats=["Electronics","Clothing","Home","Beauty","Toys","Books",
          "Sports","Outdoors","Grocery","Health","Automotive",
          "Jewelry","Garden"]
    df=pd.DataFrame({
      "CustomerID":cust,
      "Date":dates,
      "ProductCategory":np.random.choice(cats,n,replace=True),
      "OrderValue":np.round(np.random.uniform(20,500,n),2),
      "Discount":np.round(np.random.uniform(0,0.5,n),2),
      "BrowseTime":np.round(np.random.uniform(1,60,n),1),
      "ReturnFlag":np.random.choice(["Yes","No"],n,p=[0.3,0.7])
    })
    df.to_csv(output_csv,index=False)

    df=pd.read_csv(output_csv,parse_dates=['Date'])
    df['ProductCategory']=df['ProductCategory'].str.strip().str.title()
    df=df[df['OrderValue']>=0]
    df=df[(df['Discount']>=0)&(df['Discount']<=1)]
    df=df[df['BrowseTime']>=0]
    df['ReturnFlag']=df['ReturnFlag'].str.strip().str.title()

    df['NetOrderValue']=df['OrderValue']*(1-df['Discount'])
    rev=df.groupby('ProductCategory')['NetOrderValue'].sum()
    ret_rate=df.groupby('ProductCategory')['ReturnFlag']\
               .apply(lambda s:(s=="Yes").mean())
    top_ret=ret_rate.idxmax()
    df['Day']=df['Date'].dt.to_period('D').astype(str)
    daily=df.groupby('Day').agg(
        total_order=('OrderValue','sum'),
        total_discount_amt=('OrderValue',lambda s:(s*df.loc[s.index,'Discount']).sum()),
        avg_browse=('BrowseTime','mean'),
        return_rate=('ReturnFlag',lambda s:(s=="Yes").mean())
    )
    df['PurchaseEff']=df['NetOrderValue']/df['BrowseTime']
    corr_br=np.corrcoef(df['BrowseTime'],(df['ReturnFlag']=="Yes").astype(int))[0,1]
    df['DiscountImpact']=df['OrderValue']*df['Discount']
    avg_disc=df['DiscountImpact'].mean()

    cust_dict={row.CustomerID:(row.ProductCategory,row.NetOrderValue,row.PurchaseEff)
               for row in df.itertuples()}

    imgs2=[]
    # line daily order
    fig,ax=plt.subplots(figsize=(6,4))
    daily['total_order'].plot(marker='o',ax=ax)
    ax.set_title("Daily Total Order"); ax.set_xlabel("Day"); ax.set_ylabel("$")
    plt.xticks(rotation=45); plt.tight_layout()
    imgs2.append("t2_line.png"); fig.savefig(imgs2[-1]); plt.close(fig)
    # bar return rate
    fig,ax=plt.subplots(figsize=(6,4))
    ret_rate.plot(kind='bar',ax=ax)
    ax.set_title("Return Rate by Category"); ax.set_xlabel("Category"); ax.set_ylabel("Rate")
    plt.xticks(rotation=45); plt.tight_layout()
    imgs2.append("t2_bar.png"); fig.savefig(imgs2[-1]); plt.close(fig)
    # scatter browse vs net
    x,y=df['BrowseTime'],df['NetOrderValue']; m,b=np.polyfit(x,y,1)
    fig,ax=plt.subplots(figsize=(6,4))
    ax.scatter(x,y); ax.plot(x,m*x+b,'--')
    ax.set_title("Browse vs NetOrder"); ax.set_xlabel("Time"); ax.set_ylabel("$")
    plt.tight_layout()
    imgs2.append("t2_scatter.png"); fig.savefig(imgs2[-1]); plt.close(fig)

    doc=SimpleDocTemplate(output_pdf,pagesize=letter)
    st=getSampleStyleSheet()
    elems=[Paragraph("E-commerce Behavior Analysis",st['Title']),Spacer(1,12),
           Paragraph("Key Metrics",st['Heading2'])]
    tbl=[["Metric","Value"]]
    for cat,rv in rev.items():
        tbl.append([f"Revenue–{cat}",f"${rv:,.2f}"])
    tbl+=[["Top Return Cat",top_ret],
          ["Browse-Return Corr",f"{corr_br:.3f}"],
          ["Avg Discount Impact",f"{avg_disc:.2f}"]]
    t=Table(tbl,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightblue),
                           ('GRID',(0,0),(-1,-1),0.5,colors.grey),
                           ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                           ('ALIGN',(0,0),(-1,-1),'CENTER')]))
    elems+=[t,Spacer(1,12),Paragraph("Visualizations",st['Heading2'])]
    for img in imgs2:
        elems+=[Image(img,400,300),Spacer(1,12)]
    analysis=(
      "Over the sample period, net revenue peaked in "+rev.idxmax()+
      f" (${rev.max():,.2f}). {top_ret} had the highest return rate "+
      f"{ret_rate[top_ret]*100:.1f}%. Daily orders showed steady trends, "+
      "purchase efficiency averaged "+
      f"${df['PurchaseEff'].mean():.2f}, and browse-return corr is "+
      f"{corr_br:.3f}. Discounts reduced values by avg "+
      f"${avg_disc:.2f}. Future segmentation by category/time can refine."
    )
    elems+=[Paragraph("Analysis",st['Heading2']),Paragraph(analysis,st['Normal'])]
    doc.build(elems)

    return {'csv':output_csv,'pdf':output_pdf,'images':imgs2}

# --- Task 3: Urban Planning ---
def task3(
    output_csv="sample_resource_10.csv",
    output_pdf="urban_plan_report.pdf",
    seed:int=2
):
    np.random.seed(seed)
    n=10
    zones=[f"Z{str(i).zfill(3)}" for i in range(1,n+1)]
    dates=pd.date_range("2025-01-01",periods=n)
    svs=["Water","Transport","Healthcare","Education","Waste Management"]
    df=pd.DataFrame({
      "ZoneID":zones,
      "Date":dates,
      "ServiceType":np.random.choice(svs,n,replace=True),
      "Budget":np.round(np.random.uniform(10000,100000,n),2),
      "Population":np.random.randint(1000,10000,n),
      "InfrastructureScore":np.round(np.random.uniform(0,1,n),2),
      "Complaints":np.random.randint(0,100,n)
    })
    df.to_csv(output_csv,index=False)

    df=pd.read_csv(output_csv,parse_dates=['Date'])
    df['ServiceType']=df['ServiceType'].str.strip().str.title()
    df.loc[(df['InfrastructureScore']<0)|(df['InfrastructureScore']>1),'InfrastructureScore']=df['InfrastructureScore'].mean()
    df=df[(df['Budget']>=0)&(df['Population']>0)&(df['Complaints']>=0)]

    df['BudgetPerCapita']=df['Budget']/df['Population']
    tb=df.groupby('ServiceType')['Budget'].sum()
    df['CompPerCap']=df['Complaints']/df['Population']
    mostc=df.groupby('ZoneID')['CompPerCap'].mean().idxmax()
    df['Month']=df['Date'].dt.to_period('M').astype(str)
    monthly=df.groupby('Month').agg(
        total_budget=('Budget','sum'),
        total_population=('Population','sum'),
        avg_infra=('InfrastructureScore','mean'),
        total_complaints=('Complaints','sum')
    )
    df['ServiceEff']=df['InfrastructureScore']/df['Budget']*1000
    corr_cp=np.corrcoef(df['InfrastructureScore'],df['Complaints'])[0,1]
    daily=df.groupby('Date')['Budget'].sum().pct_change().fillna(0)

    zone_dict={row.ZoneID:(row.ServiceType,row.BudgetPerCapita,row.ServiceEff)
               for row in df.itertuples()}

    imgs3=[]
    fig,ax=plt.subplots(figsize=(6,4))
    monthly['total_budget'].plot(marker='o',ax=ax)
    ax.set_title("Monthly Total Budget"); ax.set_xlabel("Month"); ax.set_ylabel("$")
    plt.xticks(rotation=45); plt.tight_layout()
    imgs3.append("t3_line.png"); fig.savefig(imgs3[-1]); plt.close(fig)

    fig,ax=plt.subplots(figsize=(6,4))
    df.groupby('ServiceType')['CompPerCap'].mean().plot(kind='bar',ax=ax)
    ax.set_title("Avg Complaints per Capita"); ax.set_xlabel("Service"); ax.set_ylabel("Complaints")
    plt.xticks(rotation=45); plt.tight_layout()
    imgs3.append("t3_bar.png"); fig.savefig(imgs3[-1]); plt.close(fig)

    x,y=df['InfrastructureScore'],df['Complaints']; m,b=np.polyfit(x,y,1)
    fig,ax=plt.subplots(figsize=(6,4))
    ax.scatter(x,y); ax.plot(x,m*x+b,'--')
    ax.set_title("Infra Score vs Complaints"); ax.set_xlabel("Score"); ax.set_ylabel("Complaints")
    plt.tight_layout()
    imgs3.append("t3_scatter.png"); fig.savefig(imgs3[-1]); plt.close(fig)

    doc=SimpleDocTemplate(output_pdf,pagesize=letter)
    st=getSampleStyleSheet()
    elems=[Paragraph("Urban Planning Resource Allocation",st['Title']),Spacer(1,12),
           Paragraph("Key Metrics",st['Heading2'])]
    tbl=[["Metric","Value"]]
    for svc,bud in tb.items():
        tbl.append([f"Total Budget – {svc}",f"${bud:,.2f}"])
    tbl+=[["Most Critical Zone",mostc],
          ["Infra–Complaints Corr",f"{corr_cp:.3f}"],
          ["Budget Trend Day-Over-Day",
           ", ".join(f"{d.strftime('%Y-%m-%d')}: {pct:.1%}"
                    for d,pct in daily.items())]]
    t=Table(tbl,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightblue),
                           ('GRID',(0,0),(-1,-1),0.5,colors.grey),
                           ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                           ('ALIGN',(0,0),(-1,-1),'CENTER')]))
    elems+=[t,Spacer(1,12),Paragraph("Visualizations",st['Heading2'])]
    for img in imgs3:
        elems+=[Image(img,400,300),Spacer(1,12)]
    analysis=(
      "Over this period, budget allocations and complaint patterns varied per service. "
      "Water and Transport received the largest shares. Zone "+mostc+
      " exhibited the highest complaints per capita. The infra-complaint correlation "
      f"is {corr_cp:.3f}, suggesting inverse relation. Daily budget changes highlight "
      "reallocations. Service efficiency averaged "+
      f"{df['ServiceEff'].mean():.3f}. These findings can inform targeted investments "
      "to improve city services and citizen satisfaction."
    )
    elems+=[Paragraph("Analysis",st['Heading2']),Paragraph(analysis,st['Normal'])]
    doc.build(elems)

    return {'csv':output_csv,'pdf':output_pdf,'images':imgs3}

def run_task1():
    res = task1()
    # Build a Markdown summary of the key metrics
    total = res['total_yield']             # pandas Series
    most  = res['most_productive']         # string
    corr  = res['weather_impact']          # float
    teff  = res['temp_effect']             # float or None

    md = "### Task 1 Metrics: Agricultural Yield Analysis\n\n"
    md += "**Total Yield per Crop:**  " + ", ".join(f"{crop}: {yt:.1f} tons" for crop,yt in total.items()) + "\n\n"
    md += f"**Most Productive Crop:** **{most}**\n\n"
    md += f"**Rainfall–Yield Correlation:** `{corr:.3f}`\n\n"
    md += f"**Temperature Effect (>25 °C):** `{teff:.1f}%`" if teff is not None else \
"**Temperature Effect (>25 °C):** `N/A`"

    return md, res['images'], res['csv'], res['pdf']

def run_task2():
    res = task2_v2()
    rev  = res['metrics']['total_revenue']       # dict
    top  = res['metrics']['highest_return_cat']  # string
    corr = res['metrics']['browse_return_corr']  # float
    di   = res['metrics']['avg_discount_impact'] # float

    md = "### Task 2 Metrics: E-commerce Customer Behavior\n\n"
    md += "**Total Net Revenue per Category:**  " +  ", ".join(f"{cat}: ${val:,.2f}" for cat,val in rev.items()) + "\n\n"
    md += f"**Highest Return Category:** **{top}**\n\n"
    md += f"**Browse–Return Correlation:** `{corr:.3f}`\n\n"
    md += f"**Average Discount Impact:** `${di:.2f}`"

    return md, res['images'], res['csv'], res['pdf']

def run_task3():
    res = task3()
    tb    = res['total_budget']           # dict
    crit  = res['most_critical']          # string
    corr  = res['complaint_corr']         # float
    trend = res['budget_trend']           # dict of {date: pct}

    md = "### Task 3 Metrics: Urban Planning Allocation\n\n"
    md += "**Total Budget per Service:**  " +  ", ".join(f"{svc}: ${bd:,.2f}" for svc,bd in tb.items()) + "\n\n"
    md += f"**Most Critical Zone:** **{crit}**\n\n"
    md += f"**Infra–Complaints Correlation:** `{corr:.3f}`\n\n"
    md += "**Budget Trend Day-over-Day:**  " +  ", ".join(f"{d}: {pct:.1%}" for d,pct in trend.items())

    return md, res['images'], res['csv'], res['pdf']
# --- Dashboard ---
def main(task_choice):
    if task_choice == "Task 1":
        res = task1()
    elif task_choice == "Task 2":
        res = task2_v2()
    else:
        res = task3()

    # Prepare outputs
    msg = (
        f"✅ **{task_choice} complete!**\n\n"
        f"- CSV: `{res['csv']}`\n"
        f"- PDF: `{res['pdf']}`"
    )
    return msg, res['images'], res['csv'], res['pdf']

iface = gr.Interface(
    fn=main,
    inputs=gr.Radio(
        choices=["Task 1", "Task 2", "Task 3"],
        label="🔎 **Select an analysis task:**"
    ),
    outputs=[
        gr.Markdown(label="📝 **Status**"),
        gr.Gallery(label="📊 **Charts**", type="filepath"),
        gr.File(label="⬇️ Download CSV"),
        gr.File(label="⬇️ Download PDF")
    ],
    title="📈 Multi-Task Analysis Dashboard",
    description="""
**Welcome!** This dashboard runs three advanced data‐analysis pipelines.
**Steps to use:**
1. **Select** one of the tasks above.
2. **Click** the **Submit** button.
3. **View** the generated charts below.
4. **Download** your CSV and PDF reports using the buttons.

**Tasks Available**
- **Task 1**: Agricultural Yield Analysis
- **Task 2**: E-commerce Customer Behavior Analysis
- **Task 3**: Urban Planning Resource Allocation
""",
    allow_flagging="never"
)

iface.launch(share=True)
