"""
UEBA Threat Intelligence Dashboard — "Fortexa" style
Built on real UNSW_NB15_training-set.parquet data
Team Technologia | SIH 2025

INSTALL:
  pip install streamlit pandas numpy plotly scikit-learn pyarrow

RUN:
  streamlit run ueba_dashboard_v2.py

PUT IN SAME FOLDER:
  UNSW_NB15_training-set.parquet
  baseline_isolation_forest.pkl  (optional — auto-detected)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle, os
from datetime import datetime

st.set_page_config(
    page_title="UEBA · Threat Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── PURPLE SAAS THEME ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

html,body,[class*="css"]{
    background-color:#0e0c16!important;
    color:#e4e1ec!important;
    font-family:'Inter',sans-serif!important;
}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:0.8rem!important; max-width:100% !important;}

/* gradient glow backdrop */
body::before{
    content:'';position:fixed;top:-20%;left:10%;width:60%;height:40%;
    background:radial-gradient(ellipse at center, rgba(168,85,247,0.12) 0%, transparent 70%);
    pointer-events:none;z-index:0;filter:blur(60px);
}

/* sidebar */
[data-testid="stSidebar"]{
    background:#15121f!important;
    border-right:1px solid #2a2438!important;
}
[data-testid="stSidebar"] *{color:#e4e1ec!important;}

/* metric cards */
[data-testid="metric-container"]{
    background:linear-gradient(145deg,#1a1726,#15121f)!important;
    border:1px solid #2a2438!important;
    border-radius:14px!important;
    padding:1rem 1.1rem!important;
    box-shadow:0 4px 20px rgba(0,0,0,0.25)!important;
}
[data-testid="metric-container"] label{
    font-family:'Inter',sans-serif!important;font-size:0.68rem!important;
    color:#9b94b3!important;text-transform:uppercase!important;letter-spacing:0.12em!important;
    font-weight:600!important;
}
[data-testid="stMetricValue"]{
    font-family:'Inter',sans-serif!important;font-size:1.7rem!important;
    font-weight:800!important;color:#ffffff!important;
}
[data-testid="stMetricDelta"]{
    font-family:'JetBrains Mono',monospace!important;font-size:0.7rem!important;
}

/* selects, sliders, multiselect labels */
.stSelectbox label,.stSlider label,.stMultiSelect label,.stRadio label,.stCheckbox label{
    font-family:'Inter',sans-serif!important;font-size:0.75rem!important;
    color:#9b94b3!important;text-transform:uppercase!important;letter-spacing:0.08em!important;
    font-weight:600!important;
}

/* radio nav items */
[data-testid="stSidebar"] [role="radiogroup"] label{
    background:transparent;border-radius:10px;padding:0.5rem 0.8rem;
    margin-bottom:2px;transition:background 0.15s;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{ background:#211d31; }

/* dataframe */
[data-testid="stDataFrame"]{ border:1px solid #2a2438!important; border-radius:12px!important; overflow:hidden;}

::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#15121f;}
::-webkit-scrollbar-thumb{background:#3d3554;border-radius:3px;}

/* Force plotly chart wrappers to be transparent / dark, not white */
[data-testid="stPlotlyChart"],
[data-testid="stPlotlyChart"] > div,
.js-plotly-plot,
.plot-container,
.svg-container {
    background: transparent !important;
}
[data-testid="stVerticalBlock"] { background: transparent !important; }
.main .block-container { background: #0e0c16 !important; }
[data-testid="stAppViewContainer"] { background: #0e0c16 !important; }
[data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }

/* Ensure all generic text stays light on dark */
p, span, div, label, h1, h2, h3, h4, h5, h6 { color: #e4e1ec; }
</style>
""", unsafe_allow_html=True)

# ── HELPERS ──────────────────────────────────────────────────
def card_open(title="", subtitle="", height_px=None):
    h = f"min-height:{height_px}px;" if height_px else ""
    sub = f'<div style="font-size:0.72rem;color:#7c7593;margin-top:2px">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div style="background:linear-gradient(145deg,#1a1726,#15121f);border:1px solid #2a2438;
    border-radius:16px;padding:1.1rem 1.2rem;margin-bottom:0.8rem;{h}
    box-shadow:0 4px 24px rgba(0,0,0,0.22)">
    <div style="font-size:0.92rem;font-weight:700;color:#f4f2f8;letter-spacing:0.01em">{title}</div>
    {sub}
    """, unsafe_allow_html=True)

def card_close():
    st.markdown("</div>", unsafe_allow_html=True)

PLOT = dict(
    paper_bgcolor='#15121f', plot_bgcolor='#15121f',
    font=dict(color='#9b94b3', family='Inter', size=11),
    margin=dict(l=10,r=10,t=10,b=10),
    xaxis=dict(gridcolor='#26213a',showgrid=True,color='#7c7593',zeroline=False,linecolor='#2a2438',showline=False),
    yaxis=dict(gridcolor='#26213a',showgrid=True,color='#7c7593',zeroline=False,linecolor='#2a2438',showline=False),
    legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#9b94b3',size=10)),
)

# Purple/violet palette + accent colors echoing the Fortexa screenshots
PALETTE = {
    'primary':'#a855f7',   # violet
    'primary2':'#7c3aed',  # deep purple
    'pink':'#ec4899',
    'blue':'#3b82f6',
    'cyan':'#22d3ee',
    'amber':'#f59e0b',
    'red':'#ef4444',
    'green':'#22c55e',
    'gray':'#6b7280',
}

ATTACK_COLORS = {
    'Normal':'#22c55e','Generic':'#3b82f6','Exploits':'#ef4444',
    'Fuzzers':'#f59e0b','DoS':'#ec4899','Reconnaissance':'#22d3ee',
    'Analysis':'#a855f7','Backdoor':'#fb923c','Shellcode':'#f472b6','Worms':'#c084fc',
}

MVP = ['dur','sbytes','dbytes','sload','dload']

# ── LOAD DATA ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_parquet('UNSW_NB15_training-set.parquet')
    df['attack_cat'] = df['attack_cat'].astype(str).str.strip()
    df['attack_cat'] = df['attack_cat'].replace({'nan':'Normal','':'Normal'})
    X = df[MVP].fillna(0).values.astype(np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if os.path.exists('baseline_isolation_forest.pkl'):
        with open('baseline_isolation_forest.pkl','rb') as f:
            model = pickle.load(f)
    else:
        model = IsolationForest(contamination=0.01,random_state=42,n_estimators=50)
        model.fit(Xs)
    raw   = model.decision_function(Xs)
    score = (1-(raw-raw.min())/(raw.max()-raw.min()+1e-9))*100
    df['risk_score'] = score.astype(np.float32)
    df['anomaly']    = (model.predict(Xs)==-1).astype(int)
    df['threat_level'] = pd.cut(df['risk_score'],bins=[0,40,60,80,100],
                                 labels=['Low','Medium','High','Critical'],right=True).astype(str)
    return df

with st.spinner("Loading UNSW-NB15 · scoring 175,341 connections..."):
    df = load_data()

model_present = os.path.exists('baseline_isolation_forest.pkl')

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.6rem 0 1rem;display:flex;align-items:center;gap:0.5rem">
        <div style="width:34px;height:34px;border-radius:9px;
        background:linear-gradient(135deg,#a855f7,#3b82f6);
        display:flex;align-items:center;justify-content:center;font-size:1.1rem">🛡️</div>
        <div>
            <div style="font-weight:800;font-size:1.05rem;color:#f4f2f8">UEBA</div>
            <div style="font-size:0.62rem;color:#7c7593;letter-spacing:0.1em">Team Technologia</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", ["📊  Overview","🌐  Threats","🔍  Exposures","📡  Feed","🧠  Model"],
                     label_visibility="collapsed")

    st.markdown('<div style="border-top:1px solid #2a2438;margin:0.8rem 0"></div>', unsafe_allow_html=True)
    threshold = st.slider("Alert Threshold", 0, 100, 50)
    flagged   = int((df['risk_score'] >= threshold).sum())

    mdl_color = "#22c55e" if model_present else "#f59e0b"
    mdl_label = "Loaded" if model_present else "Demo mode"

    st.markdown(f"""
    <div style="background:#1a1726;border:1px solid #2a2438;border-radius:12px;
    padding:0.9rem;font-size:0.75rem;line-height:2">
        <div style="color:#7c7593;font-weight:600;margin-bottom:0.2rem">SYSTEM</div>
        <div>Dataset &nbsp;<span style="float:right;color:#f4f2f8;font-weight:600">UNSW-NB15</span></div>
        <div>Rows &nbsp;&nbsp;&nbsp;&nbsp;<span style="float:right;color:#f4f2f8;font-weight:600">{len(df):,}</span></div>
        <div>Model &nbsp;&nbsp;&nbsp;<span style="float:right;color:{mdl_color};font-weight:600">{mdl_label}</span></div>
        <div>Algorithm <span style="float:right;color:#f4f2f8;font-weight:600">IsoForest</span></div>
        <div style="border-top:1px solid #2a2438;margin-top:0.4rem;padding-top:0.4rem">
        Attacks &nbsp;<span style="float:right;color:#ef4444;font-weight:700">{int(df['label'].sum()):,}</span></div>
        <div>Flagged &nbsp;<span style="float:right;color:#ef4444;font-weight:700">{flagged:,}</span></div>
        <div>Updated &nbsp;<span style="float:right;color:#a855f7;font-weight:600">{datetime.now().strftime('%H:%M:%S')}</span></div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if "Overview" in page:

    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:1rem">
        <div>
            <div style="font-size:1.5rem;font-weight:800;color:#f4f2f8">Threat Overview</div>
            <div style="font-size:0.8rem;color:#7c7593;margin-top:2px">
            UNSW-NB15 · {len(df):,} connections analyzed · {datetime.now().strftime('%b %d, %Y · %H:%M')}</div>
        </div>
        <div style="background:linear-gradient(135deg,#a855f7,#3b82f6);padding:0.5rem 1rem;
        border-radius:10px;font-size:0.78rem;font-weight:700;color:white">
        ● Isolation Forest Active</div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Connections", f"{len(df):,}")
    c2.metric("Attacks", f"{int(df['label'].sum()):,}", f"{df['label'].mean()*100:.1f}% of traffic")
    c3.metric("Normal", f"{int((df['label']==0).sum()):,}", f"{(df['label']==0).mean()*100:.1f}%")
    c4.metric("Critical", f"{int((df['threat_level']=='Critical').sum()):,}", "risk ≥ 80", delta_color="inverse")
    c5.metric("Flagged", f"{flagged:,}", f"threshold {threshold}")
    c6.metric("Avg Risk", f"{df['risk_score'].mean():.1f}", f"max {df['risk_score'].max():.1f}")

    col1, col2 = st.columns([2,1])

    with col1:
        card_open("Open alerts by classification", "Attack categories across the dataset", 360)
        cc = df['attack_cat'].value_counts().reset_index()
        cc.columns = ['category','count']
        fig = go.Figure(go.Bar(
            x=cc['category'], y=cc['count'],
            marker=dict(color=[ATTACK_COLORS.get(c,'#6b7280') for c in cc['category']],
                        line=dict(color='rgba(0,0,0,0)',width=0),
                        cornerradius=6),
        ))
        l = {**PLOT,'height':280}
        l['xaxis'] = dict(**PLOT['xaxis'])
        l['yaxis'] = dict(**PLOT['yaxis'],title='Connections')
        fig.update_layout(**l)
        st.plotly_chart(fig, use_container_width=True)
        card_close()

    with col2:
        card_open("Threat status", "Distribution by risk level", 360)
        lvc = df['threat_level'].value_counts()
        order = ['Low','Medium','High','Critical']
        lvc = lvc.reindex(order).fillna(0)
        lcolors_map = {'Critical':'#ef4444','High':'#f59e0b','Medium':'#a855f7','Low':'#22c55e'}
        fig2 = go.Figure(go.Pie(
            labels=lvc.index, values=lvc.values, hole=0.65,
            marker=dict(colors=[lcolors_map[l] for l in lvc.index],
                        line=dict(color='#15121f',width=3)),
            textfont=dict(family='Inter',size=10,color='white'),
            sort=False
        ))
        fig2.add_annotation(text=f"<b>{len(df):,}</b><br><span style='font-size:10px'>Total</span>",
                            x=0.5,y=0.5,showarrow=False,
                            font=dict(size=15,color='#f4f2f8',family='Inter'))
        l2 = {**PLOT,'height':280}
        l2.pop('xaxis',None); l2.pop('yaxis',None)
        l2['legend'] = dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#9b94b3',size=10),orientation='h',y=-0.1)
        fig2.update_layout(**l2)
        st.plotly_chart(fig2, use_container_width=True)
        card_close()

    col3, col4 = st.columns(2)
    with col3:
        card_open("Risk score distribution", "Normal vs attack · Isolation Forest scores", 320)
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=df[df['label']==0]['risk_score'],name='Normal',
                                    nbinsx=50,marker=dict(color=PALETTE['green'],opacity=0.55)))
        fig3.add_trace(go.Histogram(x=df[df['label']==1]['risk_score'],name='Attack',
                                    nbinsx=50,marker=dict(color=PALETTE['primary'],opacity=0.65)))
        fig3.add_vline(x=threshold,line_dash='dash',line_color='#f4f2f8',
                       annotation_text=f'  threshold',
                       annotation_font=dict(color='#f4f2f8',size=10,family='Inter'))
        l3 = {**PLOT,'height':260,'barmode':'overlay'}
        l3['xaxis'] = dict(**PLOT['xaxis'],title='Risk score')
        l3['yaxis'] = dict(**PLOT['yaxis'],title='Connections')
        fig3.update_layout(**l3)
        st.plotly_chart(fig3, use_container_width=True)
        card_close()

    with col4:
        card_open("Threats coverage by protocol", "Top network protocols", 320)
        pc = df['proto'].value_counts().head(8).reset_index()
        pc.columns = ['proto','count']
        colors8 = ['#a855f7','#3b82f6','#22d3ee','#22c55e','#f59e0b','#ef4444','#ec4899','#c084fc']
        fig4 = go.Figure(go.Bar(
            x=pc['count'], y=pc['proto'].astype(str), orientation='h',
            marker=dict(color=colors8[:len(pc)], cornerradius=6,
                        line=dict(color='rgba(0,0,0,0)',width=0))
        ))
        l4 = {**PLOT,'height':260}
        l4['xaxis'] = dict(**PLOT['xaxis'],title='Connections')
        l4['yaxis'] = dict(**PLOT['yaxis'])
        fig4.update_layout(**l4)
        st.plotly_chart(fig4, use_container_width=True)
        card_close()

    card_open("Feature signals", "MVP features used by the Isolation Forest model")
    descs = {'dur':'Attacks last longer','sbytes':'Attacks send far more data',
             'dbytes':'Attacks receive little back','sload':'Attacks push data faster',
             'dload':'Low return = exfiltration'}
    rows = []
    for f in MVP:
        nm = df[df['label']==0][f].mean()
        am = df[df['label']==1][f].mean()
        rows.append({'Feature':f,'Normal Mean':f"{nm:,.2f}",'Attack Mean':f"{am:,.2f}",
                     'Ratio':f"{am/nm:.2f}×" if nm>0 else 'N/A','Insight':descs[f]})
    st.dataframe(pd.DataFrame(rows).style.set_properties(**{
        'background-color':'#1a1726','color':'#e4e1ec',
        'border-color':'#2a2438','font-family':'Inter','font-size':'13px'
    }), use_container_width=True, hide_index=True)
    card_close()


# ══════════════════════════════════════════════════════════════
# PAGE 2 — THREATS (attack explorer)
# ══════════════════════════════════════════════════════════════
elif "Threats" in page:

    st.markdown(f"""
    <div style="margin-bottom:1rem">
        <div style="font-size:1.5rem;font-weight:800;color:#f4f2f8">Threat Tactics</div>
        <div style="font-size:0.8rem;color:#7c7593;margin-top:2px">
        9 attack categories from UNSW-NB15 · DoS, Exploits, Backdoor, Shellcode, Fuzzers, Generic, Worms, Analysis, Reconnaissance</div>
    </div>
    """, unsafe_allow_html=True)

    cats = sorted(df['attack_cat'].unique().tolist())
    sel  = st.multiselect("Categories", cats, default=cats)
    sub  = df[df['attack_cat'].isin(sel)] if sel else df

    col1, col2 = st.columns([1,1])
    with col1:
        card_open("Threats tactics", "Share of total connections by category", 380)
        cc2 = sub['attack_cat'].value_counts().reset_index()
        cc2.columns = ['cat','count']
        fig = go.Figure(go.Pie(
            labels=cc2['cat'], values=cc2['count'], hole=0.5,
            marker=dict(colors=[ATTACK_COLORS.get(c,'#6b7280') for c in cc2['cat']],
                        line=dict(color='#15121f',width=2)),
            textfont=dict(family='Inter',size=10,color='white'),
            textinfo='label+percent'
        ))
        l = {**PLOT,'height':340}
        l.pop('xaxis',None); l.pop('yaxis',None)
        l['showlegend'] = False
        fig.update_layout(**l)
        st.plotly_chart(fig, use_container_width=True)
        card_close()

    with col2:
        card_open("Risk score by category", "Box plots showing score spread per attack type", 380)
        fig2 = go.Figure()
        for cat in sorted(sub['attack_cat'].unique()):
            d = sub[sub['attack_cat']==cat]['risk_score']
            color = ATTACK_COLORS.get(cat,'#6b7280')
            r,g,b = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
            fig2.add_trace(go.Box(y=d,name=cat,marker_color=color,
                                   line=dict(color=color,width=1.5),
                                   fillcolor=f'rgba({r},{g},{b},0.15)'))
        l2 = {**PLOT,'height':340,'showlegend':False}
        l2['xaxis'] = dict(**PLOT['xaxis'])
        l2['yaxis'] = dict(**PLOT['yaxis'],title='Risk score')
        fig2.update_layout(**l2)
        st.plotly_chart(fig2, use_container_width=True)
        card_close()

    card_open("sbytes vs dbytes", "Each point = one connection · log scale · 3000 sampled", 380)
    sample = sub.sample(min(3000,len(sub)),random_state=42)
    fig3 = go.Figure()
    for cat in sorted(sample['attack_cat'].unique()):
        d = sample[sample['attack_cat']==cat]
        fig3.add_trace(go.Scatter(
            x=np.log1p(d['sbytes']), y=np.log1p(d['dbytes']), mode='markers', name=cat,
            marker=dict(color=ATTACK_COLORS.get(cat,'#6b7280'),
                        size=4 if cat=='Normal' else 6,
                        opacity=0.35 if cat=='Normal' else 0.7)
        ))
    l3 = {**PLOT,'height':350}
    l3['xaxis'] = dict(**PLOT['xaxis'],title='log(sbytes) — data sent out')
    l3['yaxis'] = dict(**PLOT['yaxis'],title='log(dbytes) — data received')
    fig3.update_layout(**l3)
    st.plotly_chart(fig3, use_container_width=True)
    card_close()

    card_open("Security threats table", "Per-category statistics")
    stat = sub.groupby('attack_cat').agg(
        Count=('label','count'),
        Pct=('label',lambda x: f"{len(x)/len(df)*100:.2f}%"),
        Avg_Risk=('risk_score','mean'),Max_Risk=('risk_score','max'),
        Avg_dur=('dur','mean'),Avg_sbytes=('sbytes','mean'),
        Avg_dbytes=('dbytes','mean'),Avg_sload=('sload','mean'),Avg_dload=('dload','mean'),
    ).round(2).reset_index()
    stat.columns=['Category','Count','% Total','Avg Risk','Max Risk',
                  'Avg dur','Avg sbytes','Avg dbytes','Avg sload','Avg dload']
    st.dataframe(stat.style
        .background_gradient(subset=['Avg Risk','Max Risk'],cmap='Purples')
        .format({'Avg Risk':'{:.1f}','Max Risk':'{:.1f}','Avg dur':'{:.3f}',
                 'Avg sbytes':'{:,.0f}','Avg dbytes':'{:,.0f}',
                 'Avg sload':'{:,.0f}','Avg dload':'{:,.0f}'})
        .set_properties(**{'background-color':'#1a1726','color':'#e4e1ec',
                           'border-color':'#2a2438','font-family':'Inter','font-size':'13px'}),
        use_container_width=True, hide_index=True)
    card_close()


# ══════════════════════════════════════════════════════════════
# PAGE 3 — EXPOSURES (feature analysis)
# ══════════════════════════════════════════════════════════════
elif "Exposures" in page:

    st.markdown(f"""
    <div style="margin-bottom:1rem">
        <div style="font-size:1.5rem;font-weight:800;color:#f4f2f8">Feature Exposures</div>
        <div style="font-size:0.8rem;color:#7c7593;margin-top:2px">
        dur · sbytes · dbytes · sload · dload — the 5 features powering the Isolation Forest model</div>
    </div>
    """, unsafe_allow_html=True)

    feat_labels = {'dur':'dur — Duration (seconds)','sbytes':'sbytes — Source Bytes',
                   'dbytes':'dbytes — Destination Bytes','sload':'sload — Source Load',
                   'dload':'dload — Destination Load'}
    feat_sel = st.selectbox("Feature", MVP, format_func=lambda x: feat_labels[x])

    n_s = df[df['label']==0][feat_sel]
    a_s = df[df['label']==1][feat_sel]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Normal mean",  f"{n_s.mean():,.2f}")
    c2.metric("Attack mean",  f"{a_s.mean():,.2f}",
              f"{a_s.mean()/n_s.mean():.1f}× higher" if n_s.mean()>0 else "")
    c3.metric("Normal median",f"{n_s.median():,.2f}")
    c4.metric("Attack median",f"{a_s.median():,.2f}")

    col_a, col_b = st.columns(2)
    with col_a:
        card_open(f"{feat_sel} distribution", "Log scale · normal vs attack", 320)
        clip = float(df[feat_sel].quantile(0.995))
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=np.log1p(n_s.clip(0,clip)),name='Normal',
                                   nbinsx=60,marker=dict(color=PALETTE['green'],opacity=0.55)))
        fig.add_trace(go.Histogram(x=np.log1p(a_s.clip(0,clip)),name='Attack',
                                   nbinsx=60,marker=dict(color=PALETTE['primary'],opacity=0.65)))
        l = {**PLOT,'height':260,'barmode':'overlay'}
        l['xaxis'] = dict(**PLOT['xaxis'],title=f'log(1+{feat_sel})')
        l['yaxis'] = dict(**PLOT['yaxis'],title='Count')
        fig.update_layout(**l)
        st.plotly_chart(fig, use_container_width=True)
        card_close()

    with col_b:
        card_open(f"{feat_sel} vs risk score", "3000 sampled connections", 320)
        samp = df.sample(min(3000,len(df)),random_state=1)
        fig2 = go.Figure()
        for label, color, name in [(0,PALETTE['green'],'Normal'),(1,PALETTE['primary'],'Attack')]:
            s2 = samp[samp['label']==label]
            fig2.add_trace(go.Scatter(x=np.log1p(s2[feat_sel]),y=s2['risk_score'],
                                      mode='markers',name=name,
                                      marker=dict(color=color,size=3,opacity=0.5)))
        fig2.add_hline(y=threshold,line_dash='dash',line_color='#f4f2f8',
                       annotation_text=f'  threshold',
                       annotation_font=dict(color='#f4f2f8',size=10,family='Inter'))
        l2 = {**PLOT,'height':260}
        l2['xaxis'] = dict(**PLOT['xaxis'],title=f'log(1+{feat_sel})')
        l2['yaxis'] = dict(**PLOT['yaxis'],title='Risk score')
        fig2.update_layout(**l2)
        st.plotly_chart(fig2, use_container_width=True)
        card_close()

    card_open(f"{feat_sel} by attack category", "Violin plots per attack type", 340)
    clip2 = float(df[feat_sel].quantile(0.99))
    fig3 = go.Figure()
    for cat in sorted(df['attack_cat'].unique()):
        d = np.log1p(df[df['attack_cat']==cat][feat_sel].clip(0,clip2))
        color = ATTACK_COLORS.get(cat,'#6b7280')
        r,g,b = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
        fig3.add_trace(go.Violin(y=d,name=cat,line_color=color,
                                  fillcolor=f'rgba({r},{g},{b},0.15)',points=False))
    l3 = {**PLOT,'height':300,'showlegend':False}
    l3['xaxis'] = dict(**PLOT['xaxis'])
    l3['yaxis'] = dict(**PLOT['yaxis'],title=f'log(1+{feat_sel})')
    fig3.update_layout(**l3)
    st.plotly_chart(fig3, use_container_width=True)
    card_close()

    card_open("Feature correlation matrix", "All 5 MVP features", 350)
    corr = df[MVP].corr()
    fig4 = go.Figure(go.Heatmap(
        z=corr.values, x=MVP, y=MVP,
        colorscale=[[0,'#ef4444'],[0.5,'#1a1726'],[1,'#a855f7']],
        text=corr.round(2).values, texttemplate='%{text}',
        textfont=dict(size=12,family='Inter',color='white'), showscale=True, zmin=-1,zmax=1
    ))
    l4 = {**PLOT,'height':310}
    l4.pop('xaxis',None); l4.pop('yaxis',None)
    fig4.update_layout(**l4)
    st.plotly_chart(fig4, use_container_width=True)
    card_close()


# ══════════════════════════════════════════════════════════════
# PAGE 4 — FEED (live threat feed)
# ══════════════════════════════════════════════════════════════
elif "Feed" in page:

    st.markdown(f"""
    <div style="margin-bottom:1rem">
        <div style="font-size:1.5rem;font-weight:800;color:#f4f2f8">Live Threat Feed</div>
        <div style="font-size:0.8rem;color:#7c7593;margin-top:2px">
        Real UNSW-NB15 connections · sorted by risk score</div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1: min_score = st.slider("Min risk score", 0, 100, threshold)
    with c2:
        cat_filter = st.multiselect("Attack category",
                                     sorted(df['attack_cat'].unique()),
                                     default=sorted(df['attack_cat'].unique()))
    with c3: show_n = st.selectbox("Show rows",[50,100,200,500],index=1)

    filtered = df[(df['risk_score']>=min_score)&(df['attack_cat'].isin(cat_filter))]\
                 .sort_values('risk_score',ascending=False).head(show_n)

    card_open(f"{len(filtered)} connections", "Sorted by risk score, highest first")

    lc  = {'Critical':'#ef4444','High':'#f59e0b','Medium':'#a855f7','Low':'#22c55e'}

    st.markdown("""<div style="display:grid;grid-template-columns:70px 100px 110px 80px 100px 90px 90px 90px 80px;
    gap:6px;padding:0.5rem 0.6rem;font-size:0.7rem;color:#7c7593;
    text-transform:uppercase;letter-spacing:0.06em;font-weight:700;
    border-bottom:1px solid #2a2438;margin-bottom:0.3rem">
    <div>Proto</div><div>State</div><div>Category</div><div>Dur(s)</div>
    <div>sbytes</div><div>dbytes</div><div>sload</div><div>dload</div><div>Risk</div></div>""",
    unsafe_allow_html=True)

    for _, row in filtered.iterrows():
        lvl   = row['threat_level']
        color = lc.get(lvl,'#22c55e')
        cat   = row['attack_cat']
        cc    = ATTACK_COLORS.get(cat,'#6b7280')
        st.markdown(f"""<div style="display:grid;grid-template-columns:70px 100px 110px 80px 100px 90px 90px 90px 80px;
        gap:6px;padding:0.45rem 0.6rem;font-size:0.78rem;
        background:#1a1726;border-radius:8px;margin:2px 0;align-items:center;
        border-left:3px solid {color}">
        <div style="color:#e4e1ec">{str(row['proto'])[:6]}</div>
        <div style="color:#9b94b3">{str(row['state'])[:8]}</div>
        <div style="color:{cc};font-weight:700">{cat[:12]}</div>
        <div style="color:#e4e1ec">{row['dur']:.3f}</div>
        <div style="color:#e4e1ec">{int(row['sbytes']):,}</div>
        <div style="color:#e4e1ec">{int(row['dbytes']):,}</div>
        <div style="color:#e4e1ec">{row['sload']:.0f}</div>
        <div style="color:#e4e1ec">{row['dload']:.0f}</div>
        <div style="color:{color};font-weight:800">{row['risk_score']:.0f}</div>
        </div>""", unsafe_allow_html=True)
    card_close()


# ══════════════════════════════════════════════════════════════
# PAGE 5 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════
elif "Model" in page:

    st.markdown(f"""
    <div style="margin-bottom:1rem">
        <div style="font-size:1.5rem;font-weight:800;color:#f4f2f8">Model Insights</div>
        <div style="font-size:0.8rem;color:#7c7593;margin-top:2px">
        Isolation Forest · contamination=0.01 · random_state=42</div>
    </div>
    """, unsafe_allow_html=True)

    y_true = df['label'].values
    y_pred = df['anomaly'].values
    tn = int(((y_pred==0)&(y_true==0)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    tp = int(((y_pred==1)&(y_true==1)).sum())
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Precision",  f"{prec*100:.1f}%", "of flagged = real attacks")
    c2.metric("Recall",     f"{rec*100:.1f}%",  "of attacks caught")
    c3.metric("F1 Score",   f"{f1*100:.1f}%",   "harmonic mean")
    c4.metric("False alarms",f"{fp:,}",         f"{fp/len(df)*100:.1f}% of traffic", delta_color="inverse")

    col1, col2 = st.columns(2)
    with col1:
        card_open("Confusion matrix", "Predicted vs actual · real UNSW-NB15 labels", 320)
        cm_vals = [[tn,fp],[fn,tp]]
        cm_text = [[f"TN  {tn:,}",f"FP  {fp:,}"],[f"FN  {fn:,}",f"TP  {tp:,}"]]
        fig = go.Figure(go.Heatmap(
            z=cm_vals,
            x=['Pred: Normal','Pred: Attack'],
            y=['Act: Normal', 'Act: Attack'],
            colorscale=[[0,'#1a1726'],[1,'#a855f7']],
            text=cm_text, texttemplate='%{text}',
            textfont=dict(size=14,family='Inter',color='white'),
            showscale=False
        ))
        l = {**PLOT,'height':260}
        l.pop('xaxis',None); l.pop('yaxis',None)
        l['xaxis'] = dict(color='#e4e1ec',tickfont=dict(family='Inter',size=11))
        l['yaxis'] = dict(color='#e4e1ec',tickfont=dict(family='Inter',size=11))
        fig.update_layout(**l)
        st.plotly_chart(fig, use_container_width=True)
        card_close()

    with col2:
        card_open("Risk score CDF", "Cumulative distribution · normal vs attack", 320)
        n_scores = np.sort(df[df['label']==0]['risk_score'].values)
        a_scores = np.sort(df[df['label']==1]['risk_score'].values)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=n_scores,y=np.linspace(0,1,len(n_scores)),
                                   name='Normal',line=dict(color=PALETTE['green'],width=2)))
        fig2.add_trace(go.Scatter(x=a_scores,y=np.linspace(0,1,len(a_scores)),
                                   name='Attack',line=dict(color=PALETTE['primary'],width=2)))
        fig2.add_vline(x=threshold,line_dash='dash',line_color='#f4f2f8',
                       annotation_text=f'  threshold',
                       annotation_font=dict(color='#f4f2f8',size=10,family='Inter'))
        l2 = {**PLOT,'height':260}
        l2['xaxis'] = dict(**PLOT['xaxis'],title='Risk score')
        l2['yaxis'] = dict(**PLOT['yaxis'],title='Cumulative fraction')
        fig2.update_layout(**l2)
        st.plotly_chart(fig2, use_container_width=True)
        card_close()

    card_open("Manual connection scorer", "Enter values to get an instant risk score")
    c1,c2,c3,c4,c5 = st.columns(5)
    i_dur    = c1.number_input("dur",    0.0, 10000.0, 0.12,   step=0.01)
    i_sbytes = c2.number_input("sbytes", 0,   10000000, 258,   step=100)
    i_dbytes = c3.number_input("dbytes", 0,   10000000, 172,   step=100)
    i_sload  = c4.number_input("sload",  0.0, 5000000.0,14158.,step=100.)
    i_dload  = c5.number_input("dload",  0.0, 5000000.0, 8495.,step=100.)

    if st.button("Analyze connection", use_container_width=True):
        score  = 0; reasons = []
        if i_dur > 10:       score += 20; reasons.append(f"long duration: {i_dur:.2f}s")
        if i_sbytes > 50000: score += 30; reasons.append(f"high sbytes: {i_sbytes:,}")
        if i_dbytes < 500:   score += 20; reasons.append(f"low dbytes: {i_dbytes}")
        if i_sload > 50000:  score += 15; reasons.append(f"high sload: {i_sload:.0f}")
        if i_dload < 1000:   score += 10; reasons.append(f"low dload: {i_dload:.0f}")
        score = min(100, score)
        lvl   = "Critical" if score>=80 else "High" if score>=60 else "Medium" if score>=40 else "Low"
        color = {'Critical':'#ef4444','High':'#f59e0b','Medium':'#a855f7','Low':'#22c55e'}[lvl]
        st.markdown(f"""<div style="background:#1a1726;border:1px solid {color};border-radius:14px;
        padding:1.2rem 1.5rem;margin-top:0.8rem;display:flex;justify-content:space-between;align-items:center;">
        <div><div style="font-size:1.4rem;font-weight:800;color:{color}">{lvl} risk</div>
        <div style="font-size:0.8rem;color:#9b94b3;margin-top:0.4rem">
        {'<br>'.join([f'• {r}' for r in reasons]) if reasons else '• All features within normal range'}
        </div></div>
        <div style="font-size:3rem;font-weight:900;color:{color}">{score}</div></div>""",
        unsafe_allow_html=True)
    card_close()