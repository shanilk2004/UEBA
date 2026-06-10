

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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{background-color:#020b08!important;color:#a0ffc8!important;font-family:'Exo 2',sans-serif!important;}
body::before{content:'';position:fixed;top:0;left:0;right:0;bottom:0;
background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,255,100,0.012) 2px,rgba(0,255,100,0.012) 4px);
pointer-events:none;z-index:9999;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:0.6rem!important;}
[data-testid="stSidebar"]{background:#010a06!important;border-right:1px solid #0d3320!important;}
[data-testid="stSidebar"] *{color:#a0ffc8!important;}
[data-testid="metric-container"]{background:#010f07!important;border:1px solid #0d4025!important;
border-top:2px solid #00ff64!important;border-radius:4px!important;box-shadow:0 0 18px rgba(0,255,100,0.06)!important;}
[data-testid="metric-container"] label{font-family:'Share Tech Mono',monospace!important;font-size:0.62rem!important;
color:#3a8c58!important;text-transform:uppercase!important;letter-spacing:0.18em!important;}
[data-testid="stMetricValue"]{font-family:'Orbitron',monospace!important;font-size:1.55rem!important;
font-weight:700!important;color:#00ff64!important;text-shadow:0 0 18px rgba(0,255,100,0.5)!important;}
[data-testid="stMetricDelta"]{font-family:'Share Tech Mono',monospace!important;font-size:0.68rem!important;}
.stSelectbox label,.stSlider label,.stMultiSelect label,.stRadio label,.stCheckbox label{
font-family:'Share Tech Mono',monospace!important;font-size:0.72rem!important;
color:#3a8c58!important;text-transform:uppercase!important;letter-spacing:0.1em!important;}
::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-track{background:#010a06;}
::-webkit-scrollbar-thumb{background:#0d4025;border-radius:2px;}
</style>
""", unsafe_allow_html=True)

# ── HELPERS ──────────────────────────────────────────────────
def sec(title, sub=""):
    s = f'<span style="color:#1a5c30;font-size:0.68rem;margin-left:0.7rem;font-family:Share Tech Mono">{sub}</span>' if sub else ""
    st.markdown(f'<div style="margin:0.7rem 0 0.4rem;border-bottom:1px solid #0d3320;padding-bottom:0.25rem;">'
                f'<span style="font-family:Orbitron;font-size:0.72rem;color:#00ff64;text-transform:uppercase;'
                f'letter-spacing:0.18em;text-shadow:0 0 8px rgba(0,255,100,0.35)">{title}</span>{s}</div>',
                unsafe_allow_html=True)

PLOT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#3a8c58', family='Share Tech Mono', size=10),
    margin=dict(l=10,r=10,t=30,b=10),
    xaxis=dict(gridcolor='#0a2818',showgrid=True,color='#3a8c58',zeroline=False,linecolor='#0d3320',showline=True),
    yaxis=dict(gridcolor='#0a2818',showgrid=True,color='#3a8c58',zeroline=False,linecolor='#0d3320',showline=True),
    legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#3a8c58',size=9)),
)

ATTACK_COLORS = {
    'Normal':'#00ff64','Generic':'#4488ff','Exploits':'#ff4d4d',
    'Fuzzers':'#ffcc00','DoS':'#ff2020','Reconnaissance':'#44ffcc',
    'Analysis':'#aa44ff','Backdoor':'#ff6600','Shellcode':'#ff88aa','Worms':'#ff00ff',
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
                                 labels=['LOW','MEDIUM','HIGH','CRITICAL'],right=True).astype(str)
    return df

with st.spinner("🔍 Loading UNSW-NB15 · scoring 175,341 connections..."):
    df = load_data()

model_present = os.path.exists('baseline_isolation_forest.pkl')

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="padding:0.4rem 0 0.8rem">
    <div style="font-family:Orbitron;font-size:1.05rem;font-weight:900;color:#00ff64;
    text-shadow:0 0 14px rgba(0,255,100,0.6);letter-spacing:0.08em">⚡ UEBA</div>
    <div style="font-family:Share Tech Mono;font-size:0.6rem;color:#3a8c58;letter-spacing:0.12em;margin-top:0.1rem">
    NETWORK THREAT INTELLIGENCE</div>
    <div style="font-family:Share Tech Mono;font-size:0.55rem;color:#1a5c30;margin-top:0.1rem">
    Team Technologia · SIH 2025</div></div>
    <div style="border-top:1px solid #0d3320;margin-bottom:0.7rem"></div>""", unsafe_allow_html=True)

    page = st.radio("", ["⬡  OVERVIEW","◈  ATTACK EXPLORER","◉  FEATURE ANALYSIS",
                          "⬢  THREAT FEED","△  MODEL INSIGHTS"], label_visibility="collapsed")

    st.markdown('<div style="border-top:1px solid #0d3320;margin:0.7rem 0"></div>', unsafe_allow_html=True)
    threshold = st.slider("ALERT THRESHOLD", 0, 100, 50)
    flagged   = int((df['risk_score'] >= threshold).sum())
    mdl_color = "#00ff64" if model_present else "#ffcc00"
    mdl_label = "LOADED ✓" if model_present else "DEMO MODE"

    st.markdown(f"""<div style="background:#010f07;border:1px solid #0d3320;border-radius:4px;
    padding:0.75rem;font-family:Share Tech Mono;font-size:0.66rem;line-height:1.9">
    <div style="color:#3a8c58;margin-bottom:0.2rem">── SYSTEM STATUS ──</div>
    <div>DATASET &nbsp;: <span style="color:#00ff64">UNSW-NB15</span></div>
    <div>ROWS &nbsp;&nbsp;&nbsp;&nbsp;: <span style="color:#00ff64">{len(df):,}</span></div>
    <div>FEATURES : <span style="color:#00ff64">36 COLUMNS</span></div>
    <div>MODEL &nbsp;&nbsp;&nbsp;: <span style="color:{mdl_color}">{mdl_label}</span></div>
    <div>ALGO &nbsp;&nbsp;&nbsp;&nbsp;: <span style="color:#00ff64">ISOLATION FOREST</span></div>
    <div>CONTAM &nbsp;&nbsp;: <span style="color:#00ff64">0.01</span></div>
    <div style="border-top:1px solid #0d3320;margin-top:0.3rem;padding-top:0.3rem">
    ATTACKS &nbsp;: <span style="color:#ff4d4d">{int(df['label'].sum()):,} ({df['label'].mean()*100:.1f}%)</span></div>
    <div>FLAGGED &nbsp;: <span style="color:#ff4d4d">{flagged:,}</span></div>
    <div>TIME &nbsp;&nbsp;&nbsp;&nbsp;: <span style="color:#00ff64">{datetime.now().strftime('%H:%M:%S')}</span></div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if "OVERVIEW" in page:
    st.markdown(f"""<div style="background:linear-gradient(90deg,#010f07,#020b08);border:1px solid #0d3320;
    border-left:3px solid #00ff64;border-radius:4px;padding:0.75rem 1.2rem;margin-bottom:0.8rem">
    <div style="font-family:Orbitron;font-size:1.35rem;font-weight:900;color:#00ff64;
    text-shadow:0 0 20px rgba(0,255,100,0.45)">NETWORK THREAT OVERVIEW</div>
    <div style="font-family:Share Tech Mono;font-size:0.66rem;color:#3a8c58;margin-top:0.15rem">
    UNSW-NB15 · {len(df):,} CONNECTIONS · ISOLATION FOREST · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div></div>""", unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("TOTAL CONNECTIONS", f"{len(df):,}")
    c2.metric("ATTACKS",   f"{int(df['label'].sum()):,}",  f"{df['label'].mean()*100:.1f}% of traffic")
    c3.metric("NORMAL",    f"{int((df['label']==0).sum()):,}", f"{(df['label']==0).mean()*100:.1f}%")
    c4.metric("CRITICAL",  f"{int((df['threat_level']=='CRITICAL').sum()):,}", "risk ≥ 80")
    c5.metric("FLAGGED",   f"{flagged:,}", f"threshold {threshold}")
    c6.metric("AVG RISK",  f"{df['risk_score'].mean():.1f}", f"max {df['risk_score'].max():.1f}")

    st.markdown("")
    col1, col2 = st.columns([3,1])

    with col1:
        sec("ATTACK CATEGORY DISTRIBUTION", f"real UNSW-NB15 labels · {df['attack_cat'].nunique()} categories")
        cc = df['attack_cat'].value_counts().reset_index()
        cc.columns = ['category','count']
        fig = go.Figure(go.Bar(
            x=cc['count'], y=cc['category'], orientation='h',
            text=cc['count'].apply(lambda x: f"{x:,}"),
            textposition='inside',
            textfont=dict(family='Share Tech Mono',size=10,color='#020b08'),
            marker=dict(color=[ATTACK_COLORS.get(c,'#3a8c58') for c in cc['category']],
                        line=dict(color='rgba(0,0,0,0)',width=0))
        ))
        l = {**PLOT,'height':300}
        l.pop('xaxis',None); l.pop('yaxis',None)
        l['xaxis'] = dict(gridcolor='#0a2818',color='#3a8c58',showgrid=True)
        l['yaxis'] = dict(color='#a0ffc8',showgrid=False,tickfont=dict(family='Share Tech Mono',size=10))
        fig.update_layout(**l)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("THREAT LEVELS")
        lvc = df['threat_level'].value_counts()
        lcolors_map = {'CRITICAL':'#ff2020','HIGH':'#ff6600','MEDIUM':'#ffcc00','LOW':'#00ff64'}
        fig2 = go.Figure(go.Pie(
            labels=lvc.index, values=lvc.values, hole=0.62,
            marker=dict(colors=[lcolors_map.get(l,'#3a8c58') for l in lvc.index],
                        line=dict(color='#020b08',width=2)),
            textfont=dict(family='Share Tech Mono',size=9,color='#020b08')
        ))
        fig2.add_annotation(text=f"<b>{len(df):,}</b><br>TOTAL",
                            x=0.5,y=0.5,showarrow=False,
                            font=dict(size=11,color='#00ff64',family='Orbitron'))
        l2 = {**PLOT,'height':300}
        l2.pop('xaxis',None); l2.pop('yaxis',None)
        l2['margin'] = dict(l=0,r=0,t=30,b=0)
        fig2.update_layout(**l2)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        sec("RISK SCORE DISTRIBUTION", "normal vs attack · isolation forest scores")
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=df[df['label']==0]['risk_score'],name='Normal',
                                    nbinsx=60,marker=dict(color='#00ff64',opacity=0.55)))
        fig3.add_trace(go.Histogram(x=df[df['label']==1]['risk_score'],name='Attack',
                                    nbinsx=60,marker=dict(color='#ff4d4d',opacity=0.75)))
        fig3.add_vline(x=threshold,line_dash='dash',line_color='#fff',
                       annotation_text=f'  T={threshold}',
                       annotation_font=dict(color='#fff',size=9,family='Share Tech Mono'))
        l3 = {**PLOT,'height':260,'barmode':'overlay'}
        l3['xaxis'] = dict(**PLOT['xaxis'],title='Risk Score (0–100)')
        l3['yaxis'] = dict(**PLOT['yaxis'],title='Count')
        fig3.update_layout(**l3)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        sec("PROTOCOL BREAKDOWN", "top 8 network protocols")
        pc = df['proto'].value_counts().head(8).reset_index()
        pc.columns = ['proto','count']
        fig4 = go.Figure(go.Bar(
            x=pc['proto'].astype(str), y=pc['count'],
            text=pc['count'].apply(lambda x: f"{x:,}"), textposition='outside',
            textfont=dict(family='Share Tech Mono',size=9,color='#3a8c58'),
            marker=dict(color=['#00ff64','#4488ff','#ffcc00','#ff4d4d','#44ffcc','#aa44ff','#ff6600','#ff88aa'],
                        line=dict(color='rgba(0,0,0,0)',width=0))
        ))
        l4 = {**PLOT,'height':260}
        l4['xaxis'] = dict(**PLOT['xaxis'],title='Protocol')
        l4['yaxis'] = dict(**PLOT['yaxis'],title='Count')
        fig4.update_layout(**l4)
        st.plotly_chart(fig4, use_container_width=True)

    sec("MVP FEATURE AVERAGES: NORMAL vs ATTACK", "5 features used by isolation forest")
    rows = []
    descs = {'dur':'Attacks last longer','sbytes':'Attacks send far more data',
             'dbytes':'Attacks receive little back','sload':'Attacks push data faster',
             'dload':'Low return = exfiltration'}
    for f in MVP:
        nm = df[df['label']==0][f].mean()
        am = df[df['label']==1][f].mean()
        rows.append({'Feature':f.upper(),'Normal Mean':f"{nm:,.2f}",'Attack Mean':f"{am:,.2f}",
                     'Ratio':f"{am/nm:.2f}×" if nm>0 else 'N/A','Insight':descs[f]})
    st.dataframe(pd.DataFrame(rows).style.set_properties(**{
        'background-color':'#010f07','color':'#a0ffc8',
        'border-color':'#0d3320','font-family':'Share Tech Mono','font-size':'12px'
    }), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — ATTACK EXPLORER
# ══════════════════════════════════════════════════════════════
elif "EXPLORER" in page:
    st.markdown("""<div style="border:1px solid #0d3320;border-left:3px solid #ff4d4d;border-radius:4px;
    padding:0.75rem 1.2rem;margin-bottom:0.8rem;background:#0f0101">
    <div style="font-family:Orbitron;font-size:1.3rem;font-weight:900;color:#ff4d4d;
    text-shadow:0 0 14px rgba(255,77,77,0.4)">◈ ATTACK CATEGORY EXPLORER</div>
    <div style="font-family:Share Tech Mono;font-size:0.66rem;color:#8c3a3a">
    9 ATTACK TYPES · DoS · Exploits · Backdoor · Shellcode · Fuzzers · Generic · Worms · Analysis · Reconnaisance
    </div></div>""", unsafe_allow_html=True)

    cats = sorted(df['attack_cat'].unique().tolist())
    sel  = st.multiselect("SELECT CATEGORIES", cats, default=cats)
    sub  = df[df['attack_cat'].isin(sel)] if sel else df

    col1, col2 = st.columns(2)
    with col1:
        sec("CONNECTIONS PER CATEGORY")
        cc2 = sub['attack_cat'].value_counts().reset_index()
        cc2.columns = ['cat','count']
        fig = go.Figure(go.Pie(
            labels=cc2['cat'], values=cc2['count'], hole=0.45,
            marker=dict(colors=[ATTACK_COLORS.get(c,'#3a8c58') for c in cc2['cat']],
                        line=dict(color='#020b08',width=2)),
            textfont=dict(family='Share Tech Mono',size=9,color='white'),
            textinfo='label+percent'
        ))
        l = {**PLOT,'height':320}
        l.pop('xaxis',None); l.pop('yaxis',None)
        l['margin'] = dict(l=0,r=0,t=30,b=0)
        fig.update_layout(**l)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("RISK SCORE BY CATEGORY", "box plots showing score spread per attack type")
        fig2 = go.Figure()
        for cat in sorted(sub['attack_cat'].unique()):
            d = sub[sub['attack_cat']==cat]['risk_score']
            color = ATTACK_COLORS.get(cat,'#3a8c58')
            r,g,b = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
            fig2.add_trace(go.Box(y=d,name=cat,marker_color=color,
                                   line=dict(color=color,width=1.5),
                                   fillcolor=f'rgba({r},{g},{b},0.12)'))
        l2 = {**PLOT,'height':320,'showlegend':False}
        l2['xaxis'] = dict(gridcolor='#0a2818',color='#a0ffc8',
                           tickfont=dict(family='Share Tech Mono',size=9))
        l2['yaxis'] = dict(gridcolor='#0a2818',color='#3a8c58',title='Risk Score')
        fig2.update_layout(**l2)
        st.plotly_chart(fig2, use_container_width=True)

    sec("SBYTES vs DBYTES", "log scale · coloured by attack category · 3000 sampled connections")
    sample = sub.sample(min(3000,len(sub)),random_state=42)
    fig3 = go.Figure()
    for cat in sorted(sample['attack_cat'].unique()):
        d = sample[sample['attack_cat']==cat]
        fig3.add_trace(go.Scatter(
            x=np.log1p(d['sbytes']), y=np.log1p(d['dbytes']), mode='markers', name=cat,
            marker=dict(color=ATTACK_COLORS.get(cat,'#3a8c58'),
                        size=3 if cat=='Normal' else 5,
                        opacity=0.35 if cat=='Normal' else 0.7)
        ))
    l3 = {**PLOT,'height':320}
    l3['xaxis'] = dict(**PLOT['xaxis'],title='log(sbytes) — data sent out')
    l3['yaxis'] = dict(**PLOT['yaxis'],title='log(dbytes) — data received')
    fig3.update_layout(**l3)
    st.plotly_chart(fig3, use_container_width=True)

    sec("PER-CATEGORY STATISTICS TABLE")
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
        .background_gradient(subset=['Avg Risk','Max Risk'],cmap='RdYlGn_r')
        .format({'Avg Risk':'{:.1f}','Max Risk':'{:.1f}','Avg dur':'{:.3f}',
                 'Avg sbytes':'{:,.0f}','Avg dbytes':'{:,.0f}',
                 'Avg sload':'{:,.0f}','Avg dload':'{:,.0f}'})
        .set_properties(**{'background-color':'#010f07','color':'#a0ffc8',
                           'border-color':'#0d3320','font-family':'Share Tech Mono','font-size':'12px'}),
        use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════
elif "FEATURE" in page:
    st.markdown("""<div style="border:1px solid #0d3320;border-left:3px solid #ffcc00;border-radius:4px;
    padding:0.75rem 1.2rem;margin-bottom:0.8rem;background:#0f0e01">
    <div style="font-family:Orbitron;font-size:1.3rem;font-weight:900;color:#ffcc00;
    text-shadow:0 0 14px rgba(255,204,0,0.4)">◉ FEATURE ANALYSIS</div>
    <div style="font-family:Share Tech Mono;font-size:0.66rem;color:#8c7a3a">
    dur · sbytes · dbytes · sload · dload — the 5 features powering isolation forest</div></div>""",
    unsafe_allow_html=True)

    feat_labels = {'dur':'dur — Duration (s)','sbytes':'sbytes — Src Bytes',
                   'dbytes':'dbytes — Dst Bytes','sload':'sload — Src Load','dload':'dload — Dst Load'}
    feat_sel = st.selectbox("SELECT FEATURE", MVP, format_func=lambda x: feat_labels[x])

    n_s = df[df['label']==0][feat_sel]
    a_s = df[df['label']==1][feat_sel]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("NORMAL MEAN",  f"{n_s.mean():,.2f}")
    c2.metric("ATTACK MEAN",  f"{a_s.mean():,.2f}",
              f"{a_s.mean()/n_s.mean():.1f}× higher" if n_s.mean()>0 else "")
    c3.metric("NORMAL MEDIAN",f"{n_s.median():,.2f}")
    c4.metric("ATTACK MEDIAN",f"{a_s.median():,.2f}")
    st.markdown("")

    col_a, col_b = st.columns(2)
    with col_a:
        sec(f"{feat_sel.upper()} DISTRIBUTION", "log scale · normal vs attack")
        clip = float(df[feat_sel].quantile(0.995))
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=np.log1p(n_s.clip(0,clip)),name='Normal',
                                   nbinsx=70,marker=dict(color='#00ff64',opacity=0.55)))
        fig.add_trace(go.Histogram(x=np.log1p(a_s.clip(0,clip)),name='Attack',
                                   nbinsx=70,marker=dict(color='#ff4d4d',opacity=0.75)))
        l = {**PLOT,'height':270,'barmode':'overlay'}
        l['xaxis'] = dict(**PLOT['xaxis'],title=f'log(1+{feat_sel})')
        l['yaxis'] = dict(**PLOT['yaxis'],title='Count')
        fig.update_layout(**l)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        sec(f"{feat_sel.upper()} vs RISK SCORE", "3000 sampled connections")
        samp = df.sample(min(3000,len(df)),random_state=1)
        fig2 = go.Figure()
        for label, color, name in [(0,'#00ff64','Normal'),(1,'#ff4d4d','Attack')]:
            s2 = samp[samp['label']==label]
            fig2.add_trace(go.Scatter(x=np.log1p(s2[feat_sel]),y=s2['risk_score'],
                                      mode='markers',name=name,
                                      marker=dict(color=color,size=3,opacity=0.5)))
        fig2.add_hline(y=threshold,line_dash='dash',line_color='#fff',
                       annotation_text=f'  threshold={threshold}',
                       annotation_font=dict(color='#fff',size=9,family='Share Tech Mono'))
        l2 = {**PLOT,'height':270}
        l2['xaxis'] = dict(**PLOT['xaxis'],title=f'log(1+{feat_sel})')
        l2['yaxis'] = dict(**PLOT['yaxis'],title='Risk Score')
        fig2.update_layout(**l2)
        st.plotly_chart(fig2, use_container_width=True)

    sec(f"{feat_sel.upper()} BY ATTACK CATEGORY", "violin plots per attack type")
    clip2 = float(df[feat_sel].quantile(0.99))
    fig3 = go.Figure()
    for cat in sorted(df['attack_cat'].unique()):
        d = np.log1p(df[df['attack_cat']==cat][feat_sel].clip(0,clip2))
        color = ATTACK_COLORS.get(cat,'#3a8c58')
        r,g,b = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
        fig3.add_trace(go.Violin(y=d,name=cat,line_color=color,
                                  fillcolor=f'rgba({r},{g},{b},0.15)',points=False))
    l3 = {**PLOT,'height':290,'showlegend':False}
    l3['xaxis'] = dict(gridcolor='#0a2818',color='#a0ffc8',tickfont=dict(family='Share Tech Mono',size=9))
    l3['yaxis'] = dict(gridcolor='#0a2818',color='#3a8c58',title=f'log(1+{feat_sel})')
    fig3.update_layout(**l3)
    st.plotly_chart(fig3, use_container_width=True)

    sec("FEATURE CORRELATION MATRIX", "all 5 MVP features")
    corr = df[MVP].corr()
    fig4 = go.Figure(go.Heatmap(
        z=corr.values, x=[f.upper() for f in MVP], y=[f.upper() for f in MVP],
        colorscale=[[0,'#ff2020'],[0.5,'#010f07'],[1,'#00ff64']],
        text=corr.round(2).values, texttemplate='%{text}',
        textfont=dict(size=12,family='Share Tech Mono'), showscale=True, zmin=-1,zmax=1
    ))
    l4 = {**PLOT,'height':310}
    l4.pop('xaxis',None); l4.pop('yaxis',None)
    fig4.update_layout(**l4)
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — THREAT FEED
# ══════════════════════════════════════════════════════════════
elif "FEED" in page:
    st.markdown("""<div style="border:1px solid #0d3320;border-left:3px solid #ff2020;border-radius:4px;
    padding:0.75rem 1.2rem;margin-bottom:0.8rem;background:#0f0101">
    <div style="font-family:Orbitron;font-size:1.3rem;font-weight:900;color:#ff2020;
    text-shadow:0 0 14px rgba(255,32,32,0.45)">⬢ THREAT FEED</div>
    <div style="font-family:Share Tech Mono;font-size:0.66rem;color:#8c1a1a">
    REAL UNSW-NB15 CONNECTIONS SORTED BY RISK SCORE</div></div>""", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1: min_score = st.slider("MIN RISK SCORE", 0, 100, threshold)
    with c2:
        cat_filter = st.multiselect("ATTACK CATEGORY",
                                     sorted(df['attack_cat'].unique()),
                                     default=sorted(df['attack_cat'].unique()))
    with c3: show_n = st.selectbox("SHOW ROWS",[50,100,200,500],index=1)

    filtered = df[(df['risk_score']>=min_score)&(df['attack_cat'].isin(cat_filter))]\
                 .sort_values('risk_score',ascending=False).head(show_n)
    sec(f"{len(filtered)} CONNECTIONS · SORTED BY RISK")

    lc  = {'CRITICAL':'#ff2020','HIGH':'#ff6600','MEDIUM':'#ffcc00','LOW':'#00ff64'}
    lbg = {'CRITICAL':'#1a0000','HIGH':'#1a0800','MEDIUM':'#1a1200','LOW':'#001a08'}

    st.markdown("""<div style="display:grid;grid-template-columns:65px 100px 110px 85px 110px 90px 85px 85px 80px;
    gap:3px;padding:0.3rem 0.5rem;font-family:Share Tech Mono;font-size:0.6rem;color:#3a8c58;
    text-transform:uppercase;letter-spacing:0.05em;border-bottom:1px solid #0d3320;margin-bottom:0.2rem">
    <div>PROTO</div><div>STATE</div><div>ATTACK CAT</div><div>DUR(s)</div>
    <div>SBYTES</div><div>DBYTES</div><div>SLOAD</div><div>DLOAD</div><div>RISK</div></div>""",
    unsafe_allow_html=True)

    for _, row in filtered.iterrows():
        lvl   = row['threat_level']
        color = lc.get(lvl,'#00ff64')
        bg    = lbg.get(lvl,'#010f07')
        cat   = row['attack_cat']
        cc    = ATTACK_COLORS.get(cat,'#3a8c58')
        st.markdown(f"""<div style="display:grid;grid-template-columns:65px 100px 110px 85px 110px 90px 85px 85px 80px;
        gap:3px;padding:0.28rem 0.5rem;font-family:Share Tech Mono;font-size:0.65rem;
        background:{bg};border:1px solid {color}25;border-left:2px solid {color};
        border-radius:3px;margin:1px 0;align-items:center;">
        <div style="color:#a0ffc8">{str(row['proto'])[:6]}</div>
        <div style="color:#3a8c58">{str(row['state'])[:8]}</div>
        <div style="color:{cc};font-weight:700">{cat[:12]}</div>
        <div style="color:#a0ffc8">{row['dur']:.3f}</div>
        <div style="color:#a0ffc8">{int(row['sbytes']):,}</div>
        <div style="color:#a0ffc8">{int(row['dbytes']):,}</div>
        <div style="color:#a0ffc8">{row['sload']:.0f}</div>
        <div style="color:#a0ffc8">{row['dload']:.0f}</div>
        <div style="color:{color};font-family:Orbitron;font-weight:700;font-size:0.78rem">{row['risk_score']:.0f}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 5 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════
elif "MODEL" in page:
    st.markdown("""<div style="border:1px solid #0d3320;border-left:3px solid #44ffcc;border-radius:4px;
    padding:0.75rem 1.2rem;margin-bottom:0.8rem;background:#010f0e">
    <div style="font-family:Orbitron;font-size:1.3rem;font-weight:900;color:#44ffcc;
    text-shadow:0 0 14px rgba(68,255,204,0.4)">△ MODEL INSIGHTS</div>
    <div style="font-family:Share Tech Mono;font-size:0.66rem;color:#1a8c7a">
    ISOLATION FOREST · contamination=0.01 · random_state=42 · n_estimators=100</div></div>""",
    unsafe_allow_html=True)

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
    c1.metric("PRECISION",  f"{prec*100:.1f}%", "of flagged = real attacks")
    c2.metric("RECALL",     f"{rec*100:.1f}%",  "of attacks caught")
    c3.metric("F1 SCORE",   f"{f1*100:.1f}%",   "harmonic mean P+R")
    c4.metric("FALSE ALARMS",f"{fp:,}",         f"{fp/len(df)*100:.1f}% of traffic")

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        sec("CONFUSION MATRIX", "predicted vs actual · real UNSW-NB15 labels")
        cm_vals = [[tn,fp],[fn,tp]]
        cm_text = [[f"TN\n{tn:,}",f"FP\n{fp:,}"],[f"FN\n{fn:,}",f"TP\n{tp:,}"]]
        fig = go.Figure(go.Heatmap(
            z=cm_vals,
            x=['Pred: Normal','Pred: Attack'],
            y=['Act: Normal', 'Act: Attack'],
            colorscale=[[0,'#010f07'],[1,'#00ff64']],
            text=cm_text, texttemplate='%{text}',
            textfont=dict(size=13,family='Share Tech Mono',color='white'),
            showscale=False
        ))
        l = {**PLOT,'height':270}
        l.pop('xaxis',None); l.pop('yaxis',None)
        l['xaxis'] = dict(color='#a0ffc8',tickfont=dict(family='Share Tech Mono',size=10))
        l['yaxis'] = dict(color='#a0ffc8',tickfont=dict(family='Share Tech Mono',size=10))
        fig.update_layout(**l)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("RISK SCORE CDF", "cumulative distribution · normal vs attack")
        n_scores = np.sort(df[df['label']==0]['risk_score'].values)
        a_scores = np.sort(df[df['label']==1]['risk_score'].values)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=n_scores,y=np.linspace(0,1,len(n_scores)),
                                   name='Normal',line=dict(color='#00ff64',width=2)))
        fig2.add_trace(go.Scatter(x=a_scores,y=np.linspace(0,1,len(a_scores)),
                                   name='Attack',line=dict(color='#ff4d4d',width=2)))
        fig2.add_vline(x=threshold,line_dash='dash',line_color='#fff',
                       annotation_text=f'  threshold={threshold}',
                       annotation_font=dict(color='#fff',size=9,family='Share Tech Mono'))
        l2 = {**PLOT,'height':270}
        l2['xaxis'] = dict(**PLOT['xaxis'],title='Risk Score')
        l2['yaxis'] = dict(**PLOT['yaxis'],title='Cumulative Fraction')
        fig2.update_layout(**l2)
        st.plotly_chart(fig2, use_container_width=True)

    sec("MANUAL CONNECTION SCORER", "enter any values to get instant risk score")
    with st.expander("▶ SCORE A CONNECTION", expanded=True):
        c1,c2,c3,c4,c5 = st.columns(5)
        i_dur    = c1.number_input("dur",    0.0, 10000.0, 0.12,   step=0.01)
        i_sbytes = c2.number_input("sbytes", 0,   10000000, 258,   step=100)
        i_dbytes = c3.number_input("dbytes", 0,   10000000, 172,   step=100)
        i_sload  = c4.number_input("sload",  0.0, 5000000.0,14158.,step=100.)
        i_dload  = c5.number_input("dload",  0.0, 5000000.0, 8495.,step=100.)

        if st.button("▶ ANALYZE CONNECTION", use_container_width=True):
            score  = 0; reasons = []
            if i_dur > 10:       score += 20; reasons.append(f"long duration: {i_dur:.2f}s")
            if i_sbytes > 50000: score += 30; reasons.append(f"high sbytes: {i_sbytes:,}")
            if i_dbytes < 500:   score += 20; reasons.append(f"low dbytes: {i_dbytes}")
            if i_sload > 50000:  score += 15; reasons.append(f"high sload: {i_sload:.0f}")
            if i_dload < 1000:   score += 10; reasons.append(f"low dload: {i_dload:.0f}")
            score = min(100, score)
            lvl   = "CRITICAL" if score>=80 else "HIGH" if score>=60 else "MEDIUM" if score>=40 else "NORMAL"
            color = {'CRITICAL':'#ff2020','HIGH':'#ff6600','MEDIUM':'#ffcc00','NORMAL':'#00ff64'}[lvl]
            st.markdown(f"""<div style="background:#010f07;border:1px solid {color};border-radius:6px;
            padding:1rem 1.5rem;margin-top:0.5rem;display:flex;justify-content:space-between;align-items:center;">
            <div><div style="font-family:Orbitron;font-size:1.4rem;font-weight:900;
            color:{color};text-shadow:0 0 15px {color}55">{lvl}</div>
            <div style="font-family:Share Tech Mono;font-size:0.7rem;color:#3a8c58;margin-top:0.3rem">
            {'<br>'.join([f'→ {r}' for r in reasons]) if reasons else '→ All features within normal range'}
            </div></div>
            <div style="font-family:Orbitron;font-size:3rem;font-weight:900;
            color:{color};text-shadow:0 0 25px {color}44">{score}</div></div>""",
            unsafe_allow_html=True)