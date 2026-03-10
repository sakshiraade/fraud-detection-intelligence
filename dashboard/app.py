import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score,
                             precision_recall_curve, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────
# Page Config
# ─────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Intelligence",
    page_icon="🔍",
    layout="wide"
)

# ─────────────────────────────
# Global Dark Theme CSS
# ─────────────────────────────
st.markdown("""
<style>
    /* ── Base backgrounds ── */
    [data-testid="stAppViewContainer"] { background-color: #0e1117; }
    [data-testid="stSidebar"]          { background-color: #161b27; }
    [data-testid="stHeader"]           { background-color: #0e1117; }

    /* ── Metric cards ── */
    .stMetric {
        background-color: #1c2130;
        border-radius: 10px;
        padding: 16px !important;
        border: 1px solid #2d3748;
    }
    .stMetricLabel  { color: #a0aec0 !important; font-size: 13px !important; }
    .stMetricValue  { color: #fafafa  !important; }
    .stMetricDelta  { color: #a0aec0  !important; }

    /* ── Typography ── */
    h1, h2, h3, h4  { color: #fafafa !important; }
    p, li, label    { color: #c8d0e0  !important; }
    .stMarkdown p   { color: #c8d0e0  !important; }
    hr              { border-color: #2d3748; }
    [data-testid="stCaption"] { color: #718096 !important; }

    /* ── Sidebar text ── */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span { color: #c8d0e0 !important; }

    /* ── Expander ── */
    div[data-testid="stExpander"] {
        background-color: #1c2130;
        border-radius: 8px;
        border: 1px solid #2d3748;
    }
    div[data-testid="stExpander"] summary p { color: #fafafa !important; }

    /* ── ALL buttons — default (secondary) ── */
    div[data-testid="stButton"] > button {
        background-color: #1c2130   !important;
        color:            #fafafa   !important;
        border:           1px solid #4a5568 !important;
        border-radius:    8px       !important;
        font-weight:      500       !important;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #2d3748 !important;
        border-color:     #718096 !important;
    }

    /* ── Primary buttons (type="primary") ── */
    div[data-testid="stButton"] > button[kind="primaryFormSubmit"],
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {
        background-color: #E1306C !important;
        color:            #ffffff !important;
        border:           none    !important;
    }
    div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover {
        background-color: #c0255a !important;
    }

    /* ── Selectbox / dropdowns ── */
    div[data-testid="stSelectbox"] > div > div {
        background-color: #1c2130 !important;
        color:            #fafafa !important;
        border:           1px solid #4a5568 !important;
    }

    /* ── Slider labels ── */
    div[data-testid="stSlider"] p   { color: #c8d0e0 !important; }
    div[data-testid="stSlider"] span{ color: #c8d0e0 !important; }

    /* ── Info / success / warning boxes ── */
    div[data-testid="stAlert"] p    { color: #fafafa !important; }

    /* ── DataFrames — force dark background ── */
    div[data-testid="stDataFrame"] > div {
        background-color: #1c2130 !important;
        border-radius: 8px;
    }
    div[data-testid="stDataFrame"] iframe {
        background-color: #1c2130 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# Dark chart helper
# ─────────────────────────────
DARK_BG   = '#1c2130'
PAPER_BG  = '#0e1117'
FONT_CLR  = '#fafafa'
GRID_CLR  = '#2d3748'

def dk(fig, height=350):
    fig.update_layout(
        plot_bgcolor  = DARK_BG,
        paper_bgcolor = PAPER_BG,
        height        = height,
        font          = dict(color=FONT_CLR, size=12),
        title_font    = dict(color=FONT_CLR, size=15),
        legend        = dict(font=dict(color=FONT_CLR), bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(
            gridcolor    = GRID_CLR,
            zerolinecolor= GRID_CLR,
            tickfont     = dict(color=FONT_CLR),
            title_font   = dict(color=FONT_CLR),
        ),
        yaxis=dict(
            gridcolor    = GRID_CLR,
            zerolinecolor= GRID_CLR,
            tickfont     = dict(color=FONT_CLR),
            title_font   = dict(color=FONT_CLR),
        ),
    )
    return fig

# ─────────────────────────────
# Dark HTML table helper
# replaces st.dataframe for
# tables that resist CSS theming
# ─────────────────────────────
def dark_table(df, fmt=None):
    if fmt:
        styler = df.style.format(fmt)
    else:
        styler = df.style
    styler = styler.set_properties(**{
        'background-color': '#1c2130',
        'color':            '#e2e8f0',
        'border':           '1px solid #2d3748',
        'padding':          '6px 12px',
        'font-size':        '13px',
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#161b27'),
            ('color',            '#a0aec0'),
            ('border',           '1px solid #2d3748'),
            ('padding',          '8px 12px'),
            ('font-size',        '13px'),
        ]},
        {'selector': 'table', 'props': [
            ('width',            '100%'),
            ('border-collapse',  'collapse'),
        ]},
        {'selector': 'tr:hover td', 'props': [
            ('background-color', '#2d3748'),
        ]},
    ])
    st.markdown(styler.to_html(), unsafe_allow_html=True)

# ─────────────────────────────
# Neural Network definition
# ─────────────────────────────
class FraudNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

# ─────────────────────────────
# Data loaders
# ─────────────────────────────
@st.cache_data
def load_data():
    p = os.path.join(BASE_DIR, 'data/processed/sample_features.csv')
    if not os.path.exists(p):
        p = os.path.join(BASE_DIR, 'data/processed/creditcard_features.csv')
    return pd.read_csv(p)

@st.cache_data
def load_hourly_stats():
    import sqlite3
    db = os.path.join(BASE_DIR, 'data/processed/fraud_warehouse.db')
    if os.path.exists(db):
        conn = sqlite3.connect(db)
        df = pd.read_sql("SELECT * FROM hourly_stats ORDER BY hour", conn)
        conn.close()
        return df
    data = load_data().copy()
    data['hour'] = (data.index // 100).astype(int) % 48
    h = data.groupby('hour').agg(
        total_transactions=('Class','count'),
        fraud_count=('Class','sum'),
        total_amount=('Amount_scaled','sum'),
        avg_amount=('Amount_scaled','mean'),
        max_amount=('Amount_scaled','max')
    ).reset_index()
    h['fraud_rate'] = h['fraud_count'] / h['total_transactions']
    return h

@st.cache_data
def load_narratives():
    p = os.path.join(BASE_DIR, 'reports/risk_narratives.json')
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return []

@st.cache_resource
def load_model():
    data = load_data()
    fcols = [c for c in data.columns if c not in ['Class','Time','Amount']]
    mp = os.path.join(BASE_DIR, 'src/fraud_model.pth')
    if not os.path.exists(mp):
        return None, fcols
    device = torch.device('cpu')
    m = FraudNet(input_dim=len(fcols)).to(device)
    m.load_state_dict(torch.load(mp, map_location=device))
    m.eval()
    return m, fcols

@st.cache_data
def get_predictions():
    data  = load_data()
    model, fcols = load_model()
    if model is None:
        return None, None, None
    X = data[fcols];  y = data['Class']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    with torch.no_grad():
        proba = model(
            torch.FloatTensor(X_test.values)
        ).squeeze().cpu().numpy()
    result = X_test.copy().reset_index(drop=True)
    result['fraud_probability'] = proba
    result['actual_fraud']      = y_test.values
    result['risk_level'] = pd.cut(
        proba, bins=[0,0.3,0.6,0.8,1.0],
        labels=['Low','Medium','High','Critical']
    )
    return result, proba, y_test.values

# ─────────────────────────────
# Live Claude narrative
# ─────────────────────────────
def generate_live_narrative(res, fraud_prob, decision):
    try:
        import anthropic
        api_key = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY",""))
        if not api_key:
            return "⚠️ Add ANTHROPIC_API_KEY to Streamlit secrets to enable live narratives."
        client = anthropic.Anthropic(api_key=api_key)
        time_label   = "(Late Night ⚠️)" if res['is_night'] else "(Daytime)"
        amount_label = (
            "(Unusually high ⚠️)" if res['amount_z'] > 2
            else "(Unusually low — possible card-test ⚠️)" if res['amount_z'] < -1
            else "(Normal range)"
        )
        prompt = f"""You are a fraud risk analyst at a major financial institution.

Analyze this flagged credit card transaction and produce a compliance-ready risk assessment.

Transaction Signals:
- Fraud Probability : {fraud_prob:.1%}
- Hour of Day       : {res['hour']} {time_label}
- Amount Z-Score    : {res['amount_z']:.2f} {amount_label}
- V14 Signal        : {res['v14']:.3f}  (strongest fraud predictor in this model)
- V4  Signal        : {res['v4']:.3f}
- Decision          : {decision}

Write 3-4 sentences of plain-English narrative covering:
1. Why this transaction was flagged (cite specific signals)
2. What fraud pattern it most resembles
3. Recommended action

End with: **DECISION: {decision}**"""
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role":"user","content":prompt}]
        )
        return msg.content[0].text
    except Exception as e:
        return f"⚠️ Could not generate narrative: {e}"

# ─────────────────────────────
# Bootstrap
# ─────────────────────────────
df         = load_data()
hourly     = load_hourly_stats()
narratives = load_narratives()
nn_model, feature_cols       = load_model()
preds, nn_proba, nn_labels   = get_predictions()

# ─────────────────────────────
# Sidebar
# ─────────────────────────────
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Overview",
    "🎯 Live Fraud Detector",
    "🔍 Transaction Explorer",
    "📊 Model Comparison",
    "🚨 Risk Monitor",
    "🤖 AI Risk Narratives"
])
st.sidebar.divider()
st.sidebar.markdown("**Dataset Stats**")
st.sidebar.metric("Total Transactions", f"{len(df):,}")
st.sidebar.metric("Fraud Cases",        f"{df['Class'].sum():,}")
st.sidebar.metric("Fraud Rate",         f"{df['Class'].mean():.3%}")

# ══════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🔍 Credit Card Fraud Detection Intelligence")
    st.markdown("*Real-time fraud detection using ML + AI risk narratives — 10K transaction sample from ULB dataset*")
    st.markdown("""
    <div style='background:#1c2130;border-left:4px solid #4267B2;padding:14px 18px;
    border-radius:6px;margin-bottom:12px'>
    <p style='color:#c8d0e0;margin:0;font-size:14px;line-height:1.6'>
    This dashboard presents an end-to-end credit card fraud detection system built on the
    <b style='color:#fafafa'>ULB Credit Card Fraud dataset</b> (284,807 real European transactions, 0.17% fraud rate).
    Explore transaction patterns in the Overview, test the live model in the Fraud Detector,
    compare ML models, monitor high-risk alerts, and read AI-generated compliance narratives —
    all in one place.
    </p></div>
    """, unsafe_allow_html=True)
    st.divider()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Fraud Cases",        f"{df['Class'].sum():,}")
    c3.metric("Fraud Rate",         f"{df['Class'].mean():.3%}")
    c4.metric("Peak Fraud Hour",    "2 AM", help="Hour with highest fraud rate")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(hourly, x='hour', y='total_transactions',
                      title='Transaction Volume by Hour',
                      color_discrete_sequence=['#4267B2'])
        fig.update_layout(xaxis_title='Hour', yaxis_title='Transactions')
        st.plotly_chart(dk(fig), use_container_width=True)
    with c2:
        hourly['rolling_fraud'] = hourly['fraud_rate'].rolling(6, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly['hour'], y=hourly['fraud_rate']*100,
            mode='lines', name='Hourly',
            line=dict(color='#E1306C', width=1), opacity=0.5
        ))
        fig.add_trace(go.Scatter(
            x=hourly['hour'], y=hourly['rolling_fraud']*100,
            mode='lines', name='6-hr Rolling Avg',
            line=dict(color='#ff6b9d', width=2.5)
        ))
        fig.update_layout(title='Fraud Rate Over Time (%)',
                          xaxis_title='Hour', yaxis_title='Fraud Rate (%)')
        st.plotly_chart(dk(fig), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[df.Class==0]['Amount_scaled'].clip(upper=5),
            nbinsx=50, name='Legitimate',
            marker_color='#4267B2', opacity=0.7, histnorm='probability density'
        ))
        fig.add_trace(go.Histogram(
            x=df[df.Class==1]['Amount_scaled'].clip(upper=5),
            nbinsx=50, name='Fraud',
            marker_color='#E1306C', opacity=0.7, histnorm='probability density'
        ))
        fig.update_layout(barmode='overlay',
                          title='Transaction Amount Distribution',
                          xaxis_title='Amount (scaled)')
        st.plotly_chart(dk(fig), use_container_width=True)
    with c2:
        fbh = df.groupby('hour_of_day')['Class'].agg(['sum','count'])
        fbh['rate'] = fbh['sum'] / fbh['count'] * 100
        fig = px.bar(fbh.reset_index(), x='hour_of_day', y='rate',
                     title='Fraud Rate by Hour of Day (%)',
                     color='rate', color_continuous_scale='Reds')
        fig.update_layout(
            xaxis_title='Hour of Day', yaxis_title='Fraud Rate (%)',
            coloraxis_colorbar=dict(tickfont=dict(color=FONT_CLR),
                                    title=dict(text='rate', font=dict(color=FONT_CLR)))
        )
        st.plotly_chart(dk(fig), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — Live Fraud Detector
# ══════════════════════════════════════════════════════════════
elif page == "🎯 Live Fraud Detector":
    st.title("🎯 Live Fraud Detector")
    st.markdown("*Sample a real transaction, tweak behavioral signals, get an AI-powered verdict*")
    st.markdown("""
    <div style='background:#1c2130;border-left:4px solid #E1306C;padding:14px 18px;
    border-radius:6px;margin-bottom:12px'>
    <p style='color:#c8d0e0;margin:0;font-size:14px;line-height:1.6'>
    Interact directly with the trained neural network. Sample any transaction from the dataset,
    adjust the <b style='color:#fafafa'>hour of day</b> and <b style='color:#fafafa'>amount anomaly score</b>,
    and watch the fraud probability update in real time. Then generate a
    <b style='color:#fafafa'>Claude-powered compliance narrative</b> explaining exactly why the
    transaction was flagged — the same kind of output a fraud analyst would use to make a decision.
    </p></div>
    """, unsafe_allow_html=True)
    st.divider()

    st.info(
        "**How it works:** Each transaction is sampled from the real ULB dataset. "
        "Adjust Hour and Amount signals, then run the model for a live fraud probability "
        "and a Claude-generated compliance narrative."
    )

    left, right = st.columns([1, 2])

    with left:
        st.subheader("⚙️ Transaction Input")

        # Styled sample button
        st.markdown("""
        <style>
        div[data-testid="stButton"]:first-of-type > button {
            background-color: #2d3748 !important;
            color: #fafafa !important;
            border: 1px solid #718096 !important;
            width: 100%;
        }
        </style>""", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🎲 Sample Random", use_container_width=True):
                st.session_state.sampled_idx = int(np.random.randint(0, len(df)))
                for k in ['live_narrative','last_result']:
                    st.session_state.pop(k, None)
        with col_b:
            if st.button("🔴 Sample Fraud", use_container_width=True,
                         help="Pick a known fraud transaction for demo"):
                fraud_indices = df[df['Class'] == 1].index.tolist()
                st.session_state.sampled_idx = int(np.random.choice(fraud_indices))
                for k in ['live_narrative','last_result']:
                    st.session_state.pop(k, None)

        if 'sampled_idx' not in st.session_state:
            st.session_state.sampled_idx = int(np.random.randint(0, len(df)))

        sample = df.iloc[st.session_state.sampled_idx].copy()
        actual = int(sample['Class'])

        st.markdown(
            f"<p style='color:#a0aec0;font-size:13px;margin-top:8px'>"
            f"<b style='color:#fafafa'>Transaction #{st.session_state.sampled_idx:,}</b></p>",
            unsafe_allow_html=True
        )
        st.caption(f"Ground truth: {'🔴 Fraud' if actual==1 else '✅ Legitimate'} *(hidden from model)*")
        st.divider()

        hour = st.slider("🕐 Hour of Day", 0, 23,
                         int(sample.get('hour_of_day', 12)))
        is_night_val = 1 if (hour >= 22 or hour <= 5) else 0
        st.caption("🌙 Late Night — elevated risk window" if is_night_val else "☀️ Daytime transaction")

        amount_z = st.slider(
            "📊 Amount vs Recent Average (Z-Score)", -3.0, 3.0,
            float(np.clip(sample.get('amount_zscore', 0.0), -3.0, 3.0)), step=0.1,
            help="0 = typical  |  +2 = unusually large  |  -2 = suspiciously small"
        )
        st.divider()
        st.markdown("<p style='color:#a0aec0;font-size:13px;font-weight:600'>🧬 Behavioral Signals</p>",
                    unsafe_allow_html=True)
        v14 = float(sample.get('V14', 0))
        v4  = float(sample.get('V4',  0))
        v12 = float(sample.get('V12', 0))

        def badge(v, t1=-5, t2=-2):
            return "🔴 High Risk" if v < t1 else "🟡 Moderate" if v < t2 else "🟢 Normal"

        st.markdown(
            f"<p style='color:#c8d0e0;font-size:12px'>V14 (top predictor): {v14:.3f} &nbsp; {badge(v14)}</p>"
            f"<p style='color:#c8d0e0;font-size:12px'>V4  (behavioral):    {v4:.3f} &nbsp; "
            f"{'🔴 High Risk' if abs(v4)>5 else '🟡 Moderate' if abs(v4)>2 else '🟢 Normal'}</p>"
            f"<p style='color:#c8d0e0;font-size:12px'>V12 (pattern):       {v12:.3f} &nbsp; {badge(v12)}</p>",
            unsafe_allow_html=True
        )
        st.divider()

        analyze_btn = st.button("🔍 Analyze Transaction", type="primary", use_container_width=True)

    with right:
        if analyze_btn and nn_model is not None:
            fv = sample[feature_cols].copy()
            fv['hour_of_day']   = hour
            fv['is_night']      = is_night_val
            fv['amount_zscore'] = amount_z
            x = torch.FloatTensor(fv.values.reshape(1, -1))
            with torch.no_grad():
                fp = nn_model(x).item()
            st.session_state.last_result = dict(
                fraud_prob=fp, hour=hour, is_night=is_night_val,
                amount_z=amount_z, v14=v14, v4=v4, actual=actual
            )
            st.session_state.pop('live_narrative', None)

        if 'last_result' in st.session_state:
            res = st.session_state.last_result
            fp  = res['fraud_prob']
            if   fp >= 0.8: decision, col, emo = "BLOCK",  "#C73E1D", "🚫"
            elif fp >= 0.5: decision, col, emo = "REVIEW", "#F18F01", "⚠️"
            else:           decision, col, emo = "APPROVE","#44BBA4", "✅"

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(fp*100, 1),
                number={'suffix':'%',
                        'font':{'size':48, 'color':col, 'family':'Arial Black'}},
                domain={'x':[0,1],'y':[0,1]},
                title={'text':"Fraud Probability",
                       'font':{'size':16,'color':FONT_CLR}},
                gauge={
                    'axis':{'range':[0,100],'ticksuffix':'%',
                            'tickfont':{'color':FONT_CLR,'size':11},
                            'tickcolor':FONT_CLR},
                    'bar':{'color':col, 'thickness':0.3},
                    'bgcolor':'#1c2130',
                    'bordercolor':'#2d3748',
                    'steps':[
                        {'range':[0, 30], 'color':'#1a3a2a'},
                        {'range':[30,60], 'color':'#3a2a0a'},
                        {'range':[60,80], 'color':'#3a1020'},
                        {'range':[80,100],'color':'#2a0808'},
                    ],
                    'threshold':{'line':{'color':'white','width':3},
                                 'thickness':0.75,'value':round(fp*100,1)}
                }
            ))
            fig.update_layout(
                height=280, paper_bgcolor=PAPER_BG,
                font={'color':FONT_CLR}, margin=dict(t=60,b=0,l=20,r=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Decision badge
            st.markdown(
                f"<div style='text-align:center;padding:16px;border-radius:12px;"
                f"background:{col}22;border:2px solid {col};"
                f"font-size:24px;font-weight:bold;color:{col};margin-bottom:12px'>"
                f"{emo}&nbsp; DECISION: {decision}</div>",
                unsafe_allow_html=True
            )

            # Ground truth reveal
            if res['actual'] == 1:
                st.error(
                    f"Ground truth: **Fraud** — model said {decision} "
                    f"{'✅ Correct' if decision in ['BLOCK','REVIEW'] else '❌ Missed'}"
                )
            else:
                st.success(
                    f"Ground truth: **Legitimate** — model said {decision} "
                    f"{'✅ Correct' if decision=='APPROVE' else '⚠️ False Positive'}"
                )

            st.divider()
            st.subheader("🤖 AI Risk Assessment")
            gen_btn = st.button("✨ Generate Claude Narrative", use_container_width=True)
            if gen_btn:
                with st.spinner("Claude is analyzing the transaction signals..."):
                    st.session_state.live_narrative = \
                        generate_live_narrative(res, fp, decision)

            if 'live_narrative' in st.session_state:
                st.markdown(
                    f"<div style='background:#1c2130;padding:20px;border-radius:10px;"
                    f"border-left:4px solid {col};color:#e2e8f0;font-size:14px;line-height:1.7'>"
                    f"{st.session_state.live_narrative}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                "<div style='text-align:center;padding:80px;color:#4a5568'>"
                "<div style='font-size:60px'>🎯</div>"
                "<h3 style='color:#718096'>Sample a transaction and click Analyze</h3>"
                "<p style='color:#4a5568'>Get a real-time fraud prediction + AI-generated risk narrative</p>"
                "</div>",
                unsafe_allow_html=True
            )
            if nn_model is None:
                st.warning("Model file not found. Ensure src/fraud_model.pth exists.")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — Transaction Explorer
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Transaction Explorer":
    st.title("🔍 Transaction Explorer")
    st.markdown("*Filter transactions — watch the insights update dynamically*")
    st.markdown("""
    <div style='background:#1c2130;border-left:4px solid #44BBA4;padding:14px 18px;
    border-radius:6px;margin-bottom:12px'>
    <p style='color:#c8d0e0;margin:0;font-size:14px;line-height:1.6'>
    Slice the dataset by transaction type, time of day, and amount anomaly score.
    The <b style='color:#fafafa'>insight cards update dynamically</b> as you filter —
    showing how fraud rate, night-time concentration, and amount patterns shift across segments.
    This mirrors how a fraud analyst would isolate high-risk cohorts before building detection rules.
    </p></div>
    """, unsafe_allow_html=True)
    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        class_filter = st.selectbox("Transaction Type",
                                    ["All","Legitimate Only","Fraud Only"])
    with c2:
        mn, mx = st.slider("Amount Anomaly Score Range", -5.0, 5.0, (-5.0, 5.0))
    with c3:
        night_filter = st.selectbox("Time of Day",
                                    ["All","Night Only (10pm-5am)","Day Only"])

    filtered = df.copy()
    if class_filter == "Legitimate Only": filtered = filtered[filtered.Class==0]
    elif class_filter == "Fraud Only":    filtered = filtered[filtered.Class==1]
    if night_filter == "Night Only (10pm-5am)": filtered = filtered[filtered.is_night==1]
    elif night_filter == "Day Only":            filtered = filtered[filtered.is_night==0]
    filtered = filtered[
        (filtered.amount_zscore >= mn) & (filtered.amount_zscore <= mx)
    ]

    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions Shown", f"{len(filtered):,}")
    c2.metric("Fraud in Selection", f"{filtered['Class'].sum():,}")
    c3.metric("Fraud Rate",         f"{filtered['Class'].mean():.3%}")

    # Dynamic insight cards
    st.divider()
    st.subheader("💡 What this filter is telling you")
    if len(filtered) > 0:
        fr       = filtered['Class'].mean()
        baseline = df['Class'].mean()
        delta    = fr - baseline
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            if delta > 0.02:
                st.error(f"🔴 **{fr:.1%} fraud rate** — {delta/baseline:.0%} above baseline. High-risk segment.")
            elif delta < -0.02:
                st.success(f"🟢 **{fr:.1%} fraud rate** — below baseline. Lower-risk segment.")
            else:
                st.info(f"⚪ **{fr:.1%} fraud rate** — near baseline.")
        with ic2:
            if filtered['Class'].sum() > 0:
                np_ = filtered[filtered.Class==1]['is_night'].mean()
                st.info(f"🌙 **{np_:.0%}** of fraud in this segment occurs at night")
            else:
                st.info("No fraud cases in current selection")
        with ic3:
            if filtered['Class'].sum() > 0:
                az = filtered[filtered.Class==1]['amount_zscore'].mean()
                if az < -0.5:
                    st.warning(f"⚠️ Avg fraud amount score: **{az:.2f}** — small amounts suggest card-testing")
                else:
                    st.info(f"Avg fraud amount score: **{az:.2f}**")
            else:
                st.info("Filter to Fraud Only to see amount patterns")

    st.divider()
    st.subheader("Sample Transactions")
    display = filtered[['Amount_scaled','hour_of_day','is_night',
                         'amount_zscore','Class']].head(100).rename(columns={
        'Amount_scaled': 'Txn Amount',
        'hour_of_day':   'Hour',
        'is_night':      'Late Night?',
        'amount_zscore': 'Amount Anomaly Score',
        'Class':         'Fraud'
    })
    styler = display.style.format({
        'Txn Amount':          '{:.3f}',
        'Amount Anomaly Score':'{:.3f}'
    }).set_properties(**{
        'background-color': '#1c2130',
        'color':            '#e2e8f0',
        'border':           '1px solid #2d3748',
        'padding':          '8px 16px',
        'font-size':        '13px',
        'white-space':      'nowrap',       # prevent column wrapping
        'min-width':        '140px',        # each column has breathing room
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#161b27'),
            ('color',            '#a0aec0'),
            ('border',           '1px solid #2d3748'),
            ('padding',          '10px 16px'),
            ('font-size',        '13px'),
            ('white-space',      'nowrap'),
            ('min-width',        '140px'),
        ]},
        {'selector': 'table', 'props': [
            ('width',           'auto'),        # fit to content, no stretching
            ('border-collapse', 'collapse'),
        ]},
        {'selector': 'tr:hover td', 'props': [
            ('background-color', '#2d3748'),
        ]},
    ])
    # Wrap in a div with both horizontal scroll + fixed-height vertical scroll
    html = styler.to_html()
    st.markdown(
        f"""<div style='
            overflow-x: auto;
            overflow-y: auto;
            max-height: 420px;
            border: 1px solid #2d3748;
            border-radius: 8px;
            width: 100%;
        '>{html}</div>""",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════
# PAGE 4 — Model Comparison
# ══════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.markdown("*Logistic Regression vs XGBoost vs Neural Network*")
    st.markdown("""
    <div style='background:#1c2130;border-left:4px solid #F18F01;padding:14px 18px;
    border-radius:6px;margin-bottom:12px'>
    <p style='color:#c8d0e0;margin:0;font-size:14px;line-height:1.6'>
    Three models were benchmarked — each chosen deliberately.
    <b style='color:#fafafa'>Logistic Regression</b> sets the interpretable baseline.
    <b style='color:#fafafa'>XGBoost</b> was selected for production based on
    Average Precision (0.853) and SHAP explainability.
    <b style='color:#fafafa'>PyTorch Neural Network</b> achieved the highest ROC-AUC but was ruled out
    for production due to slower inference and limited explainability.
    The SHAP chart, Precision-Recall curve, and interactive Confusion Matrix show exactly
    why these trade-offs matter at a 0.17% fraud rate.
    </p></div>
    """, unsafe_allow_html=True)
    st.divider()

    rdf = pd.DataFrame({
        'Model':             ['Logistic Regression','XGBoost','Neural Network'],
        'ROC-AUC':           [0.9667, 0.9755, 0.9828],
        'Avg Precision':     [0.7411, 0.8525, 0.8226],
        'Recall (Fraud)':    [0.89,   0.88,   0.86],
        'Precision (Fraud)': [0.51,   0.41,   0.73]
    })
    st.subheader("Performance Summary")
    dark_table(rdf.set_index('Model').reset_index(), fmt={
        'ROC-AUC':           '{:.4f}',
        'Avg Precision':     '{:.4f}',
        'Recall (Fraud)':    '{:.2f}',
        'Precision (Fraud)': '{:.2f}',
    })
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(rdf, x='Model', y='ROC-AUC', title='ROC-AUC Score by Model',
                     color='Model',
                     color_discrete_sequence=['#4267B2','#44BBA4','#E1306C'])
        fig.update_layout(yaxis_range=[0.95,1.0], showlegend=False)
        fig.update_traces(text=rdf['ROC-AUC'].round(4), textposition='outside',
                          textfont=dict(color=FONT_CLR))
        st.plotly_chart(dk(fig), use_container_width=True)
    with c2:
        fig = px.bar(rdf, x='Model', y='Avg Precision',
                     title='Average Precision (Key Metric for Fraud)',
                     color='Model',
                     color_discrete_sequence=['#4267B2','#44BBA4','#E1306C'])
        fig.update_layout(yaxis_range=[0.6,1.0], showlegend=False)
        fig.update_traces(text=rdf['Avg Precision'].round(4), textposition='outside',
                          textfont=dict(color=FONT_CLR))
        st.plotly_chart(dk(fig), use_container_width=True)

    st.divider()
    st.subheader("🏆 Model Selection Verdict")
    c1, c2 = st.columns(2)
    with c1:
        st.success("**Recommended for Production: XGBoost**")
        st.markdown("""
        - Highest Average Precision (0.8525)
        - Best balance of precision and recall
        - Fast inference — critical for real-time fraud detection
        - SHAP explainability for compliance requirements
        """)
    with c2:
        st.info("**Neural Network: Best ROC-AUC (0.9828)**")
        st.markdown("""
        - Highest overall discrimination ability
        - Better precision (0.73) — fewer false positives
        - Slower inference than XGBoost
        - Harder to explain to compliance officers
        """)

    # SHAP
    st.divider()
    st.subheader("🔍 XGBoost Feature Importance (SHAP)")
    st.markdown("*V14 is the strongest signal — consistent with published fraud detection research on this dataset*")
    shap_df = pd.DataFrame({
        'Feature': ['V14','V4','V11','V12','Amount Anomaly Score',
                    'V10','V17','V3','Hour of Day','V7'],
        'Mean |SHAP Value|': [0.42,0.31,0.28,0.25,0.19,
                               0.17,0.15,0.13,0.11,0.09]
    })
    fig = px.bar(shap_df, x='Mean |SHAP Value|', y='Feature', orientation='h',
                 title='Top 10 Features Driving Fraud Predictions (XGBoost)',
                 color='Mean |SHAP Value|', color_continuous_scale='Reds')
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        coloraxis_showscale=False,
        coloraxis_colorbar=dict(tickfont=dict(color=FONT_CLR))
    )
    fig.update_traces(textfont=dict(color=FONT_CLR))
    st.plotly_chart(dk(fig, height=400), use_container_width=True)

    sc1, sc2, sc3 = st.columns(3)
    sc1.info("**V14** — top SHAP feature, consistently identified in fraud detection literature")
    sc2.info("**Hour of Day** — engineered from raw Time field, confirms late-night fraud pattern")
    sc3.info("**Amount Anomaly Score** — velocity z-score captures card-testing invisible in raw amounts")

    # PR Curve
    st.divider()
    st.subheader("📈 Precision-Recall Curve")
    st.markdown(
        "*Area under this curve = Average Precision. "
        "Unlike ROC-AUC, not inflated by true negatives — "
        "making it the correct tool at 0.17% fraud prevalence.*"
    )
    if nn_proba is not None:
        pv, rv, _ = precision_recall_curve(nn_labels, nn_proba)
        ap_nn     = average_precision_score(nn_labels, nn_proba)
        bl        = nn_labels.mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rv, y=pv, mode='lines',
            name=f'Neural Network (AP={ap_nn:.3f})',
            line=dict(color='#E1306C', width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=[0.88], y=[0.853], mode='markers+text',
            name='XGBoost (AP=0.853) ✅ Production',
            marker=dict(color='#44BBA4', size=14, symbol='star'),
            text=['XGBoost AP=0.853'], textposition='top right',
            textfont=dict(color=FONT_CLR, size=11)
        ))
        fig.add_hline(y=bl, line_dash='dash', line_color='#718096',
                      annotation_text=f'Random Classifier (AP={bl:.3f})',
                      annotation_position='bottom right',
                      annotation_font_color='#a0aec0')
        fig.update_layout(
            xaxis_title='Recall (Fraud Caught)',
            yaxis_title='Precision (Flagged = Actually Fraud)',
            legend=dict(yanchor='top',y=0.99,xanchor='right',x=0.99,
                        font=dict(color=FONT_CLR)),
            yaxis=dict(range=[0,1.05]),
            xaxis=dict(range=[0,1.05])
        )
        st.plotly_chart(dk(fig, height=420), use_container_width=True)
        st.caption("XGBoost AP of 0.853 exceeds the NN curve — confirming it as the stronger production model.")
    else:
        st.warning("Model file not found — PR curve unavailable.")

    # Confusion Matrix
    st.divider()
    st.subheader("🎯 Confusion Matrix — Threshold Explorer")
    st.markdown("*Catching more fraud always means more false positives. Slide to explore the trade-off.*")
    if nn_proba is not None:
        thresh = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05,
                           help="Lower = catch more fraud, more false positives")
        yp = (nn_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(nn_labels, yp).ravel()

        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("✅ True Positives",  f"{tp:,}")
        mc2.metric("❌ False Negatives", f"{fn:,}",
                   delta=f"-{fn} missed", delta_color="inverse")
        mc3.metric("⚠️ False Positives", f"{fp:,}",
                   delta=f"+{fp} blocked legit", delta_color="inverse")
        mc4.metric("✅ True Negatives",  f"{tn:,}")

        cm_df = pd.DataFrame(
            [[tp,fn],[fp,tn]],
            index=['Predicted Fraud','Predicted Legit'],
            columns=['Actual Fraud','Actual Legit']
        )
        fig = px.imshow(cm_df, text_auto=True,
                        color_continuous_scale='Reds',
                        title=f'Confusion Matrix at Threshold = {thresh}')
        fig.update_traces(textfont=dict(color=FONT_CLR, size=16))
        fig.update_layout(
            coloraxis_colorbar=dict(tickfont=dict(color=FONT_CLR),
                                    title=dict(font=dict(color=FONT_CLR)))
        )
        st.plotly_chart(dk(fig, height=350), use_container_width=True)

        fcr = tp/(tp+fn) if (tp+fn)>0 else 0
        fpr = fp/(fp+tn) if (fp+tn)>0 else 0
        if thresh <= 0.3:
            st.warning(f"⚠️ Aggressive: catching {fcr:.1%} of fraud but blocking {fp:,} "
                       f"legitimate transactions ({fpr:.2%} FPR). High customer friction.")
        elif thresh >= 0.7:
            st.warning(f"⚠️ Conservative: only catching {fcr:.1%} of fraud. "
                       f"Missing {fn:,} transactions. High financial exposure.")
        else:
            st.success(f"✅ Balanced: catching {fcr:.1%} of fraud with "
                       f"{fp:,} false positives ({fpr:.2%} FPR).")
    else:
        st.warning("Model file not found — confusion matrix unavailable.")

# ══════════════════════════════════════════════════════════════
# PAGE 5 — Risk Monitor
# ══════════════════════════════════════════════════════════════
elif page == "🚨 Risk Monitor":
    st.title("🚨 Risk Monitor")
    st.markdown("*High-risk transactions flagged by the neural network*")
    st.markdown("""
    <div style='background:#1c2130;border-left:4px solid #C73E1D;padding:14px 18px;
    border-radius:6px;margin-bottom:12px'>
    <p style='color:#c8d0e0;margin:0;font-size:14px;line-height:1.6'>
    This page simulates a <b style='color:#fafafa'>live fraud operations triage queue</b>.
    The neural network scores every transaction in the test set and surfaces those above 80% probability
    for immediate review. Metrics show how many critical, high-risk, and borderline transactions
    the model flagged — and how many actual frauds it caught vs. missed.
    This is the operational layer that sits between model output and analyst action.
    </p></div>
    """, unsafe_allow_html=True)
    st.divider()

    if preds is None:
        st.warning("Model file not found. Please ensure src/fraud_model.pth exists.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Critical Risk (>80%)",
                  f"{(preds.fraud_probability>0.8).sum():,}")
        c2.metric("High Risk (60–80%)",
                  f"{((preds.fraud_probability>0.6)&(preds.fraud_probability<=0.8)).sum():,}")
        c3.metric("True Frauds Caught",
                  f"{((preds.fraud_probability>0.5)&(preds.actual_fraud==1)).sum():,}")
        c4.metric("False Positives",
                  f"{((preds.fraud_probability>0.5)&(preds.actual_fraud==0)).sum():,}")

        c1, c2 = st.columns(2)
        with c1:
            rc = preds['risk_level'].value_counts()
            fig = px.pie(
                values=rc.values, names=rc.index,
                title='Transaction Risk Distribution',
                color_discrete_sequence=['#44BBA4','#F18F01','#E1306C','#C73E1D']
            )
            fig.update_layout(
                paper_bgcolor=PAPER_BG,
                plot_bgcolor =DARK_BG,
                font=dict(color=FONT_CLR, size=13),
                title_font=dict(color=FONT_CLR),
                legend=dict(font=dict(color=FONT_CLR, size=12),
                            bgcolor='rgba(0,0,0,0)'),
                height=350
            )
            fig.update_traces(textfont=dict(color=FONT_CLR, size=13))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(preds, x='fraud_probability', nbins=50,
                               title='Fraud Probability Distribution',
                               color_discrete_sequence=['#E1306C'])
            fig.add_vline(x=0.5, line_dash='dash', line_color='#fafafa',
                          annotation_text='Decision Threshold',
                          annotation_font_color=FONT_CLR)
            st.plotly_chart(dk(fig), use_container_width=True)

        st.success(
            "🔍 **Key Finding:** The model flags fraud most aggressively during late-night hours (10PM–5AM), "
            "consistent with automated card-testing behavior where fraudsters probe stolen credentials "
            "while cardholders are asleep. A production system could apply heightened sensitivity "
            "thresholds during these hours with minimal daytime false positives."
        )

        st.subheader("🚨 Critical Risk Transactions (>80% probability)")
        critical = preds[preds.fraud_probability>0.8][
            ['amount_zscore','hour_of_day','is_night','fraud_probability','actual_fraud']
        ].sort_values('fraud_probability', ascending=False).head(20).copy()
        critical.columns = ['Amount Anomaly Score','Hour','Late Night?',
                             'Fraud Prob','Actual Fraud']
        dark_table(critical, fmt={
            'Fraud Prob':           '{:.1%}',
            'Amount Anomaly Score': '{:.2f}'
        })

# ══════════════════════════════════════════════════════════════
# PAGE 6 — AI Risk Narratives
# ══════════════════════════════════════════════════════════════
elif page == "🤖 AI Risk Narratives":
    st.title("🤖 AI-Generated Risk Narratives")
    st.markdown("*Claude API translates model signals into plain-English risk assessments*")
    st.markdown("""
    <div style='background:#1c2130;border-left:4px solid #9b59b6;padding:14px 18px;
    border-radius:6px;margin-bottom:12px'>
    <p style='color:#c8d0e0;margin:0;font-size:14px;line-height:1.6'>
    A fraud probability score alone isn't enough — a risk analyst needs to know <i>why</i>.
    This page uses the <b style='color:#fafafa'>Claude API</b> to translate raw SHAP signals
    into compliance-ready narratives: what pattern was detected, what fraud behavior it resembles,
    and what action to take. Each narrative ends with a clear
    <b style='color:#fafafa'>APPROVE / REVIEW / BLOCK</b> decision.
    This is the <b style='color:#fafafa'>"last mile"</b> of the fraud detection pipeline —
    bridging model output to human judgment.
    </p></div>
    """, unsafe_allow_html=True)
    st.divider()

    st.info("""
    **How this works:** The neural network flags high-risk transactions and assigns a fraud probability.
    Claude then analyzes the top risk signals and generates a compliance-ready narrative explaining
    WHY the transaction was flagged and what action to take.
    """)

    if narratives:
        for n in narratives:
            prob  = n.get('fraud_probability', n.get('fraud_prob', 0))
            color = "🔴" if prob > 0.9 else "🟠"
            with st.expander(
                f"{color} Transaction #{n['transaction']} — "
                f"Fraud Probability: {prob:.1%}"
            ):
                st.markdown(
                    f"<div style='color:#e2e8f0;font-size:14px;line-height:1.7'>"
                    f"{n['narrative']}</div>",
                    unsafe_allow_html=True
                )
    else:
        st.warning("No narratives found. Run notebook 05_llm_risk_narrative.ipynb first.")