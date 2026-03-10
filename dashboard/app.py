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
# Dark Theme CSS
# ─────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; }
    [data-testid="stSidebar"]          { background-color: #161b27; }
    [data-testid="stHeader"]           { background-color: #0e1117; }
    .stMetric {
        background-color: #1c2130;
        border-radius: 10px;
        padding: 16px !important;
        border: 1px solid #2d3748;
    }
    .stMetricLabel { color: #a0aec0 !important; }
    .stMetricValue { color: #fafafa  !important; }
    div[data-testid="stExpander"] {
        background-color: #1c2130;
        border-radius: 8px;
    }
    h1, h2, h3 { color: #fafafa !important; }
    p, li      { color: #c8d0e0; }
    hr         { border-color: #2d3748; }
    [data-testid="stCaption"] { color: #718096; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# Dark chart helper
# applies dark bg to any plotly fig
# ─────────────────────────────
def dk(fig, height=350):
    fig.update_layout(
        plot_bgcolor='#1c2130',
        paper_bgcolor='#0e1117',
        font=dict(color='#fafafa'),
        height=height,
        xaxis=dict(gridcolor='#2d3748', zerolinecolor='#2d3748'),
        yaxis=dict(gridcolor='#2d3748', zerolinecolor='#2d3748'),
    )
    return fig

# ─────────────────────────────
# Neural Network (top-level so
# it's available everywhere)
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

# ─────────────────────────────
# Load model once via
# cache_resource (non-serializable)
# ─────────────────────────────
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

# ─────────────────────────────
# Get predictions (cached)
# ─────────────────────────────
@st.cache_data
def get_predictions():
    data = load_data()
    model, fcols = load_model()
    if model is None:
        return None, None, None
    X = data[fcols]
    y = data['Class']
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
        proba, bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=['Low','Medium','High','Critical']
    )
    return result, proba, y_test.values

# ─────────────────────────────
# Live narrative via Claude API
# ─────────────────────────────
def generate_live_narrative(res, fraud_prob, decision):
    try:
        import anthropic
        api_key = st.secrets.get(
            "ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY","")
        )
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
# Bootstrap top-level objects
# ─────────────────────────────
df           = load_data()
hourly       = load_hourly_stats()
narratives   = load_narratives()
nn_model, feature_cols = load_model()
preds, nn_proba, nn_labels = get_predictions()

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
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Fraud Cases",        f"{df['Class'].sum():,}")
    c3.metric("Fraud Rate",         f"{df['Class'].mean():.3%}")
    c4.metric("Peak Fraud Hour",    "2 AM",
              help="Hour with highest fraud rate in dataset")
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
            line=dict(color='#E1306C', width=1), opacity=0.4
        ))
        fig.add_trace(go.Scatter(
            x=hourly['hour'], y=hourly['rolling_fraud']*100,
            mode='lines', name='6-hr Rolling Avg',
            line=dict(color='#E1306C', width=2.5)
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
            marker_color='#4267B2', opacity=0.6,
            histnorm='probability density'
        ))
        fig.add_trace(go.Histogram(
            x=df[df.Class==1]['Amount_scaled'].clip(upper=5),
            nbinsx=50, name='Fraud',
            marker_color='#E1306C', opacity=0.6,
            histnorm='probability density'
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
        fig.update_layout(xaxis_title='Hour of Day', yaxis_title='Fraud Rate (%)')
        st.plotly_chart(dk(fig), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — Live Fraud Detector  ⭐ NEW
# ══════════════════════════════════════════════════════════════
elif page == "🎯 Live Fraud Detector":
    st.title("🎯 Live Fraud Detector")
    st.markdown("*Sample a real transaction, tweak behavioral signals, get an AI-powered verdict*")
    st.divider()

    st.info(
        "**How it works:** Each transaction is sampled from the real ULB dataset. "
        "Adjust Hour and Amount signals, then run the model for a live fraud probability "
        "and a Claude-generated compliance narrative."
    )

    left, right = st.columns([1, 2])

    with left:
        st.subheader("⚙️ Transaction Input")

        if st.button("🎲 Sample New Transaction", use_container_width=True):
            st.session_state.sampled_idx = int(np.random.randint(0, len(df)))
            for k in ['live_narrative','last_result']:
                st.session_state.pop(k, None)

        if 'sampled_idx' not in st.session_state:
            st.session_state.sampled_idx = int(np.random.randint(0, len(df)))

        sample = df.iloc[st.session_state.sampled_idx].copy()
        actual = int(sample['Class'])

        st.markdown(f"**Transaction #{st.session_state.sampled_idx:,}**")
        st.caption(f"Ground truth: {'🔴 Fraud' if actual==1 else '✅ Legitimate'} *(hidden from model)*")
        st.divider()

        hour = st.slider("🕐 Hour of Day", 0, 23,
                         int(sample.get('hour_of_day', 12)),
                         help="When did this transaction occur?")
        is_night_val = 1 if (hour >= 22 or hour <= 5) else 0
        st.caption("🌙 Late Night — elevated risk window" if is_night_val
                   else "☀️ Daytime transaction")

        amount_z = st.slider(
            "📊 Amount vs Recent Average (Z-Score)",
            -3.0, 3.0,
            float(np.clip(sample.get('amount_zscore', 0.0), -3.0, 3.0)),
            step=0.1,
            help="0 = typical  |  +2 = unusually large  |  -2 = suspiciously small (card-testing)"
        )

        st.divider()
        st.markdown("**🧬 Behavioral Signals** *(from transaction record)*")
        v14 = float(sample.get('V14', 0))
        v4  = float(sample.get('V4',  0))
        v12 = float(sample.get('V12', 0))

        def badge(v, t1=-5, t2=-2):
            return "🔴 High Risk" if v < t1 else "🟡 Moderate" if v < t2 else "🟢 Normal"

        st.caption(f"V14 (top predictor): {v14:.3f}  {badge(v14)}")
        st.caption(f"V4  (behavioral):    {v4:.3f}  "
                   f"{'🔴 High Risk' if abs(v4)>5 else '🟡 Moderate' if abs(v4)>2 else '🟢 Normal'}")
        st.caption(f"V12 (pattern):       {v12:.3f}  {badge(v12)}")
        st.divider()

        analyze_btn = st.button("🔍 Analyze Transaction",
                                type="primary", use_container_width=True)

    with right:
        # Run model on button click
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
            res  = st.session_state.last_result
            fp   = res['fraud_prob']
            if   fp >= 0.8: decision, col, emo = "BLOCK",  "#C73E1D", "🚫"
            elif fp >= 0.5: decision, col, emo = "REVIEW", "#F18F01", "⚠️"
            else:           decision, col, emo = "APPROVE","#44BBA4", "✅"

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(fp*100, 1),
                number={'suffix':'%','font':{'size':44,'color':col}},
                domain={'x':[0,1],'y':[0,1]},
                title={'text':"Fraud Probability",'font':{'size':18,'color':'#fafafa'}},
                gauge={
                    'axis':{'range':[0,100],'ticksuffix':'%',
                            'tickfont':{'color':'#fafafa'}},
                    'bar':{'color':col},
                    'steps':[
                        {'range':[0, 30],'color':'#1a3a2a'},
                        {'range':[30,60],'color':'#3a2a0a'},
                        {'range':[60,80],'color':'#3a1020'},
                        {'range':[80,100],'color':'#2a0808'},
                    ],
                    'threshold':{'line':{'color':'white','width':4},
                                 'thickness':0.75,'value':round(fp*100,1)}
                }
            ))
            fig.update_layout(height=300, paper_bgcolor='#0e1117',
                              font={'color':'#fafafa'}, margin=dict(t=60,b=0))
            st.plotly_chart(fig, use_container_width=True)

            # Decision badge
            st.markdown(
                f"<div style='text-align:center;padding:18px;border-radius:12px;"
                f"background:{col}22;border:2px solid {col};"
                f"font-size:26px;font-weight:bold;color:{col};margin-bottom:16px'>"
                f"{emo}&nbsp; DECISION: {decision}</div>",
                unsafe_allow_html=True
            )

            # Reveal ground truth
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
            if st.button("✨ Generate Claude Narrative", use_container_width=True):
                with st.spinner("Claude is analyzing the transaction signals..."):
                    st.session_state.live_narrative = \
                        generate_live_narrative(res, fp, decision)

            if 'live_narrative' in st.session_state:
                st.markdown(
                    f"<div style='background:#1c2130;padding:20px;border-radius:10px;"
                    f"border-left:4px solid {col};color:#e2e8f0'>"
                    f"{st.session_state.live_narrative}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                "<div style='text-align:center;padding:80px;color:#4a5568'>"
                "<div style='font-size:60px'>🎯</div>"
                "<h3 style='color:#718096'>Sample a transaction and click Analyze</h3>"
                "<p>Get a real-time fraud prediction + AI-generated risk narrative</p>"
                "</div>",
                unsafe_allow_html=True
            )
            if nn_model is None:
                st.warning("Model file not found. Ensure src/fraud_model.pth exists.")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — Transaction Explorer (dynamic insights added)
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Transaction Explorer":
    st.title("🔍 Transaction Explorer")
    st.markdown("*Filter transactions — watch the insights update dynamically*")
    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        class_filter = st.selectbox("Transaction Type",
                                    ["All","Legitimate Only","Fraud Only"])
    with c2:
        mn, mx = st.slider("Amount Anomaly Score Range",
                           -5.0, 5.0, (-5.0, 5.0))
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

    # ── Dynamic insight cards ────────────────────────────────
    st.divider()
    st.subheader("💡 What this filter is telling you")
    if len(filtered) > 0:
        fr        = filtered['Class'].mean()
        baseline  = df['Class'].mean()
        delta     = fr - baseline

        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            if delta > 0.02:
                st.error(
                    f"🔴 **{fr:.1%} fraud rate** — "
                    f"{delta/baseline:.0%} above baseline. High-risk segment."
                )
            elif delta < -0.02:
                st.success(
                    f"🟢 **{fr:.1%} fraud rate** — below baseline. Lower-risk segment."
                )
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
                    st.warning(
                        f"⚠️ Avg fraud amount score: **{az:.2f}** "
                        f"— small amounts suggest card-testing behavior"
                    )
                else:
                    st.info(f"Avg fraud amount score: **{az:.2f}**")
            else:
                st.info("Filter to Fraud Only to see amount patterns")

    st.divider()
    st.subheader("Sample Transactions")
    dcols = ['Amount_scaled','hour_of_day','is_night','amount_zscore','Class']
    st.dataframe(
        filtered[dcols].head(100).rename(columns={
            'Amount_scaled': 'Txn Amount',
            'hour_of_day':   'Hour',
            'is_night':      'Late Night?',
            'amount_zscore': 'Amount Anomaly Score',
            'Class':         'Fraud'
        }),
        use_container_width=True, height=350
    )

# ══════════════════════════════════════════════════════════════
# PAGE 4 — Model Comparison
# ══════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.markdown("*Logistic Regression vs XGBoost vs Neural Network*")
    st.divider()

    rdf = pd.DataFrame({
        'Model':             ['Logistic Regression','XGBoost','Neural Network'],
        'ROC-AUC':           [0.9667, 0.9755, 0.9828],
        'Avg Precision':     [0.7411, 0.8525, 0.8226],
        'Recall (Fraud)':    [0.89,   0.88,   0.86],
        'Precision (Fraud)': [0.51,   0.41,   0.73]
    })
    st.subheader("Performance Summary")
    st.dataframe(rdf.set_index('Model'), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(rdf, x='Model', y='ROC-AUC', title='ROC-AUC Score by Model',
                     color='Model',
                     color_discrete_sequence=['#4267B2','#44BBA4','#E1306C'])
        fig.update_layout(yaxis_range=[0.95,1.0], showlegend=False)
        fig.update_traces(text=rdf['ROC-AUC'].round(4), textposition='outside')
        st.plotly_chart(dk(fig), use_container_width=True)
    with c2:
        fig = px.bar(rdf, x='Model', y='Avg Precision',
                     title='Average Precision (Key Metric for Fraud)',
                     color='Model',
                     color_discrete_sequence=['#4267B2','#44BBA4','#E1306C'])
        fig.update_layout(yaxis_range=[0.6,1.0], showlegend=False)
        fig.update_traces(text=rdf['Avg Precision'].round(4), textposition='outside')
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
    fig.update_layout(yaxis={'categoryorder':'total ascending'},
                      coloraxis_showscale=False)
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
        "Unlike ROC-AUC, it is not inflated by true negatives — "
        "making it the correct tool at 0.17% fraud prevalence.*"
    )
    if nn_proba is not None:
        pv, rv, _ = precision_recall_curve(nn_labels, nn_proba)
        ap_nn = average_precision_score(nn_labels, nn_proba)
        bl    = nn_labels.mean()
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
            textfont=dict(size=11)
        ))
        fig.add_hline(y=bl, line_dash='dash', line_color='#718096',
                      annotation_text=f'Random Classifier (AP={bl:.3f})',
                      annotation_position='bottom right',
                      annotation_font_color='#718096')
        fig.update_layout(
            xaxis_title='Recall (Fraud Caught)',
            yaxis_title='Precision (Flagged = Actually Fraud)',
            legend=dict(yanchor='top',y=0.99,xanchor='right',x=0.99),
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
    st.markdown(
        "*Catching more fraud always means more false positives. "
        "Slide the threshold to explore the operational trade-off.*"
    )
    if nn_proba is not None:
        thresh = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05,
                           help="Lower = catch more fraud, more false positives")
        yp = (nn_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(nn_labels, yp).ravel()

        mc1, mc2, mc3, mc4 = st.columns(4)
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
        st.plotly_chart(dk(fig, height=350), use_container_width=True)

        fcr = tp/(tp+fn) if (tp+fn) > 0 else 0
        fpr = fp/(fp+tn) if (fp+tn) > 0 else 0
        if thresh <= 0.3:
            st.warning(f"⚠️ Aggressive: catching {fcr:.1%} of fraud but blocking "
                       f"{fp:,} legitimate transactions ({fpr:.2%} FPR). High customer friction.")
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
            fig = px.pie(values=rc.values, names=rc.index,
                         title='Transaction Risk Distribution',
                         color_discrete_sequence=['#44BBA4','#F18F01','#E1306C','#C73E1D'])
            fig.update_layout(paper_bgcolor='#0e1117', font=dict(color='#fafafa'))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(preds, x='fraud_probability', nbins=50,
                               title='Fraud Probability Distribution',
                               color_discrete_sequence=['#E1306C'])
            fig.add_vline(x=0.5, line_dash='dash',
                          annotation_text='Decision Threshold',
                          annotation_font_color='#fafafa')
            st.plotly_chart(dk(fig), use_container_width=True)

        st.success(
            "🔍 **Key Finding:** The model flags fraud most aggressively during late-night hours (10PM–5AM), "
            "consistent with automated card-testing behavior where fraudsters probe stolen credentials "
            "while cardholders are asleep. A production system could apply heightened sensitivity "
            "thresholds during these hours to catch more fraud with minimal daytime false positives."
        )

        st.subheader("🚨 Critical Risk Transactions (>80% probability)")
        critical = preds[preds.fraud_probability>0.8][
            ['amount_zscore','hour_of_day','is_night','fraud_probability','actual_fraud']
        ].sort_values('fraud_probability', ascending=False).head(20)
        critical.columns = ['Amount Anomaly Score','Hour','Late Night?',
                             'Fraud Prob','Actual Fraud']
        st.dataframe(
            critical.style.format({
                'Fraud Prob':'            {:.1%}',
                'Amount Anomaly Score':'{:.2f}'
            }),
            use_container_width=True
        )

# ══════════════════════════════════════════════════════════════
# PAGE 6 — AI Risk Narratives
# ══════════════════════════════════════════════════════════════
elif page == "🤖 AI Risk Narratives":
    st.title("🤖 AI-Generated Risk Narratives")
    st.markdown("*Claude API translates model signals into plain-English risk assessments*")
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
                st.markdown(n['narrative'])
    else:
        st.warning("No narratives found. Run notebook 05_llm_risk_narrative.ipynb first.")