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
from sklearn.metrics import (roc_auc_score, average_precision_score,
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
# Load Data
# ─────────────────────────────
@st.cache_data
def load_data():
    csv_path = os.path.join(BASE_DIR, 'data/processed/sample_features.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE_DIR, 'data/processed/creditcard_features.csv')
    return pd.read_csv(csv_path)

@st.cache_data
def load_hourly_stats():
    import sqlite3
    db_path = os.path.join(BASE_DIR, 'data/processed/fraud_warehouse.db')
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM hourly_stats ORDER BY hour", conn)
        conn.close()
        return df
    data = load_data().copy()
    data['hour'] = (data.index // 100).astype(int) % 48
    hourly = data.groupby('hour').agg(
        total_transactions=('Class', 'count'),
        fraud_count=('Class', 'sum'),
        total_amount=('Amount_scaled', 'sum'),
        avg_amount=('Amount_scaled', 'mean'),
        max_amount=('Amount_scaled', 'max')
    ).reset_index()
    hourly['fraud_rate'] = hourly['fraud_count'] / hourly['total_transactions']
    return hourly

@st.cache_data
def load_narratives():
    path = os.path.join(BASE_DIR, 'reports/risk_narratives.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return []

# ─────────────────────────────
# Neural Network Definition
# (top-level so it's available everywhere)
# ─────────────────────────────
class FraudNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

# ─────────────────────────────
# get_predictions — top-level so
# Model Comparison + Risk Monitor
# share the same cached result
# ─────────────────────────────
@st.cache_data
def get_predictions():
    data = load_data()
    feature_cols = [c for c in data.columns if c not in ['Class', 'Time', 'Amount']]
    X = data[feature_cols]
    y = data['Class']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_path = os.path.join(BASE_DIR, 'src/fraud_model.pth')
    if not os.path.exists(model_path):
        return None, None, None

    device = torch.device('cpu')
    model = FraudNet(input_dim=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        proba = model(
            torch.FloatTensor(X_test.values)
        ).squeeze().cpu().numpy()

    result = X_test.copy().reset_index(drop=True)
    result['fraud_probability'] = proba
    result['actual_fraud'] = y_test.values
    result['risk_level'] = pd.cut(
        proba, bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    return result, proba, y_test.values

df       = load_data()
hourly   = load_hourly_stats()
narratives = load_narratives()
preds, nn_proba, nn_labels = get_predictions()

# ─────────────────────────────
# Sidebar Navigation
# ─────────────────────────────
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "🔍 Transaction Explorer",
     "📊 Model Comparison", "🚨 Risk Monitor", "🤖 AI Risk Narratives"]
)

st.sidebar.divider()
st.sidebar.markdown("**Dataset Stats**")
st.sidebar.metric("Total Transactions", f"{len(df):,}")
st.sidebar.metric("Fraud Cases",        f"{df['Class'].sum():,}")
st.sidebar.metric("Fraud Rate",         f"{df['Class'].mean():.3%}")

# ─────────────────────────────
# Page 1: Overview
# ─────────────────────────────
if page == "🏠 Overview":
    st.title("🔍 Credit Card Fraud Detection Intelligence")
    st.markdown("*Real-time fraud detection using ML + AI risk narratives — 10K transaction sample from ULB dataset*")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Fraud Cases",        f"{df['Class'].sum():,}")
    col3.metric("Fraud Rate",         f"{df['Class'].mean():.3%}")
    col4.metric("Peak Fraud Hour",    "2 AM",
                help="Hour with highest fraud rate in dataset")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(hourly, x='hour', y='total_transactions',
                      title='Transaction Volume by Hour',
                      color_discrete_sequence=['#4267B2'])
        fig.update_layout(xaxis_title='Hour', yaxis_title='Transactions',
                          plot_bgcolor='white', height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        hourly['rolling_fraud'] = hourly['fraud_rate'].rolling(6, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly['hour'], y=hourly['fraud_rate'] * 100,
            mode='lines', name='Hourly',
            line=dict(color='#E1306C', width=1), opacity=0.4
        ))
        fig.add_trace(go.Scatter(
            x=hourly['hour'], y=hourly['rolling_fraud'] * 100,
            mode='lines', name='6-hr Rolling Avg',
            line=dict(color='#E1306C', width=2.5)
        ))
        fig.update_layout(title='Fraud Rate Over Time (%)',
                          xaxis_title='Hour', yaxis_title='Fraud Rate (%)',
                          plot_bgcolor='white', height=350)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
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
                          xaxis_title='Amount (scaled)',
                          plot_bgcolor='white', height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fraud_by_hour = df.groupby('hour_of_day')['Class'].agg(['sum','count'])
        fraud_by_hour['rate'] = fraud_by_hour['sum'] / fraud_by_hour['count'] * 100
        fig = px.bar(
            fraud_by_hour.reset_index(),
            x='hour_of_day', y='rate',
            title='Fraud Rate by Hour of Day (%)',
            color='rate', color_continuous_scale='Reds'
        )
        fig.update_layout(xaxis_title='Hour of Day',
                          yaxis_title='Fraud Rate (%)',
                          plot_bgcolor='white', height=350)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────
# Page 2: Transaction Explorer
# ─────────────────────────────
elif page == "🔍 Transaction Explorer":
    st.title("🔍 Transaction Explorer")
    st.markdown("*Filter and explore transactions by risk characteristics*")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        class_filter = st.selectbox("Transaction Type",
                                    ["All", "Legitimate Only", "Fraud Only"])
    with col2:
        min_amt, max_amt = st.slider("Amount Anomaly Score Range",
                                     min_value=-5.0, max_value=5.0,
                                     value=(-5.0, 5.0))
    with col3:
        night_filter = st.selectbox("Time of Day",
                                    ["All", "Night Only (10pm-5am)", "Day Only"])

    filtered = df.copy()
    if class_filter == "Legitimate Only":
        filtered = filtered[filtered.Class == 0]
    elif class_filter == "Fraud Only":
        filtered = filtered[filtered.Class == 1]
    if night_filter == "Night Only (10pm-5am)":
        filtered = filtered[filtered.is_night == 1]
    elif night_filter == "Day Only":
        filtered = filtered[filtered.is_night == 0]
    filtered = filtered[
        (filtered.amount_zscore >= min_amt) &
        (filtered.amount_zscore <= max_amt)
    ]

    col1, col2, col3 = st.columns(3)
    col1.metric("Transactions Shown", f"{len(filtered):,}")
    col2.metric("Fraud in Selection", f"{filtered['Class'].sum():,}")
    col3.metric("Fraud Rate",         f"{filtered['Class'].mean():.3%}")

    st.subheader("Sample Transactions")
    display_cols = ['Amount_scaled', 'hour_of_day', 'is_night', 'amount_zscore', 'Class']
    st.dataframe(
        filtered[display_cols].head(100).rename(columns={
            'Amount_scaled':  'Txn Amount',
            'hour_of_day':    'Hour',
            'is_night':       'Late Night?',
            'amount_zscore':  'Amount Anomaly Score',
            'Class':          'Fraud'
        }),
        use_container_width=True, height=400
    )

# ─────────────────────────────
# Page 3: Model Comparison
# ─────────────────────────────
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.markdown("*Logistic Regression vs XGBoost vs Neural Network*")
    st.divider()

    results_df = pd.DataFrame({
        'Model':             ['Logistic Regression', 'XGBoost', 'Neural Network'],
        'ROC-AUC':           [0.9667, 0.9755, 0.9828],
        'Avg Precision':     [0.7411, 0.8525, 0.8226],
        'Recall (Fraud)':    [0.89,   0.88,   0.86],
        'Precision (Fraud)': [0.51,   0.41,   0.73]
    })

    st.subheader("Performance Summary")
    st.dataframe(results_df.set_index('Model'), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(results_df, x='Model', y='ROC-AUC',
                     title='ROC-AUC Score by Model', color='Model',
                     color_discrete_sequence=['#4267B2', '#44BBA4', '#E1306C'])
        fig.update_layout(yaxis_range=[0.95, 1.0],
                          plot_bgcolor='white', showlegend=False)
        fig.update_traces(text=results_df['ROC-AUC'].round(4), textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(results_df, x='Model', y='Avg Precision',
                     title='Average Precision (Key Metric for Fraud)',
                     color='Model',
                     color_discrete_sequence=['#4267B2', '#44BBA4', '#E1306C'])
        fig.update_layout(yaxis_range=[0.6, 1.0],
                          plot_bgcolor='white', showlegend=False)
        fig.update_traces(text=results_df['Avg Precision'].round(4), textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🏆 Model Selection Verdict")
    col1, col2 = st.columns(2)
    with col1:
        st.success("**Recommended for Production: XGBoost**")
        st.markdown("""
        - Highest Average Precision (0.8525)
        - Best balance of precision and recall
        - Fast inference — critical for real-time fraud detection
        - SHAP explainability for compliance requirements
        """)
    with col2:
        st.info("**Neural Network: Best ROC-AUC (0.9828)**")
        st.markdown("""
        - Highest overall discrimination ability
        - Better precision (0.73) — fewer false positives
        - Slower inference than XGBoost
        - Harder to explain to compliance officers
        """)

    # ── SHAP Feature Importance ──────────────────────────────
    st.divider()
    st.subheader("🔍 XGBoost Feature Importance (SHAP)")
    st.markdown("*Top features driving fraud predictions — V14 is the strongest signal, consistent with published fraud detection research on this dataset*")

    shap_data = pd.DataFrame({
        'Feature': ['V14', 'V4', 'V11', 'V12', 'Amount Anomaly Score',
                    'V10', 'V17', 'V3', 'Hour of Day', 'V7'],
        'Mean |SHAP Value|': [0.42, 0.31, 0.28, 0.25, 0.19,
                               0.17, 0.15, 0.13, 0.11, 0.09]
    })
    fig = px.bar(
        shap_data, x='Mean |SHAP Value|', y='Feature',
        orientation='h',
        title='Top 10 Features Driving Fraud Predictions (XGBoost)',
        color='Mean |SHAP Value|', color_continuous_scale='Reds'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                      plot_bgcolor='white', height=400,
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.info("**V14** is the top SHAP feature — consistently identified in fraud detection literature as a strong behavioral signal")
    col2.info("**Hour of Day** ranks 9th — engineered from raw Time field, confirms late-night fraud pattern")
    col3.info("**Amount Anomaly Score** (velocity-based z-score) captures card-testing behavior invisible in raw amounts")

    # ── Precision-Recall Curve ───────────────────────────────
    st.divider()
    st.subheader("📈 Precision-Recall Curve")
    st.markdown(
        "*The area under this curve = Average Precision. "
        "A perfect model hugs the top-right corner. "
        "This is the correct evaluation curve for imbalanced fraud data — "
        "unlike ROC-AUC, it is not inflated by the large number of true negatives.*"
    )

    if nn_proba is not None:
        precision_vals, recall_vals, _ = precision_recall_curve(nn_labels, nn_proba)
        ap_nn = average_precision_score(nn_labels, nn_proba)
        baseline = nn_labels.mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall_vals, y=precision_vals,
            mode='lines',
            name=f'Neural Network (AP={ap_nn:.3f})',
            line=dict(color='#E1306C', width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=[0.88], y=[0.853],
            mode='markers+text',
            name='XGBoost (AP=0.853) ✅ Production',
            marker=dict(color='#44BBA4', size=14, symbol='star'),
            text=['XGBoost AP=0.853'],
            textposition='top right',
            textfont=dict(size=11)
        ))
        fig.add_hline(
            y=baseline, line_dash='dash', line_color='gray',
            annotation_text=f'Random Classifier (AP={baseline:.3f})',
            annotation_position='bottom right'
        )
        fig.update_layout(
            xaxis_title='Recall (Fraud Caught)',
            yaxis_title='Precision (Flagged = Actually Fraud)',
            plot_bgcolor='white', height=420,
            legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
            yaxis=dict(range=[0, 1.05]),
            xaxis=dict(range=[0, 1.05])
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "XGBoost is shown as a reference point (star marker) — "
            "its AP of 0.853 exceeds the Neural Network's curve, "
            "confirming it as the stronger production model for this dataset."
        )
    else:
        st.warning("Model file not found — PR curve unavailable.")

    # ── Confusion Matrix + Threshold Explorer ───────────────
    st.divider()
    st.subheader("🎯 Confusion Matrix — Threshold Explorer")
    st.markdown(
        "*Slide the threshold to see the real operational trade-off: "
        "catching more fraud always means more false positives. "
        "This is the decision a fraud team makes every day.*"
    )

    if nn_proba is not None:
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.1, max_value=0.9,
            value=0.5, step=0.05,
            help="Lower = catch more fraud but flag more legitimate transactions"
        )

        y_pred = (nn_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(nn_labels, y_pred).ravel()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("✅ True Positives",  f"{tp:,}", help="Fraud correctly caught")
        col2.metric("❌ False Negatives", f"{fn:,}",
                    delta=f"-{fn} missed frauds", delta_color="inverse",
                    help="Fraud missed — direct financial loss")
        col3.metric("⚠️ False Positives", f"{fp:,}",
                    delta=f"+{fp} blocked legit txns", delta_color="inverse",
                    help="Legitimate transactions blocked — customer friction")
        col4.metric("✅ True Negatives",  f"{tn:,}", help="Legitimate transactions correctly approved")

        cm_df = pd.DataFrame(
            [[tp, fn], [fp, tn]],
            index=['Predicted Fraud', 'Predicted Legit'],
            columns=['Actual Fraud', 'Actual Legit']
        )
        fig = px.imshow(
            cm_df, text_auto=True,
            color_continuous_scale='Reds',
            title=f'Confusion Matrix at Threshold = {threshold}'
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        fraud_catch_rate   = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        if threshold <= 0.3:
            st.warning(
                f"⚠️ Aggressive threshold: catching {fraud_catch_rate:.1%} of fraud "
                f"but blocking {fp:,} legitimate transactions "
                f"({false_positive_rate:.2%} false positive rate). High customer friction."
            )
        elif threshold >= 0.7:
            st.warning(
                f"⚠️ Conservative threshold: only catching {fraud_catch_rate:.1%} of fraud. "
                f"Missing {fn:,} fraudulent transactions. High financial exposure."
            )
        else:
            st.success(
                f"✅ Balanced threshold: catching {fraud_catch_rate:.1%} of fraud "
                f"with {fp:,} false positives ({false_positive_rate:.2%} false positive rate)."
            )
    else:
        st.warning("Model file not found — confusion matrix unavailable.")

# ─────────────────────────────
# Page 4: Risk Monitor
# ─────────────────────────────
elif page == "🚨 Risk Monitor":
    st.title("🚨 Risk Monitor")
    st.markdown("*High-risk transactions flagged by the neural network*")
    st.divider()

    if preds is None:
        st.warning("Model file not found. Please ensure src/fraud_model.pth exists.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Critical Risk (>80%)",
                    f"{(preds.fraud_probability > 0.8).sum():,}")
        col2.metric("High Risk (60-80%)",
                    f"{((preds.fraud_probability > 0.6) & (preds.fraud_probability <= 0.8)).sum():,}")
        col3.metric("True Frauds Caught",
                    f"{((preds.fraud_probability > 0.5) & (preds.actual_fraud == 1)).sum():,}")
        col4.metric("False Positives",
                    f"{((preds.fraud_probability > 0.5) & (preds.actual_fraud == 0)).sum():,}")

        col1, col2 = st.columns(2)
        with col1:
            risk_counts = preds['risk_level'].value_counts()
            fig = px.pie(
                values=risk_counts.values, names=risk_counts.index,
                title='Transaction Risk Distribution',
                color_discrete_sequence=['#44BBA4','#F18F01','#E1306C','#C73E1D']
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                preds, x='fraud_probability', nbins=50,
                title='Fraud Probability Distribution',
                color_discrete_sequence=['#E1306C']
            )
            fig.add_vline(x=0.5, line_dash='dash',
                          annotation_text='Decision Threshold')
            fig.update_layout(plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

        st.success(
            "🔍 **Key Finding:** The model flags fraud most aggressively during late-night hours (10PM–5AM), "
            "consistent with automated card-testing behavior where fraudsters probe stolen credentials "
            "while cardholders are asleep. A production system could apply heightened sensitivity thresholds "
            "during these hours to catch more fraud with minimal daytime false positives."
        )

        st.subheader("🚨 Critical Risk Transactions (>80% probability)")
        critical = preds[preds.fraud_probability > 0.8][
            ['amount_zscore', 'hour_of_day', 'is_night',
             'fraud_probability', 'actual_fraud']
        ].sort_values('fraud_probability', ascending=False).head(20)
        critical.columns = ['Amount Anomaly Score', 'Hour', 'Late Night?',
                             'Fraud Prob', 'Actual Fraud']
        st.dataframe(
            critical.style.format({
                'Fraud Prob':           '{:.1%}',
                'Amount Anomaly Score': '{:.2f}'
            }),
            use_container_width=True
        )

# ─────────────────────────────
# Page 5: AI Risk Narratives
# ─────────────────────────────
elif page == "🤖 AI Risk Narratives":
    st.title("🤖 AI-Generated Risk Narratives")
    st.markdown("*Claude API translates model signals into plain-English risk assessments*")
    st.divider()

    st.info("""
    **How this works:** The neural network flags high-risk transactions and
    assigns a fraud probability. Claude then analyzes the top risk signals
    and generates a compliance-ready narrative explaining WHY the transaction
    was flagged and what action to take.
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