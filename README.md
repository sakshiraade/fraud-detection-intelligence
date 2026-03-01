# 🔍 Credit Card Fraud Detection & Risk Intelligence Platform
### *End-to-end fraud detection using ML, Deep Learning, and AI-powered risk narratives*

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-orange)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)

---

## 🔗 Live Dashboard
👉 **[View the deployed dashboard here](https://fraud-detection-intelligence.streamlit.app/)**

---

## 🎯 Business Problem

> *"284,807 credit card transactions came in over 2 days. 492 are fraudulent (0.17%). Build a system that catches them before losses occur — and explains why each transaction was flagged."*

This project uses the [ULB Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — the industry benchmark for fraud detection research, referenced in hundreds of academic papers.

---

## 🔍 Key Findings

- **Late night fraud spike** — Hours 2am and 26hr (2am day 2) showed 10x the baseline fraud rate, identified via SQL window functions
- **XGBoost won on Average Precision (0.853)** — the correct metric for imbalanced fraud data, outperforming Logistic Regression by +15%
- **Neural Network won on ROC-AUC (0.983)** — but XGBoost chosen for production due to better precision and explainability
- **V14 was the top SHAP feature** — consistent with published fraud detection research on this dataset
- **Probe transaction pattern detected** — AI narratives identified low-amount + high-risk signal combinations as card-testing behavior

---

## 🗂️ Project Modules

### 1. 🗄️ SQL Data Engineering Pipeline
- SQLite data warehouse with multi-table schema
- Window functions for rolling fraud rate analysis
- CTE-based high-risk hour identification
- 48-hour time series of transaction behavior

### 2. ⚙️ Feature Engineering
- Velocity features — rolling amount averages (10 and 50 transaction windows)
- Amount z-score vs recent transaction history
- Time features — hour of day, night flag
- Engineered 8 new features on top of 28 PCA components

### 3. 🤖 Model Comparison
- **Logistic Regression** — baseline (AUC: 0.967, AP: 0.741)
- **XGBoost** — production model (AUC: 0.976, AP: 0.853)
- **PyTorch Neural Network** — deep learning (AUC: 0.983, AP: 0.823)
- SMOTE for class imbalance — 0.17% fraud rate
- SHAP explainability on XGBoost
- Apple Silicon MPS acceleration for PyTorch training

### 4. 🧠 LLM Risk Narrative Generator
- Claude API generates compliance-ready risk assessments
- Identifies fraud patterns (probe transactions, velocity anomalies)
- Outputs APPROVE / REVIEW / BLOCK decisions
- Plain-English explanations for non-technical compliance teams

### 5. 📊 Streamlit Dashboard (5 pages)
- Overview — transaction volume, fraud rate time series
- Transaction Explorer — interactive filters by amount, time, type
- Model Comparison — side-by-side performance metrics
- Risk Monitor — high probability flagged transactions
- AI Risk Narratives — Claude-generated assessments

---

## 💡 Why XGBoost Over Neural Network?

| Factor | XGBoost | Neural Network |
|---|---|---|
| Average Precision | **0.853** ✅ | 0.823 |
| ROC-AUC | 0.976 | **0.983** ✅ |
| Precision (Fraud) | 0.41 | **0.73** ✅ |
| Inference Speed | **Fast** ✅ | Slower |
| Explainability | **SHAP** ✅ | Complex |

> *XGBoost chosen for production — Average Precision is the correct metric for severe class imbalance, and SHAP explainability meets compliance requirements.*

---

## 🛠️ Tech Stack

| Area | Tools |
|---|---|
| Data Engineering | Python, SQLite, SQL Window Functions |
| Feature Engineering | Pandas, NumPy, Scikit-learn |
| Machine Learning | XGBoost, Logistic Regression, SHAP |
| Deep Learning | PyTorch, BatchNorm, Dropout, MPS |
| Class Imbalance | SMOTE (imbalanced-learn) |
| GenAI | Anthropic Claude API |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```
fraud-detection-intelligence/
│
├── data/
│   ├── raw/                        # Original Kaggle dataset (not tracked)
│   └── processed/                  # Feature engineered data (not tracked)
│
├── notebooks/
│   ├── 01_eda_and_sql_pipeline.ipynb   # EDA + SQLite warehouse
│   ├── 02_feature_engineering.ipynb    # Velocity + time features
│   ├── 03_baseline_models.ipynb        # LR + XGBoost + SHAP
│   ├── 04_deep_learning.ipynb          # PyTorch neural network
│   └── 05_llm_risk_narrative.ipynb     # Claude API narratives
│
├── src/
│   └── fraud_model.pth             # Saved PyTorch model weights
│
├── dashboard/
│   └── app.py                      # Streamlit dashboard
│
├── reports/                        # Charts and risk narratives
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/sakshiraade/fraud-detection-intelligence.git
cd fraud-detection-intelligence

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Get creditcard.csv from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place it in: data/raw/creditcard.csv

# Run notebooks in order (1 → 5) to generate processed data
# Then run dashboard:
cd dashboard
streamlit run app.py
```

---

## 📓 Run Order

1. `01_eda_and_sql_pipeline.ipynb` — generates SQLite warehouse
2. `02_feature_engineering.ipynb` — generates processed CSV
3. `03_baseline_models.ipynb` — trains LR and XGBoost
4. `04_deep_learning.ipynb` — trains PyTorch model
5. `05_llm_risk_narrative.ipynb` — generates risk narratives (requires Anthropic API key)

---

## 🔑 Environment Variables

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_key_here
```

---

## 👩‍💻 About

Built by **Sakshi Aade** as a portfolio project demonstrating end-to-end fraud detection combining classical ML, deep learning, and LLM integration.

🔗 [LinkedIn](https://www.linkedin.com/in/sakshi-aade/) | 📧 sakshiaade03@gmail
