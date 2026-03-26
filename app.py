import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="B2B Risk & Churn Dashboard", layout="wide")
st.title("B2B Client Risk & Churn Intelligence Dashboard")

# -----------------------------
# Load data (Part A)
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("B2B_Client_Churn_5000.csv")

df = load_data()

st.caption("Dataset loaded successfully.")

# -----------------------------
# Part B: Risk Scoring Logic (business rule formula)
# -----------------------------
# Risk logic:
# - Payment delay high => higher risk
# - Usage low => higher risk
# - Contract short => higher risk
# - Support tickets high => higher risk

def calc_risk_score(row):
    score = 0

    # Payment Delay
    if row["Payment_Delay_Days"] > 30:
        score += 3
    elif row["Payment_Delay_Days"] > 10:
        score += 2
    elif row["Payment_Delay_Days"] > 0:
        score += 1

    # Low Usage
    if row["Monthly_Usage_Score"] < 40:
        score += 3
    elif row["Monthly_Usage_Score"] < 60:
        score += 2
    elif row["Monthly_Usage_Score"] < 75:
        score += 1

    # Short Contract
    if row["Contract_Length_Months"] < 6:
        score += 3
    elif row["Contract_Length_Months"] < 12:
        score += 2
    elif row["Contract_Length_Months"] < 18:
        score += 1

    # Support Tickets (complaints)
    if row["Support_Tickets_Last30Days"] > 6:
        score += 3
    elif row["Support_Tickets_Last30Days"] > 3:
        score += 2
    elif row["Support_Tickets_Last30Days"] > 0:
        score += 1

    return score

df["Risk_Score"] = df.apply(calc_risk_score, axis=1)

def risk_bucket(score):
    if score >= 9:
        return "High Risk"
    elif score >= 5:
        return "Medium Risk"
    else:
        return "Low Risk"

df["Risk_Category_New"] = df["Risk_Score"].apply(risk_bucket)

# -----------------------------
# Part D: Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

region_filter = st.sidebar.multiselect(
    "Region", sorted(df["Region"].dropna().unique()), default=sorted(df["Region"].dropna().unique())
)

industry_filter = st.sidebar.multiselect(
    "Industry", sorted(df["Industry"].dropna().unique()), default=sorted(df["Industry"].dropna().unique())
)

risk_filter = st.sidebar.multiselect(
    "Risk Category", ["Low Risk", "Medium Risk", "High Risk"], default=["Low Risk", "Medium Risk", "High Risk"]
)

filtered = df[
    (df["Region"].isin(region_filter)) &
    (df["Industry"].isin(industry_filter)) &
    (df["Risk_Category_New"].isin(risk_filter))
].copy()

# -----------------------------
# KPI Cards (Part D)
# -----------------------------
total_clients = len(filtered)
high_risk_clients = (filtered["Risk_Category_New"] == "High Risk").sum()
avg_revenue = filtered["Monthly_Revenue_USD"].mean()

# Churn rate based on Renewal_Status ("No" means churn)
churn_rate_pct = (filtered["Renewal_Status"].eq("No").mean() * 100) if total_clients > 0 else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Clients", f"{total_clients}")
k2.metric("High Risk Clients", f"{high_risk_clients}")
k3.metric("Churn Rate % (from Renewal_Status)", f"{churn_rate_pct:.2f}%")
k4.metric("Avg Revenue per Client", f"${avg_revenue:,.2f}")

st.divider()

# -----------------------------
# Visualizations (Part D)
# -----------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Risk Category Distribution (Bar Chart)")
    counts = filtered["Risk_Category_New"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0)

    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Risk Category")
    ax.set_ylabel("Number of Clients")
    st.pyplot(fig)

with c2:
    st.subheader("Industry-wise Risk Analysis")
    # simple pivot count
    pivot = pd.pivot_table(
        filtered,
        index="Industry",
        columns="Risk_Category_New",
        values="Client_ID",
        aggfunc="count",
        fill_value=0
    ).reindex(columns=["Low Risk", "Medium Risk", "High Risk"], fill_value=0)

    st.dataframe(pivot, use_container_width=True)

st.divider()

c3, c4 = st.columns(2)

with c3:
    st.subheader("Revenue vs Risk (Scatter)")
    fig2, ax2 = plt.subplots()
    ax2.scatter(filtered["Monthly_Revenue_USD"], filtered["Risk_Score"])
    ax2.set_xlabel("Monthly Revenue (USD)")
    ax2.set_ylabel("Risk Score")
    st.pyplot(fig2)

with c4:
    st.subheader("Contract Length vs Churn")
    # churn = Renewal_Status == "No"
    churn_df = filtered.copy()
    churn_df["Churned"] = churn_df["Renewal_Status"].map({"Yes": 0, "No": 1})

    fig3, ax3 = plt.subplots()
    ax3.scatter(churn_df["Contract_Length_Months"], churn_df["Churned"])
    ax3.set_xlabel("Contract Length (Months)")
    ax3.set_ylabel("Churned (1=Yes, 0=No)")
    st.pyplot(fig3)

st.divider()

# -----------------------------
# Part C: Machine Learning (Decision Tree)
# Predict Renewal_Status
# -----------------------------
st.subheader("Machine Learning: Decision Tree Churn Prediction")

# We will use both numeric + categorical using get_dummies
target = df["Renewal_Status"].map({"Yes": 1, "No": 0})

features = df[[
    "Industry", "Region", "Plan", "Lead_Source",
    "Account_Age_Months", "Contract_Length_Months",
    "Monthly_Usage_Score", "Support_Tickets_Last30Days",
    "Payment_Delay_Days", "Monthly_Revenue_USD",
    "Risk_Score"
]].copy()

X = pd.get_dummies(features, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, target, test_size=0.2, random_state=42, stratify=target
)

model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)
st.write(f"**Accuracy:** {acc:.4f}")

cm = confusion_matrix(y_test, pred)
st.write("**Confusion Matrix:** (rows = actual, columns = predicted)")
st.write(cm)

# Feature Importance (Top 10)
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
st.write("**Top Factors Influencing Churn (Feature Importance - Top 10):**")
st.bar_chart(importances)

st.info(
    "Interpretation: Higher importance means that factor is more influential in predicting renewal/churn."
)

st.divider()

# -----------------------------
# Part D: Table - Top 20 High Risk Clients
# -----------------------------
st.subheader("Top 20 High-Risk Clients")
top20 = filtered.sort_values(["Risk_Score", "Monthly_Revenue_USD"], ascending=[False, False]).head(20)

show_cols = [
    "Client_ID", "Company_Name", "Industry", "Region",
    "Monthly_Usage_Score", "Payment_Delay_Days",
    "Contract_Length_Months", "Support_Tickets_Last30Days",
    "Monthly_Revenue_USD", "Risk_Score", "Risk_Category_New", "Renewal_Status"
]

st.dataframe(top20[show_cols], use_container_width=True)

st.divider()

# -----------------------------
# Part E: AI-Based Retention Suggestions (button)
# -----------------------------
st.subheader("AI-Based Retention Suggestions")

if st.button("Generate Retention Strategy"):
    st.write("Here are practical retention actions based on common risk patterns:")
    st.write("1) If payment delay > 30 days: offer flexible payment plan + early-payment discount.")
    st.write("2) If usage is low: provide onboarding refresh, training session, and usage nudges.")
    st.write("3) If support tickets are high: assign dedicated account manager + priority support.")
    st.write("4) If contract is short: offer longer-term contract incentive (discount or added features).")
    st.write("5) For high-revenue high-risk clients: schedule executive check-in + custom success plan.")

st.divider()

# -----------------------------
# Part F: Responsible AI Section
# -----------------------------
st.subheader("Responsible AI: Ethical Implications of Predicting Client Churn")

st.write("""
**1) Bias in Predictive Models:**  
If some regions/industries historically churn more (maybe due to market conditions), the model may learn unfair patterns and label them as risky.

**2) Harm of Labeling Clients as “High Risk”:**  
Teams might treat “high risk” clients differently (less attention or stricter rules), which could itself increase churn. So the score should guide support, not punish clients.

**3) Data Privacy & Trust:**  
Client data (usage, payments, tickets) must be protected. Only authorized people should access this dashboard and data should be handled securely.

**4) Responsible Decision-Making:**  
Predictions are not facts. They should be used with human judgment and business context (client relationships, market changes, special situations).

**5) Transparency & Audits:**  
The company should explain what factors drive risk, review model performance regularly, and check whether any group is being unfairly targeted.
""")
