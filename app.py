import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="UIDAI Aadhaar Enrolment Decision Support", layout="wide")
st.title("📊 UIDAI Aadhaar Enrolment Decision Support System")
st.caption("Status → Risk → Decision Framework")

# -------------------------------------------------
# Load Data
# -------------------------------------------------
@st.cache_data
def load_data():
    files = ["data/uidai_part1.csv", "data/uidai_part2.csv", "data/uidai_part3.csv"]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df = load_data()

# -------------------------------------------------
# Data Cleaning
# -------------------------------------------------
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
df = df.dropna(subset=["date"])

# ---- State Normalisation (ROBUST) ----
STATE_FIX = {
    "DELHI": "Delhi",
    "NCT OF DELHI": "Delhi",
    "ORISSA": "Odisha",
    "ODISHA": "Odisha",
    "JAMMU & KASHMIR": "Jammu And Kashmir",
    "JAMMU AND KASHMIR": "Jammu And Kashmir",
    "ANDAMAN & NICOBAR ISLANDS": "Andaman And Nicobar Islands",
    "DADRA & NAGAR HAVELI": "Dadra And Nagar Haveli And Daman And Diu",
    "DAMAN & DIU": "Dadra And Nagar Haveli And Daman And Diu",

    # ✅ West Bengal variants
    "WEST BENGAL": "West Bengal",
    "WEST  BENGAL": "West Bengal",
    "WESTBENGAL": "West Bengal",
    "WEST BANGAL": "West Bengal"
}

df["state"] = (
    df["state"]
    .astype(str)
    .str.upper()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
    .replace(STATE_FIX)
    .str.title()
)

df["district"] = df["district"].astype(str).str.strip().str.title()

# ❌ Remove corrupted states like "100000"
df = df[df["state"].apply(lambda x: isinstance(x, str) and not x.isdigit())]

df = df.drop_duplicates()

# -------------------------------------------------
# Feature Engineering
# -------------------------------------------------
df["total_enrolments"] = df["age_0_5"] + df["age_5_17"] + df["age_18_greater"]
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# ❌ Remove extreme enrolment outliers
MAX_ENROLMENT = 50000
df = df[df["total_enrolments"] < MAX_ENROLMENT]

# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
st.sidebar.header("⚙️ Controls")
analysis_level = st.sidebar.radio("Analysis Level", ["State Level", "District Level"])
high_risk_prob = st.sidebar.slider("High Risk Threshold", 0.05, 0.5, 0.30, 0.05)
medium_risk_prob = st.sidebar.slider("Medium Risk Threshold", 0.30, 0.8, 0.50, 0.05)

states = sorted(df["state"].unique())
selected_states = st.sidebar.multiselect("Select States", states)

df_filtered = df[df["state"].isin(selected_states)] if selected_states else df.copy()

if analysis_level == "District Level" and selected_states:
    districts = sorted(df_filtered["district"].unique())
    selected_districts = st.sidebar.multiselect("Select Districts", districts)
    if selected_districts:
        df_filtered = df_filtered[df_filtered["district"].isin(selected_districts)]

# -------------------------------------------------
# Key Metrics
# -------------------------------------------------
st.header("📌 Key Metrics")
c1, c2, c3 = st.columns(3)
c1.metric("Records Analysed", len(df_filtered))
c2.metric("Total Enrolments", int(df_filtered["total_enrolments"].sum()))
c3.metric("Districts Covered", df_filtered["district"].nunique())

# -------------------------------------------------
# Age-wise Enrolment
# -------------------------------------------------
st.header("📊 Age Group-wise Aadhaar Enrolment")
age_df = pd.DataFrame({
    "Age Group": ["0–5", "5–17", "18+"],
    "Enrolments": [
        df_filtered["age_0_5"].sum(),
        df_filtered["age_5_17"].sum(),
        df_filtered["age_18_greater"].sum()
    ]
})
st.plotly_chart(px.bar(age_df, x="Age Group", y="Enrolments", text="Enrolments"),
                use_container_width=True)

# -------------------------------------------------
# Early Age Enrolment Analysis
# -------------------------------------------------
st.header("👶 Early-Age Enrolment Analysis")
early_df = df_filtered.groupby("state")[["age_0_5", "age_5_17"]].sum().reset_index()
early_df["early_ratio"] = early_df["age_0_5"] / (early_df["age_0_5"] + early_df["age_5_17"])
early_df["early_ratio"] = early_df["early_ratio"].fillna(0)

def early_reco(r):
    if r < 0.30:
        return "Urgent: Hospital-based enrolment"
    elif r < 0.35:
        return "Mobile enrolment & awareness"
    return "Maintain"

early_df["recommendation"] = early_df["early_ratio"].apply(early_reco)

early_df = early_df.sort_values("early_ratio")

st.plotly_chart(
    px.bar(
        pd.concat([early_df.head(10), early_df.tail(10)]),
        x="state",
        y="early_ratio",
        text_auto=".2f",
        title="Early-Age Enrolment Ratio (Lowest & Highest States)"
    ),
    use_container_width=True
)

st.dataframe(early_df, use_container_width=True, hide_index=True)


# -------------------------------------------------
st.header("🚨 Enrolment Risk Assessment & Decision Support")

# -----------------------------
# Grouping columns
# -----------------------------
group_cols = ["state", "year", "month"]
if analysis_level == "District Level":
    group_cols.insert(1, "district")

# -----------------------------
# Monthly aggregation
# -----------------------------
monthly_df = (
    df_filtered
    .groupby(group_cols)["total_enrolments"]
    .sum()
    .reset_index()
    .sort_values(group_cols)
)

# -----------------------------
# Time-series features
# -----------------------------
monthly_df["lag_1"] = monthly_df.groupby(group_cols[:-2])["total_enrolments"].shift(1)
monthly_df["lag_2"] = monthly_df.groupby(group_cols[:-2])["total_enrolments"].shift(2)
monthly_df["rolling_mean_3"] = (
    monthly_df
    .groupby(group_cols[:-2])["total_enrolments"]
    .transform(lambda x: x.rolling(3).mean())
)
monthly_df["next_month"] = monthly_df.groupby(group_cols[:-2])["total_enrolments"].shift(-1)

monthly_df = monthly_df.dropna()

# -----------------------------
# Trend target
# -----------------------------
monthly_df["trend"] = (
    monthly_df["next_month"] > monthly_df["total_enrolments"]
).astype(int)

# -----------------------------
# Encode state
# -----------------------------
le = LabelEncoder()
monthly_df["state_encoded"] = le.fit_transform(monthly_df["state"])

# -----------------------------
# Model
# -----------------------------
X = monthly_df[["lag_1", "lag_2", "rolling_mean_3", "month", "state_encoded"]]
y = monthly_df["trend"]

if y.nunique() > 1 and len(y) > 10:
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    monthly_df["prob_increase"] = model.predict_proba(X)[:, 1]
else:
    monthly_df["prob_increase"] = 0.5

# -----------------------------
# Risk & action logic
# -----------------------------
def risk_fn(p):
    if p < high_risk_prob:
        return "🔴 High Risk"
    elif p < medium_risk_prob:
        return "🟡 Medium Risk"
    return "🟢 Low Risk"

def action_fn(r):
    if r == "🔴 High Risk":
        return "Deploy mobile camps & audit"
    elif r == "🟡 Medium Risk":
        return "Awareness & monitoring"
    return "Maintain"

monthly_df["risk"] = monthly_df["prob_increase"].apply(risk_fn)
monthly_df["action"] = monthly_df["risk"].apply(action_fn)

# =================================================
# ✅ CRITICAL FIX: Use ONLY latest record per state
# =================================================
latest_df = (
    monthly_df
    .sort_values(["state", "year", "month"])
    .groupby("state", as_index=False)
    .tail(1)
)

# -----------------------------
# Risk summary (CORRECT COUNTS)
# -----------------------------
risk_counts = latest_df.groupby("risk")["state"].nunique()

c1, c2, c3 = st.columns(3)
c1.metric("🔴 High Risk States", int(risk_counts.get("🔴 High Risk", 0)))
c2.metric("🟡 Medium Risk States", int(risk_counts.get("🟡 Medium Risk", 0)))
c3.metric("🟢 Low Risk States", int(risk_counts.get("🟢 Low Risk", 0)))

# -----------------------------
# Balanced decision table
# -----------------------------
table_df = pd.concat([
    latest_df[latest_df["risk"] == "🔴 High Risk"]
        .sort_values("prob_increase")
        .head(10),
    latest_df[latest_df["risk"] == "🟡 Medium Risk"]
        .sort_values("prob_increase")
        .head(8),
    latest_df[latest_df["risk"] == "🟢 Low Risk"]
        .sort_values("prob_increase")
        .head(7)
])

# -----------------------------
# Column ordering
# -----------------------------
priority_cols = ["state"]
if "district" in table_df.columns:
    priority_cols.append("district")

priority_cols += ["prob_increase", "risk", "action"]

display_cols = priority_cols + [
    c for c in table_df.columns if c not in priority_cols
]

table_df = table_df[display_cols].reset_index(drop=True)

# -----------------------------
# Final clean column selection
# -----------------------------
final_cols = ["state"]

# include district only if it exists (district-level mode)
if "district" in table_df.columns:
    final_cols.append("district")

# decision-focused columns
final_cols += [
    "prob_increase",
    "risk",
    "action",
    "total_enrolments",
    "year",
    "month"
]

# keep only existing columns (safety check)
final_cols = [c for c in final_cols if c in table_df.columns]

# apply column filter
table_df = table_df[final_cols]

# -----------------------------
# Sort ascending by probability (default view)
# -----------------------------
table_df = table_df.sort_values("prob_increase", ascending=True).reset_index(drop=True)

# -----------------------------
# Display
# -----------------------------
st.dataframe(table_df, use_container_width=True)






st.header("🗺️ State-wise Aadhaar Enrolment Intensity ")
state_summary = df.groupby("state")["total_enrolments"].sum().reset_index()
india_map_img = Image.open("data/india_map.jpg")

STATE_POSITIONS = {
    "Jammu And Kashmir": (0.50, 0.09),
    "Punjab": (0.47, 0.22),
    "Haryana": (0.49, 0.27),
    "Delhi": (0.50, 0.28),
    "Uttar Pradesh": (0.60, 0.34),
    "Rajasthan": (0.42, 0.37),
    "Gujarat": (0.38, 0.45),
    "Madhya Pradesh": (0.50, 0.46),
    "Bihar": (0.75, 0.37),
    "West Bengal": (0.8, 0.45),
    "Odisha": (0.75, 0.50),
    "Maharashtra": (0.46, 0.58),
    "Telangana": (0.55, 0.62),
    "Andhra Pradesh": (0.55, 0.72),
    "Karnataka": (0.46, 0.72),
    "Tamil Nadu": (0.55, 0.82),
    "Kerala": (0.47, 0.82),
    "Assam": (0.953, 0.35)
}

map_df = state_summary.copy()
map_df["x"] = map_df["state"].map(lambda s: STATE_POSITIONS.get(s, (None, None))[0])
map_df["y"] = map_df["state"].map(lambda s: STATE_POSITIONS.get(s, (None, None))[1])
map_df = map_df.dropna()

fig = px.scatter(
    map_df,
    x="x",
    y="y",
    size="total_enrolments",
    color="total_enrolments",
    hover_name="state",
    hover_data={"x": False, "y": False, "total_enrolments": ":,"}
)

fig.update_xaxes(visible=False, range=[0, 1],fixedrange=True)
fig.update_yaxes(
    visible=False,
    range=[1, 0],
    fixedrange=True,
    scaleanchor="x",
    scaleratio=1
)

fig.add_layout_image(dict(source=india_map_img, xref="paper", yref="paper", x=0.4, y=1, sizex=1, sizey=1, layer="below"))
fig.update_layout(
    autosize=False,
    margin=dict(l=0, r=0, t=30, b=0)
)

st.plotly_chart(
    fig,
    config={
        "displayModeBar": False,
        "scrollZoom": False
    }
)



st.header("🧠 District-Level Enrolment Pattern Discovery")

district_df = df.groupby(["state", "district"]).agg({
    "age_0_5": "sum",
    "age_5_17": "sum",
    "age_18_greater": "sum"
}).reset_index()

district_df["total_enrolment"] = district_df["age_0_5"] + district_df["age_5_17"] + district_df["age_18_greater"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(district_df[["age_0_5", "age_5_17", "age_18_greater", "total_enrolment"]])

kmeans = KMeans(n_clusters=4, random_state=42)
district_df["cluster"] = kmeans.fit_predict(X_scaled)

cluster_labels = {
    0: "Low Enrolment Rural Districts",
    1: "Youth-Dominant Districts",
    2: "Balanced Growth Districts",
    3: "High Enrolment Urban Hubs"
}

district_df["cluster_label"] = district_df["cluster"].map(cluster_labels)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 6), facecolor="black")
ax = plt.gca()
ax.set_facecolor("black")

for cid, label in cluster_labels.items():
    subset = X_pca[district_df["cluster"] == cid]
    plt.scatter(
        subset[:, 0],
        subset[:, 1],
        label=label,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.5
    )

plt.xlabel("PCA Component 1(Overall Enrolment Pattern)", color="white")
plt.ylabel("PCA Component 2(Demographic Variation)", color="white")
plt.title("District-wise Aadhaar Enrolment Clusters (PCA View)", color="white")

plt.tick_params(colors="white")

for spine in ax.spines.values():
    spine.set_color("white")

legend = plt.legend(title="Cluster Type")
legend.get_frame().set_facecolor("black")
legend.get_frame().set_edgecolor("white")
legend.get_title().set_color("white")
for text in legend.get_texts():
    text.set_color("white")

st.pyplot(plt)


st.download_button(
    "⬇️ Download Decision Dataset",
    monthly_df.to_csv(index=False),
    "uidai_decision_ready_data.csv",
    "text/csv"
)
