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

st.set_page_config(page_title="UIDAI Aadhaar Enrolment Decision Support", layout="wide")
st.title("📊 UIDAI Aadhaar Enrolment Decision Support System")
st.caption("Status → Risk → Decision Framework")

@st.cache_data
def load_data():
    files = ["data/uidai_part1.csv", "data/uidai_part2.csv", "data/uidai_part3.csv"]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df = load_data()

df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
df = df.dropna(subset=["date"])

STATE_FIX = {
    "DELHI": "Delhi",
    "NCT OF DELHI": "Delhi",
    "ORISSA": "Odisha",
    "ODISHA": "Odisha",
    "JAMMU & KASHMIR": "Jammu And Kashmir",
    "JAMMU AND KASHMIR": "Jammu And Kashmir",
    "ANDAMAN & NICOBAR ISLANDS": "Andaman And Nicobar Islands",
    "DADRA & NAGAR HAVELI": "Dadra And Nagar Haveli And Daman And Diu",
    "DAMAN & DIU": "Dadra And Nagar Haveli And Daman And Diu"
}

df["state"] = df["state"].str.upper().str.strip().replace(STATE_FIX).str.title()
df["district"] = df["district"].str.strip().str.title()
df = df.drop_duplicates()

df["total_enrolments"] = df["age_0_5"] + df["age_5_17"] + df["age_18_greater"]
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

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

st.header("📌 Key Metrics")
c1, c2, c3 = st.columns(3)
c1.metric("Records", len(df_filtered))
c2.metric("Total Enrolments", int(df_filtered["total_enrolments"].sum()))
c3.metric("Districts Covered", df_filtered["district"].nunique())

st.header("📊 Age-wise Aadhaar Enrolments")
age_df = pd.DataFrame({
    "Age Group": ["0–5", "5–17", "18+"],
    "Enrolments": [
        df_filtered["age_0_5"].sum(),
        df_filtered["age_5_17"].sum(),
        df_filtered["age_18_greater"].sum()
    ]
})
st.plotly_chart(px.bar(age_df, x="Age Group", y="Enrolments", text="Enrolments"), use_container_width=True)

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

st.plotly_chart(px.bar(early_df, x="state", y="early_ratio", text_auto=".2f"), use_container_width=True)
st.dataframe(early_df, use_container_width=True)

st.header("🚨 Enrolment Risk Assessment & Decision Support")

group_cols = ["state", "year", "month"]
if analysis_level == "District Level":
    group_cols.insert(1, "district")

monthly_df = df_filtered.groupby(group_cols)["total_enrolments"].sum().reset_index().sort_values(group_cols)
monthly_df["lag_1"] = monthly_df.groupby(group_cols[:-2])["total_enrolments"].shift(1)
monthly_df["lag_2"] = monthly_df.groupby(group_cols[:-2])["total_enrolments"].shift(2)
monthly_df["rolling_mean_3"] = monthly_df.groupby(group_cols[:-2])["total_enrolments"].transform(lambda x: x.rolling(3).mean())
monthly_df["next_month"] = monthly_df.groupby(group_cols[:-2])["total_enrolments"].shift(-1)
monthly_df = monthly_df.dropna()

monthly_df["trend"] = (monthly_df["next_month"] > monthly_df["total_enrolments"]).astype(int)

le = LabelEncoder()
monthly_df["state_encoded"] = le.fit_transform(monthly_df["state"])

X = monthly_df[["lag_1", "lag_2", "rolling_mean_3", "month", "state_encoded"]]
y = monthly_df["trend"]

if y.nunique() > 1 and len(y) > 10:
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    monthly_df["prob_increase"] = model.predict_proba(X)[:, 1]
else:
    monthly_df["prob_increase"] = 0.5

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

st.dataframe(
    monthly_df.sort_values("prob_increase").head(25)[
        group_cols + ["total_enrolments", "prob_increase", "risk", "action"]
    ],
    use_container_width=True
)

st.header("🗺️ State-wise Aadhaar Enrolment Intensity Map")
state_summary = df.groupby("state")["total_enrolments"].sum().reset_index()
india_map_img = Image.open("data/india_map.jpg")

STATE_POSITIONS = {
    "Jammu And Kashmir": (0.50, 0.09),
    "Punjab": (0.48, 0.22),
    "Haryana": (0.49, 0.27),
    "Delhi": (0.50, 0.28),
    "Uttar Pradesh": (0.54, 0.34),
    "Rajasthan": (0.44, 0.37),
    "Gujarat": (0.44, 0.45),
    "Madhya Pradesh": (0.50, 0.46),
    "Bihar": (0.60, 0.37),
    "West Bengal": (0.614, 0.45),
    "Odisha": (0.58, 0.50),
    "Maharashtra": (0.46, 0.58),
    "Telangana": (0.51, 0.62),
    "Andhra Pradesh": (0.51, 0.72),
    "Karnataka": (0.48, 0.72),
    "Tamil Nadu": (0.51, 0.84),
    "Kerala": (0.49, 0.82),
    "Assam": (0.676, 0.35)
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

fig.update_xaxes(visible=False, range=[0, 1])
fig.update_yaxes(visible=False, range=[1, 0])
fig.add_layout_image(dict(source=india_map_img, xref="paper", yref="paper", x=0.4, y=1, sizex=1, sizey=1, layer="below"))
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(
    fig,
    use_container_width=True,
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

plt.xlabel("PCA Component 1", color="white")
plt.ylabel("PCA Component 2", color="white")
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
