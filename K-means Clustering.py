import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

files = [
    "C:/Users/Dell/Documents/api_data_aadhar_enrolment_0_500000.csv",
    "C:/Users/Dell/Documents/api_data_aadhar_enrolment_500000_1000000.csv",
    "C:/Users/Dell/Documents/api_data_aadhar_enrolment_1000000_1006029.csv"
]

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

print("Raw dataset shape:", df.shape)

district_df = df.groupby(["state", "district"]).agg({
    "age_0_5": "sum",
    "age_5_17": "sum",
    "age_18_greater": "sum"
}).reset_index()

district_df["total_enrolment"] = (
    district_df["age_0_5"] +
    district_df["age_5_17"] +
    district_df["age_18_greater"]
)

print("District-level dataset shape:", district_df.shape)

features = [
    "age_0_5",
    "age_5_17",
    "age_18_greater",
    "total_enrolment"
]

X = district_df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
K = range(2, 11)

for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

final_kmeans = KMeans(n_clusters=4, random_state=42)
district_df["cluster"] = final_kmeans.fit_predict(X_scaled)

print("\nCluster Distribution:")
print(district_df["cluster"].value_counts())

cluster_profile = district_df.groupby("cluster")[features].mean()

print("\nCluster Profiles (Mean values):")
print(cluster_profile)

cluster_labels = {
    0: "Mega Urban Hubs (Outliers)",
    1: "Youth-Centric Districts",
    2: "Low Enrolment Rural Districts",
    3: "High Enrolment Urban Growth Districts"
}

district_df["cluster_label"] = district_df["cluster"].map(cluster_labels)
