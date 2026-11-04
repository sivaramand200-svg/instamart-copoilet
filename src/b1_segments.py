"""Part B tasks 1-3: RFM, K-means, segment profile + names, PCA, customer_segments.csv."""
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

SNAP = pd.Timestamp("2025-10-01 00:00:00")
o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
comp = o[o.status == "completed"]
rfm = comp.groupby("customer_id").agg(last=("order_ts", "max"), frequency=("order_id", "nunique"),
                                      monetary=("order_amount", "sum")).reset_index()
rfm["recency"] = (SNAP - rfm["last"]).dt.days
rfm["monetary"] = rfm["monetary"].round(2)
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]
print(f"RFM customers: {len(rfm)}")
print(f"recency median={rfm.recency.median()} min={rfm.recency.min()} max={rfm.recency.max()}")
print(f"frequency mean={rfm.frequency.mean():.2f} max={rfm.frequency.max()}")
print(f"monetary mean={rfm.monetary.mean():.2f} max={rfm.monetary.max():.2f}")

X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
print("\nK | inertia | silhouette")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    print(f"{k} | {km.inertia_:.1f} | {silhouette_score(X, km.labels_):.3f}")

K = 4
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), mean_recency=("recency", "mean"),
                                     mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean")).round(2)
print("\nProfile:\n", prof.to_string())

# names assigned from the profile (cluster numbers themselves carry no order)
names, left = {}, set(prof.index)
premium = prof.mean_monetary.idxmax(); names[premium] = "Premium High-Value"; left.discard(premium)
dormant = prof.loc[list(left)].mean_recency.idxmax(); names[dormant] = "Dormant / At-Risk"; left.discard(dormant)
loyal = prof.loc[list(left)].mean_frequency.idxmax(); names[loyal] = "Loyal Regulars"; left.discard(loyal)
for s in left:
    names[s] = "Occasional Shoppers"
rfm["segment_name"] = rfm.segment_id.map(names)
rfm["is_premium"] = rfm.segment_id == premium
print("\nNames:", {int(k): v for k, v in names.items()}, "| premium segment id:", int(premium))

pca = PCA(n_components=2).fit(X)
ev = pca.explained_variance_ratio_
print(f"\nPCA explained variance: {ev[0]:.3f} {ev[1]:.3f} total {ev.sum():.3f}")
rfm.to_csv("outputs/customer_segments.csv", index=False)
print("premium customers:", int(rfm.is_premium.sum()))
