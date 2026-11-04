"""Part B tasks 4-8: lapse problem, leakage-free preprocessing, tuning, balancing, persistence."""
import joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SELECTED = "rf"      # model chosen after reading the test table below (justification in README)
CUT, END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")
o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
cust = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
prod = pd.read_csv("data/clean/products.csv")
o = o.merge(prod[["product_id", "category"]], on="product_id")

pre = o[o.order_ts < CUT]                       # ONLY orders before the cutoff feed the features
pre_c = pre[pre.status == "completed"]
feat = pre_c.groupby("customer_id").agg(last=("order_ts", "max"), frequency=("order_id", "nunique"),
                                        monetary=("order_amount", "sum"), avg_discount=("discount_pct", "mean"),
                                        avg_delivery=("delivery_minutes", "mean")).reset_index()
feat["recency"] = (CUT - feat["last"]).dt.days
canc = pre[pre.status == "cancelled"].groupby("customer_id").size().rename("n_canc")
fav = (pre_c.groupby(["customer_id", "category"]).size().reset_index(name="n")
       .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
       .drop_duplicates("customer_id").set_index("customer_id")["category"].rename("fav_category"))
feat = feat.merge(canc, on="customer_id", how="left").merge(fav, on="customer_id", how="left")
feat["n_canc"] = feat["n_canc"].fillna(0)
feat["cancel_rate"] = feat["n_canc"] / (feat["frequency"] + feat["n_canc"])
feat = feat.merge(cust, on="customer_id", how="left")
feat["tenure_days"] = (CUT - feat["signup_date"]).dt.days
# label: no completed order in September 2025
sept = o[(o.status == "completed") & (o.order_ts >= CUT) & (o.order_ts < END)].customer_id.unique()
feat["lapsed"] = (~feat.customer_id.isin(sept)).astype(int)

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X, y = feat[NUM + CAT], feat["lapsed"]
print(f"cohort={len(feat)} lapsed={int(y.sum())} ({y.mean()*100:.2f}%)")
assert X.isna().sum().sum() == 0
print("missing values in any feature column:", int(X.isna().sum().sum()))
assert feat["last"].max() < CUT          # nothing on/after the cutoff in features

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"train={len(Xtr)} test={len(Xte)}")
base = np.zeros(len(yte), int)
print(f"baseline (always not-lapsed): acc={accuracy_score(yte, base)*100:.1f}% F1={f1_score(yte, base, zero_division=0):.2f}")

def make_pre():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])
p = make_pre().fit(Xtr)                         # fitted on TRAIN rows only
print(f"scaler mean(frequency) learned on train={p.named_transformers_['num'].mean_[1]:.3f} | full cohort mean={feat.frequency.mean():.3f}")

def metrics(m, Xe, ye):
    pr = m.predict(Xe)
    return dict(tn_fp_fn_tp=confusion_matrix(ye, pr).ravel().tolist(), precision=precision_score(ye, pr, zero_division=0),
                recall=recall_score(ye, pr, zero_division=0), f1=f1_score(ye, pr, zero_division=0))

lr_pipe = Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000))])
lr = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(Xtr, ytr)
rf_pipe = Pipeline([("pre", make_pre()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
rf = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(Xtr, ytr)
print("LR best:", lr.best_params_, f"cv F1={lr.best_score_:.3f}")
print("RF best:", rf.best_params_, f"cv F1={rf.best_score_:.3f}")
res = {"logreg": metrics(lr.best_estimator_, Xte, yte), "rf": metrics(rf.best_estimator_, Xte, yte)}
for k, v in res.items():
    print(f"TEST {k}: [tn,fp,fn,tp]={v['tn_fp_fn_tp']} precision={v['precision']:.3f} recall={v['recall']:.3f} F1={v['f1']:.3f}")

# ---- class balance experiment (training rows only)
tr = Xtr.copy(); tr["lapsed"] = ytr
pos, neg = tr[tr.lapsed == 1], tr[tr.lapsed == 0]
extra = pos.sample(n=len(neg) - len(pos), replace=True, random_state=42)
bal = pd.concat([tr, extra]); print(f"balanced train counts: {bal.lapsed.value_counts().to_dict()}")
lr_bal = clone(lr.best_estimator_).fit(bal[NUM + CAT], bal["lapsed"])
mb = metrics(lr_bal, Xte, yte)
mu = res["logreg"]
print(f"LR un-balanced: P={mu['precision']:.3f} R={mu['recall']:.3f} F1={mu['f1']:.3f}")
print(f"LR balanced   : P={mb['precision']:.3f} R={mb['recall']:.3f} F1={mb['f1']:.3f}  cm={mb['tn_fp_fn_tp']}")

# ---- interpretation
names = rf.best_estimator_.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf.best_estimator_.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\nRF top-5 importances:\n", imp.head(5).round(4).to_string())
coef = pd.Series(lr.best_estimator_.named_steps["clf"].coef_[0], index=names)
print("\nLR coefficients (numeric, standardised):\n", coef[[f"num__{c}" for c in NUM]].round(3).to_string())

# ---- persist + reload
model = {"logreg": lr, "rf": rf}[SELECTED].best_estimator_
joblib.dump(model, "outputs/lapse_model.joblib")
re = joblib.load("outputs/lapse_model.joblib")
same = bool((re.predict(Xte) == model.predict(Xte)).all()) and bool(np.allclose(re.predict_proba(Xte), model.predict_proba(Xte)))
print("\nreloaded model reproduces test predictions exactly:", same); assert same

# ---- retention list (Premium flag is post-hoc, whole-window)
seg = pd.read_csv("outputs/customer_segments.csv")
prem = set(seg[seg.is_premium].customer_id)
te = feat.loc[Xte.index, ["customer_id"]].copy()
te["lapse_probability"] = model.predict_proba(Xte)[:, 1].round(4)
te["actually_lapsed"] = yte.values
cand = te[te.customer_id.isin(prem)].sort_values("lapse_probability", ascending=False)
print(f"premium customers in test set: {len(cand)}")
top = cand.head(10); top.to_csv("outputs/retention_list.csv", index=False)
print(f"retention list rows={len(top)} truly lapsed={int(top.actually_lapsed.sum())}")
print(top.to_string(index=False))
