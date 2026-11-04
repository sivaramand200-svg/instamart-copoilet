"""Part A tasks 3-5: clean customers and orders, write CSVs + SQLite."""
import sqlite3
import pandas as pd

R, C = "data/raw/", "data/clean/"
log = {}

# ---------------- customers ----------------
cust = pd.read_csv(R + "customers.csv")
before = cust["city"].nunique()
cust["city"] = cust["city"].str.strip().str.title()
log["city: spellings before -> after"] = (before, cust["city"].nunique())
gmap = {"m": "male", "male": "male", "f": "female", "female": "female"}
cust["gender"] = cust["gender"].fillna("").str.strip().str.lower().map(gmap).fillna("unknown")
bad_age = (cust["age"] < 10) | (cust["age"] > 100)
log["age: out-of-range set to missing"] = int(bad_age.sum())
cust.loc[bad_age, "age"] = float("nan")
log["age: missing before fill"] = int(cust["age"].isna().sum())
cust["age"] = cust["age"].fillna(cust.groupby("city")["age"].transform("median"))
cust["age"] = cust["age"].round().astype(int)       # medians such as 38.5 -> whole years
cust["signup_date"] = pd.to_datetime(cust["signup_date"])

# ---------------- products ----------------
prod = pd.read_csv(R + "products.csv")

# ---------------- orders ----------------
o = pd.read_csv(R + "orders.csv", dtype={"order_amount": "string"}, keep_default_na=False, na_values=[""])
n0 = len(o)
o["order_ts"] = pd.to_datetime(o["order_ts"])                      # rule 1
o = o.drop_duplicates(); log["orders: exact duplicates removed"] = n0 - len(o)   # rule 2
n1 = len(o)
o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")  # rule 3
log["orders: older records dropped (latest order_ts kept)"] = n1 - len(o)
o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")             # rule 4
miss = o["order_amount"].isna()
log["orders: unparseable/missing amounts refilled from price"] = int(miss.sum())
o = o.merge(prod[["product_id", "category", "price_inr"]], on="product_id", how="left")
calc = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
o["order_amount"] = o["order_amount"].fillna(calc)
neg = o["delivery_minutes"] < 0                                                    # rule 5
log["orders: negative delivery_minutes set to missing"] = int(neg.sum())
o.loc[neg, "delivery_minutes"] = float("nan")
comp = o["status"] == "completed"
log["orders: completed rows with missing delivery before fill"] = int((comp & o["delivery_minutes"].isna()).sum())
med = o["delivery_minutes"].where(comp).groupby(o["category"]).transform("median")
o["delivery_minutes"] = o["delivery_minutes"].where(~(comp & o["delivery_minutes"].isna()), med)
o = o.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)

# ---------------- store ----------------
out_c, out_o = cust.copy(), o.copy()
out_c["signup_date"] = out_c["signup_date"].dt.strftime("%Y-%m-%d")
out_o["order_ts"] = out_o["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
out_c.to_csv(C + "customers.csv", index=False)
prod.to_csv(C + "products.csv", index=False)
out_o.to_csv(C + "orders.csv", index=False)
import os
if os.path.exists("data/instamart.db"):
    os.remove("data/instamart.db")
con = sqlite3.connect("data/instamart.db")
out_c.to_sql("customers", con, index=False)
prod.to_sql("products", con, index=False)
out_o.to_sql("orders", con, index=False)
con.close()

print("Cleaning log:")
for k, v in log.items():
    print(f"  {k}: {v}")
print("cities:", cust["city"].value_counts().to_dict())
print("genders:", cust["gender"].value_counts().to_dict())
print("age min/max/missing:", cust["age"].min(), cust["age"].max(), int(cust["age"].isna().sum()))
print("orders:", len(o), o["status"].value_counts().to_dict())
c = o[o["status"] == "completed"]
print("completed amount sum:", round(c["order_amount"].sum(), 2))
print("missing amount:", int(o["order_amount"].isna().sum()),
      "| missing delivery completed:", int(c["delivery_minutes"].isna().sum()),
      "| cancelled:", int(o.loc[o.status == "cancelled", "delivery_minutes"].isna().sum()))
