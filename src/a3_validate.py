"""Part A task 5: assert the reference checkpoints."""
import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
c = pd.read_sql_query("SELECT * FROM customers", con)
o = pd.read_sql_query("SELECT * FROM orders", con)
assert {"customers", "products", "orders"} <= set(pd.read_sql_query("SELECT name FROM sqlite_master", con)["name"])
comp = o[o.status == "completed"]
checks = {
    "city counts": c.city.value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184},
    "gender counts": c.gender.value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30},
    "no missing ages": int(c.age.isna().sum()) == 0,
    "ages 18-60": c.age.between(18, 60).all(),
    "12,309 order rows": len(o) == 12309,
    "unique order_id": o.order_id.is_unique,
    "11,110 completed": len(comp) == 11110,
    "1,199 cancelled": int((o.status == "cancelled").sum()) == 1199,
    "no pending": int((o.status == "pending").sum()) == 0,
    "0 missing amount": int(o.order_amount.isna().sum()) == 0,
    "0 missing delivery (completed)": int(comp.delivery_minutes.isna().sum()) == 0,
    "1,199 missing delivery (cancelled)": int(o[o.status == "cancelled"].delivery_minutes.isna().sum()) == 1199,
    "completed revenue 2,998,668.80": round(float(comp.order_amount.sum()), 2) == 2998668.80,
}
for k, v in checks.items():
    assert v, f"checkpoint failed: {k}"
print("ALL CHECKPOINTS PASSED")
