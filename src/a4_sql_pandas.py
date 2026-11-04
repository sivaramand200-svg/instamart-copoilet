"""Part A tasks 6-7: run the seven SQL queries and the pandas cross-check / EDA."""
import re, sqlite3
import numpy as np, pandas as pd

con = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql", encoding="utf-8").read()
parts = re.split(r"^-- (Q\d):.*$", text, flags=re.M)
Q = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}
R = {k: pd.read_sql_query(Q[k].rstrip(";"), con) for k in sorted(Q)}

print("Q1 customers:", len(R["Q1"]))
q2 = R["Q2"]; print("Q2 rows:", len(q2), "sum:", round(q2.revenue.sum(), 2)); print("   largest:", q2.iloc[0].to_dict())
print("Q3 products:", len(R["Q3"]))
print("Q4 customers:", len(R["Q4"]))
print("Q5 customers:", len(R["Q5"]), "avg:", R["Q5"].avg_total.iloc[0])
q6 = R["Q6"]
print("Q6 top per city:"); print(q6[q6.row_num == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q6 rows RANK != ROW_NUMBER:", int((q6.rnk != q6.row_num).sum()))
q7 = R["Q7"]; print("Q7 rows:", len(q7), "last cumulative:", q7.cumulative_revenue.iloc[-1])

# ---- pandas cross-check
o = pd.read_sql_query("SELECT o.*, c.city, p.category FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id)", con)
comp = o[o.status == "completed"]
pv = pd.pivot_table(comp, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True, margins_name="Total")
print("\npivot grand total:", round(pv.loc["Total", "Total"], 2), "== SQL total:", round(q2.revenue.sum(), 2))
g = comp.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"), count=("order_amount", "count")).round(2)
print(g.to_string())

# ---- EDA on completed order_amount
a = comp["order_amount"]
q1, q3 = a.quantile(.25), a.quantile(.75); iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\nmean={a.mean():.2f} median={a.median():.2f} skew={a.skew():.2f} Q1={q1:.2f} Q3={q3:.2f}")
print(f"fences low={lo:.2f} high={hi:.2f} outliers below={int((a < lo).sum())} above={int((a > hi).sum())}")
cm = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(cm.round(3).to_string())
pairs = cm.where(np.triu(np.ones(cm.shape, bool), 1)).stack()
print("largest |corr| pair:", pairs.abs().idxmax(), round(pairs[pairs.abs().idxmax()], 3))
