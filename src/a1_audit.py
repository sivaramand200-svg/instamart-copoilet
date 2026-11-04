"""Part A task 2: audit the raw tables."""
import pandas as pd
R = "data/raw/"
cust = pd.read_csv(R + "customers.csv")
prod = pd.read_csv(R + "products.csv")
orders = pd.read_csv(R + "orders.csv")
for name, df in [("customers", cust), ("products", prod), ("orders", orders)]:
    print(f"\n=== {name}: shape={df.shape}")
    t = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                      "missing_pct": (df.isna().mean() * 100).round(2)})
    print(t.to_string())
amt = pd.to_numeric(orders["order_amount"], errors="coerce")
dm = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
age = pd.to_numeric(cust["age"], errors="coerce")
rows = [("raw orders rows", len(orders)),
        ("exact duplicate rows", int(orders.duplicated().sum())),
        ("rows with status pending", int((orders["status"] == "pending").sum())),
        ("distinct order_id", int(orders["order_id"].nunique())),
        ("non-numeric order_amount", int((amt.isna() & orders["order_amount"].notna()).sum() + orders["order_amount"].isna().sum())),
        ("negative delivery_minutes", int((dm < 0).sum())),
        ("blank delivery_minutes", int(dm.isna().sum())),
        ("customers: blank age", int(age.isna().sum())),
        ("customers: age outside 10-100", int(((age < 10) | (age > 100)).sum())),
        ("customers: distinct raw city spellings", int(cust["city"].nunique())),
        ("customers: distinct non-blank gender spellings", int(cust["gender"].dropna().nunique())),
        ("customers: blank gender", int(cust["gender"].isna().sum()))]
print("\n=== audit figures")
print(pd.DataFrame(rows, columns=["check", "value"]).to_string(index=False))
