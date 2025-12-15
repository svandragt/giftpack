import pandas as pd
import pulp as pl

MAX_KG = 2.0
MAX_GBP = 39.0

df = pd.read_csv("gifts.csv")
items = list(df.index)
kg = df["kg"].to_dict()
gbp = df["gbp"].to_dict()
B = list(range(len(items)))

x = pl.LpVariable.dicts("x", (items, B), 0, 1, pl.LpBinary)
y = pl.LpVariable.dicts("y", B, 0, 1, pl.LpBinary)

prob = pl.LpProblem("GiftPacking", pl.LpMinimize)
prob += pl.lpSum(y[b] for b in B)

for i in items:
    prob += pl.lpSum(x[i][b] for b in B) == 1

for b in B:
    prob += pl.lpSum(kg[i] * x[i][b] for i in items) <= MAX_KG * y[b]
    prob += pl.lpSum(gbp[i] * x[i][b] for i in items) <= MAX_GBP * y[b]

for b in range(len(B) - 1):
    prob += y[b] >= y[b + 1]

prob.solve(pl.PULP_CBC_CMD(msg=False))

used = [b for b in B if pl.value(y[b]) > 0.5]
print(f"Minimum boxes: {len(used)}\n")

for b in used:
    box_items = [i for i in items if pl.value(x[i][b]) > 0.5]
    w = sum(kg[i] for i in box_items)
    v = sum(gbp[i] for i in box_items)
    names = df.loc[box_items, "item"].tolist()
    print(f"Box {b+1}: {w:.2f} kg, £{v:.2f} -> {', '.join(names)}")
