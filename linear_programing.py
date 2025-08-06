import sys
import numpy as np
from scipy.optimize import linprog
sys.stdout.reconfigure(encoding='utf-8')

# -----------------------------------------------------------------------
# 1)  BUSINESS DATA  (unchanged)
# -----------------------------------------------------------------------
price_kg  = np.array([6.25, 6.10, 6.10, 5.25, 4.00])
profit_kg = np.array([1.20, 1.40, 1.60, 1.30, 0.90])

KG_TON = 1_000
price_t  = price_kg  * KG_TON
profit_t = profit_kg * KG_TON

inventory = np.array([708_289.13,  38_392.92, 271_416.12,
                      490_729.26, 1_703_390.77])
min_sales = np.array([321_949.61,  17_451.33, 123_370.97,
                      223_058.76, 774_268.53])
max_sales = np.array([643_899.21,  34_902.65, 246_741.93,
                      446_117.51, 1_548_537.06])
upper = np.minimum(max_sales, inventory)

shares = np.array([[0.19203, 0.06108, 0.48351, 0.08322, 0.01214],
                   [0.02847, 0.01799, 0.02438, 0.00128, 0.00000],
                   [0.00000, 0.02570, 0.00463, 0.00000, 0.00000],
                   [0.00000, 0.04800, 0.01776, 0.00000, 0.00000]])
caps = np.array([2_429_521.6, 210_078.47, 88_577.76, 192_020.53])

# -----------------------------------------------------------------------
# 2)  PHASE 1 – minimise total slack with **equalities**
# -----------------------------------------------------------------------
n_prod, n_plants = 5, 4
c1 = np.concatenate([np.zeros(n_prod), np.ones(n_plants)])      # min Σ slack
A_eq_cap = np.hstack([shares, np.eye(n_plants)])                # load+slack = C
b_eq_cap = caps

bounds = [(lo, hi) for lo, hi in zip(min_sales, upper)] + \
         [(0, None)] * n_plants

res1 = linprog(c1, A_eq=A_eq_cap, b_eq=b_eq_cap,
               bounds=bounds, method="highs")
if not res1.success:
    raise RuntimeError("Phase 1 failed: " + res1.message)

min_slack_total = res1.fun
x_feasible      = res1.x[:n_prod]
slack_eq        = res1.x[n_prod:]

# -----------------------------------------------------------------------
# 3)  PHASE 2 – maximise profit at that slack level
# -----------------------------------------------------------------------
c2 = np.concatenate([-profit_t, np.zeros(n_plants)])            # maximise profit
# keep the capacity equalities; add one more equality to fix Σ slack
A_eq2 = np.vstack([A_eq_cap, np.hstack([np.zeros(n_prod), np.ones(n_plants)])])
b_eq2 = np.concatenate([b_eq_cap, [min_slack_total]])

res2 = linprog(c2, A_eq=A_eq2, b_eq=b_eq2,
               bounds=bounds, method="highs")
if not res2.success:
    raise RuntimeError("Phase 2 failed: " + res2.message)

x_opt   = res2.x[:n_prod]
slack   = res2.x[n_prod:]
profit  = profit_t @ x_opt
revenue = price_t  @ x_opt
used    = shares  @ x_opt

# -----------------------------------------------------------------------
# 4)  REPORT
# -----------------------------------------------------------------------
prods  = ["Pain", "Morceau", "Lingot", "GR-PCDT", "GR-GCDT"]
plants = ["COSUMAR", "SUTA", "SURAC", "SUNABEL"]

print("\nOptimal Sales with Minimum Slack")
print(f"{'Product':<10} {'Sales (t)':>15} {'Revenue':>15} {'Profit':>15}")
print("-"*58)
for n, s, r, p in zip(prods, x_opt, price_t*x_opt, profit_t*x_opt):
    print(f"{n:<10} {s:>15,.0f} {r:>15,.0f} {p:>15,.0f}")
print("-"*58)
print(f"{'TOTAL':<10} {x_opt.sum():>15,.0f} {revenue:>15,.0f} {profit:>15,.0f}")

print("\nCapacity Utilisation (slack = capacity − load)")
print(f"{'Plant':<10} {'Used (t)':>12} {'Limit (t)':>12} {'Slack (t)':>12}")
print("-"*50)
for n, u, c, s in zip(plants, used, caps, slack):
    print(f"{n:<10} {u:>12,.0f} {c:>12,.0f} {s:>12,.0f}")

print(f"\nTotal slack (minimum): {min_slack_total:,.0f} t")
print(f"Maximum profit at that slack: {profit:,.0f} MAD")
