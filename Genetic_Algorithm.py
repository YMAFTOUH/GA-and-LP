import numpy as np
import pygad

# 1)  DATA  ----------------------------------------------------------
# Values PER KILOGRAM
price_per_kg  = np.array([6.25, 6.10, 6.10, 5.25, 4.00])   # MAD/kg
profit_per_kg = np.array([1.20, 1.40, 1.60, 1.30, 0.90])   # MAD/kg

# Convert to per-tonne for internal maths
KG_PER_TON      = 1_000
price_per_ton   = price_per_kg  * KG_PER_TON
profit_per_ton  = profit_per_kg * KG_PER_TON

# >>> NEW INVENTORY / MIN / MAX SALES (t) <<<
inventory = np.array([708_289.13,  38_392.92, 271_416.12,
                      490_729.26, 1_703_390.77])

min_sales = np.array([321_949.61,  17_451.33, 123_370.97,
                      223_058.76, 774_268.53])

max_sales = np.array([643_899.21,  34_902.65, 246_741.93,
                      446_117.51, 1_548_537.06])

product_names = ["Pain", "Morceau", "Lingot", "GR-PCDT", "GR-GCDT"]
socs          = ["COSUMAR", "SUTA", "SURAC", "SUNABEL"]

# Capacity-share matrix (unchanged)
shares = np.array([
    [0.19203, 0.06108, 0.48351, 0.08322, 0.01214],
    [0.02847, 0.01799, 0.02438, 0.00128, 0.00000],
    [0.00000, 0.02570, 0.00463, 0.00000, 0.00000],
    [0.00000, 0.04800, 0.01776, 0.00000, 0.00000]
])
capacities = np.array([2_429_521.6, 210_078.47, 88_577.76, 192_020.53])

# 2)  GENE SPACE -----------------------------------------------------
cap_high = []
for j in range(5):
    feas = [capacities[i] / shares[i, j]
            for i in range(shares.shape[0]) if shares[i, j] > 0]
    cap_high.append(min(feas) if feas else capacities.sum())
cap_high = np.array(cap_high)

high_bound = np.minimum.reduce([cap_high, max_sales, inventory])
gene_space = [{"low": lo, "high": hi} for lo, hi in zip(min_sales, high_bound)]

# 3)  FITNESS  (slack-first: minimise slack, then maximise profit) ----
def fitness_func(_, solution, __):
    x     = np.asarray(solution, float)
    loads = shares @ x
    if (np.any(x > inventory) or np.any(x > max_sales) or
        np.any(loads > capacities)):
        return -1e10
    slack  = np.maximum(capacities - loads, 0.).sum()
    profit = profit_per_ton @ x
    return -slack + 1e-6 * profit

# 4)  GA SETTINGS ----------------------------------------------------
ga = pygad.GA(
    num_generations=1200,
    num_parents_mating=40,
    sol_per_pop=150,
    num_genes=5,
    gene_space=gene_space,
    fitness_func=fitness_func,
    parent_selection_type="tournament",
    keep_parents=30,
    crossover_type="uniform",
    mutation_type="random",
    mutation_percent_genes=35,
    stop_criteria=["saturate_150"]
)

# 5)  RUN ------------------------------------------------------------
ga.run()
best_solution, best_fitness, _ = ga.best_solution()

# 6)  REPORT ---------------------------------------------------------
sales   = np.round(best_solution)            # t
revenue = price_per_ton  * sales             # MAD
profit  = profit_per_ton * sales             # MAD

print("\nOptimal Sales, Revenue & Profit")
print("{:<10} {:>15} {:>15} {:>15}".format(
      "Product", "Sales (t)", "Revenue", "Profit"))
print("-" * 58)
for n, s, r, p in zip(product_names, sales, revenue, profit):
    print(f"{n:<10} {s:>15,.0f} {r:>15,.0f} {p:>15,.0f}")
print("-" * 58)
print(f"{'TOTAL':<10} {sales.sum():>15,.0f} {revenue.sum():>15,.0f} {profit.sum():>15,.0f}")

used  = shares @ sales
slack = capacities - used
print("\nCapacity Utilisation")
print("{:<10} {:>12} {:>12} {:>12}".format(
      "Societe", "Used (t)", "Limit (t)", "Slack (t)"))
print("-" * 50)
for n, u, c, s in zip(socs, used, capacities, slack):
    print(f"{n:<10} {u:>12,.0f} {c:>12,.0f} {s:>12,.0f}")

print(f"\nTotal slack: {slack.sum():,.0f} t")
print(f"Objective value (fitness): {best_fitness:,.2f}")
