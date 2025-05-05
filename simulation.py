# author: iambharatdhungana
# date: 05/04/2025
# purpose: tradetariff-semiconductors

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Read quaterly trade data and convert to PeriodIndex with quarterly frequency 

imports_q = pd.read_csv("data/quarterly_imports.csv")
exports_q = pd.read_csv("data/quarterly_exports.csv")
imports_q["Quarter"] = pd.PeriodIndex(imports_q["Quarter"], freq="Q")
exports_q["Quarter"] = pd.PeriodIndex(exports_q["Quarter"], freq="Q")

# Merge imports and exports data on the "Quarter" column
# and convert the "Quarter" column to a PeriodIndex with quarterly frequency.
# Missing values are filled with 0.

trade_idx = (imports_q.set_index("Quarter")
                        .join(exports_q.set_index("Quarter"), how="outer")
                        .fillna(0))                        


# Read the FRED data (dimestic industrial prodution, PPI and world price for semiconductors in US) and rename columns for each file

ip_semi = pd.read_csv("data/IPG3344SQ.csv", names=["DATE", "IP_semi"], header=0)
ppi_semi = pd.read_csv("data/PCU33443344.csv", names=["DATE", "PPI_semi"], header=0)
pw_semi = pd.read_csv("data/IR21320.csv", names=["DATE", "PW_semi"], header=0)

# Convert DATE to datetime and create a Quarter column using PeriodIndex with quarterly frequency

for df_item in [ip_semi, ppi_semi, pw_semi]:
    df_item["DATE"] = pd.to_datetime(df_item["DATE"])
    df_item["Quarter"] = pd.PeriodIndex(df_item["DATE"], freq="Q")

# Group by Quarter and average if data are monthly since few of the FRED datasets are monthly averages

IP = ip_semi.groupby("Quarter", as_index=True)["IP_semi"].mean().to_frame()
PPI = ppi_semi.groupby("Quarter", as_index=True)["PPI_semi"].mean().to_frame()
PW = pw_semi.groupby("Quarter", as_index=True)["PW_semi"].mean().to_frame()

# Join all FRED series & trade data to trim to 2015‑24 (the period of interest)
# and fill missing values with 0

df = (IP.join(PPI, how="inner")
        .join(PW, how="inner")
        .join(trade_idx, how="left")
        .fillna(0))

df = df.loc["2015Q1":"2024Q4"]

# Build Proxy Demand, Supply (buyers' price) & World Price

df["Cons_vol"] = df["IP_semi"] + (df["Imports_Semi_USD"] - df["Exports_Semi_USD"]) / df["PW_semi"]
df["Price_dom"] = df["PPI_semi"]   
df["Price_w"] = df["PW_semi"]

# Solving for the baseline equilibrium (2015‑24 mean)

P0  = df["Price_dom"].mean()
Qd0 = df["Cons_vol"].mean()
Qs0 = df["IP_semi"].mean()
gap0 = Qd0 - Qs0                         

# Imports & Exports flow to be used in the model

M0   = max(gap0, 0)                      
X0   = max(-gap0, 0)                     

print(f"Equilibrium price   P* = {P0:.2f}")
print(f"Equilibrium quantity Q* = {Qs0 + M0:.2f}")
print(f"Quantity simplification Q_d = {Qd0:.2f}   Q_s = {Qs0:.2f}   Imports = {M0:.2f}")

# Elastictitie value (From literature reviews; plan was to estimate them from data but its now a future work)

eps_d = -0.80          
eps_s =  2.00          
print(f"\nElasticities  E_d = {eps_d},  E_s = {eps_s}")

# Demand & Supply curves (linear) for the baseline equilibrium with elasticities

Bd = eps_d * Qd0 / P0 ;  Ad = Qd0 - Bd * P0      
Bs = eps_s * Qs0 / P0 ;  As = Qs0 - Bs * P0

def demand_inv(Q): return (Q - Ad)/Bd            
def supply_inv(Q): return (Q - As)/Bs           

# World price (mean) for semiconductors (This is the price that would prevail in the absence of tariffs)

Pw_mean = df["Price_w"].mean()                  

# Defining the function to solve for the equilibrium price, demand, supply, gap, imports and exports, tax burden according to the tariff rate
# I used 20% as a sample tariff rate, but this can be changed to any value.

def solve(tariffs):
    Pb = Pw_mean * (1 + tariffs)                    
    Q_d = Ad + Bd * Pb                          
    Q_s = As + Bs * Pb                         
    gap = Q_d - Q_s
    imports = max(gap, 0)
    exports = max(-gap, 0)
    return Pb, Q_d, Q_s, gap, imports, exports

tariffs = 0.2
Pb1, Qd1, Qs1, gap1, M1, X1 = solve(tariffs)

buyer_share  = eps_s / (eps_s - eps_d)
seller_share = 1 - buyer_share

print(f"\n—— TARIFF  τ = {tariffs*100:.0f}% ——")
print(f"Buyer price   Pb = {Pb1:.2f}")
print(f"Demand Q_d    = {Qd1:.2f}")
print(f"Supply Q_s    = {Qs1:.2f}")
print(f"Gap Q_d – Q_s = {gap1:.2f}")
print(f"Imports       = {M1:.2f}   Exports = {X1:.2f}")
print(f"burden: buyers {buyer_share:.1%} , sellers {seller_share:.1%}")

# Simulate the model for a range of tariff rates (0 to 40%) using Monte Carlo simulation.
# This will help us understand the distribution of the gap and imports across different tariff rates.
# Generate 1000 random tariff rates between 0 and 40% and capture the domestic price, demand, supply, gap, imports and exports for each rate.  
# The results are stored in lists for plots.

np.random.seed(0)
draws = np.random.uniform(0, 0.30, 1_000)
mc_price = [solve(t)[0] for t in draws]
mc_qty   = [solve(t)[1] for t in draws]


# Plots

plt.style.use("default")

# Production vs Domestic price (2015‑24) of semiconductors in the US

dates = df.index.to_timestamp()
fig, ax1 = plt.subplots(figsize=(9,3))
ax1.plot(dates, df["IP_semi"], label="Prod. index", color="tab:green")
ax1.set_ylabel("Production index")
ax2 = ax1.twinx()
ax2.plot(dates, df["Price_dom"], label="Domestic price", color="tab:red")
ax2.set_ylabel("Price index")
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
plt.title("Production & Domestic Price")
lines = ax1.get_lines() + ax2.get_lines()
plt.legend(lines,[l.get_label() for l in lines])
plt.tight_layout()
plt.savefig("plots/production_domestic_price.png")
plt.show()

# Equilibrium S‑D diagram

Qgrid = np.linspace(0.5*Qs0, 1.5*Qd0, 250)
Pd = demand_inv(Qgrid);  Ps = supply_inv(Qgrid)
plt.figure(figsize=(6,5))
plt.plot(Qgrid, Pd, label="Demand")
plt.plot(Qgrid, Ps, label="Supply")
plt.scatter([Qd0],[P0],color="black",label="Baseline (P0,Q0)")
plt.xlabel("Quantity"); plt.ylabel("Price")
plt.title("Baseline Demand & Supply"); plt.legend(); plt.tight_layout()
plt.savefig("plots/baseline_demand_supply.png")
plt.show()

# Tariff S‑D diagram (20% tariff) with tax burden

plt.figure(figsize=(6,5))
plt.plot(Qgrid, demand_inv(Qgrid), label="Demand")
plt.plot(Qgrid, supply_inv(Qgrid), label="Supply")
plt.hlines(Pw_mean, Qgrid.min(), Qd1, color="gray", linestyle=":", label="World price")
plt.hlines(Pb1, Qgrid.min(), Qd1, color="black", linestyle="--", label="Buyer price after tariff")
plt.vlines(Qd1, Pw_mean, Pb1, color="red", label="Tariff tax")
plt.scatter([Qd1], [Pb1], color="black")
plt.xlabel("Quantity")
plt.ylabel("Price")
plt.title("Tariff S–D Diagram (tariff = 20%)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/tariffs_20percent.png")
plt.show()

# Monte‑Carlo histogram of *gap* (imports positive, exports negative)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(mc_price, bins=30, color="darkblue")
plt.title("MC Buyer Price")
plt.xlabel("Price index")
plt.ylabel("Frequency")        
plt.subplot(1,2,2)
plt.hist(mc_qty, bins=30, color="darkorange")
plt.title("MC Quantity")
plt.xlabel("Quantity index")
plt.ylabel("Frequency")        
plt.tight_layout()
plt.savefig("plots/monte_carlo_histogram.png")
plt.show()


