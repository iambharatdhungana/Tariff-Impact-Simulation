# Author: iambharatdhungana
# Date: 04/29/2025


import numpy as np, pandas as pd, statsmodels.api as sm
import matplotlib.pyplot as plt


# Read quarterly trade data 

imports_q = pd.read_csv("data/quarterly_imports.csv")
exports_q = pd.read_csv("data/quarterly_exports.csv")

# Merge imports and exports data on the "Quarter" column
# and convert the "Quarter" column to a PeriodIndex with quarterly frequency

trade_q  = imports_q.merge(exports_q, on="Quarter").assign(
    Quarter=lambda d: pd.PeriodIndex(d["Quarter"], freq="Q"))

# Read FRED data files

ip_semi = pd.read_csv("data/IPG3344SQ.csv")
ip_farm = pd.read_csv("data/IPG333111SQ.csv")
ppi_semi = pd.read_csv("data/PCU33443344.csv")
ppi_farm = pd.read_csv("data/PCU333111333111P.csv")
impprice_semi = pd.read_csv("data/IR21320.csv")

# Rename  first column to DATE and second to specified column name

ip_semi.columns = ["DATE", "IP_semi"]
ip_farm.columns = ["DATE", "IP_farm"]
ppi_semi.columns = ["DATE", "PPI_semi"]
ppi_farm.columns = ["DATE", "PPI_farm"]
impprice_semi.columns = ["DATE", "IMPprice_semi"]

# Convert DATE to datetime format and create a Quarter column
# using PeriodIndex with quarterly frequency

for df in [ip_semi, ip_farm, ppi_semi, ppi_farm, impprice_semi]:
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["Quarter"] = pd.PeriodIndex(df["DATE"], freq="Q")

# Since few of the FRED datasets are monthy average, aggregate monthly data to quarterly averages (if needed)
# and group by Quarter to get the mean values


ip_semi = ip_semi.groupby("Quarter", as_index=False)["IP_semi"].mean()
ip_farm = ip_farm.groupby("Quarter", as_index=False)["IP_farm"].mean()
ppi_semi = ppi_semi.groupby("Quarter", as_index=False)["PPI_semi"].mean()
ppi_farm = ppi_farm.groupby("Quarter", as_index=False)["PPI_farm"].mean()
impprice_semi = impprice_semi.groupby("Quarter", as_index=False)["IMPprice_semi"].mean()

# Merge all series of FRED dataset into one DataFrame on 'Quarter'

fred_df = ip_semi.merge(ip_farm, on="Quarter") \
                 .merge(ppi_semi, on="Quarter") \
                 .merge(ppi_farm, on="Quarter") \
                 .merge(impprice_semi, on="Quarter")

# Filter by quarter range to include only the desired period (2015Q1 to 2024Q4)

fred_df = fred_df[(fred_df["Quarter"] >= "2015Q1") & (fred_df["Quarter"] <= "2024Q4")]

# Merge with trade_q dataset created earlier 
# and fill missing values with 0, then reset the index

data = fred_df.merge(trade_q, on="Quarter", how="left").fillna(0).reset_index(drop=True)


# Calculate consumption volumes and assign domestic/world price indexes

data["Cons_Semi_vol"] = data["IP_semi"] + (data["Imports_Semi_USD"] - data["Exports_Semi_USD"]) / data["IMPprice_semi"]
data["Cons_Farm_vol"] = data["IP_farm"] + (data["Imports_Farm_USD"] - data["Exports_Farm_USD"]) / data["PPI_farm"]

data["Price_Semi_dom"] = data["PPI_semi"]
data["Price_Farm_dom"] = data["PPI_farm"]
data["Price_Semi_world"] = data["IMPprice_semi"]
data["Price_Farm_world"] = data["PPI_farm"]

for c in ("Cons_Semi_vol", "Cons_Farm_vol", "IP_semi", "IP_farm",
          "Price_Semi_dom", "Price_Farm_dom"):
    data[c] = data[c].replace(0, 1e-6)

data = data.loc[(data["Cons_Semi_vol"] > 0) & (data["Cons_Farm_vol"] > 0) &
                (data["IP_semi"] > 0) & (data["IP_farm"] > 0)].reset_index(drop=True)

# Estimate demand and supply elasticities 

eq = {}
for prod in ("Semi", "Farm"):
    lnQd = np.log(data[f"Cons_{prod}_vol"])
    lnPd = np.log(data[f"Price_{prod}_dom"])
    E_d = sm.OLS(lnQd, sm.add_constant(lnPd)).fit().params.iloc[1]
    lnQs = np.log(data[f"IP_{prod.lower()}"])
    E_s = sm.OLS(lnQs, sm.add_constant(lnPd)).fit().params.iloc[1]
    Q_mean, P_mean = data[f"Cons_{prod}_vol"].mean(), data[f"Price_{prod}_dom"].mean()
    eq[prod] = {
        "A_d": Q_mean - E_d * Q_mean,
        "B_d": E_d * Q_mean / P_mean,
        "A_s": Q_mean - E_s * Q_mean,
        "B_s": E_s * Q_mean / P_mean
    }
    print(f"{prod}: E_d = {E_d:.2f}, E_s = {E_s:.2f}")

# Define simulation function for tariff impacts

Pw_semi = data["Price_Semi_world"].mean()
Pw_farm = data["Price_Farm_world"].mean()
def simulate(prod, t):
    Pw = Pw_semi if prod == "Semi" else Pw_farm
    P = Pw * (1 + t)
    p = eq[prod]
    Qd = p["A_d"] + p["B_d"] * P
    Qs = p["A_s"] + p["B_s"] * P
    return P, Qd, Qs, max(Qd - Qs, 0)

# Pre-tariff equilibrium (t = 0)

equilibrium_tariff = 0.0
price_eq, Qd_eq, Qs_eq, imports_eq = simulate("Semi", equilibrium_tariff)

print("Pre-Tariff Equilibrium")
print(f"Domestic Price: ${price_eq:.2f}")
print(f"Quantity Demanded: {Qd_eq:.2f}")
print(f"Quantity Supplied: {Qs_eq:.2f}")
print(f"Imports: {imports_eq:.2f}")
print()

# Tariff scenario. Using a 10% (0.10) tariff as an example

tariff_rate = 0.10  
price, quantity_demanded, quantity_supplied, imports = simulate("Semi", tariff_rate)

print(f"With {tariff_rate*100:.0f}% Tariff")
print(f"Domestic Price: ${price:.2f}")
print(f"Quantity Demanded: {quantity_demanded:.2f}")
print(f"Quantity Supplied: {quantity_supplied:.2f}")
print(f"Imports: {imports:.2f}")

# Monte Carlo simulation: simulate tariff effects 1000 times for both products
# Here we generate 1000 random tariff rates between 0 and 30% and capture the domestic price
# and consumption outcomes for each product.

np.random.seed(0)
draws = np.random.uniform(0, 0.30, 1000)
semi_results = np.array([simulate("Semi", t) for t in draws])
farm_results = np.array([simulate("Farm", t) for t in draws])
mc_df = pd.DataFrame({
    "Semi_P": semi_results[:, 0],
    "Semi_Q": semi_results[:, 1],
    "Farm_P": farm_results[:, 0],
    "Farm_Q": farm_results[:, 1]
})


# Log-log scatter for both products

for prod, color in [("Semi", "tab:blue"), ("Farm", "tab:green")]:
    plt.figure()
    lnP = np.log(data[f"Price_{prod}_dom"])
    lnQ = np.log(data[f"Cons_{prod}_vol"])
    plt.scatter(lnP, lnQ, alpha=.7, color=color)
    m, b = np.polyfit(lnP, lnQ, 1)
    plt.plot(lnP, m * lnP + b, color="red")
    plt.xlabel("ln Price")
    plt.ylabel("ln Quantity")
    plt.title(f"Demand scatter – {prod} (E ≈ {m:.2f})")
    plt.show()

# Supply–Demand diagrams (Quantity x Price) for both products
for prod in ("Semi", "Farm"):
    p = eq[prod]
    Qgrid = np.linspace(data[f"Cons_{prod}_vol"].min() * 0.5,
                        data[f"Cons_{prod}_vol"].max() * 1.5, 120)
    P_d = (Qgrid - p["A_d"]) / p["B_d"]
    P_s = (Qgrid - p["A_s"]) / p["B_s"]
    Pw = Pw_semi if prod == "Semi" else Pw_farm
    t = 0.10
    Pt = Pw * (1 + t)
    plt.figure()
    plt.plot(Qgrid, P_d, label="Demand")
    plt.plot(Qgrid, P_s, label="Supply")
    plt.hlines([Pw, Pt], Qgrid.min(), Qgrid.max(),
               colors=["gray", "black"], linestyles=[":", "--"],
               label=["World price", f"Pw(1+{t:.0%})"])
    plt.xlabel("Quantity idx")
    plt.ylabel("Price idx")
    plt.title(f"S-D diagram (t=10 %) – {prod}")
    plt.legend()
    plt.show()

# Monte Carlo price histogram (Semiconductors)
plt.figure()
plt.hist(mc_df["Semi_P"], bins=30, color="steelblue")
plt.xlabel("Domestic price idx")
plt.title("MC Price distribution – Semi")
plt.show()


# Price & quantity over time  (one panel per product, no twin axes)
for prod, color in [("Semi", "tab:blue"), ("Farm", "tab:green")]:
    plt.figure(figsize=(9,4))
    plt.plot(data["Quarter"].astype(str),
             data[f"Price_{prod}_dom"], label="Price index", color=color)
    plt.plot(data["Quarter"].astype(str),
             data[f"Cons_{prod}_vol"], label="Quantity index",
             color="tab:orange", linestyle="--")
    plt.title(f"{prod}: Price vs Quantity (2015–24)")
    plt.xlabel("Quarter")
    plt.ylabel("Index (base=100)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Log-log demand scatter with fitted line
for prod, color in [("Semi", "tab:blue"), ("Farm", "tab:green")]:
    lnP = np.log(data[f"Price_{prod}_dom"])
    lnQ = np.log(data[f"Cons_{prod}_vol"])
    m, b = np.polyfit(lnP, lnQ, 1)
    plt.figure(figsize=(5,4))
    plt.scatter(lnP, lnQ, alpha=0.6, color=color)
    plt.plot(lnP, m*lnP + b, color="red")
    plt.title(f"{prod} demand | E ≈ {m:.2f}")
    plt.xlabel("ln Price")
    plt.ylabel("ln Quantity")
    plt.tight_layout()
    plt.show()

# Imports vs tariff curve  (example – semiconductors)
t_grid = np.linspace(0, 0.30, 61)          # 0 % – 30 %
imports = [simulate("Semi", t)[3] for t in t_grid]

plt.figure(figsize=(6,4))
plt.plot(t_grid*100, imports)
plt.title("Semiconductor imports vs tariff")
plt.xlabel("Tariff rate (%)")
plt.ylabel("Import volume index")
plt.tight_layout()
plt.show()


# Monte-Carlo analysis
plt.figure(figsize=(9,3))
plt.subplot(1,2,1)
plt.hist(mc_df["Semi_P"], bins=30, color="tab:blue", alpha=0.8)
plt.title("Semiconductor price (MC)")
plt.xlabel("Price index"); plt.ylabel("Frequency")

plt.subplot(1,2,2)
plt.hist(mc_df["Farm_P"], bins=30, color="tab:green", alpha=0.8)
plt.title("Farm-machinery price (MC)")
plt.xlabel("Price index")
plt.tight_layout(); plt.show()

#  Histogram of quantities
plt.figure(figsize=(9,3))
plt.subplot(1,2,1)
plt.hist(mc_df["Semi_Q"], bins=30, color="tab:blue", alpha=0.8)
plt.title("Semiconductor quantity (MC)")
plt.xlabel("Quantity index"); plt.ylabel("Frequency")

plt.subplot(1,2,2)
plt.hist(mc_df["Farm_Q"], bins=30, color="tab:green", alpha=0.8)
plt.title("Farm-machinery quantity (MC)")
plt.xlabel("Quantity index")
plt.tight_layout(); plt.show()

