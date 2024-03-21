# %% imports
import matplotlib.ticker as mtick
from scipy.optimize import fmin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% load file
dd = pd.read_csv('ns.csv')
df = dd.copy()
df.style.format({'Maturity': '{:,.0f}'.format, 'Yield': '{:,.2%}'})

# %%
sf = df.copy()
sf = sf.dropna()
sf1 = sf.copy()
sf1['Y'] = round(sf['Yield']*100, 4)
sf = sf.style.format({'Maturity': '{:,.2f}'.format, 'Yield': '{:,.4%}'})

# %%
fontsize = 15
fig = plt.figure(figsize=(13, 7))

plt.title("Nelson-Siegel-Svensson Model - Unfitted Yield Curve", fontsize=fontsize)
plt.gca().set_facecolor("black")
fig.patch.set_facecolor('white')
X = sf1["Maturity"]
Y = sf1["Y"]
plt.scatter(X, Y, marker="o", c="red")
plt.xlabel('Period', fontsize=fontsize)
plt.ylabel('Interest', fontsize=fontsize)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.gca().yaxis.set_ticks(np.arange(0, 4, 0.25))
plt.gca().xaxis.set_ticks(np.arange(0, 30, 5))
plt.gca().legend(loc="lower right", title="Yield")
plt.grid()
plt.show()

# %%
β0 = 0.01
β1 = 0.01
β2 = 0.01
β3 = 0.01
λ0 = 1.00
λ1 = 1.00


def nss(beta0, beta1, beta2, beta3, lambda0, lambda1, df_maturity):
    result = (
        (beta0) +
        (beta1 * ((1 - np.exp(-df_maturity / lambda0)) / (df_maturity / lambda0))) +
        (beta2 * ((((1 - np.exp(-df_maturity / lambda0)) / (df_maturity / lambda0))) - (np.exp(-df_maturity / lambda0)))) +
        (beta3 * ((((1 - np.exp(-df_maturity / lambda1)) / (df_maturity / lambda1))) - (np.exp(-df_maturity / lambda1))))
    )
    return result

# %%
df['NSS'] = (β0)+(β1*((1-np.exp(-df['Maturity']/λ0))/(df['Maturity']/λ0)))+(β2*((((1-np.exp(-df['Maturity']/λ0))/(df['Maturity']/λ0))) -
                                                                                (np.exp(-df['Maturity']/λ0))))+(β3*((((1-np.exp(-df['Maturity']/λ1))/(df['Maturity']/λ1)))-(np.exp(-df['Maturity']/λ1))))
df.style.format({'Maturity': '{:,.0f}'.format, 'Yield': '{:,.2%}','NSS': '{:,.2%}'})

# %%
df1 = df.copy()
df['Y'] = round(df['Yield']*100, 4)
df['NSS'] = (β0)+(β1*((1-np.exp(-df['Maturity']/λ0))/(df['Maturity']/λ0)))+(β2*((((1-np.exp(-df['Maturity']/λ0))/(df['Maturity']/λ0))) -
                                                                                (np.exp(-df['Maturity']/λ0))))+(β3*((((1-np.exp(-df['Maturity']/λ1))/(df['Maturity']/λ1)))-(np.exp(-df['Maturity']/λ1))))
df['N'] = round(df['NSS']*100, 4)
df2 = df.copy()
df2 = df2.style.format({'Maturity': '{:,.2f}'.format,'Y': '{:,.2%}', 'N': '{:,.2%}'})
fontsize = 15
fig = plt.figure(figsize=(13, 7))
plt.title("Nelson-Siegel-Svensson Model - Unfitted Yield Curve", fontsize=fontsize)
ax = plt.axes()
plt.gca().set_facecolor("black")
fig.patch.set_facecolor('white')
X = df["Maturity"]
Y = df["Y"]
x = df["Maturity"]
y = df["N"]
ax.plot(x, y, color="orange", label="NSS")
plt.scatter(x, y, marker="o", c="orange")
plt.scatter(X, Y, marker="o", c="blue")
plt.xlabel('Period', fontsize=fontsize)
plt.ylabel('Interest', fontsize=fontsize)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.gca().xaxis.set_ticks(np.arange(0, 30, 5))
plt.gca().yaxis.set_ticks(np.arange(0, 4, 0.5))
plt.gca().legend(loc="lower right", title="Yield")
plt.grid()
plt.show()

# %%
df['Residual'] = (df['Yield'] - df['NSS'])**2
df22 = df[['Maturity', 'Yield', 'NSS', 'Residual']]
df22.style.format({'Maturity': '{:,.0f}'.format, 'Yield': '{:,.2%}','NSS': '{:,.2%}','Residual': '{:,.9f}'})
# %%
np.sum(df['Residual'])


# %%
def myval(c):
    df = dd.copy()
    df['NSS'] = (c[0])+(c[1]*((1-np.exp(-df['Maturity']/c[4]))/(df['Maturity']/c[4])))+(c[2]*((((1-np.exp(-df['Maturity']/c[4]))/(df['Maturity']/c[4]))
                                                                                               )-(np.exp(-df['Maturity']/c[4]))))+(c[3]*((((1-np.exp(-df['Maturity']/c[5]))/(df['Maturity']/c[5])))-(np.exp(-df['Maturity']/c[5]))))
    df['Residual'] = (df['Yield'] - df['NSS'])**2
    val = np.sum(df['Residual'])
    print("[β0, β1, β2, β3, λ0, λ1]=", c, ", SUM:", val)
    return (val)


c = fmin(myval, [0.0001, 0.0001, 0.0001, 0.0001, 1.00, 1.00])
        #  xtol=0.0001, ftol=0.0001)

# %%
β0 = c[0]
β1 = c[1]
β2 = c[2]
β3 = c[3]
λ0 = c[4]
λ1 = c[5]
print("[β0, β1, β2, β3, λ0, λ1]=", [c[0].round(2), c[1].round(2),
      c[2].round(2), c[3].round(2), c[4].round(2), c[5].round(2)])

# %%
df = df1.copy()
df['NSS'] = (β0)+(β1*((1-np.exp(-df['Maturity']/λ0))/(df['Maturity']/λ0)))+(β2*((((1-np.exp(-df['Maturity']/λ0))/(df['Maturity']/λ0))) -
                                                                                (np.exp(-df['Maturity']/λ0))))+(β3*((((1-np.exp(-df['Maturity']/λ1))/(df['Maturity']/λ1)))-(np.exp(-df['Maturity']/λ1))))
sf4 = df.copy()
sf5 = sf4.copy()
sf5['Y'] = round(sf4['Yield']*100, 4)
sf5['N'] = round(sf4['NSS']*100, 4)
sf4 = sf4.style.format({'Maturity': '{:,.2f}'.format,
                       'Yield': '{:,.2%}', 'NS': '{:,.2%}'})
# M0 = 0.00
# M1 = 3.50
fontsize = 15
fig = plt.figure(figsize=(13, 7))
plt.title("Nelson-Siegel-Svensson Model - Fitted Yield Curve", fontsize=fontsize)
ax = plt.axes()
plt.gca().set_facecolor("black")
fig.patch.set_facecolor('white')
X = sf5["Maturity"]
Y = sf5["Y"]
x = sf5["Maturity"]
y = sf5["N"]
ax.plot(x, y, color="orange", label="NSS")
plt.scatter(x, y, marker="o", c="orange")
plt.scatter(X, Y, marker="o", c="blue")
plt.xlabel('Period', fontsize=fontsize)
plt.ylabel('Interest', fontsize=fontsize)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.gca().xaxis.set_ticks(np.arange(0, 30, 5))
plt.gca().yaxis.set_ticks(np.arange(0, 4, 0.5))
plt.gca().legend(loc="lower right", title="Yield")
plt.grid()
plt.show()

# %%
df.style.format({'Maturity': '{:,.1f}'.format, 'Yield': '{:,.2%}','NSS': '{:,.2%}'})
# %%
