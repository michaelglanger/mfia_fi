# %% imports
import matplotlib.ticker as mtick
from scipy.optimize import fmin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# date_of_values = '12/28/23'
date_of_values = '7/17/14'

# %% load file
ycdf = pd.read_csv('yield-curve-rates-1990-2023.csv', header=0, index_col=0)


def create_maturity_yield_tuple(df: pd.DataFrame, date: str) -> tuple:
    maturity = [1, 2, 3, 4, 6, 12, 24, 36, 60, 84, 120, 240, 360]  # TODO: calculate this.
    yields = (df.loc[[date]].values / 100).flatten().tolist()
    return maturity, yields


mat, yld = create_maturity_yield_tuple(ycdf, date_of_values)
dd = pd.DataFrame(data={'Maturity': mat, 'Yield': yld})

df = dd.copy()
df.style.format({'Maturity': '{:,.0f}'.format, 'Yield': '{:,.2%}'})

# %%
sf = df.copy()
sf = sf.dropna()
sf1 = sf.copy()
sf1['Y'] = round(sf['Yield']*100, 4)
sf = sf.style.format({'Maturity': '{:,.0f}'.format, 'Yield': '{:,.4%}'})

# %%


# f"Nelson-Siegel-Svensson Model - Unfitted Yield Curve for {date_of_values}"
def plot_ssn(x_maturity: pd.Series, y_yield: pd.Series, y_nss=None, title='Nelson-Siegel-Svensson Model - Unfitted Yield Curve', fontsize=15, logx=False):
    fig = plt.figure(figsize=(13, 7))
    plt.title(title, fontsize=fontsize)
    plt.gca().set_facecolor("black")
    fig.patch.set_facecolor('white')
    plt.scatter(x_maturity, y_yield, marker="o", c="blue")
    plt.gca().plot(x_maturity, y_yield, color="blue", label="Yield")
    if y_nss is not None:
        plt.scatter(x_maturity, y_nss, marker="o", c="orange")
        plt.gca().plot(x_maturity, y_nss, color="orange", label="NSS")

    plt.xlabel('Period', fontsize=fontsize)
    plt.ylabel('Interest', fontsize=fontsize)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.gca().yaxis.set_ticks(np.arange(0, 6, 0.2))  # TODO: calculate the max and round up instead of the 6
    x_ticks = x_maturity.to_list()
    plt.gca().xaxis.set_ticks(x_ticks)
    if logx:  # true if we need a logaritmic x axis
        plt.gca().set_xscale('log')
    plt.gca().legend(loc="lower right", title="Yield")
    plt.grid()
    plt.show()


plot_ssn(sf1["Maturity"], sf1["Y"],
         title=f"Nelson-Siegel-Svensson Model - Unfitted Yield Curve for {date_of_values}")

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
df['NSS'] = nss(β0, β1, β2, β3, λ0, λ1, df['Maturity'])
df.style.format({'Maturity': '{:,.0f}'.format, 'Yield': '{:,.2%}','NSS': '{:,.2%}'})

# %%
df1 = df.copy()
df['Y'] = round(df['Yield']*100, 4)
df['NSS'] = nss(β0, β1, β2, β3, λ0, λ1, df['Maturity'])
df['N'] = round(df['NSS']*100, 4)
df2 = df.copy()
df2 = df2.style.format({'Maturity': '{:,.2f}'.format,'Y': '{:,.2%}', 'N': '{:,.2%}'})

plot_ssn(df["Maturity"], df["Y"], df["N"])

# %%
df['Residual'] = (df['Yield'] - df['NSS'])**2
df22 = df[['Maturity', 'Yield', 'NSS', 'Residual']]
df22.style.format({'Maturity': '{:,.0f}'.format, 'Yield': '{:,.2%}','NSS': '{:,.2%}','Residual': '{:,.9f}'})
# %%
np.sum(df['Residual'])


# %%
def myval(c):
    df = dd.copy()
    df['NSS'] = nss(c[0], c[1], c[2], c[3], c[4], c[5], df['Maturity'])
    df['Residual'] = (df['Yield'] - df['NSS'])**2
    val = np.sum(df['Residual'])
    print("[β0, β1, β2, β3, λ0, λ1]=", c, ", SUM:", val)
    return (val)


β0, β1, β2, β3, λ0, λ1 = fmin(myval, [0.0005, 0.0005, 0.0005, 0.0005, 1.00, 1.00])
# β0, β1, β2, β3, λ0, λ1 = fmin(myval, [0.0001, 0.0001, 0.0001, 0.0001, 1.00, 1.00])
print("[β0, β1, β2, β3, λ0, λ1]=", [β0.round(2), β1.round(2), β2.round(2), β3.round(2), λ0.round(2), λ1.round(2)])

# %%
df = df1.copy()
df['NSS'] = nss(β0, β1, β2, β3, λ0, λ1, df['Maturity'])

sf4 = df.copy()
sf5 = sf4.copy()
sf5['Y'] = round(sf4['Yield']*100, 4)
sf5['N'] = round(sf4['NSS']*100, 4)
sf4 = sf4.style.format({'Maturity': '{:,.2f}', 'Yield': '{:,.2%}', 'NS': '{:,.2%}'})

plot_ssn(sf5["Maturity"], sf5["Y"], sf5["N"])

# %%
df['D'] = df['NSS'] - df['Yield']
df.style.format({'Maturity': '{:,.0f}'.format,
                'Yield': '{:,.2%}', 'NSS': '{:,.2%}', 'D': '{:,.2%}'})
# %%
