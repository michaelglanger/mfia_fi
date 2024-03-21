# %%
import matplotlib.ticker as mtick
import matplotlib.markers as mk
from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
raw_data = pd.read_csv('qs3.csv')
raw_data

# %%
a0 = raw_data.Duration[(len(raw_data.Duration)-2)]
a1 = raw_data.Duration[(len(raw_data.Duration)-1)]
b0 = raw_data.YtM[(len(raw_data.YtM)-2)]
b1 = raw_data.YtM[(len(raw_data.YtM)-1)]
a = (a1-a0)
b = (b1-b0)
c0 = 361.00
c = (c0-a1)
d = (c*b/a)
e = c0
f = b1 + d
raw_data.loc[len(raw_data.index)] = [e, f]
raw_data.rename(columns={"YtM": "Spot ex post"}, inplace=True)
raw_data

# %%
raw_data["Forward ex post"] = raw_data["Spot ex post"]
for i in range(1, len(raw_data["Spot ex post"])):
    raw_data["Forward ex post"][i] = (raw_data["Duration"][i]*raw_data["Spot ex post"][i]-raw_data["Duration"]
                                      [i-1]*raw_data["Spot ex post"][i-1])/(raw_data["Duration"][i]-raw_data["Duration"][i-1])
x1 = np.array(raw_data.Duration)
y1 = np.array(raw_data["Forward ex post"])
xvals = np.linspace(0, 360, 121).round(2)
yinterp = np.interp(xvals, x1, y1)
a = {'Duration': xvals, 'Yield': yinterp}
data = pd.DataFrame(data=a)

Anchor = [0, 4, 8, 20, 40, 80, 120]
Duration = [data.Duration[i] for i in Anchor]
Yield = [data.Yield[i] for i in Anchor]

x = np.array(Duration)
y = np.array(Yield)
anchor_points = pd.DataFrame(data={'Duration': x, 'Forward ex post': y})
df = anchor_points.copy()


# %%
x = np.array(df.Duration)
y = np.array(df["Forward ex post"])

# use bc_type = 'natural' adds the constraints as we described above
f = CubicSpline(x, y, bc_type='natural')
x_new = np.linspace(0.25, 360, 100)
y_new = f(x_new)

df = pd.DataFrame(data={'Period': x_new, 'Forward ex post': y_new})
df1 = df.copy()

df1['Spot ex ante'] = df1['Forward ex post']
for i in range(1, len(df1['Forward ex post'])):
    df1['Spot ex ante'][i] = (df1['Forward ex post'][i]*(df1['Period'][i]-df1['Period'][i-1]) +
                              df1['Period'][i-1]*df1['Spot ex ante'][i-1])/df1['Period'][i]
df1.rename(columns={"Forward ex post": "Forward Rate"}, inplace=True)
df2 = df1.copy()


# %%
df2.rename(columns={"Spot ex ante": "Spot Rate"}, inplace=True)
df2.drop(['Forward Rate'], axis=1, inplace=True)
df1 = df2.copy()
df1



# %%
# M0 = np.min(df1['Spot Rate']) - 0.05
# M1 = np.max(df1['Spot Rate']) + 0.05
fontsize = 15
fig = plt.figure(figsize=(13, 7))
plt.title("Forward Spline Interpolation - Israeli Real RF Spot Yield Curve 30.09.2023", fontsize=fontsize)
ax = plt.axes()
ax.set_facecolor("moccasin")
fig.patch.set_facecolor('aliceblue')
X = df1['Period']
# Y = df1['RF_S']
Y = df1['Spot Rate']
ax.plot(X, Y, color="limegreen", label="Intrinsic Value")
plt.scatter(X, Y, marker="o", c="blue")
plt.xlabel('Period', fontsize=fontsize)
plt.ylabel('Spot Rate', fontsize=fontsize)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
# plt.xlim(0.25, 25)
# plt.ylim(M0, M1)
plt.legend(loc="center right")
plt.grid()
plt.show()

# %%
