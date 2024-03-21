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
c0 = 372.00
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
Duration = []
Yield = []
for i in Anchor:
    Duration.append(data.Duration[i])
    Yield.append(data.Yield[i])

x = np.array(Duration)
y = np.array(Yield)
b = {'Duration': x, 'Forward ex post': y}
anchor_points = pd.DataFrame(data=b)
df = anchor_points.copy()
df['RF'] = round(anchor_points['Forward ex post']*100, 4)
anchor_points = anchor_points.style.format({'Duration': '{:,.2f}'.format, 'Forward ex post': '{:,.4%}'})
anchor_points

# %%
x = np.array(df.Duration)
y = np.array(df["Forward ex post"])
z = np.array(df.RF)
x1 = x
# use bc_type = 'natural' adds the constraints as we described above
f = CubicSpline(x, y, bc_type='natural')
x_new = np.linspace(0.25, 360, 20)
y_new = f(x_new)

e = {'Period': x_new, 'Forward ex post': y_new}
df = pd.DataFrame(data=e)
df1 = df.copy()

df1['Spot ex ante'] = df1['Forward ex post']
for i in range(1, len(df1['Forward ex post'])):
    df1['Spot ex ante'][i] = (df1['Forward ex post'][i]*(df1['Period'][i]-df1['Period'][i-1]) +
                              df1['Period'][i-1]*df1['Spot ex ante'][i-1])/df1['Period'][i]
df1.rename(columns={"Forward ex post": "Forward Rate"}, inplace=True)
df2 = df1.copy()
df1.drop(['Spot ex ante'], axis=1, inplace=True)

a1 = df1.copy()
a2 = df1.copy()
a3 = df1.copy()
a4 = df1.copy()
a5 = df1.copy()
a6 = df1.copy()

a1 = a1.head(17)

a2 = a2.tail(83)
a2 = a2.head(17)
a2 = a2.reset_index(drop=True)

a3 = a3.tail(66)
a3 = a3.head(17)
a3 = a3.reset_index(drop=True)

a4 = a4.tail(49)
a4 = a4.head(17)
a4 = a4.reset_index(drop=True)

a5 = a5.tail(32)
a5 = a5.head(17)
a5 = a5.reset_index(drop=True)

a6 = a6.tail(15)
a6.loc[len(a6.index)] = ['', '']
a6.loc[len(a6.index)] = ['', '']
a6 = a6.head(17)
a6 = a6.reset_index(drop=True)

frames = [a1, a2, a3, a4, a5, a6]
f_vector = pd.concat(frames, axis=1)
f_vector

# %%
df2.rename(columns={"Spot ex ante": "Spot Rate"}, inplace=True)
df2.drop(['Forward Rate'], axis=1, inplace=True)
df1 = df2.copy()
df1

a1 = df1.copy()
a2 = df1.copy()
a3 = df1.copy()
a4 = df1.copy()
a5 = df1.copy()
a6 = df1.copy()

a1 = a1.head(17)

a2 = a2.tail(83)
a2 = a2.head(17)
a2 = a2.reset_index(drop=True)

a3 = a3.tail(66)
a3 = a3.head(17)
a3 = a3.reset_index(drop=True)

a4 = a4.tail(49)
a4 = a4.head(17)
a4 = a4.reset_index(drop=True)

a5 = a5.tail(32)
a5 = a5.head(17)
a5 = a5.reset_index(drop=True)

a6 = a6.tail(15)
a6.loc[len(a6.index)] = ['', '']
a6.loc[len(a6.index)] = ['', '']
a6 = a6.head(17)
a6 = a6.reset_index(drop=True)

frames = [a1, a2, a3, a4, a5, a6]
s_vector = pd.concat(frames, axis=1)
s_vector

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
