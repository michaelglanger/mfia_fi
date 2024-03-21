#!/usr/bin/env python
""" NSS calculation with downhill simplex algorithm and save to file.

Longer description of this module.
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Michael G. Langer-Monsalve"
__contact__ = "michael@langer-monsalve.com"
__copyright__ = "Copyright 2024, MGL-M"
__credits__ = ["Michael G. Langer-Monsalve"]
__date__ = "2024/03/18"
__deprecated__ = False
__email__ = "michael@langer-monsalve.com"
__license__ = "GPLv3"
__maintainer__ = "developer"
__status__ = "Development"
__version__ = "0.1.1"

# %% imports
import functools
import time
from scipy.optimize import fmin
import pandas as pd
import numpy as np


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("Finished {} in {} secs".format(repr(func.__name__), round(run_time, 3)))
        return value

    return wrapper


# %% load file
ycdf = pd.read_csv('./data/in/yield-curve-rates-2023.csv', header=0, index_col=0)
available_dates = ycdf.index.to_list()
ycdf['β0_dsa'] = 0.0
ycdf['β1_dsa'] = 0.0
ycdf['β2_dsa'] = 0.0
ycdf['β3_dsa'] = 0.0
ycdf['tau0_dsa'] = 0.0
ycdf['tau1_dsa'] = 0.0
ycdf['residual_dsa'] = 0.0


# %% load file
def create_maturity_yield_tuple(df: pd.DataFrame, date: str) -> tuple:
    maturity = [1, 2, 3, 4, 6, 12, 24, 36, 60, 84, 120, 240, 360]  # TODO: calculate this.
    yields = (df.loc[[date]].values / 100).flatten().tolist()[0:13]
    return maturity, yields


def nss(beta0, beta1, beta2, beta3, lambda0, lambda1, df_maturity):
    result = (
        (beta0) +
        (beta1 * ((1 - np.exp(-df_maturity / lambda0)) / (df_maturity / lambda0))) +
        (beta2 * ((((1 - np.exp(-df_maturity / lambda0)) / (df_maturity / lambda0))) - (np.exp(-df_maturity / lambda0)))) +
        (beta3 * ((((1 - np.exp(-df_maturity / lambda1)) / (df_maturity / lambda1))) - (np.exp(-df_maturity / lambda1))))
    )
    return result


@timer
def run():
    def myval(c):
        df = dd.copy()
        df['NSS'] = nss(c[0], c[1], c[2], c[3], c[4], c[5], df['Maturity'])
        df['Residual'] = (df['Yield'] - df['NSS'])**2
        val = np.sum(df['Residual'])
        # print("[β0, β1, β2, β3, λ0, λ1]=", c, ", SUM:", val)
        return (val)
    
    for ad in available_dates:
        mat, yld = create_maturity_yield_tuple(ycdf, str(ad))
        dd = pd.DataFrame(data={'Maturity': mat, 'Yield': yld})
        df = dd.copy()
        sf = df.copy()
        sf = sf.dropna()
        sf1 = sf.copy()
        sf1['Y'] = round(sf['Yield']*100, 4)
        β0 = 0.01
        β1 = 0.01
        β2 = 0.01
        β3 = 0.01
        tau0 = 1.00
        tau1 = 1.00
        df['Y'] = round(df['Yield']*100, 4)
        df['NSS'] = nss(β0, β1, β2, β3, tau0, tau1, df['Maturity'])
        df['N'] = round(df['NSS']*100, 4)
        df['Residual'] = (df['Yield'] - df['NSS'])**2
        residual_sum = np.sum(df['Residual'])  # TODO: save this in the table
        β0, β1, β2, β3, tau0, tau1 = fmin(myval, [0.0005, 0.0005, 0.0005, 0.0005, 1.00, 1.00])
        ycdf.loc[str(ad), 'β0_dsa'] = β0
        ycdf.loc[str(ad), 'β1_dsa'] = β1
        ycdf.loc[str(ad), 'β2_dsa'] = β2
        ycdf.loc[str(ad), 'β3_dsa'] = β3
        ycdf.loc[str(ad), 'tau0_dsa'] = tau0
        ycdf.loc[str(ad), 'tau1_dsa'] = tau1
        ycdf.loc[str(ad), 'residual_dsa'] = residual_sum


run()
ycdf.to_csv('./data/out/dsa.csv', index=True, index_label='Date')
