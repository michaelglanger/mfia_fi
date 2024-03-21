#!/usr/bin/env python
""" Yield, NS, NSS and QS file viewer

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
__date__ = "2024/03/19"
__deprecated__ = False
__email__ = "michael@langer-monsalve.com"
__license__ = "GPLv3"
__maintainer__ = "developer"
__status__ = "Development"
__version__ = "0.1.2"
# %%
import pandas as pd
from fi_nss.nss_common import plot_all, ns, nss

IN_FILE_NAME = '../data/out/nss-up-22-23-d.csv'
# IN_FILE_NAME = '../data/out/nss.csv'     # "../data/out/nss-2023-c.csv"
# IN_FILE_NAME = '../data/out/nss-up-22-23-d.csv'     # "../data/out/nss-2023-c.csv"
# date_of_values = '12/28/23'  # '12/26/23'  # '6/6/22'                           # '6/14/22'  # '12/22/23'
date_of_values = '12/7/21'  # '12/26/23'  # '6/6/22'                           # '6/14/22'  # '12/22/23'
NS = True
NSS = True
QS = False


def create_maturity_yield_tuple(df: pd.DataFrame, date: str) -> tuple:
    maturity = [1, 2, 3, 4, 6, 12, 24, 36, 60, 84, 120, 240, 360]  # TODO: calculate this.
    yields = (df.loc[[date]].values / 100).flatten().tolist()[0:13]
    return maturity, yields


ycdf = pd.read_csv(IN_FILE_NAME, header=0, index_col=0)
mat, yld = create_maturity_yield_tuple(ycdf, date_of_values)
dd = pd.DataFrame(data={'Maturity': mat, 'Yield': yld})
df = dd.copy()
df['Yield'] = round(dd['Yield']*100, 4)


if NS:
    beta0_ns = ycdf.loc[date_of_values, 'β0_ns']
    beta1_ns = ycdf.loc[date_of_values, 'β1_ns']
    beta2_ns = ycdf.loc[date_of_values, 'β2_ns']
    tau_ns = ycdf.loc[date_of_values, 'tau_ns']
    df['NS'] = round(ns(beta0_ns, beta1_ns, beta2_ns, tau_ns, df['Maturity'])*100, 4)
    y_ns = df['NS']
else:
    y_ns = None

if NSS:
    beta0_nss = ycdf.loc[date_of_values, 'β0_nss']
    beta1_nss = ycdf.loc[date_of_values, 'β1_nss']
    beta2_nss = ycdf.loc[date_of_values, 'β2_nss']
    beta3_nss = ycdf.loc[date_of_values, 'β3_nss']
    tau0_nss = ycdf.loc[date_of_values, 'tau0_nss']
    tau1_nss = ycdf.loc[date_of_values, 'tau1_nss']
    df['NSS'] = round(nss(beta0_nss, beta1_nss, beta2_nss, beta3_nss, tau0_nss, tau1_nss, df['Maturity'])*100, 4)
    y_nss = df['NSS']
else:
    y_nss = None


plot_all(df['Maturity'], df['Yield'], y_ns=y_ns, y_nss=y_nss)

# %%
