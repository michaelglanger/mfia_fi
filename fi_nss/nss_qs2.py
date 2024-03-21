#!/usr/bin/env python
""" Short description of this Python module.

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
__credits__ = ["Michael G. Langer-Monsalve", "Roi Polanitzer"]
__date__ = "2024/03/13"
__deprecated__ = False
__email__ = "michael@langer-monsalve.com"
__license__ = "GPLv3"
__maintainer__ = "developer"
__status__ = "Development"
__version__ = "0.4.1"


# %% imports
import pandas as pd
import numpy as np
import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from rich import print as rprint
from nss_common import plot_all

random.seed(55)


def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
                        halloffame=None, verbose=__debug__, max_fitness=None):
    """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
    halloffame is used to implement an elitism mechanism. The individuals contained in the
    halloffame are directly injected into the next generation and are not subject to the
    genetic operators of selection, crossover and mutation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        rprint(logbook.stream)

    # Begin the generational process
    # for gen in range(1, 10):
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the best back to population:
        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            rprint(logbook.stream)    
        
        if max_fitness and halloffame.items[0].fitness.values[0] < max_fitness:
            # rprint(f'OK {gen}')
            return population, logbook, gen
        
        # rprint(f'BEST {gen} - {halloffame.items[0].fitness.values[0]}')
        
    return population, logbook, ngen


def create_maturity_yield_tuple(df: pd.DataFrame, date: str) -> tuple:
    maturity = [1, 2, 3, 4, 6, 12, 24, 36, 60, 84, 120, 240, 360]  # TODO: calculate this.
    yields = (df.loc[[date]].values / 100).flatten().tolist()
    return maturity, yields


def ns(beta0, beta1, beta2, lambda0, df_maturity):
    result = (
        (beta0) +
        (beta1 * ((1 - np.exp(-df_maturity / lambda0)) / (df_maturity / lambda0))) +
        (beta2 * ((((1 - np.exp(-df_maturity / lambda0)) / (df_maturity / lambda0))) - (np.exp(-df_maturity / lambda0))))
    )
    return result


def nss(beta0, beta1, beta2, beta3, lambda0, lambda1, df_maturity):
    result = (
        (beta0) +
        (beta1 * ((1 - np.exp(-df_maturity / lambda0)) / (df_maturity / lambda0))) +
        (beta2 * ((((1 - np.exp(-df_maturity / lambda0)) / (df_maturity / lambda0))) - (np.exp(-df_maturity / lambda0)))) +
        (beta3 * ((((1 - np.exp(-df_maturity / lambda1)) / (df_maturity / lambda1))) - (np.exp(-df_maturity / lambda1))))
    )
    return result


# %% load file
# date_of_values = '12/28/23'
# date_of_values = '10/29/21'
date_of_values = '7/17/14'
ycdf = pd.read_csv('yield-curve-rates-1990-2023.csv', header=0, index_col=0)
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
plot_all(sf1["Maturity"], sf1["Y"], title=f"Yield Curve for {date_of_values}")

# %%
β0 = 0.01
β1 = 0.01
β2 = 0.01
β3 = 0.01
λ0 = 1.00
λ1 = 1.00

# %%
df['NSS'] = nss(β0, β1, β2, β3, λ0, λ1, df['Maturity'])
df.style.format({'Maturity': '{:,.0f}'.format, 'Yield': '{:,.2%}', 'NSS': '{:,.2%}'})

# %%
df1 = df.copy()
df['Y'] = round(df['Yield']*100, 4)
df['NSS'] = nss(β0, β1, β2, β3, λ0, λ1, df['Maturity'])
df['N'] = round(df['NSS']*100, 4)
df2 = df.copy()
df2 = df2.style.format({'Maturity': '{:,.2f}'.format, 'Y': '{:,.2%}', 'N': '{:,.2%}'})

plot_all(df["Maturity"], df["Y"], y_nss=df["N"])

# %%
df['Residual'] = (df['Yield'] - df['NSS'])**2
df22 = df[['Maturity', 'Yield', 'NSS', 'Residual']]
df22.style.format({'Maturity': '{:,.0f}'.format, 'Yield': '{:,.2%}','NSS': '{:,.2%}','Residual': '{:,.9f}'})
# %%
# np.sum(df['Residual'])


# Genetic Algorithm constants:
POPULATION_SIZE = 200
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.2   # probability for mutating an individual
MAX_GENERATIONS = 500
HALL_OF_FAME_SIZE = 40


def getLowestNS(individual):
    c = np.array(individual)
    df = dd.copy()
    df['NS'] = ns(c[0], c[1], c[2], c[3], df['Maturity'])
    df['Residual_NS'] = 100*(df['Yield'] - df['NS'])**2
    val = np.sum(df['Residual_NS'])
    # print("[β0, β1, β2, β3, λ0, λ1]=", c, ", SUM:", val)
    return val,  # This needs to be a tuple


def getLowestNSS(individual):
    c = np.array(individual)
    df = dd.copy()
    df['NSS'] = nss(c[0], c[1], c[2], c[3], c[4], c[5], df['Maturity'])
    df['Residual_NSS'] = 100*(df['Yield'] - df['NSS'])**2
    val = np.sum(df['Residual_NSS'])
    # print("[β0, β1, β2, β3, λ0, λ1]=", c, ", SUM:", val)
    return val,  # This needs to be a tuple


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


BETA_MIN = -1.1
BETA_MAX = 1.1
LAMBDA_MIN = 0
LAMBDA_MAX = 100


def createNSToolbox() -> base.Toolbox:
    toolbox = base.Toolbox()
    toolbox.register("attr_beta0", random.uniform, BETA_MIN, BETA_MAX)
    toolbox.register("attr_beta1", random.uniform, BETA_MIN, BETA_MAX)
    toolbox.register("attr_beta2", random.uniform, BETA_MIN, BETA_MAX)
    toolbox.register("attr_lambda0", random.uniform, LAMBDA_MIN, LAMBDA_MAX)
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (
            toolbox.attr_beta0,
            toolbox.attr_beta1,
            toolbox.attr_beta2,
            toolbox.attr_lambda0
        ),
    )
    toolbox.register("select", tools.selTournament, tournsize=3)
    # toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", getLowestNS)
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individual)
    # toolbox.register("mutate", tools.mutESLogNormal, low=0, up=100, indpb=0.05)
    # toolbox.register("mutate", tools.mutUniformInt, low=0, up=100, indpb=1.0/5)
    # toolbox.register("Integers", random.randint, 0, 100)
    # toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.Integers, 5)
    return toolbox


def createNSSToolbox() -> base.Toolbox:
    toolbox = base.Toolbox()
    toolbox.register("attr_beta0", random.uniform, BETA_MIN, BETA_MAX)
    toolbox.register("attr_beta1", random.uniform, BETA_MIN, BETA_MAX)
    toolbox.register("attr_beta2", random.uniform, BETA_MIN, BETA_MAX)
    toolbox.register("attr_beta3", random.uniform, BETA_MIN, BETA_MAX)
    toolbox.register("attr_lambda0", random.uniform, LAMBDA_MIN, LAMBDA_MAX)
    toolbox.register("attr_lambda1", random.uniform, LAMBDA_MIN, LAMBDA_MAX)
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (
            toolbox.attr_beta0,
            toolbox.attr_beta1,
            toolbox.attr_beta2,
            toolbox.attr_beta3,
            toolbox.attr_lambda0,
            toolbox.attr_lambda1,
        ),
    )
    toolbox.register("select", tools.selTournament, tournsize=3)
    # toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", getLowestNSS)
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individual)
    # toolbox.register("mutate", tools.mutESLogNormal, low=0, up=100, indpb=0.05)
    # toolbox.register("mutate", tools.mutUniformInt, low=0, up=100, indpb=1.0/5)
    # toolbox.register("Integers", random.randint, 0, 100)
    # toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.Integers, 5)
    return toolbox


def runGenNS():
    toolbox = createNSToolbox()
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    population, logbook, generations = eaSimpleWithElitism(
        population,
        toolbox,
        cxpb=P_CROSSOVER,
        mutpb=P_MUTATION,
        ngen=MAX_GENERATIONS,
        stats=stats,
        halloffame=hof,
        verbose=False,
        max_fitness=1.3e-05)
    best = hof.items[0]
    rprint(f"-- Best Individual NS =  {best} - Generations: {generations}")
    rprint(f"-- Best Fitness NS = {best.fitness.values[0]} - Generations: {generations}")
    minFitnessValues, meanFitnessValues, maxFitnessValues = logbook.select("min", "avg", "max")
    return best


def runGenNSS():
    toolbox = createNSSToolbox()
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    population, logbook, generations = eaSimpleWithElitism(
                                    population, 
                                    toolbox, 
                                    cxpb=P_CROSSOVER, 
                                    mutpb=P_MUTATION, 
                                    ngen=MAX_GENERATIONS, 
                                    stats=stats, 
                                    halloffame=hof, 
                                    verbose=False,
                                    max_fitness=1.5e-06)
    best = hof.items[0]
    rprint(f"-- Best Individual NSS =  {best} - Generations: {generations}")
    rprint(f"-- Best Fitness NSS = {best.fitness.values[0]} - Generations: {generations}")
    minFitnessValues, meanFitnessValues, maxFitnessValues = logbook.select("min", "avg", "max")
    return best


# %%
β0_ns, β1_ns, β2_ns, λ_ns = runGenNS()
β0_nss, β1_nss, β2_nss, β3_nss, λ0_nss, λ1_nss = runGenNSS()


DECIMALS = 5
print(
    "NS[β0, β1, β2, λ]=",
    [
        round(β0_ns, DECIMALS),
        round(β1_ns, DECIMALS),
        round(β2_ns, DECIMALS),
        round(λ_ns, DECIMALS),

    ],
)
print(
    "NSS[β0, β1, β2, β3, λ0, λ1]=",
    [
        round(β0_nss, DECIMALS),
        round(β1_nss, DECIMALS),
        round(β2_nss, DECIMALS),
        round(β3_nss, DECIMALS),
        round(λ0_nss, DECIMALS),
        round(λ1_nss, DECIMALS),
    ],
)

# %%
df = df1.copy()
df['NSS'] = nss(β0_nss, β1_nss, β2_nss, β3_nss, λ0_nss, λ1_nss, df['Maturity'])
df['NS'] = ns(β0_ns, β1_ns, β2_ns, λ_ns, df['Maturity'])


sf4 = df.copy()
sf5 = sf4.copy()
sf5['Y'] = round(sf4['Yield']*100, 4)
sf5['N'] = round(sf4['NSS']*100, 4)
sf5['N2'] = round(sf4['NS']*100, 4)
sf4 = sf4.style.format({'Maturity': '{:,.2f}', 'Yield': '{:,.2%}', 'NSS': '{:,.2%}', 'NS': '{:,.2%}'})

plot_all(sf5["Maturity"], sf5["Y"], y_nss=sf5["N"], y_ns=sf5["N2"])

# %%
df['D_NSS'] = df['NSS'] - df['Yield']
df['D_NS'] = df['NS'] - df['Yield']
df.style.format({'Maturity': '{:,.0f}'.format, 'Yield': '{:,.2%}', 'NSS': '{:,.2%}',
                'D_NS': '{:,.2%}', 'D_NSS': '{:,.2%}', 'NS': '{:,.2%}'})
# %%
