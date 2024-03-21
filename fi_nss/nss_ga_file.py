#!/usr/bin/env python
""" Genetic Algorithm NS and NSS to file

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


# %% imports
import functools
import multiprocessing
import time
import pandas as pd
import numpy as np
import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from rich import print as rprint

random.seed(55)

IN_FILE_NAME = './data/in/yield-curve-rates-up-22-23.csv'
OUT_FILE_NAME = './data/out/nss-up-22-23-d.csv'

# Genetic Algorithm constants:
POPULATION_SIZE = 200
MAX_GENERATIONS = 1000
P_CROSSOVER = 0.9       # probability for crossover
P_MUTATION = 0.2        # probability for mutating an individual
HALL_OF_FAME_SIZE = 40

BETA_MIN = -1.1
BETA_MAX = 1.1
LAMBDA_MIN = 0
LAMBDA_MAX = 100


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
            return population, logbook, gen

    return population, logbook, ngen


def create_maturity_yield_tuple(df: pd.DataFrame, date: str) -> tuple:
    maturity = [1, 2, 3, 4, 6, 12, 24, 36, 60, 84, 120, 240, 360]  # TODO: calculate this.
    yields = (df.loc[[date]].values / 100).flatten().tolist()[0:13]
    return maturity, yields


# Optimized
def ns(beta0, beta1, beta2, lambda0, df_maturity):
    df_maturity_div_lambda0 = df_maturity / lambda0
    result = (
        (beta0) +
        (beta1 * ((1 - np.exp(-df_maturity_div_lambda0)) / (df_maturity_div_lambda0))) +
        (beta2 * ((((1 - np.exp(-df_maturity_div_lambda0)) / (df_maturity_div_lambda0))) - (np.exp(-df_maturity_div_lambda0))))
    )
    return result


# Optimized
def nss(beta0, beta1, beta2, beta3, lambda0, lambda1, df_maturity):
    df_maturity_div_lambda0 = df_maturity / lambda0
    df_maturity_div_lambda1 = df_maturity / lambda1
    result = (
        (beta0) +
        (beta1 * (
            (1 - np.exp(-df_maturity_div_lambda0)) / (df_maturity_div_lambda0)
            )
         ) +
        (beta2 * (
            (((1 - np.exp(-df_maturity_div_lambda0)) / (df_maturity_div_lambda0))) - (np.exp(-df_maturity_div_lambda0))
            )
         ) +
        (beta3 * ((((1 - np.exp(-df_maturity_div_lambda1)) / (df_maturity_div_lambda1))) - (np.exp(-df_maturity_div_lambda1))))
    )
    return result


def getLowestNS(individual, df_maturity, dd):
    c = np.array(individual)
    df = dd.copy()
    df['NS'] = ns(c[0], c[1], c[2], c[3], df_maturity)
    df['Residual_NS'] = 1000*((df['Yield'] - df['NS'])**2)
    val = np.sum(df['Residual_NS'])
    return val,


def getLowestNSS(individual, df_maturity, dd):
    c = np.array(individual)
    df = dd.copy()
    df['NSS'] = nss(c[0], c[1], c[2], c[3], c[4], c[5], df_maturity)
    df['Residual_NSS'] = 1000*((df['Yield'] - df['NSS'])**2)
    val = np.sum(df['Residual_NSS'])
    return val,


def createNSToolbox(df_maturity, df) -> base.Toolbox:
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
    # toolbox.register("select", tools.selTournament, tournsize=3)  # toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", getLowestNS, df_maturity=df_maturity, dd=df)  # df['Maturity'])
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individual)
    # toolbox.register("mutate", tools.mutESLogNormal, low=0, up=100, indpb=0.05)
    # toolbox.register("mutate", tools.mutUniformInt, low=0, up=100, indpb=1.0/5)
    # toolbox.register("Integers", random.randint, 0, 100)
    # toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.Integers, 5)
    return toolbox


def createNSSToolbox(df_maturity, df) -> base.Toolbox:
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
    # toolbox.register("select", tools.selTournament, tournsize=3)
    # toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", getLowestNSS, df_maturity=df_maturity, dd=df)  # df['Maturity'])
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individual)
    # toolbox.register("mutate", tools.mutESLogNormal, low=0, up=100, indpb=0.05)
    # toolbox.register("mutate", tools.mutUniformInt, low=0, up=100, indpb=1.0/5)
    # toolbox.register("Integers", random.randint, 0, 100)
    # toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.Integers, 5)
    return toolbox


def runGenNS(df_maturity, df, pool):
    toolbox = createNSToolbox(df_maturity, df)
    toolbox.register("map", pool.map)
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
        max_fitness=0.00001)
    best = hof.items[0]
    rprint(f"-- NS Best Individual =  {best} - Generations: {generations}")
    rprint(f"-- NS Best Fitness = {best.fitness.values[0]} - Generations: {generations}")
    # minFitnessValues, meanFitnessValues, maxFitnessValues = logbook.select("min", "avg", "max")
    # rprint(f"-- NS minFitnessValues = {minFitnessValues} - meanFitnessValues: {meanFitnessValues} - maxFitnessValues: {maxFitnessValues}")
    return best


def runGenNSS(df_maturity, df, pool):
    toolbox = createNSSToolbox(df_maturity, df)
    toolbox.register("map", pool.map)
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
        max_fitness=0.00001)
    best = hof.items[0]
    rprint(f"-- NSS Best Individual =  {best} - Generations: {generations}")
    rprint(f"-- NSS Best Fitness = {best.fitness.values[0]} - Generations: {generations}")
    # minFitnessValues, meanFitnessValues, maxFitnessValues = logbook.select("min", "avg", "max")
    # rprint(f"-- NSS minFitnessValues = {minFitnessValues} - meanFitnessValues: {meanFitnessValues} - maxFitnessValues: {maxFitnessValues}")
    return best


# %% load file
ycdf = pd.read_csv(IN_FILE_NAME, header=0, index_col=0)
available_dates = ycdf.index.to_list()
ycdf['β0_nss'] = 0.0
ycdf['β1_nss'] = 0.0
ycdf['β2_nss'] = 0.0
ycdf['β3_nss'] = 0.0
ycdf['tau0_nss'] = 0.0
ycdf['tau1_nss'] = 0.0
ycdf['residual_nss'] = 0.0
ycdf['β0_ns'] = 0.0
ycdf['β1_ns'] = 0.0
ycdf['β2_ns'] = 0.0
ycdf['tau_ns'] = 0.0
ycdf['residual_ns'] = 0.0


@timer
def main():
    cpu_count = 6  # multiprocessing.cpu_count()
    rprint(f"CPUs assigned: {cpu_count}")
    pool = multiprocessing.Pool(cpu_count)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    for ad in available_dates:
        mat, yld = create_maturity_yield_tuple(ycdf, str(ad))
        dd = pd.DataFrame(data={'Maturity': mat, 'Yield': yld})
        df = dd.copy()
        
        # can this go outside?
        β0_ns, β1_ns, β2_ns, λ_ns = runGenNS(df['Maturity'], df, pool)
        β0_nss, β1_nss, β2_nss, β3_nss, λ0_nss, λ1_nss = runGenNSS(df['Maturity'], df, pool)  # redundant df
        
        df['NS'] = ns(β0_ns, β1_ns, β2_ns, λ_ns, df['Maturity'])
        df['NSS'] = nss(β0_nss, β1_nss, β2_nss, β3_nss, λ0_nss, λ1_nss, df['Maturity'])
        
        df['residual_ns'] = (df['Yield'] - df['NS'])**2
        df['residual_nss'] = (df['Yield'] - df['NSS'])**2
        
        ycdf.loc[str(ad), 'β0_ns'] = β0_ns
        ycdf.loc[str(ad), 'β1_ns'] = β1_ns
        ycdf.loc[str(ad), 'β2_ns'] = β2_ns
        ycdf.loc[str(ad), 'tau_ns'] = λ_ns
        ycdf.loc[str(ad), 'residual_ns'] = np.sum(df['residual_ns'])
        
        ycdf.loc[str(ad), 'β0_nss'] = β0_nss
        ycdf.loc[str(ad), 'β1_nss'] = β1_nss
        ycdf.loc[str(ad), 'β2_nss'] = β2_nss
        ycdf.loc[str(ad), 'β3_nss'] = β3_nss
        ycdf.loc[str(ad), 'tau0_nss'] = λ0_nss
        ycdf.loc[str(ad), 'tau1_nss'] = λ1_nss
        ycdf.loc[str(ad), 'residual_nss'] = np.sum(df['residual_nss'])

    pool.close()
    ycdf.to_csv(OUT_FILE_NAME, index=True, index_label='Date')


if __name__ == "__main__":
    main()
