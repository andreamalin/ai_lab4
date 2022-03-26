# Referencia de: https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35

from TSP_Classes.City import City
from TSP_Classes.Fitness import Fitness
import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt

def createRoute(cities):
    route = random.sample(cities, len(cities))
    return route

def initialPopulation(popSize, cities):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cities))
    return population

def routeOrdering(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).getFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse= True)

def selectionProcess(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break

    return selectionResults

def matingPool(population, selectionResults):
    matingPool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingPool.append(population[index])
    return matingPool

def breed(parent1, parent2):
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    start = min(geneA, geneB)
    end = max(geneA, geneB)

    for i in range(start, end):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    return childP1 + childP2

def breedPopulation(matingPool, eliteSize):
    children = []
    length = len(matingPool) - eliteSize
    pool = random.sample(matingPool, len(matingPool))

    for i in range(0, eliteSize):
        children.append(matingPool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingPool) - i - 1])
        children.append(child)
    return children

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWidth = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWidth]

            individual[swapped] = city2
            individual[swapWidth] = city1

    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutateInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutateInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = routeOrdering(currentGen)
    selectionResults = selectionProcess(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

#  PLOT TSP
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / routeOrdering(pop)[0][1]))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    
    print("Final distance: " + str(1 / routeOrdering(pop)[0][1]))
    bestRouteIndex = routeOrdering(pop)[0][0]
    bestRoute = pop[bestRouteIndex]

    plot_tour(bestRoute)

def plot_tour(cities):
    for i in range(len(cities)):
        x0, y0 = cities[i-1].x, cities[i-1].y
        x1, y1 = cities[i].x, cities[i].y
        plt.scatter(int(x0), int(y0), s=15, c='r')
        plt.annotate(i + 1, (int(x0), int(y0)))
        plt.plot([x1, x0], [y1, y0], c='b')         

    plt.show()


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / routeOrdering(pop)[0][1])
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / routeOrdering(pop)[0][1])
    
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

def createPopulation():
    citiesFile = open("cities.txt", "r")
    coordinates = []
    for line in citiesFile:
        lineSplit = line.split()

        if (len(lineSplit) == 1): # Reading quantity of cities
            nCities = int(lineSplit[0])
        else: # Reading coordinates
            coordinates.append(City(int(lineSplit[0]), int(lineSplit[1])))
    citiesFile.close()
    return coordinates

def main():
    cities = createPopulation()

    # Ruta
    geneticAlgorithm(population=cities, popSize=15, eliteSize=10, mutationRate=0.01, generations=500)
    # Performance
    geneticAlgorithmPlot(population=cities, popSize=15, eliteSize=10, mutationRate=0.01, generations=500)

main()