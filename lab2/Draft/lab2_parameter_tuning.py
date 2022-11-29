from operator import index
import argparse
import random
import sys
from time import time
import numpy as np
from collections import namedtuple
import logging
from collections import Counter
import itertools
from typing import Callable
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D


"""     Global Variable & Settings  """
POPULATION_SIZE = 400
OFFSPRING_SIZE = 1000

NUM_GENERATIONS = 1000

N = 100
random.seed(42)
GOAL = set(range(N))
Individual = namedtuple("Individual", ["genome", "fitness"])


"""     Functions       """

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--save',   default=True, type=bool, help="if enabled, all plots will be saved")
#     parser.add_argument('--tuning', default=True, type=bool, help="if enabled, the code will generate plots useful for paremeters tuning")
#     parser.add_argument("--clear-past", default=True, type=bool, help="Whether or not to empty routes and images content before optimization")
#     parser.add_argument("--use-best", default=False, type=bool, help="Use the best configuration for the given N")
#     parser.add_argument("--N", default=10, type=int, help="Definition of the problem size")
#     # TODO comma and plus strategies

#     return parser.parse_args()

# args = parse_args()

def remove_duplicates(list_):
  list_ = list(k for k,_ in itertools.groupby(list_))

  return list_

def sort_by(list_: list, key: Callable = None):
  list_.sort(key=key)
  return list_
  
def preproc(problem, rem_dup=False, sort=False, f:Callable = None):
  problem= sort_by(problem, key=f)
  problem= remove_duplicates(problem)
  return problem

def create_genome(problem):
    genome = []
    numbers_found = set()

    while numbers_found != GOAL:

        n_random = random.choice(range(0, len(problem)))
        genome.append(tuple(problem[n_random]))
        numbers_found |= set(problem[n_random])

        problem.pop(n_random)
        
    return genome

def fitness(genome):
    
    cnt = Counter()
    cnt.update(sum((e for e in genome), start=()))

    # Counting (Number of redundant elements, Numbers of useful elements)
    return tuple([sum(cnt[c] - 1 for c in cnt if cnt[c] > 1), -sum(cnt[c] == 1 for c in cnt)])

def tournament(population, tournament_size=2):
    return min(random.choices(population, k=tournament_size), key=lambda i: i.fitness)


def mutation(g, problem):

    # Random Number of Pops
    for _ in range(0, len(g) - 1):
        # Deleting a random Gene (= List)
        point = random.randint(0, len(g) - 1)
        g.pop(point)
   

    # Numbers covered without the Gene chosen previously
    numbers_found = set()
    for element in g:
        numbers_found != set(element)

    # Counter to avoid infinite loops
    steps = 0

    while numbers_found != GOAL:
        steps += 1

        if steps == 10000:
            # No Solution found in a reasonable number of step
            return None, g
        
        # Choosing a list from the problem randomly and
        # Adding it to the candidate solution (Genome)
        n_random = random.choice(range(0, len(problem)))

        # Avoiding to have equal lists inside the Genome
        if not any(list == tuple(problem[n_random]) for list in g):
            g.append(tuple(problem[n_random]))
            numbers_found |= set(problem[n_random])

            problem.pop(n_random)

    return g, g

def cross_over(p1, p2):
    '''This function performs the cross-over using the parents as input and generating on offspring as output
        
        Return:
            param1: offspring if the solution is valid or None if it is not
            param2: offspring
    '''
    n_random = random.randint(0, 1)
    
    if n_random == 0:
        mid_len = int(len(p1)/2)
    else:
        mid_len = int(len(p2)/2)
    
    # Security Check
    if mid_len > len(p2) or mid_len > len(p1):
        return None, p1
    
    n_random = random.randint(0, 1)
    
    # Offspring generation
    if n_random == 0:
        offspring = p1[:mid_len] + p2[mid_len:]
    else:
        offspring = p2[:mid_len] + p1[mid_len:]
    
    if set(offspring) != GOAL:
        return None, offspring 
    
    return offspring, offspring

def problem(N, seed=42):
    random.seed(seed)
    return [
        list(set(random.randint(0, N - 1) for n in range(random.randint(N // 5, N // 2))))
        for n in range(random.randint(N, N * 5))
    ]

def population_initialization(all_list):
    population = list()

    for _ in range(POPULATION_SIZE):
    
        genome = create_genome(all_list.copy())
        population.append(Individual(genome, fitness(genome)))
    
    return population

def plots():
    ''' Credit '''
    pass

def surface_plot (matrix, **kwargs):
    # acquire the cartesian coordinate matrices from the matrix
    # x is cols, y is rows
    (x, y) = np.meshgrid(np.arange(matrix.shape[1]*2, step=2), np.arange(matrix.shape[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, matrix, **kwargs)
    return (fig, ax, surf)



def main(N = N):
    st = time()
    all_list = preproc(problem(N))
    population = population_initialization(all_list)
    et = time()

    step_size = 2

    print(f"Population created in: {et - st}s")

    history = np.empty(shape = (0, int(NUM_GENERATIONS) ))
    temp_history = list()

    st = time()
    for pop in tqdm(range(0,POPULATION_SIZE, step_size)):
        for g in range(NUM_GENERATIONS):
            offspring = list()
            for i in range(OFFSPRING_SIZE):
                if random.random() < 0.5:
                    # Selection of parents
                    p = tournament(population.copy())

                    # Offspring generation
                    o, o_anyway = mutation(p.genome.copy(), all_list.copy())

                    if random.random() < 0.3:
                        
                        p2 = tournament(population.copy())
                        o, _ = list(cross_over(o_anyway, p2.genome))
                        

                else:
                    
                    p1 = tournament(population)
                    p2 = tournament(population)
                    o, o_anyway = cross_over(p1.genome, p2.genome)

                    if random.random() < 0.3:
                    
                        o, _ = mutation(o_anyway, all_list.copy())

                # Check if the mutation or cross-over returned a valid solution.
                # In this code, only valid solutions has been considered.
                # Possible Improvement: Acceptance with penalties of non-valid solutions
                if o == None:
                    continue
            
                # Fitness of Offspring
                f = fitness(o)
                
                offspring.append(Individual(o, fitness(o)))
            
            # Adding new Offspings generated to Population list
            population+=offspring

            # Sorting the Population, according to their fitness and selecting the firsts n_elements = POPULATION_SIZE
            population = sorted(population, key=lambda i: i.fitness)[:POPULATION_SIZE]

            temp_history.append(sum(len(element) for element in population[0].genome))

        history = np.vstack((history, np.array(temp_history)))
        temp_history = list()
        
        # et  = time()
        # print("Winner: ", population[0])
        # print("Cost: ", sum(len(element) for element in population[0].genome))
        # print("Bloat= ", int(sum(len(element) for element in population[0].genome)/N * 100), "%")
        # print(f"Elapsed time: {et - st}s")
    
    et  = time()
    print(f"Elapsed time: {et - st}s")
    print("Winner: ", population[0])
    print("Cost: ", sum(len(element) for element in population[0].genome))
    print("Bloat= ", int(sum(len(element) for element in population[0].genome)/N * 100), "%")


    # if args.clear_past:
    # folders = ["./images"]
    # for folder in folders: 
    #     files = os.listdir(folder)
    #     for file in files: 
    #         os.remove(folder + "/" + file)

    # if args.tuning == False:
        # assert args.save == False, f"You can save only plots generated when the 'tuning' option is enabled"
    
    # else:

    (fig, ax, surf) = surface_plot(history, cmap=plt.cm.coolwarm)

    fig.colorbar(surf)

    ax.set_xlabel('Number of Generations')
    ax.set_ylabel('Population Size')
    ax.set_zlabel('Best W (Cost)')

    plt.show()
    fig.savefig(f"N={N}-fitness.jpg")

    # X, Y = np.meshgrid(np.arange(history.shape[1]), np.arange(history.shape[0]))
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot3D(X, Y, history, 'gray')
    # fig, ax = plt.subplots()
    # ax.plot(history, lw = 2, alpha = .5)
    # ax.scatter(x = np.arange(len(history)),y = history, marker = "x", c = "grey", s = 25)
    # ax.set_title(f"Set Covering - N = {N} \nFitness in iterations", fontweight = "bold")
    # ax.set_xlabel("Generations", fontsize=12)
    # ax.set_ylabel("Fitness of Fittest Individual", fontsize=12)

    # fig.savefig(f"images/N={N}-fitness.jpg")

if __name__ == '__main__':
    main()


    
