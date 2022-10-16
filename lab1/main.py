from operator import index
import random
import sys
from time import time
import numpy as np
import logging

N_total = [5, 10, 20, 100, 500, 1000]

def problem(N, seed=42):
    random.seed(seed)
    return [
        list(set(random.randint(0, N - 1) for n in range(random.randint(N // 5, N // 2))))
        for n in range(random.randint(N, N * 5))
    ]

class State():
    """Class for states in A* alghoritm"""

    def __init__(self, state, number_found, g, h, remaining_list = []):
        self.g = g;
        self.h = h;
        self.f = g + h;

        self.remaining_list = remaining_list
        self.state = state
        self.number_found = number_found

def preprocessing(list_of_lists):

    list_of_lists =  sorted(list_of_lists, key = lambda l : len(l))

    return list_of_lists

def greedy_best_first(N):
    goal = set(range(N))
    solution = []
    start_problem = preprocessing(problem(N))
    
    number_found = set()
    n_of_visited_nodes = 0
    
    while goal > number_found:
        n_of_visited_nodes+=1    
        element = start_problem.pop(0)
        if set(element) | number_found > number_found: 
            number_found = set(element) | number_found
            solution.append(element)
    
    print(f"\nSolution using Greedy Best-First algotithm with N = {N} =>\n\t W = {sum(len(element) for element in solution)} \n\t N of VISITED NODES = {n_of_visited_nodes}")
    print(f"Solution {solution}")

#Actual Cost
def g(solution, el):
    return sum(len(element) for element in solution) + len(el)

#Euristic Cost
def h(N, el, number_found):
    return (N - (len(set(el)| number_found)))

def a_star(N):

    n_of_visited_nodes = 0
    start_problem = problem(N)
    goal = set(range(N))
    state_list = []

    open_states = []
    
    for ind, element in enumerate(start_problem):
        n_of_visited_nodes += 1
        state_list.append(element)
        temp_state = State(state_list, set(element) ,g(state_list, element), h(N,element,set(element)))
        open_states.append(temp_state)
        state_list = []
    

    while True:
    
        ind = 0
        current_state = open_states[ind]

        for indice, open_state in enumerate(open_states):

            if open_state.f < current_state.f:
                current_state = open_state
                ind = indice
        
        open_states.pop(ind)
    
        number_found = current_state.number_found      

        if number_found >= goal:
            print("solution :", current_state.state)
            print("W: ", sum(len(element) for element in current_state.state))
            return

        curr_state = current_state.state
        for element in start_problem:

            
            #Needed to not have duplicates
            if element not in current_state.state:
                n_of_visited_nodes += 1
                state_list = curr_state.copy()
                state_list.append(element)
                number_found = set(element) | current_state.number_found

                temp_state = State(state_list, number_found ,g(state_list, element), h(N, element, number_found))
                
                if temp_state.number_found >= goal:
                    print(f"\nSolution using A* algotithm with N = {N} =>\n\t W = {sum(len(element) for element in temp_state.state)} \n\t N of VISITED NODES = {n_of_visited_nodes}")
                    print(f"Solution {temp_state.state}")
                    return

                open_states.append(temp_state)

if __name__ == "__main__":

    for n in N_total:
        st = time()
        greedy_best_first(n)
        et = time()
        print(f"Elapsed Time = {et-st}seconds")

    for n in N_total:
        st = time()
        a_star(n)
        et = time()
        print(f"Elapsed Time = {et-st}seconds")