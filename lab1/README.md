# Lab 1: Set Covering

## Introduction

This lab aims to solve the **_Set Covering_** problem by finding the optimal and the most efficient solution.
To solve this problem, we used two informed strategies:

* _Greedy (Best) First_ Algorithm
* _A*_ Algorithm

## Methods

### Greedy (Best) First Algorithm

The algorithm has been implemented as follows:

```python
def greedy_best_first(N):
    goal = set(range(N))
    solution = []
    start_problem = sorted(problem(N), key = lambda l : len(l))
    
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
```

The `goal` is defined as a set containing all the numbers from $0$ to $N$. Next, we sort the list of lists from the shorter to the longer. Each loop takes and pops the first list in the queue (inside the list of lists, called `starting_problem`). Then, if the list extracted contains numbers that have not been already found, we append this list to the solution. This algorithm takes - in each loop - the most promising list measured in terms of length.

### A* Algorithm

The algorithm has been implemented as follows:

```python
class State():
    """Class for states in A* alghoritm"""

    def __init__(self, state, number_found, g, h, remaining_list = []):
        self.g = g;
        self.h = h;
        self.f = g + h;

        self.remaining_list = remaining_list
        self.state = state
        self.number_found = number_found

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
```

This algorithm aims to extract the best solution according to

$$f(n) = g(n) + h(n)$$

where $g(n)$ is the actual cost, measured as the actual solution length, and where $h(n)$ is the estimated cost (heuristic), measured as the length of the list in the solution. If $h(n)$ is **admissible** (never overestimates the cost) and if it is **consistent**, so if for every node $n$ and for every successor $n_s$ of n: $h(n) \le cost(n, n_s) + h(n_s)$, the algorithm guarantees to find the optimal solution.

## Experimental results

The experimental results for the _Greedy (Best) First_ Algorithm are reported in the following table.

| N | W(Cost) | Visited Nodes | Elapsed Time|
| --| ------- | ------------- | ----------- |
|5 | 5 | 13 | 0.0008 s |
| 10 | 13 | 14 | 0.0006 s|
|20|46|14|0.0006 s|
|100|332|23|0.04 s|
|500| 2162| 28| 0.8 s|
|1000| 4652 | 27| 2.9 s|

Next, we report the results for the _A*_ Algorithm.

| N | W(Cost) | Visited Nodes | Elapsed Time|
| --| ------- | ------------- | ----------- |
|5 |  5 | 4558 | 0.17 s |
| 10 | 10 | 27231 | 3.5 s|
|20 | 23| 49163|9.1 s|
|100|_|_|_|
|500| _| _| _|
|1000| _ | _| _|

In the last case, we do not report the results for $N$ greater than 20, because the algorithm requires a computational time extremely high.
As the tables show, the _Greedy_ algorithm gives us a _non-optimal_ solution, but it is faster in the computation. On the other hand, the _A*_ algorithm gives us optimal solutions, but it is really slow. In fact, the _A*_ visits many more nodes than the _Greedy_ one.

## Conclusions

The two algorithms give different solutions. In choosing the right algorithm, we have to consider a trade-off between computational costs and the goodness of solutions.

## Author

* [Marco Pratticò (294815)](!github.com/marcopra)
* [Samuele Pino (303332)](!github.com/samuelePino)
