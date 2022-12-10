import logging
import pickle
import random
import numpy as np
import functools
from typing import Callable
from itertools import accumulate
from copy import deepcopy
from operator import xor
from collections import namedtuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

TUNING = False
CODE_FOR_TABLE = False


N_ROWS = 3
GAMEOVER = [0 for _ in range(N_ROWS)]
POPULATION_SIZE = 10
OFFSPRING_SIZE = 10
N_GENERATIONS = 20




Nimply = namedtuple("Nimply", "row, num_objects")
Move = namedtuple("Move", "row num_objects fitness")

class Nim:
    def __init__(self, num_rows: int, k: int = None) -> None:
        self._rows = [i*2 + 1 for i in range(num_rows)]
        self._k = k
    
    def __str__(self):
        return f"{self._rows}"

    def nimming(self, row: int, num_objects: int) -> None:
        assert self._rows[row] >= num_objects
        assert self._k is None or num_objects <= self._k
        assert num_objects > 0, f"You have to pick at least one"
        self._rows[row] -= num_objects
        if sum(self._rows) == 0:
            logging.debug("Yeuch")
    
    def nimming2(self, ply: Nimply) -> None:
        row, num_objects = ply
        assert self._rows[row] >= num_objects
        assert self._k is None or num_objects <= self._k
        assert num_objects > 0, f"You have to pick at least one"
        self._rows[row] -= num_objects

    @property
    def rows(self):
        return self._rows

    @property
    def k(self) -> int:
        return self._k



def nim_sum(rows: list) -> int:
    # List XOR
    # Using reduce() + lambda + "^" operator
    res = functools.reduce(lambda x, y: x ^ y, rows)
    return res


def tournament(population, tournament_size=2):
    return min(random.choices(population, k=tournament_size), key=lambda i: i.fitness)

def mutation(p: Move, nim: Nim):
    if nim.k is None:
        elements = random.randrange(1, nim.rows[p.row] + 1)
        temp_rows = nim.rows.copy()
        temp_rows[p.row] -=elements 
        offspring = Move(p.row, elements , nim_sum(temp_rows))
    else:
        elements = min(nim.k, random.randrange(1, nim.rows[p.row] + 1))
        temp_rows = nim.rows.copy()
        temp_rows[p.row] -=elements 
        offspring = Move(p.row, elements , nim_sum(temp_rows))

    return offspring

def cross_over(p1: Move, p2: Move, nim: Nim):

    n_random = random.randint(0, 1)
    
    if n_random == 0:

        temp_rows = nim.rows.copy()
        temp_rows[p1.row] -= p2.num_objects

        if temp_rows[p1.row] < 0:
            return None

        offspring = Move(p1.row, p2.num_objects , nim_sum(temp_rows))
    else:
        temp_rows = nim.rows.copy()
        temp_rows[p2.row] -= p1.num_objects

        if temp_rows[p2.row] < 0:
            return None

        offspring = Move(p2.row, p1.num_objects , nim_sum(temp_rows))
    
    if nim.k is not None and offspring.num_objects > nim.k:
        return None

    return offspring

def cook_status(state: Nim) -> dict:
    cooked = dict()
    cooked["possible_moves"] = [
        (r, o) for r, c in enumerate(state.rows) for o in range(1, c + 1) if state.k is None or o <= state.k
    ]
    cooked["active_rows_number"] = sum(o > 0 for o in state.rows)
    cooked["shortest_row"] = min((x for x in enumerate(state.rows) if x[1] > 0), key=lambda y: y[1])[0]
    cooked["longest_row"] = max((x for x in enumerate(state.rows)), key=lambda y: y[1])[0]
    cooked["nim_sum"] = nim_sum(state.rows)

    brute_force = list()
    for m in cooked["possible_moves"]:
        tmp = deepcopy(state)
        tmp.nimming2(m)
        brute_force.append((m, nim_sum(tmp.rows)))
    cooked["brute_force"] = brute_force

    possible_new_states = list()
    for m in cooked["possible_moves"]:
            tmp = deepcopy(state)
            tmp.nimming2(m)
            # (state, move to reach the state)
            possible_new_states.append((tmp, m))
    cooked["possible_new_states"] = possible_new_states

    return cooked

class Player:
    def __init__(self, strategy = 'best') -> None:
        # Two parts for the best strategy:
        # 0 -> before all rows have one element
        # 1 -> after all rows have one element
        self._best_strategy = 0

        assert strategy in ['best', 'best_prof', 'pure_random', 'ga', 'evolvable', 'evolvable_prof', 'evolvable_tuned', 'min_max', 'rl'], f"Strategy non-available"
        self._strategy = strategy
        if strategy == 'rl':
            self.agent = Agent(pretrained=True)

    def moves(self, Nim, alpha = 0.5, beta = 0.5):
        if self._strategy == 'best':
            return self.best_strategy(Nim)
        elif self._strategy == 'best_prof':
            return self.best_strategy_by_prof(Nim)
        elif self._strategy == 'pure_random':
            return self.pure_random(Nim)
        elif self._strategy == 'ga':
            return self.evolvable_based_on_GA(Nim)
        elif self._strategy == 'evolvable':
            return self.evolvable_based_on_fixed_rules(Nim, cook_status(Nim), alpha, beta)
        elif self._strategy == 'evolvable_tuned':
            return self.evolvable_based_on_fixed_rules(Nim, cook_status(Nim), 0.4, 0.1)
        elif self._strategy == 'evolvable_prof':
            return self.evolvable_by_prof(Nim, alpha)
        elif self._strategy == 'min_max':
            return self.min_max_best_move(Nim)
        elif self._strategy == 'rl':
            return self.rl_agent(Nim)
        else: 
            assert f"Can't use a strategy"
        return


    def pure_random(self, Nim):

        # The opponent choose randomly a non-empty row 
        nonzeroind = np.nonzero(Nim.rows)[0]
        random_row = random.choice(nonzeroind)

        # The opponen choose to remove a random number of elements
        if Nim._k == None:
            random_elements = random.randint(1,Nim.rows[random_row])
        else:
            random_elements = random.randint(1,min(Nim._k,Nim.rows[random_row]))

        return Nimply(random_row, random_elements)

        
    def best_strategy(self, Nim):

        # If all the elements are equal or less then k, we can play the 'normal' nim game
        if Nim._k != None and all(v <= Nim._k for v in Nim.rows):
            temp_k = None
        else:
            temp_k = Nim._k

        if temp_k != None:

            # Try brute force:
            for ind, row in enumerate(Nim.rows):

                for el in range(1, min(row + 1, Nim._k + 1)):
                    # Reset temp_rows
                    temp_rows = Nim.rows.copy()
                    
                    # See if nim_sum == 0
                    temp_rows[ind] -= el
                    if nim_sum(temp_rows) == 0:
                        # Update table
                        # Nim.nimming(ind, el)
                        return Nimply(ind, el)
            
            equal_grater_than_k_ind = [i for i,v in enumerate(Nim.rows) if v >= Nim._k + 1]
            
            random_row = random.choice(equal_grater_than_k_ind)
            elements = Nim.rows[random_row]%(Nim._k+1) 
            
            if elements == 0:
                elements = 1

            return Nimply(random_row, elements)

        # If there is only one element greater to one, the agent picks a number of object to make
        # all the rows of the table equal to 1.
        # He can choose to remove all the objects or all the objects but one from the rows with n>1
        if sum(x >= 2 for x in Nim.rows) == 1:
            # Row with more than one element
            equal_grater_than_two_ind = [i for i,v in enumerate(Nim.rows) if v >= 2][0]

            # Change of strategy
            self._best_strategy = 1

            
            # To win, the remaing number of objects has to be even 
            if (sum(x for x in Nim.rows) - Nim.rows[equal_grater_than_two_ind]) % 2 == 0 :
        
                return Nimply(equal_grater_than_two_ind, Nim.rows[equal_grater_than_two_ind])
                
            else:

                return Nimply(equal_grater_than_two_ind, Nim.rows[equal_grater_than_two_ind]-1)        
        
        # Strategy before all rows have one element
        if self._best_strategy == 0:    
        
            res = nim_sum(Nim.rows)

            for ind, row in enumerate(Nim.rows):

                if row == 0:
                    continue

                if row ^ res < row:
                   
                    elements = row - (row ^ res)

                    return Nimply(ind, elements)
        
        # Strategy after all rows have one element
        else:

            nonzeroind = [i for i, e in enumerate(Nim.rows) if e != 0]
            random_row = random.choice(nonzeroind)

            return Nimply(random_row, 1) 
                 
        # Default move -> Random
        nonzeroind = [i for i, e in enumerate(Nim.rows) if e != 0]
        random_row = random.choice(nonzeroind)

        if Nim._k == None:
            random_elements = random.randrange(1,Nim.rows[random_row] + 1)
        else:
            random_elements = random.randrange(1,min(Nim._k,Nim.rows[random_row])+1)

          
        return Nimply(random_row, random_elements) 

    def best_strategy_by_prof(self, state: Nim):
        data = cook_status(state)
        move  = next((bf for bf in data["brute_force"] if bf[1] == 0), random.choice(data["brute_force"]))[0]
        return Nimply(move[0], move[1])

    def evolvable_by_prof(self, state: Nim, p = 0.5):
        data = cook_status(state)

        if random.random() < p:
            if state.k is not None:
                ply = Nimply(data["shortest_row"], min(state.k, random.randint(1, state.rows[data["shortest_row"]])))
            else:
                ply = Nimply(data["shortest_row"], random.randint(1, state.rows[data["shortest_row"]]))

        else:
            if state.k is not None:
                ply = Nimply(data["longest_row"], min(state.k,random.randint(1, state.rows[data["longest_row"]])))
            else:
                ply = Nimply(data["longest_row"], random.randint(1, state.rows[data["longest_row"]]))

        return ply
    
    def evolvable_based_on_fixed_rules(self, state: Nim, cook_status: dict, alpha: float = 0.5, beta: float = 0.5):
        initial_numbers = sum([i*2 + 1 for i in range(N_ROWS)])
        actual_numbers = sum(state.rows)

        # Early game strategy
        if actual_numbers > alpha * initial_numbers:
            
            if cook_status['active_rows_number'] >= beta*N_ROWS:

                row = cook_status["longest_row"]
                if state.k is not None:
                    elements = min(state.k, state.rows[row])
                    
                else:
                    elements = state.rows[row]
                        
                # state.nimming(row, elements)
                return Nimply(row, elements)

        row = cook_status["longest_row"]

        if cook_status["active_rows_number"]%2 == 0 and state.rows[row]!=1 :
            if state.k is not None:
                #Leave at least one element
                elements = min( (state.k - 1), state.rows[row] - 1)
        
            else:
                #Leave at least one element
                elements = state.rows[row] - 1

        else:
            if state.k is not None:
                #Try to remove the maximum number of elements
                elements = min(state.k, state.rows[row])
            else:
                #Try to remove the maximum number of elements
                elements = state.rows[row]
        
        if elements == 0:
            elements = 1
        # state.nimming(row, elements)
        return Nimply(row, elements)
        
    def evolvable_based_on_GA(self, state: Nim):
        
        # Population = possible moves
        population = []
        for _ in range(POPULATION_SIZE):
            # temp_rows needed to evaluate nim_sum = fitness (low nim_sum is better!)
            temp_rows = state.rows.copy()

            # Choosing a random_row
            nonzeroind = [i for i, e in enumerate(state.rows) if e != 0]
            random_row = random.choice(nonzeroind)

            #Choosing a random_move
            if state.k is None:
                random_elements = random.randrange(1, state.rows[random_row] + 1)

            else:
                random_elements =  min(random.randrange(1, state.rows[random_row] + 1), state.k)

            temp_rows[random_row] -= random_elements
            fitness = nim_sum(temp_rows)
            pop = Move(random_row, random_elements, fitness)
            population.append(pop)
        

        for g in range(N_GENERATIONS):
            offspring = list()
            for i in range(OFFSPRING_SIZE):

                if random.random() < 0.5:
                    # Selection of parents
                    p = tournament(population.copy())

                    # Offspring generation
                    o = mutation(p, state)       

                else:
                    
                    p1 = tournament(population)
                    p2 = tournament(population)
                    o = cross_over(p1, p2, state)

                # Check if cross-over returned a valid solution.
                # In this code, only valid solutions has been considered.
                # Possible Improvement: Acceptance with penalties of non-valid solutions
                if o == None:
                    continue

                offspring.append(o)
            
            # Adding new Offspings generated to Population list
            population+=offspring
            
            # Sorting the Population, according to their fitness and selecting the firsts n_elements = POPULATION_SIZE
            population = sorted(population, key=lambda i: i.fitness, reverse=False)[:POPULATION_SIZE]
            
        
        return Nimply(population[0].row, population[0].num_objects)

    # Internal function
    def _minimax(self, state: Nim, maximizing: int = True, alpha=-1, beta=1):
        
        # Check the result of the previous move
        if all(item == 0 for item in state.rows):
            # The player who made the previous move has already won
            return -1 if maximizing else 1

        cooked = cook_status(state)

        scores = []
        for new_state, move in cooked["possible_new_states"]:
            score = self._minimax(new_state, maximizing = not maximizing, alpha = alpha, beta = beta)
            if score != False:
                scores.append(score)
            if maximizing:
                alpha = max(alpha, score)
            else:
                beta = min(beta, score)
            if beta <= alpha:
                break

        return (max if maximizing else min)(scores)
    
    def min_max_best_move(self, state: Nim):
        cooked = cook_status(state)
        ply = None
        for new_state, move in cooked["possible_new_states"]:
    
            score = self._minimax(new_state, maximizing = False)

            if score > 0:
                ply = Nimply(move[0], move[1])
                break
        
        if ply is None:
            logging.debug(" No winning moves :(")
            nonzeroind = [i for i, e in enumerate(state.rows) if e != 0]
            random_row = random.choice(nonzeroind)

            ply =  Nimply(random_row, 1)
            
        return ply


    
    def rl_agent(self, state: Nim):
        ply = self.agent.choose_action(state)
        return ply

class Agent(object):
    def __init__(self, state: Nim = None, alpha=0.15, random_factor=0.2, pretrained = False):  # 80% explore, 20% exploit
        self.state_history = []  # state, reward
        self.alpha = alpha
        self.G = {}
    
        if pretrained:
            with open('lab3/RL_agent.pkl', 'rb') as f:
                self.G = pickle.load(f)
            self.random_factor = 0
            self.k = True
        else:
            assert state is not None, f'Please insert the starting state to inizialize rewards, if you want to train the agent. Otherwise, choose the `pretrained` option'
            self.G = {}
            self.init_reward(state)
            self.random_factor = random_factor

    
    def init_reward(self, state: Nim):
        cooked = cook_status(state)
        for state, move in cooked["possible_new_states"]:
            self.G[tuple(state.rows)] =  np.random.uniform(low=-1.0, high=1.0)

    def choose_action(self, state: Nim):

        maxG = -10e15
        next_move = None
        cooked = cook_status(state)

        randomN = np.random.random()
        if randomN < self.random_factor:
            # if random number below random factor, choose random action
            moves = [move for new_state, move in cooked["possible_new_states"]]
            next_move = random.choice(moves)

        else:
            # if exploiting, gather all possible actions and choose one with the highest G (reward)
            for new_state, move in cooked["possible_new_states"]:
                if tuple(new_state.rows) not in self.G:
                    self.G[tuple(new_state.rows)] = np.random.uniform(low=-1.0, high=1.0)
                    
                if self.G[tuple(new_state.rows)] > maxG:
    
                    next_move = move
                    next_move = Nimply(next_move[0], next_move[1])
                    maxG = self.G[tuple(new_state.rows)]
        
        return Nimply(next_move[0], next_move[1])

    def update_state_history(self, state, reward):
        self.state_history.append((state, reward))
    
    def get_reward_from_state(self, state: Nim ):
        return self.G[tuple(state.rows)]
        
    def learn(self):
        prev = 0

        for curr, reward in reversed(self.state_history):
            if tuple(curr.rows) not in self.G:
                self.G[tuple(curr.rows)] =  np.random.uniform(low=-1.0, high=1.0)
            prev = reward + self.alpha * (prev)
            self.G[tuple(curr.rows)] += prev

        self.state_history = []
        with open('lab3/RL_agent.pkl', 'wb') as f:
            pickle.dump(self.G, f)
        # np.save(os.path.join(os.getcwd(), 'lab3/RL_agent.npy'), self.G)

        self.random_factor -= 10e-5  # decrease random factor each episode of play
    
NUM_MATCHES = 10

def evaluate(agent_strategy = 'best', opponent_strategy = 'pure_random', parameter_dict: dict = {"alpha": None, "beta": None}) -> float:
    
    won = 0
    start = 0
    
    for m in range(NUM_MATCHES):

        agent = Player(agent_strategy)
        opponent = Player(opponent_strategy)

        # 0 -> Agent's turn
        # 1 -> Opponent's turn
        turn = start

        # the first move is equally distributed within matches
        start = 1 - start 

        
        # Training the agent using DR
        rows = random.randint(1, 10)
        K = random.randint(0, rows*2)
        if K == 0:
            K = None
        rows = 3
        K = 3
        # Nim Table Creation    
        nim = Nim(rows, K)
        

        game_over = [0 for _ in range(rows)]

        logging.debug(f"\n\n\n--------NEW GAME---------")

        # Game
        while nim.rows != game_over:

            if turn == 0:
                logging.debug(f" Actual turn: Agent")
            else:
                logging.debug(f" Actual turn: Opponent")

            logging.debug(f" \tTable before move: {nim} and Nim_sum: {nim_sum(nim._rows)}")
            
            if turn == 0:
                if agent_strategy == 'evolvable':
                    assert parameter_dict['alpha'] is not None, f"Please choose a value for alfa"
                    assert parameter_dict['beta'] is not None, f"Please choose a value for beta"
                    ply = agent.moves(nim, parameter_dict['alpha'], parameter_dict['beta'] )
                    

                elif agent_strategy == 'evolvable_by_prof':
                    assert parameter_dict['alpha'] is not None, f"Please choose a value for alfa"
                    ply = agent.moves(nim, parameter_dict['alpha'])
                else:
                    ply = agent.moves(nim)
                
                logging.debug(f" \tAgent:   <Row: {ply.row}- Elements: {ply.num_objects}>")
                
            else:
                
                if opponent_strategy == 'evolvable':
                    assert parameter_dict['alpha_opp'] is not None, f"Please choose a value for alfa used by the opponent -> 'alpha_opp'"
                    assert parameter_dict['beta_opp'] is not None, f"Please choose a value for beta used by the opponent -> 'beta_opp'"
                    ply = opponent.moves(nim, parameter_dict['alpha_opp'], parameter_dict['beta_opp'] )

                elif opponent_strategy == 'evolvable_by_prof':
                    assert parameter_dict['alpha_opp'] is not None, f"Please choose a value for alfa used by the opponent -> 'alpha_opp'"
                    ply = opponent.moves(nim, parameter_dict['alpha_opp'])

                else:
                    ply = opponent.moves(nim)
                
                logging.debug(f" \tOpponent:   <Row: {ply.row}- Elements: {ply.num_objects}>")
            if ply.num_objects == 0:
                print(f"turn = {turn} ")
            nim.nimming2(ply)
            logging.debug(f" \tTable after move: {nim} and Nim_sum: {nim_sum(nim._rows)}\n")

            
            turn = 1 - turn
        
        logging.debug(f"--------GAME OVER---------")
        # Game Over
        if turn == 1:
            won +=1
        else:
            logging.debug(f"Game Lost by the agent is the n°{m}")
            
        
    return won / NUM_MATCHES

def trainingRL():

    # Nim Table Creation    
    # This is the biggest table the RL agent can find and it is needed to inizialize the possible states
    nim = Nim(10, None)
    
    agent = Agent(nim, alpha=0.8, random_factor=0.5)
    
    win_history_to_plot = []
    indices = []

    parameter_dict= {}
    parameter_dict["alpha_opp"] = 0.5
    parameter_dict["beta_opp"] = 0.5

    start = 0

    logging.getLogger().setLevel(logging.INFO)

    for i in tqdm(range(10000)):

        turn = 1 - start
        start = turn
        # opponent_strategy = random.choice(strategies)
        opponent_strategy = 'best'
        opponent = Player(opponent_strategy)

        current_state = None
        step = 0
        
        # Training the agent using DR
        rows = random.randint(1, 10)
        K = random.randint(0, rows*2)
        if K == 0:
            K = None
        # Nim Table Creation    
        nim = Nim(rows, K)
    

        game_over = [0 for _ in range(rows)]

        while nim.rows != game_over:
            if turn == 0:
                logging.debug(f" Actual turn: Agent")
            else:
                logging.debug(f" Actual turn: Opponent")

            logging.debug(f" \tTable before move: {nim} ")
            step +=1
            if turn == 0:
                # Current state
                current_state = nim
                old_state =  nim
                # Choose an action (explore or exploit)
                ply = agent.choose_action(current_state)
                logging.debug(f" \tAgent:   <Row: {ply.row}- Elements: {ply.num_objects}>")
                nim.nimming2(ply)

                # New current state
                current_state = deepcopy(nim)

                # update the robot memory with state and reward
                if current_state.rows == GAMEOVER:
                    agent.update_state_history(current_state, 1)

            else:

                ply = opponent.moves(nim, parameter_dict['alpha_opp'], parameter_dict['beta_opp'] )
                nim.nimming2(ply)

                if current_state is not None:
                    # update the robot memory with state and reward
                    if nim.rows == GAMEOVER and turn == 1:
                        agent.update_state_history(current_state, -1)
                    else:
                        agent.update_state_history(current_state, 0)

            if step >= 1000:
                break

            logging.debug(f" \tTable after move: {nim} ")
            turn = 1 - turn

        agent.learn()  # robot should learn after every episode

        if i%5001 == 0:
            # Game Over
            if turn == 1:
                print("0")
                # win_history_to_plot.append(0)
                # indices.append(i)
            else:
                # win_history_to_plot.append(1)
                # indices.append(i)
                print("1")

    # plt.plot(indices, win_history_to_plot, "b", linewidth=1)
    # plt.show()
    return


if __name__ == '__main__':
    random.seed(42)

    trainingRL()
    parameter_dict= {}

    # # Insert here your own parameters
    # # Best values of alpha and beta against a pure random opponent are respectively 0.4 and 0.1 generally

    parameter_dict["alpha"] = 0.4
    parameter_dict["beta"] = 0.1
    parameter_dict["alpha_opp"] = 0.5
    parameter_dict["beta_opp"] = 0.5
    print(f"Agent Won: {evaluate(agent_strategy='rl', opponent_strategy='best', parameter_dict = parameter_dict)*100}% of the games")