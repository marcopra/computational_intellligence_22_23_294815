# Lab 3: Policy Search

## Goal

Write agents able to play [*Nim*](https://en.wikipedia.org/wiki/Nim), with an arbitrary number of rows and an upper bound $k$ on the number of objects that can be removed in a turn (a.k.a., *subtraction game*).

The player **taking the last object wins**.

* Task3.1: An agent using fixed rules based on *nim-sum* (i.e., an *expert system*)
* Task3.2: An agent using evolved rules

## Task 3.1 (*Expert Agent*)

### Best Possible strategy

The algorithm of the best possible strategy has been inspired by [*Nim*](https://en.wikipedia.org/wiki/Nim). This algorithm reaches optimal performances ($\approx 100\%$) against a pure random opponent. If two players choose the play with the optimal strategy, the winning agent will be the one with the first move. This strategy has been implemented in `best_strategy` function.

### Fixed Rules

The fixed rules strategy is divided into two parts:

* Early game strategy: this strategy is applied if and only if the remaining objects are more than 50% of the initial ones ( $\alpha = 0.5$ in the algorithm) and if the active rows are more than 50% of the initial ones ( $\beta = 0.5$ in the algorithm). In this strategy, the goal is to reduce as much as possible the longest row.
* Late game strategy: this strategy is applied otherwise. In this strategy, the goal is to reduce as much as possible the longest row if the active rows are odd or to leave at least one element (or $K + 1$ elements) if the active rows are even.

This strategy has been implemented in `evolvable_based_on_fixed_rules` function and it is performed using `alpha = 0.5` and `beta = 0.5`.

## Task 3.2 (*Evolvable Strategies*)

### Evolvable Strategy based on GA

This strategy is based on Genetic Algorithm paradigms.

* Genome: is a Tuple which represents a possible move (`Row`,`Object to remove`).
* Fitness: is represented by the _Nim sum_ if the possible move would be done (low fitness is better).

### Evolvable Strategy based on Fixed Rules

This strategy is based on the [one](#fixed-rules) mentioned above. In this case, the parameters $\alpha$ and $\beta$ had been tuned.
The tuned parameters had been chosen to achieve the best performances (highest win rate) over 100 matches against a pure random opponent and with a random value of $K$. The best values are $\alpha = 0.4$ e $\beta = 0.1$

## Task 3.3 (*Min-Max Strategy*)

This strategy is based on the _Min-Max_ algorithm. The implemented version of the algorithm (without a limited depth) is very slow, due to the high quantity of possible states. To make the algorithm faster, we implemented also the alpha-beta pruning to avoid considering non-necessary states. A possible improvement is the implementation of the limited depth.

## Task 3.4 (*Reinforcement Learning Agent*)

This strategy has been implemented accordingly to the Reinforcement Learning paradigms. In this case, a _State (S)_ is represented by the state of the _Nim Board_. The _Reward (R)_ of a _State_ corresponds to $+1$ if the _RL_ agent wins the game in that _State_, if the _RL_ agent loses is $-1$, otherwise is $0$. Please, note that for each _State (S)_ we compute the future discounted reward using:

```python
for curr, reward in reversed(self.state_history):
    [...]
    prev = reward + self.alpha * (prev)
    self.G[tuple(curr.rows)] += prev
```

Each state is initialized accordingly to a _Uniform_ distribution between $-1$ and $+1$. The policy $\pi$ aims to choose the new possible state with the maximum reward.
The agent has been trained using a sort of domain randomization (DR). So, it has been trained over $10000$ Games on _Nim Boards_ with $1 \leq N Rows \leq 10$ and $1 \leq K \leq N Rows \lor K = None$.

## Results

In the table below, we report the agent win ratio over 100 matches using different strategies. In this case, the games are played on a _Nim_ board with 3 rows and $K = 3$. The left column represents the strategies used by the agent, while on the row at the top, we represent the strategy used by the opponent. Please note that the starting turn is equally distributed over the matches, so the agents and the opponent will do the first 50 times each. This is important since the first moving player has an advantage.

| Agent\Opponent                                             | `pure_random` | `best_strategy` | `best_strategy_by_prof` | `evolvable_based_on_fixed_rules (alpha = 0.5, beta = 0.5)` | `evolvable_based_on_fixed_rules (alpha = 0.4, beta = 0.1)` | `evolvable_based_on_GA` | `evolvable_by_prof` | `min_max`     | `rl`          |
|------------------------------------------------------------|---------------|-----------------|-------------------------|------------------------------------------------------------|------------------------------------------------------------|-------------------------|---------------------|---------------|---------------|
| `pure_random`                                              | 54%           | 1%              | 1%                      | 19%                                                        | 16%                                                        | 1%                      | 44%                 | Not Evaluated | 4%            |
| `best_strategy`                                            | 99%           | 48%             | 47%                     | 100%                                                       | 99%                                                        | 60%                     | 96%                 | Not Evaluated | 50%           |
| `best_strategy_by_prof`                                    | 99%           | 51%             | 53%                     | 96%                                                        | 93%                                                        | 63%                     | 99%                 | Not Evaluated | 50%           |
| `evolvable_based_on_fixed_rules (alpha = 0.5, beta = 0.5)` | 94%           | 4%              | 7%                      | 50%                                                        | 46%                                                        | 4%                      | 91%                 | Not Evaluated | 50%           |
| `evolvable_based_on_fixed_rules (alpha = 0.4, beta = 0.1)` | 92%           | 4%              | 1%                      | 52%                                                        | 51%                                                        | 5%                      | 87%                 | Not Evaluated | 0%            |
| `evolvable_based_on_GA`                                    | 99%           | 34%             | 44%                     | 98%                                                        | 99%                                                        | 52%                     | 99%                 | Not Evaluated | 50%           |
| `evolvable_by_prof`                                        | 45%           | 0%              | 0%                      | 9%                                                         | 19%                                                        | 1%                      | 51%                 | Not Evaluated | 4%            |
| `min_max`                                                  | Not Evaluated | Not Evaluated   | Not Evaluated           | Not Evaluated                                              | Not Evaluated                                              | Not Evaluated           | Not Evaluated       | Not Evaluated | Not Evaluated |
| `rl`                                                       | 98%           | 50%             | 50%                     | 50%                                                        | 100%                                                       | 96%                     | 51%                 | Not Evaluated | 50%           |

Then, we report the results if using a random table ( $1 \leq N Rows \leq 10$ and $1 \leq K \leq N Rows \lor K = None$) over the 100 matches played.

| Agent\Opponent                                             | `pure_random` | `best_strategy` | `best_strategy_by_prof` | `evolvable_based_on_fixed_rules (alpha = 0.5, beta = 0.5)` | `evolvable_based_on_fixed_rules (alpha = 0.4, beta = 0.1)` | `evolvable_based_on_GA` | `evolvable_by_prof` | `min_max`     | `rl`          |
|------------------------------------------------------------|---------------|-----------------|-------------------------|------------------------------------------------------------|------------------------------------------------------------|-------------------------|---------------------|---------------|---------------|
| `pure_random`                                              | 54%           | 9%              | 12%                     | 22%                                                        | 40%                                                        | 14%                     | 54%                 | Not Evaluated | 33%           |
| `best_strategy`                                            | 85%           | 50%             | 47%                     | 87%                                                        | 88%                                                        | 63%                     | 97%                 | Not Evaluated | 92%           |
| `best_strategy_by_prof`                                    | 88%           | 54%             | 51%                     | 88%                                                        | 90%                                                        | 67%                     | 86%                 | Not Evaluated | 81%           |
| `evolvable_based_on_fixed_rules (alpha = 0.5, beta = 0.5)` | 70%           | 12%             | 16%                     | 51%                                                        | 57%                                                        | 18%                     | 68%                 | Not Evaluated | 59%           |
| `evolvable_based_on_fixed_rules (alpha = 0.4, beta = 0.1)` | 59%           | 10%             | 9%                      | 34%                                                        | 48%                                                        | 13%                     | 63%                 | Not Evaluated | 51%           |
| `evolvable_based_on_GA`                                    | 83%           | 28%             | 31%                     | 78%                                                        | 87%                                                        | 54%                     | 93%                 | Not Evaluated | 75%           |
| `evolvable_by_prof`                                        | 58%           | 11%             | 12%                     | 30%                                                        | 44%                                                        | 18%                     | 45%                 | Not Evaluated | 35%           |
| `min_max`                                                  | Not Evaluated | Not Evaluated   | Not Evaluated           | Not Evaluated                                              | Not Evaluated                                              | Not Evaluated           | Not Evaluated       | Not Evaluated | Not Evaluated |
| `rl`                                                       | 61%           | 22%             | 17%                     | 39%                                                        | 47%                                                        | 21%                     | 62%                 | Not Evaluated | 52%           |

