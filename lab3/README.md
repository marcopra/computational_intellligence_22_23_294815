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

* Early game strategy: this strategy is applied if and only if the remaining objects are more than $50%$ of the initial ones ($\alpha = 0.5$ in the algorithm$) and if the active rows are more than $50%$ of the initial ones ($\beta = 0.5$ in the algorithm$). In this strategy, the goal is to reduce as much as possible the longest row.
* Late game strategy: this strategy is applied otherwise. In this strategy, the goal is to reduce as much as possible the longest row if the active rows are odd or to leave at least one element (or $K + 1$ elements) if the active rows are even.

This strategy has been implemented in `evolvable_based_on_fixed_rules` function and it is performed using `alpha = 0.5` and `beta = 0.5`.

## Task 3.2 (*Evolvable Strategies*)

### Evolvable Strategy based on GA

This strategy is based on Genetic Algorithm paradigms.

* Genome: is a Tuple which represents a possible move (`Row`,`Object to remove`).
* Fitness: is represented by the _Nim sum_ if the possible move would be done (low fitness is better).

### Evolvable Strategy based on Fixed Rules

This strategy is based on the [one](#fixed-rules) mentioned above. In this case, the parameters $\alpha$ and $\beta$ has been tuned. to achieve .
The tuned parameters has been chosen to achive the best performances (highest win rate) over 100 matches against a pure random opponent and with a random value of $K$. The best values are $\alpha = 0.4$ e $\beta = 0.1$

## Results

In the table below, we report the agent win ration over 100 matches using different strategies. The left column represents the strategies used by the agent, while on the row at the top, we represents the strategy used by the opponet. Please note that the starting turn is equally distributes over the matches, so the agents and the opponent will do the first 50 times each. This is important since the first moving player as an advantage.

| Agent\Opponent                                             | `pure_random` | `best_strategy` | `best_strategy_by_prof` | `evolvable_based_on_fixed_rules (alpha = 0.5, beta = 0.5)` | `evolvable_based_on_fixed_rules (alpha = 0.4, beta = 0.1)` | `evolvable_based_on_GA` | `evolvable_by_prof` |
|------------------------------------------------------------|---------------|-----------------|-----------------------|------------------------------------------------------------|----------------------------------------------------------|-----------------------|-------------------|
| `pure_random`                                              | 54%           | 1%              | 1%                    | 19%                                                        | 16%                                                      | 1%                    | 44%               |
| `best_strategy`                                            | 99%           | 48%             | 47%                   | 100%                                                       | 99%                                                      | 60%                   | 96%               |
| `best_strategy_by_prof`                                      | 99%           | 51%             | 53%                   | 96%                                                        | 93%                                                      | 63%                   | 99%               |
| `evolvable_based_on_fixed_rules (alpha = 0.5, beta = 0.5)` | 94%           | 4%              | 7%                    | 50%                                                        | 46%                                                      | 4%                    | 91%               |
| `evolvable_based_on_fixed_rules (alpha = 0.4, beta = 0.1)`   | 92%           | 4%              | 1%                    | 52%                                                        | 51%                                                      | 5%                    | 87%               |
| `evolvable_based_on_GA`                                      | 99%           | 34%             | 44%                   | 98%                                                        | 99%                                                      | 52%                   | 99%               |
| `evolvable_by_prof`                                          | 45%           | 0%              | 0%                    | 9%                                                         | 19%                                                      | 1%                    | 51%               |