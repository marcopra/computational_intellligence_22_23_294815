# Lab 2: Set Covering with Genetic Algorithm

## Introduction

This lab aims to solve the **_Set Covering_** problem using the genetic algorithm.

## Method:

### Structure:

* Gene: in this solution, a gene is represented by a single list, treated as a `Tuple`.
* Genome: it is a potential solution composed of Genes.
* Individual: it is a `Tuple` containing a Genome and its cost.
* Population: it is a list containing a fixed number of individuals.

### Fitness Function:

The fitness function is defined as follows:

```python
def fitness(genome):
    cnt = Counter()
    cnt.update(sum((e for e in genome), start=()))
    return tuple([sum(cnt[c] - 1 for c in cnt if cnt[c] > 1), -sum(cnt[c] == 1 for c in cnt)])
```

The result is a tuple containing:

* the number of redundant elements.
* the negative of the useful numbers for the set covering problem.

The aim is to minimize the results of the fitness function.

## Experimental results

The experimental results for the _Greedy (Best) First_ Algorithm are reported in the following table.

| N | W(Cost) | Bloat | Population Size | Offspring size| Offspring size|
| --| ------- | -------- |------------- | ----------- |  ----------- |
|5 | 5 | 15 | 100% |3 | 0.003 s |
| 10 | 100 | 100% | 2000 | 100 | 0.09 s|
|20|26| 130% |100000 | 1000 |17.6 s|
|50|75| 150% |100000 | 1000 |18.0 s|
|100|203| 203% |100000| 1000| 18.1 s|
|500| 1607 | 321% | 100000| 1000| 36.1 s|
|1000| 3744 | 374% | 100000| 1000| 71.3 s|

## Conclusions

The experimental results show that the algorithm can find reasonable results by spending a reasonable amount of time. 
Obviously, in this algorithm, the goodness of results depends in a non-negligible way on the `POPULATION_SIZE`, `OFFSPRING_SIZE`, and `NUM_GENERATIONS`.

## Authors

This code has been developed after discussing it with [Samuele Pino 303332](https://github.com/samuelePino). However, our code is different.
