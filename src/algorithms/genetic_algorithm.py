#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    algorithms.genetic_algorithm
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Implementation of the genetic programming algorithm

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import numpy as np
from decisiontrees import Leaf
from .genetic_programming import *


class GeneticAlgorithm:
    """
    A class that implements the genetic algorithm"""

    def __init__(self, **kwargs):
        """
        Initializes the algorithm

        :pop_size: The size of the population
        :cx_prob: The crossover_probability
        :mut_prob: The mutation probability
        :tournament_size: The size of the tournament for the selection
        :c_factory: The factory for the conditions
        :l_factory: The factory for the leaves
        :bounds: dictionary containing the bounds for the two factories.
            It should contain two keys: "condition" and "leaf".
            The values must contain the bounds
            (a dict with keys (type, min, max))
            for all the parameters returned
            by "get_trainable_parameters"
        :max_depth: Maximum depth for the trees

        """
        self._pop_size = kwargs["init_pop_size"]
        self._cx_prob = kwargs["cx_prob"]
        self._mut_prob = kwargs["mut_prob"]
        self._tournament_size = kwargs["tournament_size"]
        self._c_factory = kwargs["c_factory"]
        self._l_factory = kwargs["l_factory"]
        self._bounds = kwargs["bounds"]
        self._max_depth = kwargs["max_depth"]
        self._gp = GeneticProgramming(**kwargs)
        self._pop = self._gp._init_pop()

    def ask(self):
        ask_pop = self._pop[:]
        return [p._genes for p in ask_pop]
        

    def _tournament_selection(self, fitnesses):
        n_ind = len(fitnesses)
        tournaments = np.random.choice(
            [*range(n_ind)],
            (n_ind, self._tournament_size)
        )

        selected = []

        for t in tournaments:
            max_ = float("-inf")
            argmax_ = None
            for idx in t:
                if fitnesses[idx] > max_ or argmax_ is None:
                    argmax_ = idx
                    max_ = fitnesses[idx]

            selected.append(argmax_)
        return selected

    def tell(self, fitnesses, data=None):
        selection = self._tournament_selection(fitnesses)

        new_pop = []
        n_ind = len(selection)

        for i in range(0, n_ind, 2):
            p1 = self._pop[selection[i]]

            if i + 1 < n_ind:
                p2 = self._pop[selection[i + 1]]
            else:
                p2 = None

            o1, o2 = None, None

            # Crossover
            if p2 is not None and np.random.uniform() < self._cx_prob:
                o1, o2 = self._gp._crossover(p1, p2)

            # Mutation
            if np.random.uniform() < self._mut_prob:
                o1 = self._gp._mutation(p1 if o1 is None else o1)

            if p2 is not None and np.random.uniform() < self._mut_prob:
                o2 = self._gp._mutation(p2 if o2 is None else o2)

            new_pop.append(p1 if o1 is None else o1)
            if p2 is not None:
                new_pop.append(p2 if o2 is None else o2)
                
        for i in range(self._pop_size):
            new_pop[i] = self._gp._limit_depth(new_pop[i])
        self._pop = new_pop

    def get_all_pop(self):
        return self._pop[:]