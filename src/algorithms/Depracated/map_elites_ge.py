#!/usr/bin/python3
"""
Implementation of the grammatical evolution

Author: Leonardo Lucio Custode
Creation Date: 04-04-2020
Last modified: mer 6 mag 2020, 16:30:41
"""
import re
import os
import string
import numpy as np
from typing import List
from abc import abstractmethod
from .common import OptMetaClass
from util_processing_elements.processing_element import ProcessingElementFactory, PEFMetaClass
import utils

from decisiontrees import *

TAB = " " * 4


class GrammaticalEvolutionTranslator:
    def __init__(self, grammar):
        """
        Initializes a new instance of the Grammatical Evolution
        :param n_inputs: the number of inputs of the program
        :param leaf: the leaf that can be used - a constructor
        :param constant_range: A list of constants that can be used - default is a list of integers between -10 and 10
        """
        self.operators = grammar

    def _find_candidates(self, string):
        return re.findall("<[^> ]+>", string)

    def _find_replacement(self, candidate, gene):
        key = candidate.replace("<", "").replace(">", "")
        value = self.operators[key][gene % len(self.operators[key])]
        return value

    def genotype_to_str(self, genotype):
        """ This method translates a genotype into an executable program (python) """
        string = "<bt>"
        candidates = [None]
        ctr = 0
        _max_trials = 1
        genes_used = 0

        # Generate phenotype starting from the genotype
        #   If the individual runs out of genes, it restarts from the beginning, as suggested by Ryan et al 1998
        while len(candidates) > 0 and ctr <= _max_trials:
            if ctr == _max_trials:
                return "", len(genotype)
            for gene in genotype:
                candidates = self._find_candidates(string)
                if len(candidates) > 0:
                    value = self._find_replacement(candidates[0], gene)
                    string = string.replace(candidates[0], value, 1)
                    genes_used += 1
                else:
                    break
            ctr += 1

        string = self._fix_indentation(string)
        return string, genes_used

    def _fix_indentation(self, string):
        # If parenthesis are present in the outermost block, remove them
        if string[0] == "{":
            string = string[1:-1]

        # Split in lines
        string = string.replace(";", "\n")
        string = string.replace("{", "{\n")
        string = string.replace("}", "}\n")
        lines = string.split("\n")

        fixed_lines = []
        n_tabs = 0

        # Fix lines
        for line in lines:
            if len(line) > 0:
                fixed_lines.append(TAB * n_tabs + line.replace("{", "").replace("}", ""))

                if line[-1] == "{":
                    n_tabs += 1
                while len(line) > 0 and line[-1] == "}":
                    n_tabs -= 1
                    line = line[:-1]
                if n_tabs >= 100:
                    return "None"

        return "\n".join(fixed_lines)


class Individual:
    """Represents an individual."""

    def __init__(self, genes, fitness=None, parents=None):
        """Initializes a new individual

        :genes: a list of genes
        :fitness: the fitness for the individual. Default: None.

        """
        self._genes = np.array(genes)
        self._fitness = fitness
        self._parents = parents  # A list containing the indices of the parents in the previous population
        self._id = "".join(np.random.choice([*string.ascii_lowercase], 10))

    def get_genes(self):
        return self._genes

    def set_fitness(self, fit):
        self._fitness = fit

    def get_fitness(self):
        return self._fitness

    def __repr__(self):
        return repr(self._genes).replace("array(", "").replace(")", "").replace("\n",
                                                                                "") + "; Fitness: {}; Parents: {}".format(
            self._fitness, self._parents)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return sum(self._genes != other._genes) == 0

    def copy(self):
        return Individual(self._genes[:], self._fitness, self._parents[:] if self._parents is not None else None)

    def __hash__(self):
        return hash(self._id)


class Mutator:
    """Interface for the mutation operators"""

    @abstractmethod
    def __call__(self, individual):
        pass


class UniformMutator(Mutator):
    """Uniform mutation"""

    def __init__(self, gene_probability, max_value):
        """Initializes the mutator

        :gene_probability: The probability of mutation of a single gene
        :max_value: The maximum value for a gene

        """
        Mutator.__init__(self)

        self._gene_probability = gene_probability
        self._max_value = max_value

    def __call__(self, individual):
        mutated_genes = np.random.uniform(0, 1, len(individual._genes)) < self._gene_probability
        gene_values = np.random.randint(0, self._max_value, sum(mutated_genes))
        genes = individual._genes.copy()
        genes[mutated_genes] = gene_values
        new_individual = Individual(genes, parents=individual._parents)
        return new_individual

    def __repr__(self):
        return "UniformMutator({}, {})".format(self._gene_probability, self._max_value)


class GrammaticalEvolutionME(ProcessingElementFactory, metaclass=OptMetaClass):
    """A class that implements grammatical evolution (Ryan et al. 1995) with MAP elites"""

    def __init__(self, **kwargs):
        """
        Initializes the optimizer

        :pop_size: the size of the population
        :mutation: the mutation operator
        :crossover: the crossover operator
        :selection: the selection operator
        :replacement: the replacement operator
        :mut_prob: the mutation probability
        :cx_prob: the crossover probability
        :genotype_length: the length of the genotype
        :max_int: the biggest constant that can be contained in the genotype (so random number in the range [0, max_int] are generated)

        """

        self._grammar = kwargs
        self._map_size = kwargs["map_size"]
        self._map_bound = kwargs["map_bounds"]
        self._init_pop_size = kwargs["init_pop_size"]
        self._batch_pop = kwargs["batch_pop"]
        self._maximize = kwargs["maximize"]
        self._mutation = UniformMutator(0.1, 1024)
        self._genotype_length = kwargs["genotype_length"]
        self._max_int = kwargs.get("max_int", 1024)
        self._max_depth = kwargs["max_depth"]
        self._cond_depth = kwargs.get("cond_depth", 2)
        self._map = dict()
        self._pop = []
        self._individuals = []
        self._logfile = None  # os.path.join(logdir, "grammatical_evolution.log") if logdir is not None else None
        self._init_pop()
        self._old_individuals = []
        self._updated = False  # To detect the first generation

    def _init_pop(self):
        """Initializes the population"""
        pop = []
        for i in range(self._init_pop_size):
            p = self._random_individual()
            pop.append(p)
        return pop

    def _log(self, tag, string):
        if self._logfile is not None:
            with open(self._logfile, "a") as f:
                f.write("[{}] {}\n".format(tag, string))

    def _random_individual(self):
        """ Generates a random individual """
        return Individual(np.random.randint(0, self._max_int + 1, self._genotype_length))

    def ask(self):
        """ Returns the current population """
        self._pop = []
        if len(self._map) > 0:
            archive = [ind for ind in self._map.values()]
            self._pop = [self._mutation(ind) for ind in np.random.choice(archive, self._batch_pop)]
        else:
            self._pop = self._init_pop()
        tree = list()

        for i in range(len(self._pop)):
            ind = None
            while ind is None:
                ind = utils.genotype2phenotype(self._pop[i], self._grammar)
                if ind is None:
                    self._pop[i] = self._mutation(self._pop[i])
            #print(ind)
            tree.append(ind)
        return tree[:]

    def _get_depth(self, node):
        """BFS search"""
        fringe = [(0, node)]
        max_ = 0
        #("interesting")
        while len(fringe) > 0:
            d, n = fringe.pop(0)
            #print(type(node))
            if isinstance(node, Leaf) or \
                    n is None:
                continue

            if d > max_:
                max_ = d
            if not isinstance(n, Leaf):
                fringe.append((d + 1, n.get_left()))
                fringe.append((d + 1, n.get_right()))
        return max_

    def _get_descriptor(self, ind, entropy):
        pheno_ind = utils.genotype2phenotype(ind, self._grammar)._root
        depth = self._get_depth(pheno_ind)
        return depth, entropy#self._get_cond_depth(pheno_ind)

    def _add_to_map(self, ind, fitness, data):
        desc = data #self._get_descriptor(ind, entropy)

        thr = [abs((max(self._map_bound[i]) - min(self._map_bound[i])) / self._map_size[i]) for i in
              range(len(self._map_size))]

        desc = [int((desc[i] - min(self._map_bound[i])) / thr[i]) for i in range(len(self._map_size))]

        for i in range(len(self._map_size)):
            if desc[i] < 0:
                desc[i] = 0
            elif desc[i] >= self._map_size[i]:
                desc[i] = self._map_size[i] - 1
        desc = tuple(desc)
        if desc in self._map:
            ind_old = self._map[desc]
            if self._maximize:
                if ind_old.get_fitness() < fitness:
                    ind.set_fitness(fitness)
                    self._map[desc] = ind
            else:
                if ind_old.get_fitness() > fitness:
                    ind.set_fitness(fitness)
                    self._map[desc] = ind
        else:
            ind.set_fitness(fitness)
            self._map[desc] = ind

    def tell(self, fitnesses, data=None):
        """
        Assigns the fitness for each individual

        :fitnesses: a list of numbers (the higher the better) associated (by index) to the individuals
        """
        if data is None:
            for p in zip(self._pop, fitnesses):
                self._add_to_map(p[0], p[1])
        else:
            for p in zip(self._pop, fitnesses, data):
                self._add_to_map(p[0],p[1],p[2])

    def get_all_pop(self):
        ind = list()
        for k in self._map.keys():
            ind.append((utils.genotype2phenotype(self._map[k], self._grammar), self._map[k].get_fitness()))
        return zip(self._map.keys(), ind[:])
