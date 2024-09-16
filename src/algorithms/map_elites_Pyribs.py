#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
from cmath import inf
from copy import deepcopy
from operator import add, gt, lt, mul, sub
import pickle
import matplotlib.pyplot as plt
import numpy as np
from decisiontrees import Node, Leaf, OrthogonalCondition

from ribs.archives._archive_data_frame import ArchiveDataFrame
from ribs.archives._cvt_archive import CVTArchive
from ribs.archives._grid_archive import GridArchive
from ribs.archives._sliding_boundaries_archive import SlidingBoundariesArchive
from ribs.visualize import (
    cvt_archive_heatmap,
    grid_archive_heatmap,
    sliding_boundaries_archive_heatmap,
)
from util_processing_elements.processing_element import (
    PEFMetaClass,
    ProcessingElementFactory,
)

from .common import OptMetaClass
from .individuals import *
from .genetic_programming import *


# To import the injected individual

class MapElites_Pyribs(ProcessingElementFactory, metaclass=OptMetaClass):
    def __init__(self, **kwargs):
        """
        Initializes the algorithm

        :map_size: The size of the map
        :map_bounds: List of bounds
        :init_pop_size: number of initial solutions
        :maximize: Boolean indicating if is a maximization problem
        :batch_pop: Number of population generated for iteration
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
        self._log_path = kwargs["log_path"]
        self._map_size = kwargs["map_size"]
        self._map_bound = kwargs["map_bounds"]
        self._cx_prob = kwargs["cx_prob"] if "cx_prob" in kwargs else 0
        self._mut_prob = kwargs["mut_prob"] if "mut_prob" in kwargs else 0
        self._init_pop_size = kwargs["init_pop_size"]
        self._batch_pop = kwargs["batch_pop"]
        self._maximize = kwargs["maximize"]
        if not len(self._map_bound) == len(self._map_size):
            raise Exception("number of bound must match number of dimension")

        self._c_factory = kwargs["c_factory"]
        self._l_factory = kwargs["l_factory"]
        self._bounds = kwargs["bounds"]
        self._max_depth = kwargs["max_depth"]
        self._cond_depth = kwargs.get("cond_depth", 2)
        self._pop = []
        self._injected_individual = self._get_tree(kwargs["injected_individual_path"]).get_root() if "injected_individual_path" in kwargs else None
        self._archive_type = kwargs["archive"]
        self._bins = kwargs["bins"] if "bins" in kwargs else 1
        self._bins_sliding = kwargs["sliding_bins"]if "sliding_bins" in kwargs else [0,1]
        self._solution_dim = kwargs["solution_dim"]
        self._extra_fields = {'tree': ((), object)}

        if self._archive_type == "CVT":
            self._archive = CVTArchive(self._bins, self._map_bound)
        elif self._archive_type == "Grid":
            self._archive = GridArchive(
                solution_dim=self._solution_dim,
                dims=self._map_size,
                ranges=self._map_bound,
                extra_fields=self._extra_fields,
            )
        elif self._archive_type == "SlidingBoundaries":
            self._archive = SlidingBoundariesArchive(
                self._bins_sliding, self._map_bound
            )
        else:
            raise Exception("archive not valid")

        # self._archive.initialize(1) # one dimension (counter)
        self._vmin = None
        self._vmax = None
        self._counter = 1  # number inserted in sol field of the archive
        self._gen_number = 1
        self._max_fitness = -inf
        self._selection_type = self.set_selection_type(kwargs["selection_type"])
        
        self._gp = GeneticProgramming(**kwargs)
        

    def _get_tree(self, log_file):
        tree = None
        if log_file is not None:
            with open(log_file, "rb") as f:
                tree = pickle.load(f)
        return tree
    
    def _get_entropy_bin(self, entropy):
        bin_size = (self._bounds["entropy"]["max"] - self._bounds["entropy"]["min"]) / self._max_depth
        bins = np.linspace(self._bounds["entropy"]["min"], self._bounds["entropy"]["max"], self._max_depth + 1)
        if entropy < self._bounds["entropy"]["min"]:
            return 0
        for i in range(len(bins) - 1):
            if entropy >= bins[i] and entropy < bins[i + 1]:
                return i
        return self._max_depth - 1
    
    def _get_entropy(self, visits):
        # data are the total visits made by the root
        if visits is None:
            return 0
        freq = [v / sum(visits) for v in visits]
        entropy = -sum([f * np.emath.logn(self._bounds["action"]["max"] ,f) for f in freq if f != 0])
        return entropy
    
    def _get_root_entropy(self, root):
        fringe = [root]
        entropy = 0
        total_visits = np.zeros(self._bounds["action"]["max"])
        while len(fringe) > 0:
            cur = fringe.pop(0)
            if isinstance(cur, Condition):
                fringe.append(cur.get_left())
                fringe.append(cur.get_right())
            elif isinstance(cur, Leaf):
                visits = cur.get_visits()
                if visits is not None:
                    cur.set_visits(np.zeros(self._bounds["action"]["max"]))
                total_visits = np.add(total_visits, visits)
            else:
                return 0
        entropy = self._get_entropy(total_visits)
        bin = self._get_entropy_bin(entropy)
        return bin

    def _get_descriptor(self, ind):
        
        return self._gp._get_depth(ind), self._get_root_entropy(ind)
    
    def get_all_pop(self):
        df = self._archive.as_pandas(include_metadata=True)
        dict_to_return = dict()
        for elite in df.iterelites():
            dict_to_return[(int(elite[2][0]),int(elite[2][1]))] = (elite[4]._genes,elite[1])
        return dict_to_return.items()
    
    def set_selection_type(self, selection_type="random"):
        return selection_type

    def set_pop_selection(self, coach_index=None):
        selected_pop = []
        if self._selection_type == "random":
            for _ in range(self._batch_pop):
                elites = self._archive.sample_elites(1)
                selected_pop += [IndividualGP(elites["tree"][0])]
        elif self._selection_type == "best":
            data = self._archive.data()
            objective = np.array(data["objective"])
            rank = np.argsort(objective)[::-1]
            j = 0
            for i in range(self._batch_pop):
                if j >= len(rank):
                    j = 0
                elite_tree = data["tree"][rank[j]]
                selected_pop += [IndividualGP(genes=elite_tree)]
                j += 1
        elif self._selection_type == "coach":
            if coach_index is None or len(coach_index) != self._batch_pop:
                raise Exception("coach index not valid")
            for ind in coach_index:
                elites = self._archive.retrieve_single(ind)
                selected_pop += [IndividualGP(genes= elites[1]['tree'])]
        else:
            raise Exception("selection type not valid")
        return selected_pop

    def ask(self, coach_index=None):

        ask_pop = []
        if self._archive.empty:
            ask_pop = self._gp._init_pop()
            self._pop = ask_pop
        else:
            temp = list()
            ask_pop = self.set_pop_selection(coach_index)
            for i in range(0, len(ask_pop), 2):
                p1 = ask_pop[i]
                if i + 1 < len(ask_pop):
                    p2 = ask_pop[i + 1]
                else:
                    p2 = None
                o1, o2 = None, None
                # Crossover
                if p2 is not None:
                    if np.random.uniform() < self._cx_prob:
                        o1, o2 = self._gp._crossover(p1, p2)
                        temp.append(o1)
                        temp.append(o2)
                    else:
                        temp.append(p1)
                        temp.append(p2)
                else:
                    temp.append(p1)
            
            ask_pop = [self._gp._mutation(p) for p in temp]
            self._pop += ask_pop
        return [p._genes for p in ask_pop]

    def tell(self, fitnesses, data=None):
        '''
        Tell the algorithm the fitness of the individuals
        :param fitnesses: list of fitnesses for each individual
        :param data: list of data for each individual, it is used to compute the descriptor entropy, it is the list of the created trees
        '''
        
        if data is None:
            data = [None for _ in range(len(fitnesses))]

        solutions = []
        objectives = []
        measures = []
        trees_ = []
        for p in zip(self._pop, fitnesses, data):
            if p[2] is None:
                tree = p[0]._genes
            else:
                tree = p[2]
            desc = self._get_descriptor(tree)
            p[0]._fitness = p[1]
            thr = [
                abs(
                    (max(self._map_bound[i]) - min(self._map_bound[i]))
                    / self._map_size[i]
                )
                for i in range(len(self._map_size))
            ]
            desc = [
                int((desc[i] - min(self._map_bound[i])) / thr[i])
                for i in range(len(self._map_size))
            ]
            for i in range(len(self._map_size)):
                if desc[i] < 0:
                    desc[i] = 0
                elif desc[i] >= self._map_size[i]:
                    desc[i] = self._map_size[i] - 1
            desc = tuple(desc)
            solutions.append(desc)
            objectives.append(p[0]._fitness)
            measures.append(desc)
            trees_.append(tree)
            self._counter += 1
        trees = {"tree": trees_}
        status, value = self._archive.add(solutions, objectives, measures, **trees)
            
        self._pop = []

        # Visualize archives
        if max(fitnesses) > self._max_fitness:
            self._max_fitness = max(fitnesses)
            print(
                "New best at generation: ",
                self._gen_number - 1,
                " fitness: ",
                max(fitnesses),
            )
        self._gen_number += 1
        

    def plot_archive(self, gen, vmin=None, vmax=None):
        if vmin is not None or vmax is not None:
            self._vmin = vmin
            self._vmax = vmax
        plt.figure(figsize=(8, 6))
        if self._archive_type == "CVT":
            cvt_archive_heatmap(self._archive, vmin=self._vmin, vmax=self._vmax)
        elif self._archive_type == "Grid":
            grid_archive_heatmap(self._archive, vmin=-20, vmax=30)
        elif self._archive_type == "SlidingBoundaries":
            sliding_boundaries_archive_heatmap(self._archive, vmin=self._vmin, vmax=self._vmax)
        else:
            raise Exception("archive not valid")
        if self._log_path is not None:
            plt.ylabel("Entropy Bin")
            plt.xlabel("Depth")
            plt.title(
                "Map Elites Archive Depth at Generation: " + str(gen)
            )
            os.makedirs(os.path.join(self._log_path, "archives_depth"), exist_ok=True)
            saving_path = os.path.join(
                self._log_path,
                "archives_depth/archive_depth_at_gen_" + str(gen) + ".png",
            )
            plt.savefig(saving_path)
            plt.close()