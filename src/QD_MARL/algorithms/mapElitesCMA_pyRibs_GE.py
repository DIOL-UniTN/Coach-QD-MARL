#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
from tkinter import Grid
import numpy as np
from copy import deepcopy
from .common import OptMetaClass
from decisiontrees import Leaf, Condition
from operator import gt, lt, add, sub, mul
from processing_element import ProcessingElementFactory, PEFMetaClass
from ribs.archives._cvt_archive import CVTArchive
from ribs.archives._grid_archive import GridArchive
from ribs.archives._sliding_boundaries_archive import SlidingBoundariesArchive
from ribs.archives._archive_data_frame import ArchiveDataFrame
from ribs.visualize import cvt_archive_heatmap
from ribs.visualize import grid_archive_heatmap
from ribs.visualize import sliding_boundaries_archive_heatmap
from ribs.archives import AddStatus
from ribs.emitters.opt import CMAEvolutionStrategy
import matplotlib.pyplot as plt
from .individuals import *
import os
import time

class EmitterCMA:
    def __init__(self,archive,sigma0,padding,selection_rule="filter",restart_rule="no_improvement",weight_rule="truncation",bounds=None,batch_size=None,seed=None):
        self._rng = np.random.default_rng(seed)
        self._id = "".join(np.random.choice([*string.ascii_lowercase], 10))
        self._batch_size = batch_size
        self._archive = archive
        self._sigma0 = sigma0
        self._weight_rule = weight_rule
        if selection_rule not in ["mu", "filter"]:
            raise ValueError(f"Invalid selection_rule {selection_rule}")
        self._selection_rule = selection_rule
        self._opt_seed = seed
        if restart_rule not in ["basic", "no_improvement"]:
            raise ValueError(f"Invalid restart_rule {restart_rule}")
        self._restart_rule = restart_rule
        self._bounds = bounds
        self._solution_dim = padding

       
    def initialize(self):
        self.x0 = self._archive.get_random_elite()[4]
        self.opt = CMAEvolutionStrategy(self._sigma0, self._batch_size, self._solution_dim,
                                        self._weight_rule, self._opt_seed,
                                        self._archive.dtype)
        self.opt.reset(self.x0._const)
        self._num_parents = (self.opt.batch_size //
                             2 if self._selection_rule == "mu" else None)
        self._batch_size = self.opt.batch_size
        self._restarts = False  # Currently not exposed publicly.
        self._lower_bounds = np.full(self._solution_dim, self._bounds[0], dtype=self._archive.dtype)
        self._upper_bounds = np.full(self._solution_dim, self._bounds[1], dtype=self._archive.dtype)

    def sigma0(self):
        return self._sigma0

    def batch_size(self):
        return self._batch_size

    def ask(self):
        if self._restarts:
            self._restarts = False
            self.x0 = self._archive.get_random_elite()[4]
            self.opt.reset(self.x0._const)
        evolved = self.opt.ask(self._lower_bounds, self._upper_bounds)
        tree_out = []
        for i in evolved:
            temp = self.x0.copy()
            temp._const = i
            temp.genes_to_const()
            tree_out.append(temp)
        return tree_out

    def _check_restart(self, num_parents):
        if self._restart_rule == "no_improvement":
            return num_parents == 0
        return False

    def tell(self, solutions, objective_values, behavior_values, metadata):
        ranking_data = []
        new_sols = 0
        for i, (sol, obj, beh, meta) in enumerate(
                zip(solutions, objective_values, behavior_values, metadata)):
            status, value = self._archive.add(sol, obj, beh, meta)
            ranking_data.append((status, value, i))
            if status in (AddStatus.NEW, AddStatus.IMPROVE_EXISTING):
                new_sols += 1
        # New solutions sort ahead of improved ones, which sort ahead of ones
        # that were not added.
        ranking_data.sort(reverse=True)
        indices = [d[2] for d in ranking_data]

        num_parents = (new_sols if self._selection_rule == "filter" else
                       self._num_parents)
        self.opt.tell(np.array([metadata[i]._const for i in indices]), num_parents)
        # Check for reset.
        if (self.opt.check_stop(np.array([value for status, value, i in ranking_data])) or
                self._check_restart(new_sols)):
            self._restarts = True
            return True
        return False

class MapElitesCMA_pyRibs(ProcessingElementFactory, metaclass=OptMetaClass):
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
        self._seed = kwargs["seed"]
        self._map_size = kwargs["map_size"]
        self._map_bound = kwargs["map_bounds"]
        self._cx_prob = kwargs["cx_prob"] if "cx_prob" in kwargs else 0
        self._init_pop_size = kwargs["init_pop_size"]
        self._batch_pop = kwargs["batch_pop"]
        self._maximize = kwargs["maximize"]
        if not len(self._map_bound) == len(self._map_size):
            raise Exception("number of bound must match number of dimension")
        self._bounds = kwargs["bounds"]
        self._pop = []
        self._archive_type = kwargs["archive"]
        self._bins = kwargs["bins"]
        self._bins_sliding = kwargs["sliding_bins"]
        self._num_emitters = kwargs["emitters"]
        self._generations = kwargs["generations"]
        self._logdir = kwargs["logdir"]
        self._sigma0 = kwargs["sigma0"]
        self._padding = pow(2,self._max_depth)*self._cond_depth
        self._restart = [False for _ in range(self._num_emitters)]
        if self._archive_type == "CVT":
            self._archive = CVTArchive(self._bins,self._map_bound)
        elif self._archive_type == "Grid":
            self._archive = GridArchive(self._map_size,self._map_bound)
        elif self._archive_type == "SlidingBoundaries":
            self._archive = SlidingBoundariesArchive(self._bins_sliding,self._map_bound)
        else:
            raise Exception("archive not valid")

        self._archive.initialize(1) # one dimension (counter)
        self._counter = 1 # number inserted in sol field of the archive
        self._gen_number = 1
        self._max_fitness = -250
        self._emitters = [
            EmitterCMA(self._archive,self._sigma0,self._padding,bounds=(self._bounds["float"]["min"]*10,self._bounds["float"]["max"]*10),batch_size=self._batch_pop,seed=self._seed) for _ in range(self._num_emitters)
        ]
        self._id = "".join(np.random.choice([*string.ascii_lowercase], 10))
        self._ge = GrammaticalEvolution(**kwargs["ge_kwargs"])
        #sigma la metto come parametro passato, o 1 o 0.5
        #moltiplico per 10 parametri passati a CMA 
        #le soluzioni venivano scartate e abbiamo deciso di fare così
        #noi decidiamo un intervallo x -x, CMA-ES cercherà nell'intervallo più grande e un sigma più piccolo

    def _get_descriptor(self, ind):
        return self._get_depth(ind), self._get_cond_depth(ind)

    def get_all_pop(self):
        df = self._archive.as_pandas(include_metadata=True)
        dict_to_return = dict()
        for elite in df.iterelites():
            dict_to_return[(int(elite[2][0]),int(elite[2][1]))] = (elite[4]._genes,elite[1])
        return dict_to_return.items()

    def _init_pop(self):
        pop = self._ge.ask()

        return pop

    def _mutation(self, p):
        p1 = self._ge._mutation(p)
        return GE2GP(p1,self._padding)

    def _crossover(self, par1, par2):
        p1, p2 = self._ge._crossover(par1, par2)
        return GE2GP(p1,self._padding), GE2GP(p2,self._padding)

    def ask(self):
        start = time.time()
        self._pop = []
        if self._archive.empty:
            self._pop = self._init_pop()
        else:
            for i, (e) in enumerate (self._emitters):
                if not self._restart[i]:
                    self._pop += e.ask()
                else:
                    #print("Crossover and mutation on emitter ",i)
                    temp = list()
                    pop_temp = [
                        self._archive.get_random_elite()[4] #metadata
                        for _ in range(self._batch_pop)
                    ]
                    for i in range(0, len(pop_temp), 2):
                        p1 = pop_temp[i]
                        if i + 1 < len(pop_temp):
                            p2 = pop_temp[i + 1]
                        else:
                            p2 = None

                        o1, o2 = None, None
                        # Crossover
                        if p2 is not None:
                            if  np.random.uniform() < self._cx_prob:
                                o1, o2 = self._crossover(p1, p2)
                                temp.append(o1)
                                temp.append(o2)
                            else:
                                temp.append(p1)
                                temp.append(p2)
                        else:
                            temp.append(p1)
                    pop_temp = [self._mutation(p) for p in temp]
                    for e in pop_temp:
                        e.get_genes_const()
                    self._pop += pop_temp

        end = time.time()
        return [p._genes for p in self._pop]
    def tell(self,fitnesses, data=None):
        archive_flag = self._archive.empty
        sols, objs, behavs, meta = [], [], [], [] 
        for p in zip(self._pop,fitnesses):
            desc = self._get_descriptor(p[0]._genes)
            p[0]._fitness = p[1]
            thr = [abs((max(self._map_bound[i]) - min(self._map_bound[i])) / self._map_size[i]) for i in
                range(len(self._map_size))]
            desc = [int((desc[i] - min(self._map_bound[i])) / thr[i]) for i in range(len(self._map_size))]
            for i in range(len(self._map_size)):
                if desc[i] < 0:
                    desc[i] = 0
                elif desc[i] >= self._map_size[i]:
                    desc[i] = self._map_size[i] - 1
            desc = tuple(desc)

            if archive_flag:
                self._archive.add(self._counter,p[1],desc,p[0])
            else:
                sols.append(self._counter,)
                objs.append(p[1])
                behavs.append(desc)
                meta.append(p[0])
            self._counter += 1

        for i, (e) in enumerate (self._emitters):
            if archive_flag:
                print("INITIALIZING EMITTER",e._id)
                e.initialize()
            else:
                start = i*self._batch_pop
                end = i*self._batch_pop + self._batch_pop
                if self._restart[i]:
                    for ind in range(start,end):
                        self._archive.add(sols[ind],objs[ind],behavs[ind],meta[ind])
                    self._restart[i] = False
                else:
                    self._restart[i] = e.tell(sols[start:end],objs[start:end],behavs[start:end],meta[start:end])


        #Visualize archives 
        if max(fitnesses) > self._max_fitness:
            self._max_fitness = max(fitnesses)
            print("New best at generation: ",self._gen_number-1, " fitness: ",max(fitnesses))
        if self._gen_number%self._generations == 0 :
            plt.figure(figsize=(8, 6))
            if self._archive_type == "CVT":
                cvt_archive_heatmap(self._archive,vmin=0,vmax=500)
            elif self._archive_type == "Grid":
                grid_archive_heatmap(self._archive,vmin=0,vmax=500)
            elif self._archive_type == "SlidingBoundaries":
                sliding_boundaries_archive_heatmap(self._archive,vmin=0,vmax=500)
            else:
                raise Exception("archive not valid")
            plt.ylabel("Condition Depth")
            plt.xlabel("Depth")
            plt.savefig(self._logdir+"/heatmap.png")
        self._gen_number += 1
