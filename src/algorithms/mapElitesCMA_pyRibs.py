#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import os
import time
from copy import deepcopy
from operator import add, gt, lt, mul, sub
from tkinter import Grid

import matplotlib.pyplot as plt
import numpy as np
from decisiontrees import *
from ribs.archives import AddStatus, GridArchive
from ribs.archives._archive_data_frame import ArchiveDataFrame
from ribs.archives._cvt_archive import CVTArchive
from ribs.archives._sliding_boundaries_archive import SlidingBoundariesArchive
from ribs.emitters.opt import CMAEvolutionStrategy
from ribs.emitters.rankers import ObjectiveRanker
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

class EmitterCMA:
    def __init__(
        self,
        archive,
        sigma0,
        padding,
        selection_rule="filter",
        restart_rule="no_improvement",
        weight_rule="truncation",
        bounds=None,
        batch_size=None,
        seed=None,
    ):
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
        self._ranker = ObjectiveRanker(seed=self._opt_seed)

    def initialize(self):
        self.x0 = self._archive.sample_elites(1)
        self._lower_bounds = np.full(
            self._solution_dim, self._bounds[0], dtype=self._archive.dtype
        )
        self._upper_bounds = np.full(
            self._solution_dim, self._bounds[1], dtype=self._archive.dtype
        )
        self.opt = CMAEvolutionStrategy(
            sigma0=self._sigma0,
            solution_dim=self._solution_dim,
            batch_size=self._batch_size,
            seed=self._opt_seed,
            dtype=self._archive.dtype,
        )
        self.x0 = IndividualGP(self.x0['tree'][0], self._solution_dim, fitness=self.x0['objective'])
        self.x0.get_genes_const()
        self.opt.reset(self.x0._const)
        self._num_parents = (
            self.opt.batch_size // 2 if self._selection_rule == "mu" else None
        )
        self._batch_size = self.opt.batch_size
        self._restarts = False  # Currently not exposed publicly.

    def sigma0(self):
        return self._sigma0

    def batch_size(self):
        return self._batch_size

    def ask(self):
        if self._restarts:
            self._restarts = False
            self.x0 = self._archive.sample_elites(1)
            self.x0 = IndividualGP(self.x0['tree'][0], self._solution_dim, fitness=self.x0['objective'])
            self.opt.reset(self.x0._const)
        evolved = self.opt.ask()
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
        new_sols = 0
        tree = {'tree': [meta for meta in metadata]}
        add_info = self._archive.add(behavior_values, objective_values, behavior_values, **tree)
        for i, (beh, obj) in enumerate(
            zip(behavior_values, objective_values)
        ):
            if add_info['status'][i] in (AddStatus.NEW, AddStatus.IMPROVE_EXISTING):
                new_sols += 1
        # New solutions sort ahead of improved ones, which sort ahead of ones
        # that were not added.
        
        indices = np.argsort(objective_values)
        values = np.array(objective_values)[indices]

        num_parents = (
            new_sols if self._selection_rule == "filter" else self._num_parents
        )
        self.opt.tell(indices, values, num_parents)
        self._ranker.reset(self.opt, self._archive)
        # Check for reset.
        if self.opt.check_stop(
            np.array(values)
        ) or self._check_restart(new_sols):
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
        self._log_path = kwargs["log_path"]
        self._seed = kwargs["seed"]
        self._map_size = kwargs["map_size"]
        self._map_bound = kwargs["map_bounds"]
        self._cx_prob = kwargs["cx_prob"] if "cx_prob" in kwargs else 0
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
        self._archive_type = kwargs["archive"]
        self._num_emitters = kwargs["emitters"]
        self._generations = kwargs["generations"]
        self._logdir = kwargs["logdir"]
        self._sigma0 = kwargs["sigma0"]
        self._padding = pow(2, self._max_depth) * self._cond_depth
        self._restart = [False for _ in range(self._num_emitters)]
        self._solution_dim = kwargs["solution_dim"]
        self._extra_fields = {'tree': ((), object)}
        if self._archive_type == "Grid":
            self._archive = GridArchive(
                solution_dim=self._solution_dim,
                dims=self._map_size,
                ranges=self._map_bound,
                extra_fields=self._extra_fields,
            )
        else:
            raise Exception("archive not valid")
        # self._archive.initialize(self._solution_dim) # pyribs v.0.6.3 don't have inizialization method anymore
        self._vmin = None
        self._vmax = None
        self._counter = 1  # number inserted in sol field of the archive
        self._gen_number = 1
        self._max_fitness = -250
        self._emitters = [
            EmitterCMA(
                self._archive,
                self._sigma0,
                self._padding,
                bounds=(
                    self._bounds["float"]["min"] * 10,
                    self._bounds["float"]["max"] * 10,
                ),
                batch_size=self._batch_pop,
                seed=self._seed,
            )
            for _ in range(self._num_emitters)
        ]
        self._id = "".join(np.random.choice([*string.ascii_lowercase], 10))
        self._selection_type = self.set_selection_type(kwargs["selection_type"])
        # sigma la metto come parametro passato, o 1 o 0.5
        # moltiplico per 10 parametri passati a CMA
        # le soluzioni venivano scartate e abbiamo deciso di fare così
        # noi decidiamo un intervallo x -x, CMA-ES cercherà nell'intervallo più grande e un sigma più piccolo

    def _random_var(self):
        index = np.random.randint(0, self._bounds["input_index"]["max"])
        return index

    def _random_const(self):
        index = np.random.uniform(self._bounds["float"]["min"], self._bounds["float"]["max"])
        return index
    
    def _create_random_condition(self):
        params = [self._random_var(), self._random_const()]
        return self._c_factory.create(params)

    def _next_node(self, depth = 0):
        type_ = np.random.randint(0,3)
        if type_==1 or type_==2 or depth < self._max_depth:
            params = [self._random_var(), self._random_const()]
            cond = self._c_factory.create(params)
            return cond
        else:
            left = self._next_node(depth + 1)
            right = self._next_node(depth + 1)
            params = [self._random_var(), self._random_const()]
            cond = self._c_factory.create(params)
            cond.set_left(left)
            cond.set_right(right)
            return cond
        
    def _next_node(self, depth = 0):
        if depth >= self._max_depth - 1:
            cond = self._create_random_condition()
            return cond
        
        if np.random.uniform() < 0.7:
            if np.random.uniform() < 0.5:
                left = self._next_node(depth + 1)
                right = self._next_node(depth + 1)
                cond = self._create_random_condition()
                cond.set_left(left)
                cond.set_right(right)
                return cond
            else:
                cond = self._create_random_condition()
                return cond
        else:
            return self._random_leaf()


    def _random_condition(self):
        left = self._next_node()
        right = self._next_node()
        cond = self._create_random_condition()
        cond.set_left(left)
        cond.set_right(right)
        return cond
    
    def _random_leaf(self):
        tp = self._l_factory.get_trainable_parameters()

        if len(tp) == 0:
            return self._l_factory.create()
        else:
            params = []

            for param in tp:
                min_ = self._bounds[param]["min"]
                max_ = self._bounds[param]["max"]
                if self._bounds[param]["type"] == "int":
                    params.append(np.random.randint(min_, max_))
                elif self._bounds[param]["type"] == "float":
                    params.append(np.random.uniform(min_, max_))
                else:
                    raise ValueError("Unknown type")

            return self._l_factory.create(*params)

    def _get_random_leaf_or_condition(self):
        if np.random.uniform() < 0.5:
            return self._random_leaf()
        return self._random_condition()

    def _get_depth(self, node):
        fringe = [(0, node)]
        max_ = 0
        while len(fringe) > 0:
            d, cur = fringe.pop(0)
            if isinstance(cur, Condition):
                fringe.append((d + 1, cur.get_left()))
                fringe.append((d + 1, cur.get_right()))
            if d > max_:
                max_ = d
        return max_

    def _limit_depth(self, root):
        """
        Limits the depth of the tree
        """
        fringe = [(0, root)]

        while len(fringe) > 0:
            d, cur = fringe.pop(0)

            if isinstance(cur, Condition):
                if d + 1 == self._max_depth:
                    cur.set_left(self._random_leaf())
                    cur.set_right(self._random_leaf())
                fringe.append((d + 1, cur.get_left()))
                fringe.append((d + 1, cur.get_right()))
        return root
    
    def _set_leaves(self, root):
        fringe = [(0, root)]

        while len(fringe) > 0:
            d, cur = fringe.pop(0)

            if isinstance(cur, Condition):
                if cur.get_left() is None:
                    cur.set_left(self._random_leaf())
                if cur.get_right() is None:
                    cur.set_right(self._random_leaf())
                fringe.append((d + 1, cur.get_left()))
                fringe.append((d + 1, cur.get_right()))
            elif isinstance(cur, Leaf):
                continue
            else:
                print("Error", type(cur))
                cur = self._random_leaf()
        return root
    
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
        
        return self._get_depth(ind), self._get_root_entropy(ind)
    
    def get_all_pop(self):
        df = self._archive.as_pandas(include_metadata=True)
        dict_to_return = dict()
        for elite in df.iterelites():
            dict_to_return[(int(elite[2][0]),int(elite[2][1]))] = (elite[4]._genes,elite[1])
        return dict_to_return.items()

    def _init_pop(self):
        pop = []
        grow = self._init_pop_size

        for i in range(grow):
            root = self._get_random_leaf_or_condition()
            self._set_leaves(root)
            pop.append(IndividualGP(root, self._padding))
        return pop

    def _mutation(self, p):
        p1 = p.copy()._genes
        #print(type(p1))
        cp1 = None

        p1nodes = [(None, None, p1)]

        fringe = [p1]
        while len(fringe) > 0:
            node = fringe.pop(0)

            if not isinstance(node, Leaf):
                if not isinstance(node.get_left(), Leaf):
                    fringe.append(node.get_left())
                    p1nodes.append((node, True, node.get_left()))
                if not isinstance(node.get_right(), Leaf):
                    fringe.append(node.get_right())
                    p1nodes.append((node, False, node.get_right()))

        cp1 = np.random.randint(0, len(p1nodes))

        parent = p1nodes[cp1][0]
        old_node = p1nodes[cp1][2]
        
        depth = self._get_depth(old_node)
        if depth == 0:
            new_node = self._get_random_leaf_or_condition()
            self._set_leaves(new_node)
        else:
            new_node = self._next_node()
            self._set_leaves(new_node)
        if not isinstance(new_node, Leaf):
            if not isinstance(old_node, Leaf):
                new_node.set_left(old_node.get_left())
                new_node.set_right(old_node.get_right())
            else:
                new_node.set_left(self._random_leaf())
                new_node.set_right(self._random_leaf())

        if p1nodes[cp1][1] is not None:
            if p1nodes[cp1][1]:
                parent.set_left(new_node)
            else:
                parent.set_right(new_node)
        else:
            p1 = new_node
        p1 = self._limit_depth(p1)
        p1 = self._set_leaves(p1)
        return IndividualGP(p1, self._padding)


    def _crossover(self, par1, par2):
        p1, p2 = par1.copy()._genes, par2.copy()._genes
        cp1 = None
        cp2 = None

        p1nodes = [(None, None, p1)]

        fringe = [p1]
        while len(fringe) > 0:
            node = fringe.pop(0)

            if not isinstance(node, Leaf):
                if not isinstance(node.get_left(), Leaf):
                    fringe.append(node.get_left())
                    p1nodes.append((node, True, node.get_left()))
                if not isinstance(node.get_right(), Leaf):
                    fringe.append(node.get_right())
                    p1nodes.append((node, False, node.get_right()))

        cp1 = np.random.randint(0, len(p1nodes))
        st1 = p1nodes[cp1][2]

        p2nodes = [(None, None, p2)]

        fringe = [p2]
        while len(fringe) > 0:
            node = fringe.pop(0)

            if not isinstance(node, Leaf):
                if not isinstance(node.get_left(), Leaf):
                    fringe.append(node.get_left())
                    if type(node.get_left()) == type(st1):
                        p2nodes.append((node, True, node.get_left()))
                if not isinstance(node.get_right(), Leaf):
                    fringe.append(node.get_right())
                    if type(node.get_right()) == type(st1):
                        p2nodes.append((node, False, node.get_right()))

        cp2 = np.random.randint(0, len(p2nodes))

        st2 = p2nodes[cp2][2]

        if cp1 != 0:
            if p1nodes[cp1][1]:
                p1nodes[cp1][0].set_left(st2)
            else:
                p1nodes[cp1][0].set_right(st2)
        else:
            p1 = st2

        if cp2 != 0:
            if p2nodes[cp2][1]:
                p2nodes[cp2][0].set_left(st1)
            else:
                p2nodes[cp2][0].set_right(st1)
        else:
            p2 = st1
        p1 = self._limit_depth(p1)
        p2 = self._limit_depth(p2)
        p1 = self._set_leaves(p1)
        p2 = self._set_leaves(p2)
        
        return IndividualGP(p1, self._padding), IndividualGP(p2, self._padding)


    def set_selection_type(self, selection_type="random"):
        return selection_type

    def set_pop_selection(self, coach_index=None):
        selected_pop = []
        if self._selection_type == "random":
            for _ in range(self._batch_pop):
                elites = self._archive.sample_elites(1)
                selected_pop += [IndividualGP(elites["tree"][0], self._padding)]
        elif self._selection_type == "best":
            data = self._archive.data()
            objective = np.array(data["objective"])
            rank = np.argsort(objective)[::-1]
            j = 0
            for i in range(self._batch_pop):
                if j >= len(rank):
                    j = 0
                elite_tree = data["tree"][rank[j]]
                selected_pop += [IndividualGP(elite_tree, self._padding)]
                j += 1
        elif self._selection_type == "coach":
            if coach_index is None or len(coach_index) != self._batch_pop:
                raise Exception("coach index not valid")
            for ind in coach_index:
                elites = self._archive.retrieve_single(ind)
                selected_pop += [IndividualGP(elites[1]['tree'], self._padding)]
        else:
            raise Exception("selection type not valid")
        return selected_pop

    def ask(self, coach_index=None):
        start = time.time()
        ask_pop = []
        if self._archive.empty:
            ask_pop = self._init_pop()
            self._pop = ask_pop
        else:
            for i, (e) in enumerate(self._emitters):
                if not self._restart[i]:
                    ask_pop = e.ask()
                else:
                    temp = list()
                    pop_temp = self.set_pop_selection(coach_index)
                    for i in range(0, len(pop_temp)):
                        p1 = pop_temp[i]
                        if i + 1 < len(pop_temp):
                            p2 = pop_temp[i + 1]
                        else:
                            p2 = None

                        o1, o2 = None, None
                        # Crossover
                        if p2 is not None:
                            if np.random.uniform() < self._cx_prob:
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
                    ask_pop = pop_temp
        end = time.time()
        self._pop += ask_pop
        return [p._genes for p in ask_pop]

    def tell(self, fitnesses, data=None):
        
        archive_flag = self._archive.empty
        sols, objs, behavs, meta = [], [], [], [] 
        if data is None:
            data = [None for _ in range(len(fitnesses))]
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
            sols.append(desc)
            objs.append(p[0]._fitness)
            behavs.append(desc)
            meta.append(tree)
            self._counter += 1
        trees = {"tree": meta}
        if archive_flag:
            status, value = self._archive.add(sols, objs, behavs, **trees)
            
        self._pop = []

        
        for i, (e) in enumerate(self._emitters):
            if archive_flag:
                e.initialize()
            else:
                start = i*self._batch_pop % len(sols)
                end = start + self._batch_pop

                if self._restart[i]:
                    tree = {'tree': meta[start:end]}
                    self._archive.add(behavs[start:end], objs[start:end], behavs[start:end], **tree) #TODO: set it as line 639
                    self._restart[i] = False
                else:
                    self._restart[i] = e.tell(
                        behavs[start:end], objs[start:end], behavs[start:end], meta[start:end]
                    )

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
            grid_archive_heatmap(self._archive, vmin=-20,  vmax=30)
        elif self._archive_type == "SlidingBoundaries":
            sliding_boundaries_archive_heatmap(self._archive, vmin=self._vmin, vmax=self._vmax)
        else:
            raise Exception("archive not valid")
        if self._log_path is not None:
            plt.ylabel("Condition Depth")
            plt.xlabel("Depth")
            plt.title(
                "Map Elites CMA Archive Depth at Generation: " + str(gen)
            )
            os.makedirs(os.path.join(self._log_path, "archives_depth"), exist_ok=True)
            saving_path = os.path.join(
                self._log_path,
                "archives_depth/archive_depth_at_gen_" + str(gen) + ".png",
            )
            plt.savefig(saving_path)
            plt.close()