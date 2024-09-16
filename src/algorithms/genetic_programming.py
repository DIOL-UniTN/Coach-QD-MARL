import abc
from cmath import inf
from copy import deepcopy
from operator import add, gt, lt, mul, sub
import pickle
import matplotlib.pyplot as plt
import numpy as np
from decisiontrees import Node, Leaf, OrthogonalCondition

from util_processing_elements.processing_element import (
    PEFMetaClass,
    ProcessingElementFactory,
)

from .common import OptMetaClass
from .individuals import *

class GeneticProgramming():
    def __init__(self, **kwargs) -> None:
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
        self._cx_prob = kwargs["cx_prob"] if "cx_prob" in kwargs else 0
        self._mut_prob = kwargs["mut_prob"] if "mut_prob" in kwargs else 0
        self._init_pop_size = kwargs["init_pop_size"]
        self._batch_pop = kwargs["batch_pop"]
        self._maximize = kwargs["maximize"]

        self._c_factory = kwargs["c_factory"]
        self._l_factory = kwargs["l_factory"]
        self._bounds = kwargs["bounds"]
        self._max_depth = kwargs["max_depth"]
        self._cond_depth = kwargs.get("cond_depth", 2)
        self._pop = []
        self._injected_individual = self._get_tree(kwargs["injected_individual_path"]).get_root() if "injected_individual_path" in kwargs else None
    
    def _get_tree(self, log_file):
        tree = None
        if log_file is not None:
            with open(log_file, "rb") as f:
                tree = pickle.load(f)
        return tree
    
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
            pop.append(IndividualGP(root))
        if self._injected_individual is not None:
            pop.remove(pop[0])
            pop.append(IndividualGP(self._injected_individual))
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
        return IndividualGP(p1)


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
        
        return IndividualGP(p1), IndividualGP(p2)