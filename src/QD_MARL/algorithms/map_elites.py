#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    algorithms.map_elites
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Implementation of map elites with genetic programming

    :copyright: (c) 2022 by Andrea Ferigo.
    :license: MIT, see LICENSE for more details.
"""
import abc
import numpy as np
from copy import deepcopy
from .common import OptMetaClass
from decisiontrees import Leaf, Condition
from operator import gt, lt, add, sub, mul
from processing_element import ProcessingElementFactory, PEFMetaClass


def safediv(a, b):
    if b == 0:
        return 0
    return a / b


class GPExpr:
    @abc.abstractmethod
    def get_output(self, input_):
        pass


class GPVar(GPExpr):
    """A variable"""

    def __init__(self, index):
        GPExpr.__init__(self)

        self._index = index

    def get_output(self, input_):
        return input_[self._index]

    def __repr__(self):
        return f"input_[{self._index}]"

    def __str__(self):
        return repr(self)


class GPArithNode(GPExpr):
    def __init__(self, op, left, right):
        GPExpr.__init__(self)

        self._op = op
        self._left = left
        self._right = right

    def get_output(self, input_):
        l = self._left.get_output(input_)
        r = self._right.get_output(input_)
        return self._op(l, r)

    def __repr__(self):
        return f"{self._op.__name__}({self._left}, {self._right})"

    def __str__(self):
        return repr(self)

    def get_left(self):
        return self._left

    def set_left(self, value):
        self._left = value

    def get_right(self):
        return self._right

    def set_right(self, value):
        self._right = value


class GPConst(GPExpr):
    def __init__(self, value):
        GPExpr.__init__(self)

        self._value = value

    def get_output(self, input_):
        return self._value

    def __repr__(self):
        return f"{self._value}"

    def __str__(self):
        return repr(self)


class GPNodeCondition:
    """
    A condition
    """

    def __init__(self, operator, left, right):
        """
        Initializes the node
        """
        self._operator = operator
        self._left = left
        self._right = right

    def get_output(self, input_):
        l = self._left.get_output(input_)
        r = self._right.get_output(input_)

        return self._operator(l, r)

    def __repr__(self):
        return f"{self._operator.__name__}({self._left}, {self._right})"

    def __str__(self):
        return repr(self)

    def get_left(self):
        return self._left

    def set_left(self, value):
        self._left = value

    def get_right(self):
        return self._right

    def set_right(self, value):
        self._right = value

    def get_then(self):
        return self._then

    def set_then(self, value):
        self._then = value

    def get_else(self):
        return self._else

    def set_else(self, value):
        self._else = value

    def empty_buffers(self):
        self._then.empty_buffers()
        self._else.empty_buffers()


class GPNodeIf(Condition):
    def __init__(self, condition, then, else_):
        self._condition = condition
        self._then = then
        self._else = else_

    def get_trainable_parameters(self):
        """
        Returns a list of parameters with their type
        (input_index, int or float) as a string.
        """
        return None

    def set_params_from_list(self, params):
        """
        Sets its parameters according to the parameters specified by
        the input list.

        :params: A list of params (int or float)
        """
        return None

    def get_output(self, input_):
        """
        Computes the output associated to its inputs (i.e. computes
        the path of the input vector (or vectors) in the tree and returns
        the decision associated to it).

        :input_: A 1D numpy array
        :returns: A 1D numpy array
        """
        if self._condition.get_output(input_):
            return self._then.get_output(input_)
        else:
            return self._else.get_output(input_)

    def empty_buffers(self):
        self._then.empty_buffers()
        self._else.empty_buffers()

    def copy(self):
        """
        Returns a copy of itself
        """
        new = deepcopy(self)
        return new

    def __repr__(self):
        return f"{self._condition}"

    def __str__(self):
        return repr(self)

    def get_then(self):
        return self._then

    def set_then(self, value):
        self._then = value

    def get_else(self):
        return self._else

    def set_else(self, value):
        self._else = value

    def get_left(self):
        return self._then

    def set_left(self, value):
        self._then = value

    def get_right(self):
        return self._else

    def set_right(self, value):
        self._else = value


class MapElites(ProcessingElementFactory, metaclass=OptMetaClass):

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
        self._map = dict()
        self._pop = []

    def _random_var(self):
        index = np.random.randint(0, self._bounds["input_index"]["max"])
        return GPVar(index)

    def _random_const(self):
        index = np.random.uniform(self._bounds["float"]["min"], self._bounds["float"]["max"])
        return GPConst(index)

    def _random_expr(self, depth=0):
        if depth < self._cond_depth - 1:
            type_ = np.random.randint(0, 3)
        else:
            type_ = np.random.randint(0, 2)

        if type_ == 0:
            return self._random_var()
        elif type_ == 1:
            return self._random_const()
        else:
            l = self._random_expr(depth + 1)
            r = self._random_expr(depth + 1)
            op = np.random.choice([add, sub, mul, safediv])
            return GPArithNode(op, l, r)

    def _random_condition(self):
        left = self._random_expr()
        right = self._random_expr()
        while isinstance(left, GPConst) and isinstance(right, GPConst):
            left = self._random_expr()
            right = self._random_expr()

        op = np.random.choice([gt, lt])

        return GPNodeIf(GPNodeCondition(op, left, right), None, None)

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
        """BFS search"""
        fringe = [(0, node)]
        max_ = 0
        while len(fringe) > 0:
            d, n = fringe.pop(0)
            if isinstance(node, Leaf) or \
                    isinstance(node, GPNodeCondition) or \
                    isinstance(node, GPExpr) or \
                    n is None:
                continue

            if d > max_:
                max_ = d

            if not isinstance(n, Leaf):
                fringe.append((d + 1, n._then))
                fringe.append((d + 1, n._else))
        return max_

    def _reduce_expr_len(self, expr):
        fringe = [(0, expr)]

        max_ = 0
        while len(fringe) > 0:
            d, cur = fringe.pop(0)
            if isinstance(cur, GPArithNode):
                if d + 1 > self._cond_depth:
                    cur.set_left(self._random_expr(d + 1))
                    cur.set_right(self._random_expr(d + 1))
                else:
                    fringe.append((d + 1, cur.get_left()))
                    fringe.append((d + 1, cur.get_right()))
                #print(d)
        return expr

    def _count_expr_len(self, expr):
        fringe = [(0, expr)]

        max_ = 0
        while len(fringe) > 0:
            d, cur = fringe.pop(0)
            if isinstance(cur, GPArithNode):
                fringe.append((d + 1, cur.get_left()))
                fringe.append((d + 1, cur.get_right()))
            if d > max_:
                max_=d
        return max_


    def _get_cond_depth(self, root):
        """BFS search"""

        fringe = [root]
        max_ = 0
        cc = 1
        while len(fringe) > 0:
            cur = fringe.pop(0)
            cc += 1
            if isinstance(cur, GPNodeIf):
                cond = cur._condition
                a = self._count_expr_len(cond.get_left())
                b = self._count_expr_len(cond.get_right())
                d = max(a,b )
                max_ = max(d, max_)
                fringe.append(cur.get_then())
                fringe.append(cur.get_else())
        return max_

    def _limit_cond_depth(self, root):
        """
        Limits the depth of the tree
        """
        fringe = [root]
        while len(fringe) > 0:
            cur = fringe.pop(0)

            if isinstance(cur, GPNodeIf):
                cond = cur._condition

                cond.set_left(self._reduce_expr_len(cond.get_left()))
                cond.set_right(self._reduce_expr_len(cond.get_right()))

                fringe.append(cur.get_then())
                fringe.append(cur.get_else())
        return root

    def _limit_depth(self, root):
        """
        Limits the depth of the tree
        """
        fringe = [(0, root)]

        while len(fringe) > 0:
            d, cur = fringe.pop(0)

            if isinstance(cur, GPNodeIf):
                if d + 1 == self._max_depth:
                    cur.set_then(self._random_leaf())
                    cur.set_else(self._random_leaf())
                fringe.append((d + 1, cur.get_left()))
                fringe.append((d + 1, cur.get_right()))
        return root

    def _get_descriptor(self, ind):
        return self._get_depth(ind), self._get_cond_depth(ind)

    def get_all_pop(self):
        return self._map.items()

    def _init_pop(self):
        pop = []
        grow = self._init_pop_size

        for i in range(grow):
            root = self._get_random_leaf_or_condition()
            fringe = [root]

            while len(fringe) > 0:
                node = fringe.pop(0)

                if isinstance(node, Leaf):
                    continue

                if self._get_depth(root) < self._max_depth - 1:
                    left = self._get_random_leaf_or_condition()
                    right = self._get_random_leaf_or_condition()
                else:
                    left = self._random_leaf()
                    right = self._random_leaf()

                node.set_then(left)
                node.set_else(right)

                fringe.append(left)
                fringe.append(right)

            pop.append(root)

        return pop

    def _mutation(self, p):
        p1 = p.copy()
        cp1 = None

        p1nodes = [(None, None, p1)]

        fringe = [p1]
        while len(fringe) > 0:
            node = fringe.pop(0)

            if not isinstance(node, Leaf):
                fringe.append(node.get_left())
                fringe.append(node.get_right())

                p1nodes.append((node, True, node.get_left()))
                p1nodes.append((node, False, node.get_right()))

        cp1 = np.random.randint(0, len(p1nodes))

        parent = p1nodes[cp1][0]
        old_node = p1nodes[cp1][2]
        if not isinstance(old_node, GPNodeCondition) or \
                not isinstance(old_node, GPExpr):
            new_node = self._get_random_leaf_or_condition()
        else:
            new_node = self._random_expr()

        if not isinstance(new_node, Leaf) and \
                not isinstance(new_node, GPExpr):
            if not isinstance(old_node, Leaf):
                new_node.set_then(old_node.get_left())
                new_node.set_else(old_node.get_right())
            else:
                new_node.set_then(self._random_leaf())
                new_node.set_else(self._random_leaf())

        if p1nodes[cp1][1] is not None:
            if p1nodes[cp1][1]:
                parent.set_then(new_node)
            else:
                parent.set_else(new_node)
        else:
            p1 = new_node
        p1 = self._limit_depth(p1)
        p1 = self._limit_cond_depth(p1)
        return p1


    def _crossover(self, par1, par2):
        p1, p2 = par1.copy(), par2.copy()
        cp1 = None
        cp2 = None

        p1nodes = [(None, None, p1)]

        fringe = [p1]
        while len(fringe) > 0:
            node = fringe.pop(0)

            if not isinstance(node, Leaf):
                fringe.append(node.get_left())
                fringe.append(node.get_right())

                p1nodes.append((node, True, node.get_left()))
                p1nodes.append((node, False, node.get_right()))

        cp1 = np.random.randint(0, len(p1nodes))
        st1 = p1nodes[cp1][2]

        p2nodes = [(None, None, p2)]

        fringe = [p2]
        while len(fringe) > 0:
            node = fringe.pop(0)

            if not isinstance(node, Leaf) and \
               not isinstance(node, GPVar) and \
               not isinstance(node, GPConst):
                fringe.append(node.get_left())
                fringe.append(node.get_right())

                if type(node.get_left()) == type(st1):
                    p2nodes.append((node, True, node.get_left()))
                if type(node.get_right()) == type(st1):
                    p2nodes.append((node, False, node.get_right()))

        cp2 = np.random.randint(0, len(p2nodes))

        st2 = p2nodes[cp2][2]

        if cp1 != 0:
            if p1nodes[cp1][1]:
                p1nodes[cp1][0].set_then(st2)
            else:
                p1nodes[cp1][0].set_else(st2)
        else:
            p1 = st2

        if cp2 != 0:
            if p2nodes[cp2][1]:
                p2nodes[cp2][0].set_then(st1)
            else:
                p2nodes[cp2][0].set_else(st1)
        else:
            p2 = st1
        return p1, p2



    def _add_to_map(self, ind, fitness, data=None):
        desc = self._get_descriptor(ind)
        thr = [abs(max(self._map_bound[i]) - min(self._map_bound[i])) / self._map_size[i] for i in
               range(len(self._map_size))]
        print(desc)
        print(thr)
        desc = [int(desc[i] - min(self._map_bound[i]) / thr[i]) for i in range(len(self._map_size))]
        print(desc)
        print("-----------------")
        for i in range(len(self._map_size)):
            if desc[i] < 0:
                desc[i] = 0
            elif desc[i] >= self._map_size[i]:
                desc[i] = self._map_size[i] - 1
        desc = tuple(desc)
        if desc in self._map:
            ind_old = self._map[desc]
            if self._maximize:
                if ind_old[1] < fitness:
                    self._map[desc] = (ind, fitness)
            else:
                if ind_old[1] > fitness:
                    self._map[desc] = (ind, fitness)
        else:
            self._map[desc] = (ind, fitness)

    def ask(self):
        self._pop = []
        temp = list()
        if len(self._map) > 0:
            archive = [ind[0] for ind in self._map.values()]
            self._pop = [ind for ind in np.random.choice(archive, self._batch_pop)]
            for i in range(0, len(self._pop), 2):
                p1 = self._pop[i]

                if i + 1 < len(self._pop):
                    p2 = self._pop[i + 1]
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
            self._pop = [self._mutation(p) for p in temp]
        else:
            self._pop = self._init_pop()

        return self._pop[:]

    def tell(self, fitnesses, data=None):
        if data is None:
            for p in zip(self._pop, fitnesses):
                self._add_to_map(p[0], p[1])
        else:
            for p in zip(self._pop, fitnesses, data):
                self._add_to_map(p[0], p[1], p[2])
