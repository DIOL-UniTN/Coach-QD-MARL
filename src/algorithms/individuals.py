#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
from tkinter import Grid
import numpy as np
import re
import os
import string
from copy import deepcopy
from .common import OptMetaClass
from decisiontrees import Leaf, Condition
from operator import gt, lt, add, sub, mul



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
        if self._op.__name__ == "safediv":
            return f"{self._left} / {self._right}"
        if self._op.__name__ == "mul":
            return f"{self._left} * {self._right}"
        if self._op.__name__ == "sub":
            return f"{self._left} - {self._right}"
        if self._op.__name__ == "add":
            return f"{self._left} + {self._right}"
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
        if self._operator.__name__ == "gt":
            return f"{self._left} > {self._right}"
        if self._operator.__name__ == "lt":
            return f"{self._left} < {self._right}"
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



class Individual():
    """Represents an individual."""

    def __init__(self, fitness=None, parents=None):
        """Initializes a new individual

        :genes: a list of genes
        :fitness: the fitness for the individual. Default: None.

        """
        self._genes = None
        self._fitness = fitness
        self._parents = parents  # A list containing the indices of the parents in the previous population
        self._id = "".join(np.random.choice([*string.ascii_lowercase], 10))

    def get_genes(self):
        return self._genes

    @abc.abstractmethod
    def copy(self):
        pass

    def __hash__(self):
        return hash(self._id)

class IndividualGE(Individual):
    def __init__(self, genes, fitness=None, parents=None):
        super().__init__(fitness, parents)
        self._genes = np.array(genes)
    
    def __repr__(self):
        return repr(self._genes).replace("array(", "").replace(")", "").replace("\n", "") + "; Fitness: {}; Parents: {}".format(self._fitness, self._parents)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return sum(self._genes != other._genes) == 0

    def copy(self):
        return Individual(self._genes[:], self._fitness, self._parents[:] if self._parents is not None else None)

class IndividualGP(Individual):
    def __init__(self, genes, padding=0, fitness=None, parents=None, const=None, const_len=None):
        super().__init__(fitness, parents)
        self._genes = genes
        self._padding = padding
        self._const = const 
        self._const_len = const_len
        if padding != 0:
            self.get_genes_const()
        
    
    def copy(self):
        return IndividualGP(self._genes.copy(), self._padding, self._fitness, self._parents,np.copy(self._const),self._const_len)

    def get_genes_const_nested(self,expr,const_temp):
        fringe = [expr]
        while len(fringe) > 0:
            cur = fringe.pop(0)
            if isinstance(cur, GPArithNode):
                fringe.append(cur.get_left())
                fringe.append(cur.get_right())
            if isinstance(cur,GPConst):
                const_temp.append(cur._value)

    def get_genes_const(self):
        """BFS search"""
        if not isinstance(self._genes,GPNodeIf):
            self._const = np.zeros(self._padding)
            self._const_len = 0
            return
        fringe = [self._genes]
        const_temp = []
        while len(fringe) > 0:
            cur = fringe.pop(0)
            if isinstance(cur, GPNodeIf):
                cond = cur._condition
                self.get_genes_const_nested(cond.get_left(),const_temp)
                self.get_genes_const_nested(cond.get_right(),const_temp)
                fringe.append(cur.get_then())
                fringe.append(cur.get_else())
        num = self._padding - len(const_temp)
        self._const_len = len(const_temp)
        self._const = np.pad(np.array(const_temp),(0,num))

    def genes_to_const_nested(self,expr,i):
        fringe = [expr]
        while len(fringe) > 0:
            cur = fringe.pop(0)
            if isinstance(cur, GPArithNode):
                fringe.append(cur.get_left())
                fringe.append(cur.get_right())
            if isinstance(cur,GPConst):
                if i > len(self._const) -1:
                    print(i)
                cur._value = self._const[i]
                i += 1
        return i

    def genes_to_const(self):
        """BFS search"""
        if not isinstance(self._genes,GPNodeIf):
            return
        fringe = [self._genes]
        i=0
        while len(fringe) > 0:
            cur = fringe.pop(0)
            if isinstance(cur, GPNodeIf):
                cond = cur._condition
                i = self.genes_to_const_nested(cond.get_left(),i)
                i = self.genes_to_const_nested(cond.get_right(),i)
                fringe.append(cur.get_then())
                fringe.append(cur.get_else())
