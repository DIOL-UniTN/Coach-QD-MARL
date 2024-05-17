#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    src.factories
    ~~~~~~~~~~~~~

    This module contains the implementation of a factory for decision
    trees.

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
from algorithms import OptMetaClass
from util_processing_elements.processing_element import ProcessingElementFactory, PEFMetaClass
from decisiontrees import ConditionFactory, QLearningLeafFactory, ConstantLeafFactory, \
        RLDecisionTree
# TODO: Create a MetaClass for leaves' factories <23-11-21, Leonardo Lucio Custode> #


class DecisionTreeFactory(ProcessingElementFactory, metaclass=PEFMetaClass):
    """
    This class implements a factory for decision trees.
    """

    def __init__(self, **kwargs):
        """
        Initializes the factory

        :Optimizer: A dictionary containing at least
            - name: name of the optimizer
            - kwargs: params for the optimizer
        :DecisionTree: A dictionary containing at least
            - gamma: The gamma value for Q-learning. Default: 0
        :ConditionFactory:
            - type: Name for the factory of conditions.
                Must be one of:
                    - orthogonal
                    - differentiable
                    - 2vars
                    - 2varswoffset
                    - oblique
        :LeafFactory: A dictionary containing at least:
            - name: the name of the factory.
                 Supports ConstantLeafFactory and QLearningLeafFactory
            - kwargs: the parameters for the factory
        """
        ProcessingElementFactory.__init__(self)

        c_factory = ConditionFactory(kwargs["ConditionFactory"]["type"],kwargs["ConditionFactory"]["n_inputs"] )
        lf_dict = kwargs["LeafFactory"]
        if lf_dict["class_name"] == "ConstantLeafFactory":
            l_factory = ConstantLeafFactory(lf_dict["kwargs"]["n_actions"])
        else:
            l_factory = QLearningLeafFactory(**lf_dict["kwargs"])

        opt_d = kwargs["Optimizer"]
        opt_d["kwargs"]["c_factory"] = c_factory
        opt_d["kwargs"]["l_factory"] = l_factory
        self._opt = OptMetaClass.get(opt_d["class_name"])(**opt_d["kwargs"])

        self._gamma = kwargs.get("DecisionTree", {"gamma": 0}).get("gamma", 0)

    def _make_tree(self, root):
        if root is None:
            print(root)
        return RLDecisionTree(root, self._gamma) if not isinstance(root, RLDecisionTree) else root

    def ask_pop(self):
        """
        This method returns a whole population of solutions for the factory.
        :returns: A population of solutions.
        """
        pop = self._opt.ask()
        if pop is None:
            print("ehy "+ str(pop))
        pop = map(self._make_tree, pop)
        return pop
    def get_all_pop(self):
        """
        This method returns the whole population for the factory
        :returns: all the stored population
        """
        indexes= list()
        trees = list()
        fitness = list()
        pop = list()
        for k,v  in self._opt.get_all_pop():
            indexes.append(k)
            trees.append(v[0])
            fitness.append(v[1])
        trees = map(self._make_tree, trees)
        i = 0
        for t in trees:
            pop.append((indexes[i],fitness[i],t))
            i+=1
        return pop

    def tell_pop(self, fitnesses, data=None):
        """
        This methods assigns the computed fitness for each individual of the population.
        """
        self._opt.tell(fitnesses, data)
