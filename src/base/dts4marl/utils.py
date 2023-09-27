#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    src.utils
    ~~~~~~~~~

    This module implements utilities

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import string
import numpy as np
import os
import pickle
from datetime import datetime
from joblib import Parallel, delayed
from decisiontreelibrary import RLDecisionTree
from decisiontreelibrary import ConditionFactory, QLearningLeafFactory

def get_logdir_name():
    """
    Returns a name for the dir
    :returns: a name in the format dd-mm-yyyy_:mm:ss_<random string>
    """
    time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    rand_str = "".join(np.random.choice([*string.ascii_lowercase], 8))
    return f"{time}_{rand_str}"


def get_map(n_jobs, debug=False):
    """
    Returns a function pointer that implements a parallel map function

    :n_jobs: The number of workers to spawn
    :debug: a flag that disables multiprocessing
    :returns: A function pointer

    """
    if debug:
        def fcn(function, iterable, config):
            ret_vals = []
            for i in iterable:
                ret_vals.append(function(i, config))
            return ret_vals
    else:
        def fcn(function, iterable, config):
            with Parallel(n_jobs) as p:
                return p(delayed(function)(elem, config) for elem in iterable)
    return fcn


class CircularList(list):

    """
    A list that, when indexed outside its bounds (index i), returns the
    element in position i % len(self)
    """
    def __init__(self, iterable):
        """
        Initializes the list.
        If iterable is a dict, then an arange object is created as the list
        """
        if isinstance(iterable, dict):
            list.__init__(self, np.arange(**iterable))
        else:
            list.__init__(self, iterable)

    def __getitem__(self, index):
        return super().__getitem__(index % len(self))


class Grammar(dict):
    """
    Implements a Grammar, simply a dictionary of circular lists, i.e.
    lists that return an element even if the index required is outside their
    bounds
    """

    def __init__(self, dictionary):
        """
        Initializes the grammar

        :dictionary: A dictionary containing the grammar
        """
        circular_dict = {}
        for k, v in dictionary.items():
            circular_dict[k] = CircularList(v)
        dict.__init__(self, circular_dict)

# PER RIPARAZIONE
from decisiontreelibrary import Condition

def genotype2phenotype(individual, config):
    """
    Converts a genotype in a phenotype

    :individual: An Individual (algorithms.grammatical_evolution)
    :config: A dictionary
    :returns: An instance of RLDecisionTree
    """
    genotype = individual.get_genes()
    gene = iter(genotype)
    grammar = Grammar(config["grammar"])
    cfactory = ConditionFactory(config["conditions"]["type"])
    lfactory = QLearningLeafFactory(
        config["leaves"]["params"],
        config["leaves"]["decorators"]
    )

    if grammar["root"][next(gene)] == "condition":
        params = cfactory.get_trainable_parameters()
        root = cfactory.create(
            [grammar[p][next(gene)] for p in params]
        )
    else:
        root = lfactory.create()
        return RLDecisionTree(root, config["training"]["gamma"])

    fringe = [root]

    try:
        while len(fringe) > 0:
            node = fringe.pop(0)
            for i, n in enumerate(["left", "right"]):
                if grammar["root"][next(gene)] == "condition":
                    params = cfactory.get_trainable_parameters()
                    newnode = cfactory.create(
                        [grammar[p][next(gene)] for p in params]
                    )
                    getattr(node, f"set_{n}")(newnode)
                    fringe.insert(i, newnode)
                else:
                    leaf = lfactory.create()
                    getattr(node, f"set_{n}")(leaf)
    except StopIteration:
        # tree repair
        try:
            fringe = [root]

            while len(fringe) > 0:
                node = fringe.pop(0)
                if isinstance(node, Condition):
                    for i, n in enumerate(["left", "right"]):
                        actual_node = getattr(node, f"get_{n}")()
                        if actual_node is None:
                            #print("INVALIDO")
                            actual_node = lfactory.create()
                            getattr(node, f"set_{n}")(actual_node)
                        fringe.insert(i, actual_node)
        except Exception as e:
            return None

    finally:
        return RLDecisionTree(root, config["training"]["gamma"])


def genotype2str(genotype, config):
    """
    Transforms a genotype in a string given the grammar in config

    :individual: An Individual algorithms.grammatical_evolution
    :config: A dictionary with the "grammar" key
    :returns: A string

    """
    pass

def save_tree(tree, log_dir, name):
    if log_dir is not None:
        assert isinstance(tree, RLDecisionTree), "Object passed is not a RLDecisionTree"
        log_file = os.path.join(log_dir, name + ".pickle")
        with open(log_file, "wb") as f:
            pickle.dump(tree, f)

def get_tree(log_file):
    tree = None
    if log_file is not None:
        with open(log_file, "rb") as f:
            tree = pickle.load(f)
    return tree
