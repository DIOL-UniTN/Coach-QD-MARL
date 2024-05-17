#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    src.utils
    ~~~~~~~~~

    This module implements utilities

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import os
import pickle
import string

# from memory_profiler import profile
from datetime import datetime

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from decisiontrees import ConditionFactory, QLearningLeafFactory, RLDecisionTree
from joblib import Parallel, delayed
from .print_outputs import print_debugging


def get_logdir_name():
    """
    Returns a name for the dir
    :returns: a name in the format dd-mm-yyyy_:mm:ss_<random string>
    """
    time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S-%f")
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
from decisiontrees import Condition


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
        config["leaves"]["params"], config["leaves"]["decorators"]
    )

    if grammar["root"][next(gene)] == "condition":
        params = cfactory.get_trainable_parameters()
        root = cfactory.create([grammar[p][next(gene)] for p in params])
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
                    newnode = cfactory.create([grammar[p][next(gene)] for p in params])
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
                            # print("INVALIDO")
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

def fitnesses_stats(all_fitnesses, team_fitnesses=None):
    """
    Computes the statistics of the fitnesses

    :all_fitnesses: A list of fitnesses
    :team_fitnesse: A dictionary with the fitnesses of the teams
    :returns: A dictionary with the statistics
    """
    stats = {}
    all_fitnesses = np.array(all_fitnesses)
    valid = all_fitnesses != -100000
    stats["min"] = np.min(all_fitnesses)
    stats["max"] = np.max(all_fitnesses)
    stats['max_index'] = np.argmax(all_fitnesses)
    stats["mean"] = np.mean(all_fitnesses)
    stats["std"] = np.std(all_fitnesses)
    stats["valid"] = np.sum(valid)
    stats["invalid"] = len(all_fitnesses) - stats["valid"]
    if team_fitnesses is not None:
        team_fitnesses = np.array(team_fitnesses)
        valid = team_fitnesses != -100000
        stats["teams"] = {}
        stats["teams"]["min"] = np.min(team_fitnesses)
        stats["teams"]["max"] = np.max(team_fitnesses)
        stats["teams"]['max_index'] = np.argmax(team_fitnesses)
        stats["teams"]["mean"] = np.mean(team_fitnesses)
        stats["teams"]["std"] = np.std(team_fitnesses)
        stats["teams"]["valid"] = np.sum(valid)
        stats["teams"]["invalid"] = len(team_fitnesses) - stats["teams"]["valid"]
    return stats

def remove_previous_files(path):
    for file in os.listdir(path):
        f = os.path.join(path, file)
        os.remove(f)

def plot_log(log_path=None, output_path=None, plot_name= "", file=None, gen=None):
    # the log file si a csv file with the following format:
    # <gen> <Min> <Mean> <Max> <Std>

    pop_file = os.path.join(log_path, file)
    plot_name = pop_file.split("/")[-1].split(".")[0]
    plot_dir = os.path.join(output_path, plot_name)
    os.makedirs(plot_dir, exist_ok=True)
    remove_previous_files(plot_dir)
    plot_path = os.path.join(plot_dir, f"{plot_name}_{gen}.png")
    df = pd.read_csv(pop_file)
    df = df.sort_values(by=["Generation"])
    figure, ax = plt.subplots()
    ax.plot(df["Generation"].to_list(), df["Min"].to_list(), label="Min")
    ax.errorbar(
        df["Generation"].to_list(),
        df["Mean"].to_list(),
        yerr=df["Std"].to_list(),
        label="Mean",
        marker="o",
    )
    ax.plot(df["Generation"].to_list(), df["Max"].to_list(), label="Max")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness over generations")
    ax.legend()
    ax.grid(True)
    plt.savefig(plot_path)
    plt.close(fig=figure)
    pass

def plot_actions(actions, pid, config):
    gen = config["generation"]
    actions_path = os.path.join(config["log_path"], "actions_plt", str(gen), str(pid))
    os.makedirs(
        actions_path, exist_ok=True
    )

    for agent in actions:        
        # heatmap
        if 'blue" in agent':
            action_matrix = np.zeros((len(actions[agent]), 21))
            count = 0
            for a in actions[agent]:
                action_matrix[count, a] += 1
                count += 1

            action_matrix = action_matrix.T
            plt.title(f"Heatmap of actions during generation {gen} for {agent} on pid {pid}")
            sns.heatmap(action_matrix, cmap="YlOrRd", cbar=False)
            plt.xlabel("Cycles")
            plt.ylabel("Actions")
            path = (
                actions_path
                + f"/heatmap_actions_{agent}.png"
            )
            plt.savefig(path)
            plt.close()
            x = np.arange(21)
            y = np.sum(action_matrix, axis=1)

            plt.title(f"Count of actions during generation {gen} for {agent}")
            
            path = (
                actions_path
                + f"/log_actions_{agent}.png"
            )
            ax = sns.barplot(x=x, y=y, hue=x, dodge=False, palette='husl', legend=False)
            ax.set(xlabel="Actions", ylabel="Count")
            ax.bar_label(ax.containers[0])
            plt.savefig(path)
            plt.close()
        del action_matrix
        

if __name__ == "__main__":
    log_path = "logs/qd-marl/magent_battlefield/THIS"
    log_file = "logs/qd-marl/magent_battlefield/THIS/_all_sel_log.csv"

    plot_log(log_path, log_file)
    

