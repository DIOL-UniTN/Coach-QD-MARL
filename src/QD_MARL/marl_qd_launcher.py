import os
import sys

sys.path.append(".")
import random
import time
from copy import deepcopy
from math import sqrt
from test_environments import *
import numpy as np
import pettingzoo
from pettingzoo.utils import parallel_to_aec, aec_to_parallel
import differentObservations
from utils import *
from algorithms import grammatical_evolution, map_elites, mapElitesCMA_pyRibs, map_elites_Pyribs, individuals
from agents import *
from decisiontrees import (ConditionFactory, QLearningLeafFactory,
                                 RLDecisionTree, DecisionTree)
from magent2.environments import battlefield_v5
from evaluations import *
from decisiontrees.leaves import *
import get_interpretability


def get_map_elite(config):
    """
    Produces a tree for the selected problem by using the Grammatical Evolution

    :config: a dictionary containing all the parameters
    :log_path: a path to the log directory
    """
    # Setup GE
    me_config = config["me"]["kwargs"]
    # Build classes of the operators from the config file
    me_config["c_factory"] = ConditionFactory()
    me_config["l_factory"] = QLearningLeafFactory(
        config["QLearningLeafFactory"]["kwargs"]["leaf_params"], 
        config["QLearningLeafFactory"]["kwargs"]["decorators"]
    )
    
    # me = map_elites.MapElites(**me_config)
    me = map_elites_Pyribs.MapElites_Pyribs(**me_config)
    # me = mapElitesCMA_pyRibs.MapElitesCMA_pyRibs(**me_config)

    return me

def pretrain_tree(tree, rb):
    """
    Pretrains a tree

    :t: A tree
    :rb: The replay buffer
    :returns: The pretrained tree
    """
    if tree is None:
        return None
    for e in rb:
        tree.empty_buffers()
        if len(e) > 0:
            for s, a, r, sp in e:
                tree.force_action(s, a)
                tree.set_reward(r)
            tree.set_reward_end_of_episode()
    return tree
            
def produce_tree(config, log_path=None, extra_log=False, debug=False, manual_policy=False):
    # Setup GE
    ge_config = config["ge"]
    me_config = config["me_config"]
    gram_dir = None
    for op in ["mutation", "crossover", "selection", "replacement"]:
        ge_config[op] = getattr(
            grammatical_evolution, ge_config[op]["type"]
        )(**ge_config[op]["params"])
    # ge[i] -> set_i
    ge = grammatical_evolution.GrammaticalEvolution(**ge_config, logdir=gram_dir)
    # setup log files
    evolution_dir = os.path.join(log_path, "Evolution_dir")
    os.makedirs(evolution_dir , exist_ok=False)
    trees_dir = os.path.join(log_path, "Trees_dir")
    os.makedirs(trees_dir , exist_ok=False)

    # Retrieve the map function from utils
    map_ = utils.get_map(config["training"]["jobs"], debug)
    
    # setup map elite and trees
    me = get_map_elite(me_config)
    init_trees = me._init_pop()
    # Transform the me-trees in RLDecisionTree
    trees = map_(RLDecisionTree, init_trees, config["training"]["gamma"])
    
    # init coach
    # coach_tree = RLDecisionTree(init_trees_[-1], config["coach_training"]["gamma"])
    # coach = CoachAgent("coach", None, coach_tree)
    coach = None
    # Init replay buffer
    replay_buffer = []
    # Pretrain the trees
    trees = [pretrain_tree(t, replay_buffer) for t in trees]
        # Initialize best individual
    best, best_fit, new_best = None, -float("inf"), False
    
    # Setup sets
    
    
    with open(os.path.join(log_path, "sets.log"), "w") as f:
        f.write(f"{config['sets']}\n")

    with open(os.path.join(log_path, "log.txt"), "a") as f:
        f.write(f"Generation Min Mean Max Std\n")

    # Compute the fitnesses
    # We need to return the trees in order to retrieve the
    #   correct values for the leaves when using the
    #   parallelization
    print_debugging("Evaluating initial population")
    fitnesses, trees = init_eval(trees, config, replay_buffer, map_, coach = None)
    
    print_debugging((fitnesses, len(fitnesses), len(trees)))
    
    amax = np.argmax(fitnesses)
    max_ = fitnesses[amax]
    me.tell(fitnesses, trees)
    
    # print_info(f"{'Generation' : <10} {'Min': <10} {'Mean': <10} \
    #   {'Max': <10} {'Std': <10} {'Invalid': <10} {'Best': <10}")
    
    # Check whether the best has to be updated
    # Iterate over the generations
    for i in range(config["training"]["generations"]):
        # Retrieve the current population

        #team_selection(me, config)

        trees = me.ask(random = False, best = True)
        trees = map_(RLDecisionTree, trees, config["training"]["gamma"])
        print_debugging(len(trees))
        # Compute the fitnesses
        # We need to return the trees in order to retrieve the
        #   correct values for the leaves when using the
        #   parallelization
        fitnesses, trees= evaluate(trees, config, replay_buffer, map_)
        print_debugging((fitnesses, len(fitnesses), len(trees)))
        # Check whether the best has to be updated
        amax = np.argmax(fitnesses)
        max_ = fitnesses[amax]
        
        if max_ > best_fit:
            best_fit = max_
            best = trees[amax]
            new_best = True

        # Tell the fitnesses to the GE
        me.tell(fitnesses)

        # Compute stats
        fitnesses = np.array(fitnesses)
        valid = fitnesses != -100000
        min_ = np.min(fitnesses[valid])
        mean = np.mean(fitnesses[valid])
        max_ = np.max(fitnesses[valid])
        std = np.std(fitnesses[valid])
        invalid = np.sum(fitnesses == -100000)
        print_info(f"{'Generation' : <10} {'Min': <10} {'Mean': <10} \
            {'Max': <10} {'Std': <10} {'Invalid': <10} {'Best': <10}")
        print_info(f"{i: <10} {min_: <10.4f} {mean: <10.4f} \
            {max_: <10.4f} {std: <10.4f} {invalid: <10} {best_fit: <10.4f}")

        # Update the log file
        with open(os.path.join(log_path, "log.txt"), "a") as f:
            f.write(f"{i} {min_} {mean} {max_} {std} {invalid}\n")
            if new_best:
                f.write(f"New best: {best}; Fitness: {best_fit}\n")
                with open(join("best_tree.mermaid"), "w") as f:
                    f.write(str(best))
        new_best = False
    
    return trees


if __name__ == "__main__":
    import argparse
    import json
    import shutil
    import yaml
    import utils
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path of the config file to use")
    parser.add_argument("--debug", action="store_true", help="Debug flag")
    parser.add_argument("--log", action="store_true", help="Log flag")
    parser.add_argument("seed", type=int, help="Random seed to use")
    args = parser.parse_args()
    print_info("Launching Quality Diversity MARL")
    print_configs("Environment configurations file: ", args.config)
    if args.debug:
        print_configs("DEBUG MODE")
    
    # Load the config file
    config = json.load(open(args.config))
    
    # Set the random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Setup logging
    logdir_name = utils.get_logdir_name()
    log_path = f"logs/magent_battlefield/{logdir_name}"
    join = lambda x: os.path.join(log_path, x)

    os.makedirs(log_path, exist_ok=False)
    shutil.copy(args.config, join("config.json"))
    with open(join("seed.log"), "w") as f:
        f.write(str(args.seed))
    
    squad = produce_tree(config, log_path, args.log, args.debug)

    import logging
    logging.basicConfig(filename=join("output.log"), level=logging.INFO, format='%(asctime)s %(message)s', filemode='w') 
    logger=logging.getLogger() 
    index = 0
    
    for player in squad:
        print_info(player)
    
    # return interpretability(best, config, log_path, index, logger)
    
    
    