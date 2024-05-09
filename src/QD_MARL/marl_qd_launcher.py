<<<<<<< HEAD
import os
import sys
=======
import importlib
import os
import sys
import gc
>>>>>>> aca3e01 (merged from private repo)

sys.path.append(".")
import random
import time
from copy import deepcopy
from math import sqrt
<<<<<<< HEAD
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
=======

import numpy as np
import pettingzoo
from agents.agents import *
from algorithms import (
    grammatical_evolution,
    individuals,
    map_elites,
    map_elites_Pyribs,
    mapElitesCMA_pyRibs,
)
from decisiontrees import (
    ConditionFactory,
    DecisionTree,
    QLearningLeafFactory,
    RLDecisionTree,
)
from decisiontrees.leaves import *
from magent2.environments import battlefield_v5
from training.evaluations import *
from utils import *

# from memory_profiler import profile
>>>>>>> aca3e01 (merged from private repo)


def get_map_elite(config):
    """
    Produces a tree for the selected problem by using the Grammatical Evolution

    :config: a dictionary containing all the parameters
    :log_path: a path to the log directory
    """
<<<<<<< HEAD
    # Setup GE
=======
    # Setup ME
>>>>>>> aca3e01 (merged from private repo)
    me_config = config["me"]["kwargs"]
    # Build classes of the operators from the config file
    me_config["c_factory"] = ConditionFactory()
    me_config["l_factory"] = QLearningLeafFactory(
<<<<<<< HEAD
        config["QLearningLeafFactory"]["kwargs"]["leaf_params"], 
        config["QLearningLeafFactory"]["kwargs"]["decorators"]
    )
    
    # me = map_elites.MapElites(**me_config)
    me = map_elites_Pyribs.MapElites_Pyribs(**me_config)
    # me = mapElitesCMA_pyRibs.MapElitesCMA_pyRibs(**me_config)

    return me
=======
        config["QLearningLeafFactory"]["kwargs"]["leaf_params"],
        config["QLearningLeafFactory"]["kwargs"]["decorators"],
    )
    if me_config["me_type"] == "Map_elites":
        me = map_elites.MapElites(**me_config)
    elif me_config["me_type"] == "MapElites_pyRibs":
        me = map_elites_Pyribs.MapElites_Pyribs(**me_config)
    elif me_config["me_type"] == "MapElitesCMA_pyRibs":
        me = mapElitesCMA_pyRibs.MapElitesCMA_pyRibs(**me_config)
    print_configs("ME type:", me_config["me_type"])
    if me_config["selection_type"] == "coach":
        coach_config = me_config["coach"]
        coach_config["pop_size"] = me_config["init_pop_size"]
        coach_config["batch_size"] = me_config["batch_pop"]
        coach = CoachAgent(coach_config, me)
    else:
        coach = None
    print_configs("ME selection type:", me_config["selection_type"])
    return me, coach

>>>>>>> aca3e01 (merged from private repo)

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
<<<<<<< HEAD
            
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
=======


def produce_tree(
    config, log_path=None, extra_log=False, debug=False, manual_policy=False
):
    
    # Setup ME
    me_config = config["me_config"]
    me_config["me"]["kwargs"]["log_path"] = log_path    
    
    me, coach = get_map_elite(me_config)
    coach_index = None
    selection_type = me_config["me"]["kwargs"]["selection_type"]
    #setup job manager
    map_ = utils.get_map(config["training"]["jobs"], debug)
    number_of_agents = config["n_agents"]
    number_of_sets = config["n_sets"]
    population_size = me_config["me"]["kwargs"]["init_pop_size"]
    number_of_teams = population_size // number_of_agents

     # setup log files
>>>>>>> aca3e01 (merged from private repo)
    evolution_dir = os.path.join(log_path, "Evolution_dir")
    os.makedirs(evolution_dir , exist_ok=False)
    trees_dir = os.path.join(log_path, "Trees_dir")
    os.makedirs(trees_dir , exist_ok=False)

<<<<<<< HEAD
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

=======
    # setup sets
    if config["sets"] == "random":
        sets = [[] for _ in range(number_of_sets)]
        set_full = [False for _ in range(number_of_sets)]
        agents = [i for i in range(number_of_agents)]

        agents_per_set = number_of_agents // number_of_sets
        surplus = number_of_agents // number_of_sets

        while not all(set_full):
            set_index = random.randint(0, number_of_sets -1)
            if not set_full[set_index]:
                random_index = random.randint(0, len(agents) -1)
                sets[set_index].append(agents[random_index])
                del agents[random_index]
                if len(sets[set_index]) == agents_per_set:
                    set_full[set_index] = True

        if surplus > 0:
            while len(agents) != 0:
                set_index = random.randint(0, number_of_sets -1)
                random_index = random.randint(0, len(agents) -1)
                sets[set_index].append(agents[random_index])
                del agents[random_index]

        for set_ in sets:
            set_.sort()

        config["sets"] = sets
        print_info("Sets: ", config["sets"])

    with open(os.path.join(log_path, "sets.log"), "w") as f:
        f.write(f"{config['sets']}\n")

    # Initialize best individual for each agent
    best = [None for _ in range(number_of_agents)]
    best_fit = [-float("inf") for _ in range(number_of_agents)]
    new_best = [False for _ in range(number_of_agents)]

    for i in range(number_of_agents):
            #print(f"{gen: <10} agent_{i: <4} {agent_min[i]: <10.2f} {agent_mean[i]: <10.2f} {agent_max[i]: <10.2f} {agent_std[i]: <10.2f}")
            with open(os.path.join(evolution_dir, f"agent_{i}.log"), "a") as f:
                f.write(f"Generation,Min,Mean,Max,Std\n")
    for i in range(number_of_sets):
            with open(os.path.join(evolution_dir, f"set_{i}.txt"), "a") as f:
                f.write(f"Generation,Min,Mean,Max,Std\n")
    with open(os.path.join(evolution_dir, f"bests.txt"), "a") as f:
                f.write(f"Generation,Min,Mean,Max,Std\n")
    
    print_info(f"{'Generation' : <10} {'Set': <10} {'Min': <10} {'Mean': <10} {'Max': <10} {'Std': <10}")
    sets_fitness = [ [] for _ in range(number_of_sets)]
    #start training process
    for gen in range(-1,config["training"]["generations"]):
        config["generation"] = gen
        me_pop = []
        #Ask for a new population
        if gen >=0:
            if selection_type == "coach":
                coach_index = coach.get_squad(number_of_teams)
            for i in range(number_of_teams):
                me_pop += me.ask(coach_index[0] if coach_index is not None else None) 
        else:
            me_pop = me.ask(None)  

        trees = [[RLDecisionTree(me_pop[i], config["training"]["gamma"]) for i in range(population_size)] for _ in range(number_of_sets)]
        #Form the groups
        squads = [[trees[j][i] for j in range(number_of_sets)] for i in range(population_size)]
        return_values = map_(evaluate, squads, config)
        
        agents_fitness = [ [] for _ in range(number_of_agents)]
        agents_tree = [ [] for _ in range(number_of_agents)]

        # Store trees and fitnesses
        for values in return_values:
            for i in range(number_of_agents):
                agents_fitness[i].append(values[0][i])
                agents_tree[i].append(values[1][i])

        # Check whether the best, for each agent, has to be updated
        amax = [np.argmax(agents_fitness[i]) for i in range(number_of_agents)]
        max_ = [agents_fitness[i][amax[i]] for i in range(number_of_agents)]

        for i in range(number_of_agents):
            if max_[i] > best_fit[i]:
                best_fit[i] = max_[i]
                best[i] = agents_tree[i][amax[i]]
                new_best[i] = True

                tree_text = f"{best[i]}"
                utils.save_tree(best[i], trees_dir, f"best_agent_{i}")
                with open(os.path.join(trees_dir, f"best_agent_{i}.log"), "w") as f:
                    f.write(tree_text)
        # Calculate fitnesses for each set
        sets_fitness = [ [] for _ in range(number_of_sets)]
        for index, set_ in enumerate(config["sets"]):

            set_agents_fitnesses = []
            for agent in set_:
                set_agents_fitnesses.append(agents_fitness[agent])

            set_agents_fitnesses = np.array(set_agents_fitnesses)

            # Calculate fitness for each individual in the set
            sets_fitness[index] = [getattr(np, config['statistics']['set']['type'])(a=set_agents_fitnesses[:, i], **config['statistics']['set']['params']) for i in range(set_agents_fitnesses.shape[1])]

        sets_fitness = np.array(sets_fitness).flatten()
        me.tell(sets_fitness)
        me.plot_archive(gen)
        # Compute stats for each agent
        agent_min = [np.min(agents_fitness[i]) for i in range(number_of_agents)]
        agent_mean = [np.mean(agents_fitness[i]) for i in range(number_of_agents)]
        agent_max = [np.max(agents_fitness[i]) for i in range(number_of_agents)]
        agent_std = [np.std(agents_fitness[i]) for i in range(number_of_agents)]

        for i in range(number_of_agents):
            #print(f"{gen: <10} agent_{i: <4} {agent_min[i]: <10.2f} {agent_mean[i]: <10.2f} {agent_max[i]: <10.2f} {agent_std[i]: <10.2f}")
            with open(os.path.join(evolution_dir, f"agent_{i}.log"), "a") as f:
                f.write(f"{gen} {agent_min[i]} {agent_mean[i]} {agent_max[i]} {agent_std[i]}\n")

            new_best[i] = False

        # Compute states for each set
        set_min = [np.min(sets_fitness[i]) for i in range(number_of_sets)]
        set_mean = [np.mean(sets_fitness[i]) for i in range(number_of_sets)]
        set_max = [np.max(sets_fitness[i]) for i in range(number_of_sets)]
        set_std = [np.std(sets_fitness[i]) for i in range(number_of_sets)]

        for i in range(number_of_sets):
            print_info(f"{gen: <10} set_{i: <4} {set_min[i]: <10.2f} {set_mean[i]: <10.2f} {set_max[i]: <10.2f} {set_std[i]: <10.2f}")
            with open(os.path.join(evolution_dir, f"set_{i}.txt"), "a") as f:
                f.write(f"{gen},{set_min[i]},{set_mean[i]},{set_max[i]},{set_std[i]}\n")

        # Compute stats for the bests
        best_min = np.min(best_fit)
        best_mean = np.mean(best_fit)
        best_max = np.max(best_fit)
        best_std = np.std(best_fit)

        with open(os.path.join(evolution_dir, f"bests.txt"), "a") as f:
                f.write(f"{gen},{best_min},{best_mean},{best_max},{best_std}\n")
        
        plot_log(evolution_dir, f"bests.txt", gen)
        for i in range(number_of_sets):
            plot_log(evolution_dir, f"set_{i}.txt", gen)
    return best
>>>>>>> aca3e01 (merged from private repo)

if __name__ == "__main__":
    import argparse
    import json
    import shutil
<<<<<<< HEAD
    import yaml
    import utils
=======

    import utils
    import yaml

>>>>>>> aca3e01 (merged from private repo)
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path of the config file to use")
    parser.add_argument("--debug", action="store_true", help="Debug flag")
    parser.add_argument("--log", action="store_true", help="Log flag")
    parser.add_argument("seed", type=int, help="Random seed to use")
    args = parser.parse_args()
    print_info("Launching Quality Diversity MARL")
    print_configs("Environment configurations file: ", args.config)
<<<<<<< HEAD
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

=======

    if args.debug:
        print_configs("DEBUG MODE")

    # Load the config file
    config = json.load(open(args.config))

    # Set the random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Setup logging
    logdir_name = utils.get_logdir_name()
    if config["hpc"]:
        cwd = os.getcwd()
        log_path = f"{cwd}/logs/qd-marl/hpc/with_sets/{config['me_config']['me']['kwargs']['me_type']}/{config['me_config']['me']['kwargs']['selection_type']}/magent_battlefield/{logdir_name}"
    else:
        log_path = f"logs/qd-marl/local/with_sets/{config['me_config']['me']['kwargs']['me_type']}/{config['me_config']['me']['kwargs']['selection_type']}/magent_battlefield/{logdir_name}"
    print_configs("Logs path: ", log_path)
    join = lambda x: os.path.join(log_path, x)
    config["log_path"] = log_path
>>>>>>> aca3e01 (merged from private repo)
    os.makedirs(log_path, exist_ok=False)
    shutil.copy(args.config, join("config.json"))
    with open(join("seed.log"), "w") as f:
        f.write(str(args.seed))
<<<<<<< HEAD
    
    squad = produce_tree(config, log_path, args.log, args.debug)

    import logging
    logging.basicConfig(filename=join("output.log"), level=logging.INFO, format='%(asctime)s %(message)s', filemode='w') 
    logger=logging.getLogger() 
    index = 0
    
    for player in squad:
        print_info(player)
    
    # return interpretability(best, config, log_path, index, logger)
    
    
    
=======

    squad = produce_tree(config, log_path, args.log, args.debug)

    import logging

    logging.basicConfig(
        filename=join("output.log"),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        filemode="w",
    )
    logger = logging.getLogger()
    index = 0

    for player in squad:
        print_info("\n", player)

    # return interpretability(best, config, log_path, index, logger)
>>>>>>> aca3e01 (merged from private repo)
