import importlib
import os
import sys
import gc

sys.path.append(".")
import random
import time
from copy import deepcopy
from math import sqrt

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
from training.evaluations_no_sets import *
from utils import *

# from memory_profiler import profile


def get_map_elite(config):
    """
    Produces a tree for the selected problem by using the Grammatical Evolution

    :config: a dictionary containing all the parameters
    :log_path: a path to the log directory
    """
    # Setup ME
    me_config = config["me"]["kwargs"]
    # Build classes of the operators from the config file
    me_config["c_factory"] = ConditionFactory(config["ConditionFactory"]["type"])
    me_config["l_factory"] = QLearningLeafFactory(
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
    number_of_agents = config["n_agents"]
    number_of_sets = config["n_sets"]
    population_size = me_config["me"]["kwargs"]["init_pop_size"]
    number_of_teams = population_size // number_of_agents
    map_ = utils.get_map(number_of_teams, debug)

     # setup log files
    evolution_dir = os.path.join(log_path, "Evolution_dir")
    os.makedirs(evolution_dir , exist_ok=False)
    team_dir = os.path.join(evolution_dir, "Teams")
    os.makedirs(team_dir , exist_ok=False)
    pop_dir = os.path.join(evolution_dir, "Population")
    os.makedirs(pop_dir , exist_ok=False)
    trees_dir = os.path.join(log_path, "Trees_dir")
    os.makedirs(trees_dir , exist_ok=False)

    
    # Initialize best individual for each agent
    best = [None for _ in range(number_of_agents)]
    best_fit = [-float("inf") for _ in range(number_of_agents)]
    new_best = [False for _ in range(number_of_agents)]

    for i in range(number_of_agents):
            #print(f"{gen: <10} agent_{i: <4} {agent_min[i]: <10.2f} {agent_mean[i]: <10.2f} {agent_max[i]: <10.2f} {agent_std[i]: <10.2f}")
            with open(os.path.join(evolution_dir, f"agent_{i}.log"), "a") as f:
                f.write(f"Generation,Min,Mean,Max,Std\n")
    for i in range(number_of_teams):
            with open(os.path.join(team_dir, f"team_{i}.txt"), "a") as f:
                f.write(f"Generation,Min,Mean,Max,Std\n")
    with open(os.path.join(pop_dir, f"pop.txt"), "a") as f:
                f.write(f"Generation,Min,Mean,Max,Std\n")
    with open(os.path.join(evolution_dir, f"bests.txt"), "a") as f:
                f.write(f"Generation,Min,Mean,Max,Std\n")
    
    print_info(f"{'Generation' : <10} {'Set': <10} {'Min': <10} {'Mean': <10} {'Max': <10} {'Std': <10}")
    sets_fitness = [ [] for _ in range(number_of_sets)]
    #start training process
    for gen in range(-1,config["training"]["generations"]):
        config["generation"] = gen
        me_pop = []
        squads = []
        #Ask for a new population
        if gen == -1:
            me_pop = me.ask(None)
            trees = [RLDecisionTree(me_pop[i], config["training"]["gamma"]) for i in range(len(me_pop))]
            for i in range(number_of_teams):
                squads.append(trees[i*number_of_agents:(i+1)*number_of_agents])
        else:
            if selection_type == "coach":
                coach_index = coach.get_squad(number_of_teams)
            for i in range(number_of_teams):
                me_pop = me.ask(coach_index[i] if selection_type == "coach" else None)
                trees = [RLDecisionTree(me_pop[i], config["training"]["gamma"]) for i in range(len(me_pop))]
                squads.append(trees)
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
        teams_fitness = []
        for i in range(number_of_teams):
            team = []
            for j in range(number_of_agents):
                team.append(agents_fitness[j][i])
            teams_fitness.append((team))
        individual_fitness = np.array(teams_fitness).flatten()
        me.tell(individual_fitness)
        
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
        # Compute stats for the population
        pop_min = np.min(individual_fitness)
        pop_mean = np.mean(individual_fitness)
        pop_max = np.max(individual_fitness)
        pop_std = np.std(individual_fitness)

        if gen ==-1:
             vmin = np.quantile(individual_fitness, 0.7)
             vmax = None
        else:
            vmin = None
            vmax = None

        with open(os.path.join(pop_dir, f"pop.txt"), "a") as f:
                f.write(f"{gen},{pop_min},{pop_mean},{pop_max},{pop_std}\n")
        
        plot_log(pop_dir, f"pop.txt", gen)

        # Compute states for each team

        team_min = [np.min(teams_fitness[i]) for i in range(number_of_teams)]
        team_mean = [np.mean(teams_fitness[i]) for i in range(number_of_teams)]
        team_max = [np.max(teams_fitness[i]) for i in range(number_of_teams)]
        team_std = [np.std(teams_fitness[i]) for i in range(number_of_teams)]

        for i in range(number_of_teams):
            print_info(f"{gen: <10} team_{i: <4} {team_min[i]: <10.2f} {team_mean[i]: <10.2f} {team_max[i]: <10.2f} {team_std[i]: <10.2f}")
            with open(os.path.join(team_dir, f"team_{i}.txt"), "a") as f:
                f.write(f"{gen},{team_min[i]},{team_mean[i]},{team_max[i]},{team_std[i]}\n")
            plot_log(team_dir, f"team_{i}.txt", gen)

        # Compute stats for the bests
        best_min = np.min(best_fit)
        best_mean = np.mean(best_fit)
        best_max = np.max(best_fit)
        best_std = np.std(best_fit)

        with open(os.path.join(evolution_dir, f"bests.txt"), "a") as f:
                f.write(f"{gen},{best_min},{best_mean},{best_max},{best_std}\n")
        
        plot_log(evolution_dir, f"bests.txt", gen)
        me.plot_archive(gen, vmin=vmin, vmax=vmax)
    return best

if __name__ == "__main__":
    import argparse
    import json
    import shutil

    import utils
    import yaml

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
    if config["hpc"]:
        cwd = os.getcwd()
        log_path = f"{cwd}/logs/qd-marl/hpc/no_sets/{config['me_config']['me']['kwargs']['me_type']}/{config['me_config']['me']['kwargs']['selection_type']}/magent_battlefield/{logdir_name}"
    else:
        log_path = f"logs/qd-marl/local/no_sets/{config['me_config']['me']['kwargs']['me_type']}/{config['me_config']['me']['kwargs']['selection_type']}/magent_battlefield/{logdir_name}"
    print_configs("Logs path: ", log_path)
    join = lambda x: os.path.join(log_path, x)
    config["log_path"] = log_path
    os.makedirs(log_path, exist_ok=False)
    shutil.copy(args.config, join("config.json"))
    with open(join("seed.log"), "w") as f:
        f.write(str(args.seed))

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

    # config["environment"]["render_mode"] = "human"

    # results = evaluate(squad, config)
    # print_info(results)
    # return interpretability(best, config, log_path, index, logger)
