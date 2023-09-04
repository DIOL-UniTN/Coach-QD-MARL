import os
import sys
sys.path.append(".")
import json
import time
import utils
import random
import pettingzoo
import numpy as np
from math import sqrt
from copy import deepcopy
from pip._vendor.progress.bar import Bar
from algorithms import grammatical_evolution
from decisiontreelibrary import QLearningLeafFactory, ConditionFactory, RLDecisionTree
from magent2.environments import battlefield_v5

class Agent:
    def __init__(self, name, squad, set_, tree, manual_policy, to_optimize):
        self._name = name
        self._squad = squad
        self._set = set_
        # self._tree = tree.deep_copy() if tree is not None else None
        self._manual_policy = manual_policy
        self._to_optimize = to_optimize
        self._score = []

    def get_name(self):
        return self._name

    def get_squad(self):
        return self._squad

    def get_set(self):
        return self._set

    def to_optimize(self):
        return self._to_optimize

    def get_tree(self):
        # return self._tree.deep_copy()
        pass

    def get_output(self, observation):
        if self._to_optimize:
            return self._manual_policy.get_output(observation)
        else:
            return self._manual_policy.get_output(observation)

    def set_reward(self, reward):
        # self._tree.set_reward(reward)
        self._score[-1] += reward

    def get_score_statistics(self, params):
        return getattr(np, f"{params['type']}")(a=self._score, **params['params'])

    def new_episode(self):
        self._score.append(0)

    def has_policy(self):
        return not self._manual_policy is None

    def __str__(self):
        return f"Name: {self._name}; Squad: {self._squad}; Set: {self._set}; Optimize: {str(self._to_optimize)}"

def evaluate(config):
    import differentObservations
    from manual_policies import Policies
    import numpy as np
    from magent2.environments import battlefield_v5

    # Load the function used to computer the features from the observation
    compute_features = getattr(differentObservations, f"compute_features_{config['observation']}")

    # Load manual policy if present
    policy = Policies(config['manual_policy'])

    # Load the environment
    env = battlefield_v5.env(**config['environment'])
    env.reset()
    kills = []

    # Set tree and policy to agents
    agents = {}
    for agent_name in env.agents:
        agent_squad = "_".join(agent_name.split("_")[:-1])
        if agent_squad == config["team_to_optimize"]:
            agents[agent_name] = Agent(agent_name, agent_squad, None, None, policy, True)
        else:
            agents[agent_name] = Agent(agent_name, agent_squad, None, None, policy, False)

    # Start the training
    for i in Bar('Episode').iter(range(config["training"]["episodes"])):
        kills.append(0)
        # Seed the environment
        env.reset(seed=i)
        np.random.seed(i)
        env.reset()

        # Set variabler for new episode
        for agent_name in agents:
            if agents[agent_name].to_optimize():
                agents[agent_name].new_episode()
        red_agents = 12
        blue_agents = 12
        
        # tree.empty_buffers()    # NO-BUFFER LEAFS
        # Iterate over all the agents
        for index, agent_name in enumerate(env.agent_iter()):

            obs, rew, done, trunc, _ = env.last()

            agent = agents[agent_name]

            if agent.to_optimize():
                # Register the reward
                agent.set_reward(rew)
            
            action = None
            if not done and not trunc: # if the agent is alive
                if agent.to_optimize():
                    # compute_features(observation, allies, enemies)
                    if agent.get_squad() == 'blue':
                        action = agent.get_output(compute_features(obs, blue_agents, red_agents))
                    else:
                        action = agent.get_output(compute_features(obs, red_agents, blue_agents))
                else:
                    if agent.has_policy():
                        if agent.get_squad() == 'blue':
                            action = agent.get_output(compute_features(obs, blue_agents, red_agents))
                        else:
                            action = agent.get_output(compute_features(obs, red_agents, blue_agents))
                    else:
                        #action = env.action_space(agent_name).sample()
                        action = np.random.randint(21)
            elif done: # update the number of active agents
                if agent.get_squad() == 'red':
                    red_agents -= 1
                    kills[-1] += 1
                else:
                    blue_agents -= 1

            env.step(action)
        # print(red_agents, blue_agents)

    env.close()

    scores = []
    actual_trees = []
    for agent_name in agents:
        if agents[agent_name].to_optimize():
            scores.append(agents[agent_name].get_score_statistics(config['statistics']['agent']))
            actual_trees.append(agents[agent_name].get_tree())

    return scores, kills

def produce_tree(config, log_path=None, extra_log=False, debug=False, manual_policy=False):

    number_of_agents = config["ge"]["agents"]
    number_of_sets = config["ge"]["sets"]

    # setup log files
    evolution_dir = os.path.join(log_path, "Evolution_dir")
    os.makedirs(evolution_dir , exist_ok=False)
    trees_dir = os.path.join(log_path, "Trees_dir")
    os.makedirs(trees_dir , exist_ok=False)

    gram_dir = None
    if extra_log:
        gram_dir = os.path.join(log_path, "grammatical_evolution")
        os.makedirs(gram_dir, exist_ok=False)

    # Setup sets
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

    with open(os.path.join(log_path, "sets.log"), "w") as f:
        f.write(f"{config['sets']}\n")

    # Setup GE
    ge_config = config["ge"]

    # Build classes of the operators from the config file
    for op in ["mutation", "crossover", "selection", "replacement"]:
        ge_config[op] = getattr(
            grammatical_evolution, ge_config[op]["type"]
        )(**ge_config[op]["params"])

    # ge[i] -> set_i
    ge = grammatical_evolution.GrammaticalEvolution(**ge_config, logdir=gram_dir)
    # Retrive the map function from utils
    map_ = utils.get_map(config["training"]["jobs"], debug)
    # Initialize best individual for each agent
    best = [None for _ in range(number_of_agents)]
    best_fit = [-float("inf") for _ in range(number_of_agents)]
    new_best = [False for _ in range(number_of_agents)]

    print(f"{'Generation' : <10} {'Set': <10} {'Min': <10} {'Mean': <10} {'Max': <10} {'Std': <10}")
    # Iterate over the generations
    for gen in range(config["training"]["generations"]):

        # Retrive the current population
        pop = ge.ask()

        # Convert the genotypes in phenotypes
        trees = [map_(utils.genotype2phenotype, pop[i], config) for i in range(number_of_sets)]
        # Form different groups of trees
        squads = [[trees[j][i] for j in range(number_of_sets)] for i in range(config['ge']['pop_size'])]

        # Compute the fitnesses
        # We need to return the trees in order to retrive the
        #   correct values for the leaves when using the parallelization
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

        # Tell the fitnesses to the GE
        ge.tell(sets_fitness)

        # Compute stats for each agent
        agent_min = [np.min(agents_fitness[i]) for i in range(number_of_agents)]
        agent_mean = [np.mean(agents_fitness[i]) for i in range(number_of_agents)]
        agent_max = [np.max(agents_fitness[i]) for i in range(number_of_agents)]
        agent_std = [np.std(agents_fitness[i]) for i in range(number_of_agents)]

        for i in range(number_of_agents):
            #print(f"{gen: <10} agent_{i: <4} {agent_min[i]: <10.2f} {agent_mean[i]: <10.2f} {agent_max[i]: <10.2f} {agent_std[i]: <10.2f}")
            with open(os.path.join(evolution_dir, f"agent_{i}.log"), "a") as f:
                f.write(f"{gen: <10} {agent_min[i]: <10.2f} {agent_mean[i]: <10.2f} {agent_max[i]: <10.2f} {agent_std[i]: <10.2f}\n")

            new_best[i] = False

        # Compute states for each set
        set_min = [np.min(sets_fitness[i]) for i in range(number_of_sets)]
        set_mean = [np.mean(sets_fitness[i]) for i in range(number_of_sets)]
        set_max = [np.max(sets_fitness[i]) for i in range(number_of_sets)]
        set_std = [np.std(sets_fitness[i]) for i in range(number_of_sets)]

        for i in range(number_of_sets):
            print(f"{gen: <10} set_{i: <4} {set_min[i]: <10.2f} {set_mean[i]: <10.2f} {set_max[i]: <10.2f} {set_std[i]: <10.2f}")
            with open(os.path.join(evolution_dir, f"set_{i}.log"), "a") as f:
                f.write(f"{gen: <10} {set_min[i]: <10.2f} {set_mean[i]: <10.2f} {set_max[i]: <10.2f} {set_std[i]: <10.2f}\n")

    return best

if __name__ == "__main__":
    scores, kills = evaluate(json.load(open("./battlefield.json")))
    print(scores)
    print(kills)
