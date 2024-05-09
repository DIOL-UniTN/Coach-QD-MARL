import os
import sys

sys.path.append(".")
import random
import time
from copy import deepcopy
from math import sqrt

import numpy as np
import pettingzoo

# from memory_profiler import profile
import training.differentObservations as differentObservations
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
from magent2.environments import battle_v4, battlefield_v5
from pettingzoo.utils import aec_to_parallel, parallel_to_aec
from utils import *



def evaluate(trees, config):
    # FOR LOGGING
    # Getting the pid as it is in a parallel process
    # Create the log folder
    pid = os.getpid()
    eval_logs = os.path.join(config["log_path"], "eval_log", str(config['generation']), str(pid))
    os.makedirs(eval_logs, exist_ok=True)
    
    # Starting training and evaluation process
    compute_features = getattr(
        differentObservations, f"compute_features_{config['observation']}"
    )
    policy = None
    # Setup the agents
    agents = {}
    actions = {}
    env = battlefield_v5.env(**config["environment"])
    env.reset()  # This reset lead to the problem
    for agent_name in env.agents:
        agent_squad = "_".join(agent_name.split("_")[:-1])
        if agent_squad == config["team_to_optimize"]:
            agent_number = int("_".join(agent_name.split("_")[1]))

            # Search the index of the set in which the agent belongs
            set_ = 0
            for set_index in range(len(config["sets"])):
                if agent_number in config["sets"][set_index]:
                    set_ = set_index

            # Initialize training agent
            agents[agent_name] = Agent(
                agent_name, agent_squad, set_, trees[set_], None, True
            )
        else:
            # Initialize random or policy agent
            agents[agent_name] = Agent(
                agent_name, agent_squad, None, None, policy, False
            )

    for agent_name in agents:
        actions[agent_name] = []
    
    rewards = []  
    for i in range(config["training"]["episodes"]):
        red_done = 0
        blue_done = 0
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
                # print_configs(f"Agent {agent_name} reward: {rew}")
                rewards.append(rew)

            action = None
            if not done and not trunc:  # if the agent is alive
                if agent.to_optimize():
                    # compute_features(observation, allies, enemies)
                    if agent.get_squad() == "blue":
                        action = agent.get_output(
                            compute_features(obs, blue_agents, red_agents)
                        )
                        if action is None:
                            print("None action")
                            print_debugging(type(agent.get_tree()))
                    else:
                        action = agent.get_output(
                            compute_features(obs, red_agents, blue_agents)
                        )
                else:
                    if agent.has_policy():
                        if agent.get_squad() == "blue":
                            action = agent.get_output(
                                compute_features(obs, blue_agents, red_agents)
                            )
                        else:
                            action = agent.get_output(
                                compute_features(obs, red_agents, blue_agents)
                            )
                    else:
                        # action = env.action_space(agent_name).sample()
                        action = np.random.randint(21)
            else:  # update the number of active agents
                if agent.get_squad() == "red":
                    red_agents -= 1
                else:
                    blue_agents -= 1
            env.step(action)
            # actions[agent_name].append(action)

        # Log the number of kill per in each episode
        # with open(os.path.join(eval_logs,"log_n_kills.txt"), "a") as f:
        #     f.write("Episode: " + str(i)+ " red: " + str(red_done) + " blue: " + str(blue_done) + "\n")
        # f.close()
    env.close()
    # plot_actions(actions, pid, config)
    # rewards count
    rewards = np.array(rewards)
    unique, counts = np.unique(rewards, return_counts=True)
    rewards_dict = dict(zip(unique, counts))
    
    # Log the rewards in each episode
    with open(os.path.join(eval_logs, "log_rewards.txt"), "a") as f:
        f.write(str(rewards_dict) + "\n") 
    f.close()

    # Compute the statistics and scores for each agent(Decision Tree)
    scores = []
    actual_trees = []
    for agent_name in agents:
        if agents[agent_name].to_optimize():
            scores.append(
                agents[agent_name].get_score_statistics(config["statistics"]["agent"])
            )
            actual_trees.append(agents[agent_name].get_tree())
    return scores, actual_trees
