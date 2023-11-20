import os
import sys

sys.path.append(".")
import random
import time
from copy import deepcopy
from math import sqrt

import differentObservations
import numpy as np
import pettingzoo
from agents import *
from algorithms import (grammatical_evolution, individuals, map_elites,
                        map_elites_Pyribs, mapElitesCMA_pyRibs)
from decisiontrees import (ConditionFactory, DecisionTree,
                           QLearningLeafFactory, RLDecisionTree)
from magent2.environments import battlefield_v5
from pettingzoo.utils import aec_to_parallel, parallel_to_aec
from test_environments import *
from utils import *


def set_trees_to_train(trees, beginning_index,config):
    trees_to_train = []
    for i in range(config['ge']['agents']):
        if i + beginning_index >= config['ge']['pop_size']:
            trees_to_train.append(trees[i])
        else:
            trees_to_train.append(trees[beginning_index + i])
    return trees_to_train

def set_set(config):
    number_of_agents = config["ge"]["agents"]
    number_of_sets = config["ge"]["sets"]
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
    return config["sets"]

def evaluate(trees, config, replay_buffer, map_, init_eval=False, coach = None):
    compute_features = getattr(differentObservations, f"compute_features_{config['observation']}")
    policy = None
    # Load the environment
    config["sets"] = set_set(config)
    # Setup the agents
    agents = {}
        
    env = battlefield_v5.env(**config['environment']).unwrapped
    env.reset()#This reset lead to the problem
    
    for agent_name in env.agents:
        agent_squad = "_".join(agent_name.split("_")[:-1])
        if agent_squad == config["team_to_optimize"]:
            agent_number = int("_".join(agent_name.split("_")[1]))
            
            # Search the index of the set in which the agent belongs
            set_ = 0
            for set_index in range(len(config["sets"])):
                if agent_number in config["sets"][set_index]:
                    set_ = set_index

            # Initialize trining agent
            agents[agent_name] = Agent(agent_name, agent_squad, set_, trees[set_], None, True)
        else:
            # Initialize random or policy agent
            agents[agent_name] = Agent(agent_name, agent_squad, None, None, policy, False)
            
    red_agents = 12
    blue_agents = 12
    
    count_attack_penalty = 0
    count_moves_penalty = 0
    count_dead_penalty = 0
    count_good_penalty = 0
    
    for i in range(config["training"]["episodes"]):
        for agent in agents:
            if agents[agent].to_optimize():
                agents[agent].new_episode()
        # Seed the environment
        env.reset(seed=i)
        np.random.seed(i)

        # Set variabler for new episode
        for agent_name in agents:
            if agents[agent_name].to_optimize():
                agents[agent_name].new_episode()
        
        agents_features = {}
        for agent_name in agents:
            #print_debugging(f"Agent blue_0 action: {actions['blue_0']}")
            actions = {agent: env.action_spaces[agent].sample() for agent in agents}
            obs, rew, done, trunc, info = env.step(actions)
            agent = agents[agent_name] 
            obs_ = obs[agent_name]
            rew_ = rew[agent_name]
            done_ = done[agent_name]
            trunc_ = trunc[agent_name]
            
            if rew_ > -10: count_moves_penalty += 1
            elif rew_ in range(-50,-150): count_dead_penalty += 1
            elif rew_< -500: count_attack_penalty += 1
            elif rew_ > 0 and rew < 10: count_good_penalty += 1
            if done_ or trunc_:
                print_info(f"Agent {agent_name} is dead at the of the episode {i}")
            
            if agent.to_optimize():
                # Register the reward
                agent.set_reward(rew_)
                # print_debugging(f"Agent {agent_name} reward: {rew_}")
                
            if not done_ and not trunc_: # if the agent is alive
                if agent.to_optimize():
                    # compute_features(observation, allies, enemies)
                    agents_features[agent_name] = compute_features(obs_, blue_agents, red_agents)
                    if agent.get_squad() == 'blue':
                        action = agent.get_output(agents_features[agent_name])
                    else:
                        action = agent.get_output(agents_features[agent_name])
                    # print_debugging(f"Agent {agent_name} action: {action}")
                else:
                    if agent.has_policy():
                        if agent.get_squad() == 'blue':
                            action = agent.get_output(compute_features(obs_, blue_agents, red_agents))
                        else:
                            action = agent.get_output(compute_features(obs_, red_agents, blue_agents))
                    else:
                        action = env.action_space(agent_name).sample()
                        #action = np.random.randint(21)
            else: # update the number of active agents
                if agent.get_squad() == 'red':
                    red_agents -= 1
                else:
                    blue_agents -= 1
                    
            actions[agent_name] = action
        env.step(actions)
        # print_debugging(f"End episode {i}")
        # print_debugging(f"counts: attack {count_attack_penalty}, moves {count_moves_penalty}, dead {count_dead_penalty}, good {count_good_penalty}")
    env.close()
        
    if init_eval:
        round_agents = []
        for agent_name in agents:
            agent = agents[agent_name]
            if agent.to_optimize():
                round_agents.append(agent)
        print_debugging(f"End evaluation team, red agents: {red_agents}, blue agents: {blue_agents}")
        return round_agents, agents_features
    else:
        scores = []
        actual_trees = []
        for agent_name in agents:
            if agents[agent_name].to_optimize():
                scores.append(agents[agent_name].get_score_statistics(config['statistics']['agent']))
                actual_trees.append(agents[agent_name].get_tree())
        return scores, actual_trees

def init_eval(trees, config, replay_buffer, map_, coach = None):
    
    number_of_possible_teams = int(len(trees)/(config["ge"]["agents"]))
    round_agents = []
    team_selection = {}
    for index_ in range(number_of_possible_teams):
        print_debugging(f"Evaluating team {index_}")
        round_trees = set_trees_to_train(trees, index_ * config["ge"]["agents"], config)
        round_agents_, agent_features = evaluate(round_trees, config, replay_buffer, map_, init_eval=True, coach=coach)
        round_agents.append(round_agents_)
        team_selection[index_] = agent_features
    features = get_pop_features(team_selection)
    # coach training
    if coach is not None:
        for index_ in range(len(features)):
            coach.set_reward(features[index_])
    scores = []
    actual_trees = []
    round_agents = [item for sublist in round_agents for item in sublist]
    for agent_name in round_agents:
        scores.append(agent_name.get_score_statistics(config['statistics']['agent']))
        actual_trees.append(agent_name.get_tree())
    return scores, actual_trees

def get_pop_features(features_dict):
    features_sum = []
    for n in features_dict:
        for agent_name in features_dict[n]:
            features_sum.append(features_dict[n][agent_name])
    for i in range(len(features_sum)):
        agent = np.array(features_sum[i])
        features_sum[i] = np.sum(agent, axis=0)
    return features_sum
    

# def evaluate(trees, config, replay_buffer, map_):
#     compute_features = getattr(differentObservations, f"compute_features_{config['observation']}")
#     policy = None
#     # Load the environment
#     env = battlefield_v5.env(**config['environment']).unwrapped
#     env.reset()#This reset lead to the problem
#     config["sets"] = set_set(config)
#     # Setup the agents
#     agents = {}
#     for agent_name in env.agents:
#         agent_squad = "_".join(agent_name.split("_")[:-1])
#         if agent_squad == config["team_to_optimize"]:
#             agent_number = int("_".join(agent_name.split("_")[1]))
                
#             # Search the index of the set in which the agent belongs
#             set_ = 0
#             for set_index in range(len(config["sets"])):
#                 if agent_number in config["sets"][set_index]:
#                     set_ = set_index

#             # Initialize trining agent
#             agents[agent_name] = Agent(agent_name, agent_squad, set_, trees[set_], None, True)
#         else:
#             # Initialize random or policy agent
#             agents[agent_name] = Agent(agent_name, agent_squad, None, None, policy, False)
#     for i in range(config["training"]["episodes"]):

#         # Seed the environment
#         env.reset(seed=i)
#         np.random.seed(i)
#         env.reset()

#         # Set variabler for new episode
#         for agent_name in agents:
#             if agents[agent_name].to_optimize():
#                 agents[agent_name].new_episode()
#         red_agents = 12
#         blue_agents = 12
        
#         for index, agent_name in enumerate(env.agents):
#             actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
#             obs, rew, done, trunc, info = env.step(actions)
#             agent = agents[agent_name]
#             #f = compute_features(obs, blue_agents, red_agents)
            
#             obs = obs[agent._name]
#             rew = rew[agent._name]
#             done = done[agent._name]
#             trunc = trunc[agent._name]
            
#             if agent.to_optimize():
#                 # Register the reward
#                 agent.set_reward(rew)
                
#             #action = env.action_space(agent_name).sample()
            
#             if not done and not trunc: # if the agent is alive
#                 if agent.to_optimize():
#                     # compute_features(observation, allies, enemies)
#                     if agent.get_squad() == 'blue':
#                         action = agent.get_output(compute_features(obs, blue_agents, red_agents))
#                     else:
#                         action = agent.get_output(compute_features(obs, red_agents, blue_agents))
#                 else:
#                     if agent.has_policy():
#                         if agent.get_squad() == 'blue':
#                             action = agent.get_output(compute_features(obs, blue_agents, red_agents))
#                         else:
#                             action = agent.get_output(compute_features(obs, red_agents, blue_agents))
#                     else:
#                         action = env.action_space(agent_name).sample()
#                         #action = np.random.randint(21)
#             else: # update the number of active agents
#                 if agent.get_squad() == 'red':
#                     red_agents -= 1
#                 else:
#                     blue_agents -= 1
#             actions[agent_name] = action
#         env.step(actions)
#     env.close()
#     scores = []
#     actual_trees = []
#     for agent_name in agents:
#         if agents[agent_name].to_optimize():
#             scores.append(agents[agent_name].get_score_statistics(config['statistics']['agent']))
#             actual_trees.append(agents[agent_name].get_tree())
#     return scores, actual_trees