from magent2.environments import battlefield_v5
import pettingzoo
from pettingzoo.utils import BaseParallelWrapper
from pettingzoo.utils.conversions import parallel_to_aec, aec_to_parallel
import numpy as np
import random
import pandas as pd
import os
import time
from utils.print_outputs import *

'''
['__annotations__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__',
'__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', 
'__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__',
'__str__', '__subclasshook__', '__weakref__', '_all_handles', '_calc_obs_shapes', '_calc_state_shape', '_compute_observations',
'_compute_rewards', '_compute_terminates', '_ezpickle_args', '_ezpickle_kwargs', '_minimap_features', '_renderer', 
'_zero_obs', 'action_space', 'action_spaces', 'agents', 'base_state', 'close', 'env', 'extra_features', 'frames', 
'generate_map', 'handles', 'leftID', 'map_size', 'max_cycles', 'max_num_agents', 'metadata', 'minimap_mode', 'num_agents',
'observation_space', 'observation_spaces', 'possible_agents', 'render', 'render_mode', 'reset', 'rightID', 'seed', 'state',
'state_space', 'step', 'team_sizes', 'unwrapped']
'''

class Quick_Test():
    def __init__(self, config) -> None:
        self._env = battlefield_v5.parallel_env(**config['environment'])
        print_debugging(self._env.metadata)
        self._episodes = 1000
    
    def run_env_test(self):
        total_reward = 0
        completed_episodes = 0
        render = True
        actions = {agent: self._env.action_spaces[agent].sample() for agent in self._env.agents}
        obs, reward, termination, truncation, _ = self._env.step(actions)
        
        
        while completed_episodes < self._episodes:
            obs = self._env.reset()
            for agent in self._env.agents:
                if render:
                    self._env.render()
                obs_ = obs[agent]
                rew = reward[agent]
                term = termination[agent]
                trunc = truncation[agent]                
                total_reward += rew
                
                if agent == "blue_0":
                    print_debugging(obs_)
                    print_debugging(rew)
                    print_debugging(term)
                    print_debugging(trunc)

                if term or trunc:
                    action = None
                elif isinstance(obs, dict) and "action_mask" in obs:
                    action = random.choice(np.flatnonzero(obs["action_mask"]).tolist())
                else:
                    action = self._env.action_space(agent).sample()
                actions[agent] = action
            obs, reward, termination, truncation, _ =self._env.step(actions)
            print_configs(obs['blue_0'])
            print_debugging(self._env.num_agents)
            completed_episodes += 1

        if render:
            self._env.close()

        print("Average total reward", total_reward / self._episodes)
        return 0

class Battlefield():
    def __init__(self, config) -> None:
        self._env = battlefield_v5.env(**config['environment']).unwrapped
        

if __name__ == "__main__":
    import argparse
    import json
    import shutil
    import yaml
    import utils
    
    # Load the config file
    file = "src/QD_MARL/configs/battlefield.json"
    config = json.load(open(file))
    
    # Set the random seed
    random.seed(1)
    np.random.seed(1)

    quick_test = Quick_Test(config)
    quick_test.run_env_test()