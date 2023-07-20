import gymnasium as gym
from shimmy import MeltingPotCompatibilityV0
from shimmy.utils.meltingpot import load_meltingpot
'''
In ths githb link are listed all the available substrates:
https://github.com/deepmind/meltingpot/blob/main/docs/substrate_scenario_details.md
'''

env = load_meltingpot("prisoners_dilemma_in_the_matrix__arena") 
env = MeltingPotCompatibilityV0(env, render_mode="human") # Create the environment