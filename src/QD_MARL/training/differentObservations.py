import numpy as np


def compute_features_42(obs, n_allies, n_enemies):
    
    map_h = obs.shape[0]
    map_w = obs.shape[1]

    coordinates = tuple(obs[0, 0, 7:9])
    gamma = round(0.0125 * 7, 4) # Indice ottenuto dalla coordinata per matrice 13 x 13, usata in dimensione 80
    epsilon = 0.0001
    ind_x = int(coordinates[0] / gamma + epsilon)
    ind_y = int(coordinates[1] / gamma + epsilon)

    map_h_2 = map_h // 2
    map_w_2 = map_w // 2

    new_features = []
    # Find nearby obstacles
    nearby_obstacles = [0 for _ in range(4)]
    for i in [1, 2]:
        # left
        if all(obs[map_h_2-1:map_h_2+2, map_w_2-i, 0]): nearby_obstacles[0] = 1
        # up
        if all(obs[map_h_2-i, map_w_2-1:map_w_2+2, 0]): nearby_obstacles[1] = 1
        # right
        if all(obs[map_h_2-1:map_h_2+2, map_w_2+i, 0]): nearby_obstacles[2] = 1
        # down
        if all(obs[map_h_2+i, map_w_2-1:map_w_2+2, 0]): nearby_obstacles[3] = 1

    for y in [-1, 0, 1]:
        for x in [-1, 0, 1]:
            if not(y == 0 and x == 0):
                nearby_obstacles.append(obs[map_h_2 + y, map_w_2 + x, 0])
    new_features.extend(nearby_obstacles)

    # Compute the number of teammates to the left, right, up and down
    # Compute the number of enemies to the left, right, up and down
    allies = obs[: , :, 3]
    enemis = obs[: , :, 6]
    allies[ind_y, ind_x] -= 1 + (1 / n_allies) # Beacuse in the agent coordinates it contains 1 + itself density
    if allies[ind_y, ind_x] < epsilon: allies[ind_y, ind_x] = 0
    enemis[ind_y, ind_x] -= 1 # Beacuse in the agent coordinates it contains 1 

    allies_density = np.zeros(9)
    enemies_density = np.zeros(9)

    # above the agent
    for i in range(ind_y):
        # top left
        for j in range(ind_x):
            allies_density[0] += allies[i, j]
            enemies_density[0] += enemis[i, j]
        # top
        allies_density[1] += allies[i, ind_x]
        enemies_density[1] += enemis[i, ind_x]
        # top right
        for j in range(ind_x +1, map_w):
            allies_density[2] += allies[i, j]
            enemies_density[2] += enemis[i, j]

    # to the left of the agent 
    for j in range(ind_x):
        allies_density[3] += allies[ind_y, j]
        enemies_density[3] += enemis[ind_y, j]

    # center
    allies_density[4] += allies[ind_y, ind_x]
    enemies_density[4] += enemis[ind_y, ind_x]

    # to the right of the agent 
    for j in range(ind_x+1, map_w):
        allies_density[5] += allies[ind_y, j]
        enemies_density[5] += enemis[ind_y, j]
        
    # under the agent
    for i in range(ind_y +1, map_h):
        # below left
        for j in range(ind_x):
            allies_density[6] += allies[i, j]
            enemies_density[6] += enemis[i, j]
        # below
        allies_density[7] += allies[i, ind_x]
        enemies_density[7] += enemis[i, ind_x]
        # below right
        for j in range(ind_x +1, map_w):
            allies_density[8] += allies[i, j]
            enemies_density[8] += enemis[i, j]

    new_features.extend(allies_density)
    new_features.extend(enemies_density)

    nondead = (obs[:, :, 4] * obs[:, :, 5]) > 0
    n_enemies_left = np.sum(nondead[:, :map_w_2]) / n_enemies
    n_enemies_up = np.sum(nondead[:map_h_2, :]) / n_enemies
    n_enemies_right = np.sum(nondead[:, 1 + map_w_2:]) / n_enemies
    n_enemies_down = np.sum(nondead[1 + map_h_2:, :]) / n_enemies
    new_features.extend([
        n_enemies_left,
        n_enemies_up,
        n_enemies_right,
        n_enemies_down
    ])
    enemy_presence = []
    for y in [-1, 0, 1]:
        for x in [-1, 0, 1]:
            if not(y == 0 and x == 0):
                enemy_presence.append(nondead[map_h_2 + y, map_w_2 + x])

    new_features.extend(enemy_presence)

    return np.array(new_features)

def compute_features_42_obs(obs, n_allies, n_enemies):
    
    map_h = obs.shape[0]
    map_w = obs.shape[1]

    coordinates = tuple(obs[0, 0, 7:9])
    gamma = round(0.0125 * 7, 4) # Indice ottenuto dalla coordinata per matrice 13 x 13, usata in dimensione 80
    epsilon = 0.0001
    ind_x = int(coordinates[0] / gamma + epsilon)
    ind_y = int(coordinates[1] / gamma + epsilon)

    map_h_2 = map_h // 2
    map_w_2 = map_w // 2

    new_features = []
    # Find nearby obstacles
    obstacles = (obs[:, :, 0] + ((obs[:, :, 1] * obs[:, :, 2]) > 0).astype('float32'))
    
    # two boxes up, left and two boxes down
    new_features.extend([
        # up
        obstacles[map_h_2 -2, map_w_2],
        # left
        obstacles[map_h_2, map_w_2 -2],
        # right
        obstacles[map_h_2, map_w_2 +2],
        # down
        obstacles[map_h_2 +2, map_w_2]
    ])
    
    nearby_obstacles = []
    for y in [-1, 0, 1]:
        for x in [-1, 0, 1]:
            if not(y == 0 and x == 0):
                nearby_obstacles.append(obstacles[map_h_2 + y, map_w_2 + x])
    new_features.extend(nearby_obstacles)

    # Compute the number of teammates to the left, right, up and down
    # Compute the number of enemies to the left, right, up and down
    allies = obs[: , :, 3]
    enemis = obs[: , :, 6]
    allies[ind_y, ind_x] -= 1 + (1 / n_allies) # Beacuse in the agent coordinates it contains 1 + itself density
    if allies[ind_y, ind_x] < epsilon: allies[ind_y, ind_x] = 0
    enemis[ind_y, ind_x] -= 1 # Beacuse in the agent coordinates it contains 1 

    allies_density = np.zeros(9)
    enemies_density = np.zeros(9)

    # above the agent
    for i in range(ind_y):
        # top left
        for j in range(ind_x):
            allies_density[0] += allies[i, j]
            enemies_density[0] += enemis[i, j]
        # top
        allies_density[1] += allies[i, ind_x]
        enemies_density[1] += enemis[i, ind_x]
        # top right
        for j in range(ind_x +1, map_w):
            allies_density[2] += allies[i, j]
            enemies_density[2] += enemis[i, j]

    # to the left of the agent 
    for j in range(ind_x):
        allies_density[3] += allies[ind_y, j]
        enemies_density[3] += enemis[ind_y, j]

    # center
    allies_density[4] += allies[ind_y, ind_x]
    enemies_density[4] += enemis[ind_y, ind_x]

    # to the right of the agent 
    for j in range(ind_x+1, map_w):
        allies_density[5] += allies[ind_y, j]
        enemies_density[5] += enemis[ind_y, j]
        
    # under the agent
    for i in range(ind_y +1, map_h):
        # below left
        for j in range(ind_x):
            allies_density[6] += allies[i, j]
            enemies_density[6] += enemis[i, j]
        # below
        allies_density[7] += allies[i, ind_x]
        enemies_density[7] += enemis[i, ind_x]
        # below right
        for j in range(ind_x +1, map_w):
            allies_density[8] += allies[i, j]
            enemies_density[8] += enemis[i, j]

    new_features.extend(allies_density)
    new_features.extend(enemies_density)

    nondead = (obs[:, :, 4] * obs[:, :, 5]) > 0
    n_enemies_left = np.sum(nondead[:, :map_w_2]) / n_enemies
    n_enemies_up = np.sum(nondead[:map_h_2, :]) / n_enemies
    n_enemies_right = np.sum(nondead[:, 1 + map_w_2:]) / n_enemies
    n_enemies_down = np.sum(nondead[1 + map_h_2:, :]) / n_enemies
    new_features.extend([
        n_enemies_left,
        n_enemies_up,
        n_enemies_right,
        n_enemies_down
    ])
    enemy_presence = []
    for y in [-1, 0, 1]:
        for x in [-1, 0, 1]:
            if not(y == 0 and x == 0):
                enemy_presence.append(nondead[map_h_2 + y, map_w_2 + x])

    new_features.extend(enemy_presence)

    return np.array(new_features)

def compute_features_34_old(obs, n_allies, n_enemies):
    
      
    map_h = obs.shape[0]
    map_w = obs.shape[1]

    coordinates = tuple(obs[0, 0, 7:9])
    gamma = round(0.0125 * 7, 4) # Indice ottenuto dalla coordinata per matrice 13 x 13, usata in dimensione 80
    epsilon = 0.0001
    ind_x = int(coordinates[0] / gamma + epsilon)
    ind_y = int(coordinates[1] / gamma + epsilon)

    map_h_2 = map_h // 2
    map_w_2 = map_w // 2

    new_features = []
    # Find nearby obstacles
    nearby_obstacles = [0 for _ in range(4)]
    for i in [1, 2]:
        # up
        if all(obs[map_h_2-i, map_w_2-1:map_w_2+2, 0]): nearby_obstacles[0] = 1
        # left
        if all(obs[map_h_2-1:map_h_2+2, map_w_2-i, 0]): nearby_obstacles[1] = 1
        # right
        if all(obs[map_h_2-1:map_h_2+2, map_w_2+i, 0]): nearby_obstacles[2] = 1
        # down
        if all(obs[map_h_2+i, map_w_2-1:map_w_2+2, 0]): nearby_obstacles[3] = 1

    for y in [-1, 0, 1]:
        for x in [-1, 0, 1]:
            if not(y == 0 and x == 0):
                nearby_obstacles.append(obs[map_h_2 + y, map_w_2 + x, 0])
    new_features.extend(nearby_obstacles)

    # Compute the global density of teammates to the left, right, up and down
    # Compute the global density of enemies to the left, right, up and down
    allies = obs[: , :, 3]
    enemis = obs[: , :, 6]
    allies[ind_y, ind_x] -= 1 + (1 / n_allies) # Beacuse in the agent coordinates it contains 1 + itself density
    if allies[ind_y, ind_x] < epsilon: allies[ind_y, ind_x] = 0
    enemis[ind_y, ind_x] -= 1 # Beacuse in the agent coordinates it contains 1 

    allies_density = np.zeros(5)
    enemies_density = np.zeros(5)

    # above the agent
    for i in range(ind_y):
        for j in range(map_w):
            allies_density[0] += allies[i, j]
            enemies_density[0] += enemis[i, j]

    # left of the agent 
    for i in range(ind_x):
        for j in range(map_h):
            allies_density[1] += allies[j, i]
            enemies_density[1] += enemis[j, i]

    # center
    allies_density[2] += allies[ind_y, ind_x]
    enemies_density[2] += enemis[ind_y, ind_x]

    # right of the agent 
    for i in range(ind_x+1, map_w):
        for j in range(map_h):
            allies_density[3] += allies[j, i]
            enemies_density[3] += enemis[j, i]
        
    # under the agent
    for i in range(ind_y +1, map_h):
        for j in range(map_w):
            allies_density[4] += allies[i, j]
            enemies_density[4] += enemis[i, j]

    new_features.extend(allies_density)
    new_features.extend(enemies_density)

    # Compute the local density of enemies to the left, right, up and down
    nondead = (obs[:, :, 4] * obs[:, :, 5]) > 0
    n_enemies_left = np.sum(nondead[:, :map_w_2]) / n_enemies
    n_enemies_up = np.sum(nondead[:map_h_2, :]) / n_enemies
    n_enemies_right = np.sum(nondead[:, 1 + map_w_2:]) / n_enemies
    n_enemies_down = np.sum(nondead[1 + map_h_2:, :]) / n_enemies
    new_features.extend([
        n_enemies_left,
        n_enemies_up,
        n_enemies_right,
        n_enemies_down
    ])
    enemy_presence = []
    for y in [-1, 0, 1]:
        for x in [-1, 0, 1]:
            if not(y == 0 and x == 0):
                enemy_presence.append(nondead[map_h_2 + y, map_w_2 + x])

    new_features.extend(enemy_presence)

    return np.array(new_features)

def compute_features_34_new(obs, n_allies, n_enemies):
    
    map_h = obs.shape[1]
    map_w = obs.shape[1]

    coordinates = tuple(obs[0, 0, 7:9])
    gamma = round(0.0125 * 7, 4) # Indice ottenuto dalla coordinata per matrice 13 x 13, usata in dimensione 80
    epsilon = 0.0001
    ind_x = int(coordinates[0] / gamma + epsilon)
    ind_y = int(coordinates[1] / gamma + epsilon)

    map_h_2 = map_h // 2
    map_w_2 = map_w // 2

    new_features = []
    # Find nearby obstacles
    obstacles = (obs[:, :, 0] + ((obs[:, :, 1] * obs[:, :, 2]) > 0).astype('float32'))
    
    # two boxes up, left and two boxes down
    new_features.extend([
        # up
        obstacles[map_h_2 -2, map_w_2],
        # left
        obstacles[map_h_2, map_w_2 -2],
        # right
        obstacles[map_h_2, map_w_2 +2],
        # down
        obstacles[map_h_2 +2, map_w_2]
    ])
    
    nearby_obstacles = []
    for y in [-1, 0, 1]:
        for x in [-1, 0, 1]:
            if not(y == 0 and x == 0):
                nearby_obstacles.append(obstacles[map_h_2 + y, map_w_2 + x])
    '''
    for y in range(-2, 3):
        for x in range(-2, 3):
            if not(y == 0 and x == 0) and not(abs(y) + abs(x) > 2):
                nearby_obstacles.append(obstacles[map_h_2 + y, map_w_2 + x])
    '''
    new_features.extend(nearby_obstacles)

    # Compute the global density of teammates to the left, right, up and down
    # Compute the global density of enemies to the left, right, up and down
    allies = obs[: , :, 3]
    enemis = obs[: , :, 6]
    allies[ind_y, ind_x] -= 1 + (1 / n_allies) # Beacuse in the agent coordinates it contains 1 + itself density
    if allies[ind_y, ind_x] < epsilon: allies[ind_y, ind_x] = 0
    enemis[ind_y, ind_x] -= 1 # Beacuse in the agent coordinates it contains 1 

    allies_density = np.zeros(5)
    enemies_density = np.zeros(5)

    # above the agent
    for i in range(ind_y):
        for j in range(map_w):
            allies_density[0] += allies[i, j]
            enemies_density[0] += enemis[i, j]

    # to the left of the agent 
    for i in range(ind_x):
        for j in range(map_h):
            allies_density[1] += allies[j, i]
            enemies_density[1] += enemis[j, i]

    # center
    allies_density[2] += allies[ind_y, ind_x]
    enemies_density[2] += enemis[ind_y, ind_x]

    # to the right of the agent 
    for i in range(ind_x+1, map_w):
        for j in range(map_h):
            allies_density[3] += allies[j, i]
            enemies_density[3] += enemis[j, i]
        
    # under the agent
    for i in range(ind_y +1, map_h):
        for j in range(map_w):
            allies_density[4] += allies[i, j]
            enemies_density[4] += enemis[i, j]

    new_features.extend(allies_density)
    new_features.extend(enemies_density)

    # Compute the local density of enemies to the left, right, up and down
    nondead = (obs[:, :, 4] * obs[:, :, 5]) > 0
    n_enemies_up = np.sum(nondead[:map_h_2, :]) / n_enemies
    n_enemies_left = np.sum(nondead[:, :map_w_2]) / n_enemies
    n_enemies_right = np.sum(nondead[:, 1 + map_w_2:]) / n_enemies
    n_enemies_down = np.sum(nondead[1 + map_h_2:, :]) / n_enemies
    new_features.extend([
        n_enemies_up,
        n_enemies_left,
        n_enemies_right,
        n_enemies_down
    ])
    enemy_presence = []
    for y in [-1, 0, 1]:
        for x in [-1, 0, 1]:
            if not(y == 0 and x == 0):
                enemy_presence.append(nondead[map_h_2 + y, map_w_2 + x])

    new_features.extend(enemy_presence)
    
    return np.array(new_features)

if __name__ == "__main__":


    import pettingzoo
    from magent2.environments import battle_v4, battlefield_v5
    import json
    import argparse
    import random
    import time

    class Agent:
        def __init__(self, name, squad):
            self._name = name
            self._squad = squad
        def get_name(self):
            return self._name
        def get_squad(self):
            return self._squad

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path of the config file to use")
    args = parser.parse_args()

    config = json.load(open(args.config))

    env = battlefield_v5.env(**config['environment'])

    env.reset()

    red_agents = 12
    blue_agents = 12
    agents = {}

    for agent_name in env.agents:
        agent_squad = "_".join(agent_name.split("_")[:-1])
        agents[agent_name] = Agent(agent_name, agent_squad)

    for index, agent in enumerate(env.agent_iter()):
        j = index % len(env.agents)
        if j == 0:
            env.render()
        observation, reward, done, trunc, info = env.last()
        action = 6 if not done else None
        
        #if not done:
        if agent == 'blue_0' and not done:
            features = compute_features_34_new(observation, blue_agents, red_agents)
            print(len(features))
            print(features)
            print('blue:', blue_agents)
            print('red:', red_agents)
            print()
            #print("Comando {}:".format(agent))
            print("Comando:")
            input_ = input()
            try:
                action = int(input_)
                if action < 0 or action > 20:
                    print("L'azione deve essere compresa tra 0 e 20 inclusi")
                    action = 6
            except:
                #print(input_)
                if input_ == "break" or input_ == "exit":
                    print("Closed")
                    break
                elif input_ == "vision":
                    print(features)
                    print('blue:', blue_agents)
                    print('red:', red_agents)
                    print()
                else:
                    print("pass")
                    pass
            #print(reward)
        if done:
            if agents[agent].get_squad() == 'blue':
                blue_agents -= 1
            else:
                red_agents -= 1
        
        env.step(action)
