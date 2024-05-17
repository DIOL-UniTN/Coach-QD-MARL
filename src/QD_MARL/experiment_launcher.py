
import os
import sys

sys.path.append(".")

import numpy as np
import pettingzoo
from agents.agents import *
from algorithms import (
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
import shutil
import abc

def set_type_experiment(config, extra_log=False, debug=False, manual_policy=False):
    if config["experiment"] == "me-single_me":
        experiment = Experiment_Single_ME(config, extra_log, debug, manual_policy)
    elif config["experiment"] == "me-one_tree_per_team":
        experiment = Experiment_One_Per_Team(config, extra_log, debug, manual_policy)
    elif config["experiment"] == "me-me_per_team":
        experiment = Experiment_Me_Per_Team(config, extra_log, debug, manual_policy)
    else:
        raise ValueError(f"Unknown experiment type: {config['experiment']}")
    return experiment

class Experiment():
    def __init__(self, config, extra_log=False, debug=False, manual_policy=False):
        self._config = config
        self._log_path = self.set_log_path()
        self._extra_log = extra_log
        self._debug = debug
        self._manual_policy = manual_policy
        self._me_config = self._config["me_config"]['me']["kwargs"]
        self._me_config["log_path"] = self._log_path  
        self._number_of_agents = self._config["n_agents"]
        self._number_of_teams = self._config["training"]["jobs"]
        self._selection_type = self._me_config["selection_type"]
        self._map = utils.get_map(self._number_of_teams, debug)

        # ME 
        self._me = None
        self._coach = None
        self._me_pop = []
        self._teams = []

        # Initialize best individual for each agent
        self._best = [None for _ in range(self._number_of_agents)]
        self._best_fit = [-float("inf") for _ in range(self._number_of_agents)]
        self._new_best = [False for _ in range(self._number_of_agents)]
        
        # Initialize best team
        self._best_team = None
        self._best_team_fitness = -float("inf")
        self._best_team_tree = [None for _ in range(self._number_of_agents)]
        self._best_team_fitnesses = [None for _ in range(self._number_of_agents)]
        self._best_team_min = -float("inf")
        self._best_team_mean = -float("inf")
        self._best_team_max = -float("inf")
        self._best_team_std = 0

        #init logs
        self.init_logs()

        

    def set_log_path(self):
        logdir_name = utils.get_logdir_name()
        if self._config["hpc"]:
            cwd = os.getcwd()
            log_path = f"{cwd}/logs/qd-marl/hpc/{self._config['experiment']}/{self._config['me_config']['me']['kwargs']['me_type']}/{self._config['me_config']['me']['kwargs']['selection_type']}/magent_battlefield/{logdir_name}"
        else:
            log_path = f"logs/qd-marl/local/{self._config['experiment']}/{self._config['me_config']['me']['kwargs']['me_type']}/{self._config['me_config']['me']['kwargs']['selection_type']}/magent_battlefield/{logdir_name}"
        print_configs("Logs path: ", log_path)
        join = lambda x: os.path.join(log_path, x)
        self._config["log_path"] = log_path
        os.makedirs(log_path, exist_ok=False)
        shutil.copy(self._config["original_config"], join("config.json"))
        with open(join("seed.log"), "w") as f:
            f.write(str(self._config["args_seed"]))
        return log_path

    def get_log_path(self):
        return self._log_path
    
    def init_logs(self):
        # Log files and directories
        self._evolution_dir = os.path.join(self._log_path, "Evolution_dir")
        os.makedirs(self._evolution_dir , exist_ok=False)
        self._team_dir = os.path.join(self._evolution_dir, "Teams")
        os.makedirs(self._team_dir , exist_ok=False)
        self._agent_dir = os.path.join(self._evolution_dir, "Agents")
        os.makedirs(self._agent_dir , exist_ok=False)
        self._trees_dir = os.path.join(self._log_path, "Trees_dir")
        os.makedirs(self._trees_dir , exist_ok=False)
        self._plot_dir = os.path.join(self._log_path, "Plots")
        os.makedirs(self._plot_dir , exist_ok=False)

        # Initialize logging file
        with open(os.path.join(self._team_dir, f"best_team.txt"), "a") as f:
            f.write(f"Generation,Min,Mean,Max,Std\n")

        for i in range(self._number_of_agents):
                #print(f"{gen: <10} agent_{i: <4} {agent_min[i]: <10.2f} {agent_mean[i]: <10.2f} {agent_max[i]: <10.2f} {agent_std[i]: <10.2f}")
                with open(os.path.join(self._agent_dir, f"agent_{i}.txt"), "a") as f:
                    f.write(f"Generation,Min,Mean,Max,Std\n")
        for i in range(self._number_of_teams):
                #print(f"{gen: <10} agent_{i: <4} {agent_min[i]: <10.2f} {agent_mean[i]: <10.2f} {agent_max[i]: <10.2f} {agent_std[i]: <10.2f}")
                with open(os.path.join(self._team_dir, f"team_{i}.txt"), "a") as f:
                    f.write(f"Generation,Min,Mean,Max,Std\n")
        with open(os.path.join(self._evolution_dir, f"bests.txt"), "a") as f:
                    f.write(f"Generation,Min,Mean,Max,Std\n")

    def set_map_elite(self, n=None):

        factories_config = self._config["me_config"]
        # Setup MAp Elite
        if n is not None:
            self._me_config["log_path"] = self._me_config["log_path"]+f"/team_{n}"
        else:
            pass
        # Build classes of the operators from the config file
        self._me_config["c_factory"] = ConditionFactory(factories_config["ConditionFactory"]["type"])
        self._me_config["l_factory"] = QLearningLeafFactory(
            factories_config["QLearningLeafFactory"]["kwargs"]["leaf_params"],
            factories_config["QLearningLeafFactory"]["kwargs"]["decorators"],
        )
        if self._me_config["me_type"] == "MapElites_pyRibs":
            me = map_elites_Pyribs.MapElites_Pyribs(**self._me_config)
        elif self._me_config["me_type"] == "MapElitesCMA_pyRibs":
            me = mapElitesCMA_pyRibs.MapElitesCMA_pyRibs(**self._me_config)
        else:
            raise ValueError(f"Unknown ME type: {self._me_config['me_type']}")
        print_configs("ME type:", self._me_config["me_type"])            
        print_configs("ME selection type:", self._me_config["selection_type"])
        return me
    
    def get_me(self):
        return self._me
    
    def set_coach(self, me_config, me):
        # Setup Coach, if experiment is without coach, return None
        if self._selection_type == "coach":
            coach_config = me_config["coach"]
            coach_config["pop_size"] = me_config["init_pop_size"]
            coach_config["batch_size"] = me_config["batch_pop"]
            coach = CoachAgent(coach_config, me)
        else:
            coach = None
        return coach
    
    def get_coach(self):
        return self._coach
    
    @abc.abstractmethod
    def set_experiment(self):
        pass
    
    @abc.abstractmethod
    def set_teams(self, gen):
        pass
    
    def get_teams(self):
        return self._teams    
    
    @abc.abstractmethod
    def tell_me(self, return_values, gen):
        pass      
    
    def run_experiment(self):

        self._me, self._coach = self.set_experiment()
         
        print_info(f"{'Generation' : <10} {'Set': <10} {'Min': <10} {'Mean': <10} {'Max': <10} {'Std': <10}")

        for gen in range(-1, self._config["training"]["generations"]):
            self._config["generation"] = gen
            self._me_pop = []
            self._teams = []
            self._teams = self.set_teams(gen)

            # Running training environment and evaluating the fitness
            return_values = self._map(evaluate, self._teams, self._config)

            agents_fitness, teams_fitness = self.tell_me(return_values, gen)

            # Compute stats for each agent
            agent_min = [np.min(agents_fitness[i]) for i in range(self._number_of_agents)]
            agent_mean = [np.mean(agents_fitness[i]) for i in range(self._number_of_agents)]
            agent_max = [np.max(agents_fitness[i]) for i in range(self._number_of_agents)]
            agent_std = [np.std(agents_fitness[i]) for i in range(self._number_of_agents)]

            for i in range(self._number_of_agents):
                #print(f"{gen: <10} agent_{i: <4} {agent_min[i]: <10.2f} {agent_mean[i]: <10.2f} {agent_max[i]: <10.2f} {agent_std[i]: <10.2f}")
                with open(os.path.join(self._agent_dir, f"agent_{i}.txt"), "a") as f:
                    f.write(f"{gen: <10} {agent_min[i]: <10.2f} {agent_mean[i]: <10.2f} {agent_max[i]: <10.2f} {agent_std[i]: <10.2f}\n")
            
            # Compute states for each set
            
            team_min = [np.min(teams_fitness[i]) for i in range(self._number_of_teams)]
            team_mean = [np.mean(teams_fitness[i]) for i in range(self._number_of_teams)]
            team_max = [np.max(teams_fitness[i]) for i in range(self._number_of_teams)]
            team_std = [np.std(teams_fitness[i]) for i in range(self._number_of_teams)]
            
            for i in range(self._number_of_teams):
                print(f"{gen: <10} set_{i: <4} {team_min[i]: <10.2f} {team_mean[i]: <10.2f} {team_max[i]: <10.2f} {team_std[i]: <10.2f}")
                with open(os.path.join(self._team_dir, f"team_{i}.txt"), "a") as f:
                    f.write(f"{gen: <10} {team_min[i]: <10.2f} {team_mean[i]: <10.2f} {team_max[i]: <10.2f} {team_std[i]: <10.2f}\n")

            
            amax = np.argmax(np.mean(teams_fitness))
            max_ = np.mean(teams_fitness[amax])
            if max_ > self._best_team_fitness:
                self._best_team = self._teams[amax]
                self._best_team_fitness = max_
                best_team_fitnesses = teams_fitness[amax]
                tree_text = f"{self._best_team}"
                for i in range(len(self._best_team)):
                    utils.save_tree(self._best_team[i], self._trees_dir, f"best_team_agent_{i}")
                    with open(os.path.join(self._trees_dir, f"best_team_agent_{i}.log"), "w") as f:
                        f.write(tree_text)

                with open(os.path.join(self._trees_dir, f"best_team.log"), "w") as f:
                    f.write(tree_text)
            best_team_mean = np.mean(best_team_fitnesses)
            best_team_max = np.max(best_team_fitnesses)
            best_team_min = np.min(best_team_fitnesses)
            best_team_std = np.std(best_team_fitnesses)
            print_info(f"{gen: <10} {'best_team': <10} {best_team_min: <10.2f} {best_team_mean: <10.2f} {best_team_max: <10.2f} {best_team_std: <10.2f}")
            with open(os.path.join(self._team_dir, f"best_team.txt"), "a") as f:
                    f.write(f"{gen},{best_team_min},{best_team_mean},{self._best_team_fitness},{best_team_std}\n")
            plot_log(self._team_dir, self._plot_dir, "best_team", f"best_team.txt", gen)        
                
        return self._best, self._best_team
    
        
         
class Experiment_Single_ME(Experiment):
    def __init__(self, config, extra_log=False, debug=False, manual_policy=False):
        super().__init__(config, extra_log, debug, manual_policy)

    def set_experiment(self):
        print_configs("Experiment type:", "ME - single ME")
        self._me = self.set_map_elite()
        self._coach = self.set_coach(self._me_config, self._me)
        if self._selection_type == "coach":
            self._coach.set_n_teams(self._number_of_teams)
        return self._me, self._coach

    def set_teams(self, gen):
        if gen >= 0:
                if self._selection_type == "coach":
                    me_teams = self._coach.ask()
                    for team in me_teams:
                        trees = [RLDecisionTree(team[i], self._config["training"]["gamma"]) for i in range(len(team))]
                        self._teams.append(trees)
                else:
                    for i in range(self._number_of_teams):
                        me_pop = self._me.ask()
                        trees = [RLDecisionTree(me_pop[i], self._config["training"]["gamma"]) for i in range(len(me_pop))]
                        self._teams.append(trees)            
        else:
            me_pop = self._me.ask()
            trees = [RLDecisionTree(me_pop[i], self._config["training"]["gamma"]) for i in range(len(me_pop))]
            j = 0
            for i in range(self._number_of_teams):
                if j*self._number_of_agents >= len(trees):
                    j =0
                self._teams.append(trees[j*self._number_of_agents:(j+1)*self._number_of_agents])
                j+=1 
        return self._teams

    def tell_me(self, return_values, gen):
        agents_fitness = [ [] for _ in range(self._number_of_agents)]
        agents_tree = [ [] for _ in range(self._number_of_agents)]

        # Store trees and fitnesses
        for values in return_values:
            for i in range(self._number_of_agents):
                agents_fitness[i].append(values[0][i])
                agents_tree[i].append(values[1][i])

        amax = [np.argmax(agents_fitness[i]) for i in range(self._number_of_agents)]
        max_ = [agents_fitness[i][amax[i]] for i in range(self._number_of_agents)]

        for i in range(self._number_of_agents):
            if max_[i] > self._best_fit[i]:
                self._best_fit[i] = max_[i]
                self._best[i] = agents_tree[i][amax[i]]
                self._new_best[i] = True

                tree_text = f"{self._best[i]}"
                utils.save_tree(self._best[i], self._trees_dir, f"best_agent_{i}")
                with open(os.path.join(self._trees_dir, f"best_agent_{i}.log"), "w") as f:
                    f.write(tree_text)
        # Compute stats for the bests
        best_min = np.min(self._best_fit)
        best_mean = np.mean(self._best_fit)
        best_max = np.max(self._best_fit)
        best_std = np.std(self._best_fit)

        with open(os.path.join(self._evolution_dir, f"bests.txt"), "a") as f:
                f.write(f"{gen},{best_min},{best_mean},{best_max},{best_std}\n")
        
        plot_log(self._evolution_dir, self._plot_dir, "bests",f"bests.txt", gen)


        teams_fitness = []
        teams_trees = []
        for i in range(self._number_of_teams):
            team = []
            team_trees = []
            for j in range(self._number_of_agents):
                team.append(agents_fitness[j][i])
                team_trees.append(agents_tree[j][i])
            teams_fitness.append(team)
            teams_trees.append(team_trees)

        individual_fitness = np.array(teams_fitness).flatten()
        individual_trees = np.array(teams_trees).flatten()
        individuals_roots = [t.get_root() for t in individual_trees]
        self._me.tell(individual_fitness, individuals_roots)
        self._me.plot_archive(gen)    

        teams_fitness = [np.mean(team) for team in teams_fitness]
        return agents_fitness, teams_fitness
    
    def run_experiment(self):
        return super().run_experiment()     


class Experiment_One_Per_Team(Experiment):
    def __init__(self, config, extra_log=False, debug=False, manual_policy=False):
        super().__init__(config, extra_log, debug, manual_policy)

    def set_sets(self):
        # Setup sets
        if self._config["sets"] == "random":
            sets = [[] for _ in range(self._config["n_sets"])]
            set_full = [False for _ in range(self._config["n_sets"])]
            agents = [i for i in range(self._number_of_agents)]

            agents_per_set =self._number_of_agents // self._config["n_sets"]
            surplus = self._number_of_agents // self._config["n_sets"]

            while not all(set_full):
                set_index = random.randint(0, self._config["n_sets"] -1)
                if not set_full[set_index]:
                    random_index = random.randint(0, len(agents) -1)
                    sets[set_index].append(agents[random_index])
                    del agents[random_index]
                    if len(sets[set_index]) == agents_per_set:
                        set_full[set_index] = True

            if surplus > 0:
                while len(agents) != 0:
                    set_index = random.randint(0, self._config["n_sets"] -1)
                    random_index = random.randint(0, len(agents) -1)
                    sets[set_index].append(agents[random_index])
                    del agents[random_index]

            for set_ in sets:
                set_.sort()

            self._config["sets"] = sets

    def set_experiment(self):
        print_configs("Experiment type:", "ME - one tree per team")
        self._me = self.set_map_elite()
        self._coach = self.set_coach(self._me_config, self._me)
        if self._selection_type == "coach":
            self._coach.set_n_teams(self._number_of_teams)
        self.set_sets()
        return self._me, self._coach

    def set_teams(self, gen):
        if gen >= 0:
                batch_size = self._me_config["batch_pop"]
                if self._selection_type == "coach":
                    self._me_pop = self._coach.ask()
                else:
                    self._me_pop = self._me.ask()
        else:
            self._me_pop = self._me.ask() 
            batch_size = len(self._me_pop)
            
        trees = [[RLDecisionTree(self._me_pop[i], self._config["training"]["gamma"]) for i in range(batch_size)] for _ in range(self._config["n_sets"])]
        #Form the teams
        self._teams = [[trees[j][i] for j in range(self._config["n_sets"])] for i in range(batch_size)]
        return self._teams

    def tell_me(self, return_values, gen):
        agents_fitness = [ [] for _ in range(self._number_of_agents)]
        agents_tree = [ [] for _ in range(self._number_of_agents)]

        # Store trees and fitnesses
        for values in return_values:
            for i in range(self._number_of_agents):
                agents_fitness[i].append(values[0][i])
                agents_tree[i].append(values[1][i])

        amax = [np.argmax(agents_fitness[i]) for i in range(self._number_of_agents)]
        max_ = [agents_fitness[i][amax[i]] for i in range(self._number_of_agents)]

        for i in range(self._number_of_agents):
            if max_[i] > self._best_fit[i]:
                self._best_fit[i] = max_[i]
                self._best[i] = agents_tree[i][amax[i]]
                self._new_best[i] = True

                tree_text = f"{self._best[i]}"
                utils.save_tree(self._best[i], self._trees_dir, f"best_agent_{i}")
                with open(os.path.join(self._trees_dir, f"best_agent_{i}.log"), "w") as f:
                    f.write(tree_text)
        # Compute stats for the bests
        best_min = np.min(self._best_fit)
        best_mean = np.mean(self._best_fit)
        best_max = np.max(self._best_fit)
        best_std = np.std(self._best_fit)

        with open(os.path.join(self._evolution_dir, f"bests.txt"), "a") as f:
                f.write(f"{gen},{best_min},{best_mean},{best_max},{best_std}\n")
        
        plot_log(self._evolution_dir, self._plot_dir, "bests",f"bests.txt", gen)


        teams_fitness = []
        teams_trees = []

        sets_fitness = [ [] for _ in range(self._config["n_sets"])]
        for index, set_ in enumerate(self._config["sets"]):
            set_agents_fitnesses = []
            for agent in set_:
                set_agents_fitnesses.append(agents_fitness[agent])
            set_agents_fitnesses = np.array(set_agents_fitnesses)
            # Calculate fitness for each individual in the set
            sets_fitness[index] = [getattr(np, self._config['statistics']['set']['type'])(a=set_agents_fitnesses[:, i], **self._config['statistics']['set']['params']) for i in range(set_agents_fitnesses.shape[1])]
        sets_fitness = np.array(sets_fitness).flatten()
        trees_roots = [tree.get_root() for tree in agents_tree[0]] # all agents in the team have the same root
        self._me.tell(sets_fitness, trees_roots) 
        self._me.plot_archive(gen)       
        
        for i in range(self._number_of_teams):
            team = []
            for j in range(self._number_of_agents):
                team.append(agents_fitness[j][i])
            teams_fitness.append(team)                
        
        return agents_fitness, teams_fitness
    
    def run_experiment(self):
        return super().run_experiment()


class Experiment_Me_Per_Team(Experiment):
    def __init__(self, config, extra_log=False, debug=False, manual_policy=False):
        super().__init__(config, extra_log, debug, manual_policy)

    def set_experiment(self):
        print_configs("Experiment type:", "ME - ME per team")
        self._me = [ None for _ in range(self._number_of_teams)]
        self._coach = [ None for _ in range(self._number_of_teams)]
        for n in range(self._number_of_teams):
            self._me[n] = self.set_map_elite(n)
            self._coach[n] = self.set_coach(self._me_config, self._me[n]) 
            self._coach[n].set_n_teams(1)  
        return self._me, self._coach

    def set_teams(self, gen):
        if self._selection_type == "coach" and gen >= 0:
                for i in range(self._number_of_teams):
                    me_pop = self._coach[i].ask()
                    trees = [RLDecisionTree(me_pop[0][i], self._config["training"]["gamma"]) for i in range(len(me_pop[0]))]
                    self._teams.append(trees)
        else:
            for i in range(self._number_of_teams):
                me_pop = self._me[i].ask()
                trees = [RLDecisionTree(me_pop[i], self._config["training"]["gamma"]) for i in range(len(me_pop))]
                self._teams.append(trees)
        return self._teams

    def tell_me(self, return_values, gen):
        agents_fitness = [ [] for _ in range(self._number_of_agents)]
        agents_tree = [ [] for _ in range(self._number_of_agents)]

        # Store trees and fitnesses
        for values in return_values:
            for i in range(self._number_of_agents):
                agents_fitness[i].append(values[0][i])
                agents_tree[i].append(values[1][i])

        amax = [np.argmax(agents_fitness[i]) for i in range(self._number_of_agents)]
        max_ = [agents_fitness[i][amax[i]] for i in range(self._number_of_agents)]

        for i in range(self._number_of_agents):
            if max_[i] > self._best_fit[i]:
                self._best_fit[i] = max_[i]
                self._best[i] = agents_tree[i][amax[i]]
                self._new_best[i] = True

                tree_text = f"{self._best[i]}"
                utils.save_tree(self._best[i], self._trees_dir, f"best_agent_{i}")
                with open(os.path.join(self._trees_dir, f"best_agent_{i}.log"), "w") as f:
                    f.write(tree_text)
        # Compute stats for the bests
        best_min = np.min(self._best_fit)
        best_mean = np.mean(self._best_fit)
        best_max = np.max(self._best_fit)
        best_std = np.std(self._best_fit)

        with open(os.path.join(self._evolution_dir, f"bests.txt"), "a") as f:
                f.write(f"{gen},{best_min},{best_mean},{best_max},{best_std}\n")
        
        plot_log(self._evolution_dir, self._plot_dir, "bests",f"bests.txt", gen)


        teams_fitness = []
        teams_trees = []

        for i in range(self._number_of_teams):
                team = []
                team_trees = []
                team_roots = []
                for j in range(self._number_of_agents):
                    team.append(agents_fitness[j][i])
                    team_trees.append(agents_tree[j][i])
                    team_roots.append(agents_tree[j][i].get_root())
                teams_fitness.append(team)
                teams_trees.append(team_trees)
                team_roots = [t.get_root() for t in team_trees]
                self._me[i].tell(team, team_roots)
                self._me[i].plot_archive(gen)    
            
        teams_fitness = [np.mean(team) for team in teams_fitness]
        return agents_fitness, teams_fitness
    
    def run_experiment(self):
        return super().run_experiment()