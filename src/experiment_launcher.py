
import os
import sys
import pickle
sys.path.append(".")

import numpy as np
import pettingzoo
from agents.agents import *
from algorithms import (
    map_elites_Pyribs,
    mapElitesCMA_pyRibs,
    genetic_algorithm
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
    elif config["experiment"] == "me-fully_coevolutionary":
        experiment = Experiment_Fully_Coevolutionary(config, extra_log, debug, manual_policy)
    elif config["experiment"] == "ga-baseline":
        experiment = Experiment_GA(config, extra_log, debug, manual_policy)
    else:
        raise ValueError(f"Unknown experiment type: {config['experiment']}")
    return experiment

class Experiment():
    """
    class implemets the genral structure of the experiments
    """
    def __init__(self, config, extra_log=False, debug=False, manual_policy=False):
        
        """
        constructor of the class
        :param config: the configuration file
        :param extra_log: if True, the logs will be saved in the logs folder
        :param debug: if True, the logs will be saved in the logs folder
        :param manual_policy: if True, the logs will be saved in the logs folder
        """
        self._alg_type = None
        self._config = config
        if "me_config" in self._config:
            self._alg_type = "ME"
            self._alg_config = self._config["me_config"]['me']["kwargs"]
        elif "ga_config" in self._config:
            self._alg_type = "GA"
            self._alg_config = self._config["ga_config"]['ga']["kwargs"]
        else:
            raise ValueError("Unknown algorithm configuration")
        
        
        self._log_path = self.set_log_path()
        self._extra_log = extra_log
        self._debug = debug
        self._manual_policy = manual_policy
        
        
        # alg_config: the configuration of the algorithm,
        # could be MAP-Elites or Genetic Algorithm
        
        self._alg_config["log_path"] = self._log_path  
        self._number_of_agents = self._config["n_agents"]
        self._number_of_teams = self._config["training"]["jobs"]
        self._selection_type = self._alg_config["selection_type"]
        self._map = utils.get_map(self._number_of_teams, debug)

        # ME 
        self._alg = None
        self._coach = None
        self._alg_pop = []
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
        if self._alg_type == "ME":
            if self._config["hpc"]:
                cwd = os.getcwd()
                log_path = f"{cwd}/logs/qd-marl/hpc/{self._config['experiment']}/{self._config['me_config']['me']['kwargs']['me_type']}/{self._config['me_config']['me']['kwargs']['selection_type']}/magent_battlefield/{logdir_name}"
            else:
                log_path = f"logs/qd-marl/local/{self._config['experiment']}/{self._config['me_config']['me']['kwargs']['me_type']}/{self._config['me_config']['me']['kwargs']['selection_type']}/magent_battlefield/{logdir_name}"
        elif self._alg_type == "GA":
            if self._config["hpc"]:
                cwd = os.getcwd()
                log_path = f"{cwd}/logs/qd-marl/hpc/{self._config['experiment']}/{self._config['ga_config']['ga']['kwargs']['me_type']}/{self._config['ga_config']['ga']['kwargs']['selection_type']}/magent_battlefield/{logdir_name}"
            else:
                log_path = f"logs/qd-marl/local/{self._config['experiment']}/{self._config['ga_config']['ga']['kwargs']['me_type']}/{self._config['ga_config']['ga']['kwargs']['selection_type']}/magent_battlefield/{logdir_name}"
        
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
        self._alg_dir = os.path.join(self._log_path, "Algorithm")
        os.makedirs(self._alg_dir , exist_ok=False)

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
                    
    def set_algorithm(self, n=None):
        if self._alg_type == "ME":
            alg = self.set_map_elite(n)
        elif self._alg_type == "GA":
            alg = self.set_genetic_algorithm(n)
        else:
            raise ValueError(f"Unknown algorithm type: {self._alg_type}")
        return alg

    def set_genetic_algorithm(self, n=None):
        factories_config = self._config["ga_config"]
        # Setup GA
        if n is not None:
            self._alg_config["log_path"] = self._alg_config["log_path"]+f"/team_{n}"
        else:
            pass
        # Build classes of the operators from the config file
        self._alg_config["c_factory"] = ConditionFactory(factories_config["ConditionFactory"]["type"])
        self._alg_config["l_factory"] = QLearningLeafFactory(
            factories_config["QLearningLeafFactory"]["kwargs"]["leaf_params"],
            factories_config["QLearningLeafFactory"]["kwargs"]["decorators"],
        )
        ga = genetic_algorithm.GeneticAlgorithm(**self._alg_config)
        print_configs("Genetic algorithm - Baseline")            
        return ga

    def set_map_elite(self, n=None):

        factories_config = self._config["me_config"]
        # Setup MAp Elite
        if n is not None:
            self._alg_config["log_path"] = self._alg_config["log_path"]+f"/team_{n}"
        else:
            pass
        # Build classes of the operators from the config file
        self._alg_config["c_factory"] = ConditionFactory(factories_config["ConditionFactory"]["type"])
        self._alg_config["l_factory"] = QLearningLeafFactory(
            factories_config["QLearningLeafFactory"]["kwargs"]["leaf_params"],
            factories_config["QLearningLeafFactory"]["kwargs"]["decorators"],
        )
        if self._alg_config["me_type"] == "MapElites_pyRibs":
            me = map_elites_Pyribs.MapElites_Pyribs(**self._alg_config)
        elif self._alg_config["me_type"] == "MapElitesCMA_pyRibs":
            me = mapElitesCMA_pyRibs.MapElitesCMA_pyRibs(**self._alg_config)
        else:
            raise ValueError(f"Unknown ME type: {self._alg_config['me_type']}")
        print_configs("ME type:", self._alg_config["me_type"])            
        print_configs("ME selection type:", self._alg_config["selection_type"])
        return me
    
    def get_alg(self):
        return self._alg
    
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
    def set_experiment(self, n=None):
        pass
    
    @abc.abstractmethod
    def set_teams(self, gen):
        pass
    
    def get_teams(self):
        return self._teams    
    
    @abc.abstractmethod
    def tell_alg(self, return_values, gen):
        pass
    
    def save_algorithm(self):
        if self._alg_type == "ME":
            self.save_me()
        elif self._alg_type == "GA":
            self.save_ga()
        else:
            raise ValueError(f"Unknown algorithm type: {self._alg_type}")
    
    def save_me(self):            
        if type(self._alg) == list:
            for i in range(len(self._alg)):
                name = f"me_{i}"
                log_file = os.path.join(self._alg_dir, name + ".pickle")
                with open(log_file, "wb") as f:
                    pickle.dump(self._alg[i], f)
        else:
            name = "me"
            log_file = os.path.join(self._alg_dir, name + ".pickle")
            with open(log_file, "wb") as f:
                pickle.dump(self._alg, f)
                
    def save_ga(self):
        pass            
    
    def run_experiment(self):

        self._alg, self._coach = self.set_experiment()
         
        print_info(f"{'Generation' : <10} {'Set': <10} {'Min': <10} {'Mean': <10} {'Max': <10} {'Std': <10}")

        for gen in range(-1, self._config["training"]["generations"]):
            self._config["generation"] = gen
            self._alg_pop = []
            self._teams = []
            self._teams = self.set_teams(gen)

            # Running training environment and evaluating the fitness
            return_values = self._map(evaluate, self._teams, self._config)

            agents_fitness, teams_fitness = self.tell_alg(return_values, gen)

            # Compute stats for each agent
            agent_min = [np.min(agents_fitness[i]) for i in range(self._number_of_agents)]
            agent_mean = [np.mean(agents_fitness[i]) for i in range(self._number_of_agents)]
            agent_max = [np.max(agents_fitness[i]) for i in range(self._number_of_agents)]
            agent_std = [np.std(agents_fitness[i]) for i in range(self._number_of_agents)]

            for i in range(self._number_of_agents):
                #print(f"{gen: <10} agent_{i: <4} {agent_min[i]: <10.2f} {agent_mean[i]: <10.2f} {agent_max[i]: <10.2f} {agent_std[i]: <10.2f}")
                with open(os.path.join(self._agent_dir, f"agent_{i}.txt"), "a") as f:
                    f.write(f"{gen},{agent_min[i]},{agent_mean[i]},{agent_max[i]},{agent_std[i]}\n")
            
            # Compute states for each set
            
            team_min = [np.min(teams_fitness[i]) for i in range(self._number_of_teams)]
            team_mean = [np.mean(teams_fitness[i]) for i in range(self._number_of_teams)]
            team_max = [np.max(teams_fitness[i]) for i in range(self._number_of_teams)]
            team_std = [np.std(teams_fitness[i]) for i in range(self._number_of_teams)]
            
            for i in range(self._number_of_teams):
                print(f"{gen: <10} set_{i: <4} {team_min[i]: <10.2f} {team_mean[i]: <10.2f} {team_max[i]: <10.2f} {team_std[i]: <10.2f}")
                with open(os.path.join(self._team_dir, f"team_{i}.txt"), "a") as f:
                    f.write(f"{gen},{team_min[i]},{team_mean[i]},{team_max[i]},{team_std[i]}\n")

            
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
            self.save_algorithm()
        return self._best, self._best_team
    
        
         
class Experiment_Single_ME(Experiment):
    def __init__(self, config, extra_log=False, debug=False, manual_policy=False):
        super().__init__(config, extra_log, debug, manual_policy)

    def set_experiment(self):
        print_configs("Experiment type:", "ME - single ME")
        self._alg = self.set_map_elite()
        self._coach = self.set_coach(self._alg_config, self._alg)
        if self._selection_type == "coach":
            self._coach.set_n_teams(self._number_of_teams)
        return self._alg, self._coach

    def set_teams(self, gen):
        if gen >= 0:
                if self._selection_type == "coach":
                    me_teams = self._coach.ask()
                    for team in me_teams:
                        trees = [RLDecisionTree(team[i], self._config["training"]["gamma"]) for i in range(len(team))]
                        self._teams.append(trees)
                else:
                    for i in range(self._number_of_teams):
                        me_pop = self._alg.ask()
                        trees = [RLDecisionTree(me_pop[i], self._config["training"]["gamma"]) for i in range(len(me_pop))]
                        self._teams.append(trees)            
        else:
            me_pop = self._alg.ask()
            trees = [RLDecisionTree(me_pop[i], self._config["training"]["gamma"]) for i in range(len(me_pop))]
            j = 0
            for i in range(self._number_of_teams):
                if j*self._number_of_agents >= len(trees):
                    j =0
                self._teams.append(trees[j*self._number_of_agents:(j+1)*self._number_of_agents])
                j+=1 
        return self._teams

    def tell_alg(self, return_values, gen):
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
        self._alg.tell(individual_fitness, individuals_roots)
        self._alg.plot_archive(gen)    

        teams_fitness = [np.mean(team) for team in teams_fitness]
        return agents_fitness, teams_fitness
    
    def run_experiment(self):
        return super().run_experiment()
    
class Experiment_Fully_Coevolutionary(Experiment):
    """
    Fully coevolutionary experiment
    In this experiment, are created #ME = #n_agents, then the batch size is equal to the number of teams,
    each created by taking an individual from each ME
    _summary_

    Args:
        Experiment (_type_): _description_

    Returns:
        _type_: _description_
    """
   
    def __init__(self, config, extra_log=False, debug=False, manual_policy=False):
        super().__init__(config, extra_log, debug, manual_policy)

    def set_experiment(self):
        print_configs("Experiment type:", "ME - ME per team")
        self._alg = [ None for _ in range(self._number_of_agents)]
        self._coach = [ None for _ in range(self._number_of_agents)]
        for n in range(self._number_of_agents):
            self._alg[n] = self.set_map_elite(n)
            if self._selection_type == "coach":
                self._coach[n] = self.set_coach(self._alg_config, self._alg[n]) 
                self._coach[n].set_n_teams(1)
        return self._alg, self._coach
    
    def set_teams(self, gen):
        temp_teams = []
        if gen >= 0 and self._selection_type == "coach":
            for i in range(self._number_of_agents):
                me_pop = self._coach[i].ask()
                trees = [RLDecisionTree(me_pop[0][i], self._config["training"]["gamma"]) for i in range(len(me_pop[0]))]
                temp_teams.append(trees)
        else:
            for i in range(self._number_of_agents):
                me_pop = self._alg[i].ask()
                trees = [RLDecisionTree(me_pop[i], self._config["training"]["gamma"]) for i in range(len(me_pop))]
                temp_teams.append(trees)
        teams = [[None for _ in range(self._number_of_agents)] for _ in range(self._number_of_teams)]
        for j in range(self._number_of_agents):
            for i in range(self._number_of_teams):
                teams[i][j] = temp_teams[j][i]
        self._teams = teams
        return self._teams
    
    def tell_alg(self, return_values, gen):
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
            # Tell the ME the fitness of the individuals
            trees = [agents_tree[i][j].get_root() for j in range(self._number_of_teams)]
            self._alg[i].tell(agents_fitness[i], trees)
            self._alg[i].plot_archive(gen)
            
            # Check if the new individual is better than the previous best
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
            
        teams_fitness = [np.mean(team) for team in teams_fitness]
        return agents_fitness, teams_fitness
    
    def run_experiment(self):
        return super().run_experiment()
    
    
class Experiment_GA(Experiment):
    def __init__(self, config, extra_log=False, debug=False, manual_policy=False):
        super().__init__(config, extra_log, debug, manual_policy)

    def set_experiment(self):
        print_configs("Experiment type:", "GA - Baseline")
        self._alg = [ None for _ in range(self._number_of_agents)]
        self._coach = [ None for _ in range(self._number_of_agents)]
        for n in range(self._number_of_agents):
            self._alg[n] = self.set_algorithm(n)
        return self._alg, self._coach
    
    def set_teams(self, gen):
        temp_teams = []
        for i in range(self._number_of_agents):
            ga_pop = self._alg[i].ask()
            trees = [RLDecisionTree(ga_pop[i], self._config["training"]["gamma"]) for i in range(len(ga_pop))]
            temp_teams.append(trees)
        teams = [[None for _ in range(self._number_of_agents)] for _ in range(self._number_of_teams)]
        for j in range(self._number_of_agents):
            for i in range(self._number_of_teams):
                teams[i][j] = temp_teams[j][i]
        self._teams = teams
        return self._teams
    
    def tell_alg(self, return_values, gen):
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
            # Tell the ME the fitness of the individuals
            trees = [agents_tree[i][j].get_root() for j in range(self._number_of_teams)]
            self._alg[i].tell(agents_fitness[i], trees)
            
            # Check if the new individual is better than the previous best
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
            
        teams_fitness = [np.mean(team) for team in teams_fitness]
        return agents_fitness, teams_fitness
    
    def run_experiment(self):
        return super().run_experiment()
    
    
    
