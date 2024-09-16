import argparse
import json
import shutil
import os
import utils
from training.evaluations import *

def test_all_teams(config):
    experiment_path = config["experiment_path"]
    selection_type = os.listdir(experiment_path)
    for type_ in selection_type:
            selection_type_path = os.path.join(experiment_path, type_, "magent_battlefield")
            if not os.path.isdir(experiment_path):
                continue
            else:
                runs = os.listdir(selection_type_path)
                for run in runs:
                    run_path = os.path.join(selection_type_path, run)
                    if not os.path.isdir(run_path):
                        continue
                    else:
                        log_path = f"logs/qd-marl/test/{type_}/{run}"
                        print_configs("Logs path: ", log_path)
                        join = lambda x: os.path.join(log_path, x)
                        config["log_path"] = log_path
                        config["generation"] = 0
                        os.makedirs(log_path, exist_ok=False)
                        shutil.copy(args.config, join("config.json"))

                        team_path = os.path.join(run_path, "Trees_dir")
                        team_file = os.listdir(team_path)
                        print(team_file)
                        team = []
                        for file in team_file:
                            if file.endswith(".pickle"):
                                if file.startswith("best_agent_"):
                                    tree_path = os.path.join(team_path, file)
                                    tree = get_tree(tree_path)
                                    team.append(tree)
                        test = evaluate(team, config)
    return "Finished testing all teams"

def test_single_team(config):
    team = []
    experiment_path = config["experiment_path"]
    team_path = os.path.join(experiment_path, "Trees_dir")
    team_file = os.listdir(team_path)
    log_path = f"logs/qd-marl/test/{experiment_path}"
    config["log_path"] = log_path
    for file in team_file:
        if file.endswith(".pickle"):
            if file.startswith("best_agent_"):
                tree_path = os.path.join(team_path, file)
                tree = get_tree(tree_path)
                team.append(tree)
    test = evaluate(team, config)
    return "Finished testing single team"


if __name__ == "__main__":

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
    config['args_seed'] = args.seed
    config["original_config"] = args.config
    config["generation"] = 0
    
    test_single_team(config)
    # test_all_teams(config)
