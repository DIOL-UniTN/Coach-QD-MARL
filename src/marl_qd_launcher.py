if __name__ == "__main__":
    import argparse
    import json
    import shutil
    import os
    import utils
    from utils.print_outputs import *
    from experiment_launcher import *

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
    
    experiment = set_type_experiment(config, args.log, args.debug)

    bests, best_team = experiment.run_experiment()

    print_info("Best individuals")
    for player in bests:
        print_info("\n", player)
    
    print_info("Best team")
    for player in best_team:
        print_info("\n", player)