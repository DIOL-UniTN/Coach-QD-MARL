import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("config", help="Path of the config file to use")
args = parser.parse_args()

# Load the config file
config = json.load(open(args.config))

# Load the template file
for file_name in config["files_names"]:
    new_configs = config["template"]

    names= file_name.replace(".json", "").split("_")
    print(names)

    new_configs["me_config"]["me"]["kwargs"]['selection_type'] = names[-1]
    if names[-2] == "pyribsCMA":
        new_configs["me_config"]["me"]["kwargs"]['me_type'] = "MapElitesCMA_pyRibs"
    elif names[-2] == "pyribs":
        new_configs["me_config"]["me"]["kwargs"]['me_type'] = "MapElites_pyRibs"
    else:
        raise ValueError("Unknown ME type")
    # Save the new configs in a JSON file
    if new_configs['hpc'] == True:
        new_config_file = os.path.join("src/QD_MARL/configs/hpc",file_name)
    else:
        new_config_file = os.path.join("src/QD_MARL/configs/local","test_config.json")
    with open(new_config_file, "w") as f:
        json.dump(new_configs, f, indent=4)
    print(f"New configs saved in {new_config_file}")