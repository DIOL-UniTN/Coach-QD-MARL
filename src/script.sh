#!/usr/bin/bash
echo "Hello! Here you can set environment and run codes"
echo "Please enter an integer to select an option:"
echo "[1]. Activate environment"
echo "[2]. Deactivate environment"
echo "[3]. Run code dts4marl"
echo "[4]. Run code marldts"
echo "[5]. Run code qd_marl"
echo "[6]. Exit"
read option
if [ $option -eq 1 ]
then
    echo "Activating environment..."
    source pyenv-marl-qd/bin/activate
elif [ $option -eq 2 ]
then
    echo "Deactivating environment..."
    deactivate
elif [ $option -eq 3 ]
then
    echo "Running code..."
    python3 src/base/dts4marl/launcher.py src/base/dts4marl/battlefield.json 4

elif [ $option -eq 4 ]
then
    echo "Running code..."
    python3 src/base/marl_dts/src/experiment_launchers/pz_advpursuit_reduced_obs_shared_launcher.py src/base/marl_dts/src/configs/magent_advpursuit_single_team.json 1
elif [ $option -eq 5 ]
then
    echo "Running code..."
    python3 src/QD_MARL/marl_qd_launcher.py src/QD_MARL/configs/battlefield.json 4

elif [ $option -eq 6 ]
then
    echo "Exiting..."
    exit
else
    echo "Invalid option"
    echo "Exiting..."
    exit
fi
```
```