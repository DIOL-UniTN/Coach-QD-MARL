#!/usr/bin/bash
echo "Hello! Here you can set environment and run codes"
echo "Please enter an integer to select an option:"
echo "[1]. Activate environment"
echo "[2]. Deactivate environment"
echo "[3]. Run code GA baseline"
echo "[4]. Run code qd_marl with fully coevolutionary"
echo "[5]. Run code qd_marl with a single ME"
echo "[6]. Run test final team with render mode"
echo "[7]. Plot evaluations"
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
    python3 src/marl_qd_launcher.py src/configs/local/ga_fc/battlefield.json 4
elif [ $option -eq 4 ]
then 
    echo "Running code MARL-QD with a ME per team.."
    python3 src/marl_qd_launcher.py src/configs/local/fully_coevolutionary/battlefield.json 4
elif [ $option -eq 5 ]
then
    echo "Running code MARL-QD with a single ME..."
    python3 src/marl_qd_launcher.py src/configs/local/single_me/battlefield.json 4
elif [ $option -eq 6 ]
then
    echo "Running test final team.."
    python3 src/test_team.py src/configs/test_team.json 4
elif [ $option -eq 7 ]
then
    echo "Plot evaluations.."
    python3 src/eval_runs.py 4
else
    echo "Invalid option"
    echo "Exiting..."
    exit
fi
```
```