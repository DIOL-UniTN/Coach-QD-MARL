#!/bin/bash

#PBS -l select=2:ncpus=10:mem=64gb

#set max execution time
#PBS -l walltime=4:00:00

#imposta la coda di esecuzione
#PBS -q common_cpuQ

source pyenv-marl-qd/bin/activate
python3 python3 src/QD_MARL/marl_qd_launcher.py src/QD_MARL/configs/battlefield.json 4