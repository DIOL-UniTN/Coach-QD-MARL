#!/bin/bash

#resources allcoation
#PBS -l select=1:ncpus=12:mem=100gb -l place=pack:excl

#set max execution time
#PBS -l walltime=6:00:00

#execution queue configs
#PBS -q short_cpuQ

#execution outpust name
#PBS -N marl_battlefield_hpc

#set mail notification
#PBS -M erik.nielsen@studenti.unitn.it

cd ${PBS_O_WORKDIR}

module load python-3.8.13
source $PWD/pyenv_hpc/bin/activate
python $PWD/src/QD_MARL/marl_qd_launcher_no_sets.py $PWD/src/QD_MARL/configs/hpc/no_sets/battlefield_hpc_pyribs_best.json 4