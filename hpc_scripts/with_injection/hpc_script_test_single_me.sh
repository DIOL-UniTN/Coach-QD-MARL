#!/bin/bash

#PBS -l select=6:ncpus=10:mem=15gb

#set max execution time
#PBS -l walltime=24:00:00

#imposta la coda di esecuzione
#PBS -q common_cpuQ

#execution outpust name
#PBS -N w_sing_r

#set mail notification
#PBS -M erik.nielsen@studenti.unitn.it


cd ${PBS_O_WORKDIR}

module load python-3.8.13
source $PWD/pyenv_hpc/bin/activate
python $PWD/src/marl_qd_launcher.py $PWD/src/configs/hpc/with_injection/single_me/battlefield_hpc_pyribs_random.json $PBS_ARRAY_INDEX
