#!/bin/bash

cd ${PBS_O_WORKDIR}/MARL-QD/Marl-QD_Private/
for file in $(ls $PWD/hpc_scripts/with_set/); do
    qsub $PWD/hpc_scripts/with_set/$file
done