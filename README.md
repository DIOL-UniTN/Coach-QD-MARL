# MultiAgent and QualityDiversity ReinforcementLearning
This repository contains all the main informations about the university and thesis project on Multi Agent Reinforcemnt Learning and Quality Diversity.
The project is supervised by Giovanni Iacca and Andrea Ferigo from University of Trento and follows their current researches.

## Papers and References
In [references](/references) there is a comprehensive list of references of the studied papers to complete the project.

## Source codes
In [src](/src) are stored all the scripts developed during the project. The produced scripts are based and continue the work developed in the following papers by Giovanni Iacca, Marco Crespi, Andrea Ferigo, Leonardo Lucio Custode:
- [A Population-Based Approach for Multi-Agent Interpretable Reinforcement Learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4467882)
- [Quality Diversity Evolutionary Learning of Decision Trees](https://arxiv.org/abs/2208.12758)

It is possible to run the aforementioned code by following the instructions in the README.md file in the [src/base](/src/base/) folder.
Otherwise by changing the working direcotry `cd src` and running the following command:
`chmod +x script.sh`
`source script.sh`
On the terminal output will appear the following menu:
`Hello! Here you can set environment and run codes
Please enter an integer to select an option:`
[1]. Activate environment
[2]. Deactivate environment
[3]. Run code dts4marl
[4]. Run code marldts
[5]. Run code qd_marl
[6]. Run code qd_marl in debug mode
[7]. Run test environment
[8]. Exit`
Press 1 to activate the python venv.
Then run `./script.sh` again and select one of the possible experiment.
If 3 or 4 is selected it will run the projects developed by Giovanni Iacca, Marco Crespi, Andrea Ferigo, Leonardo Lucio Custode.
if 5 or 6 (for debug and serialized mode) is selected it will run the project developed in this repository which apply a Quality Diversity approach to a Multi Agent Reinforcement Learning task.