import os
import sys
import matplotlib.pyplot as plt
sys.path.append(".")
from math import sqrt
from test_environments import *
import numpy as np


if __name__ == "__main__":
    
    import argparse
    import json
    import shutil
    import yaml
    import utils
    # the log file si a csv file with the following format:
    # <gen> <Min> <Mean> <Max> <Std>
    
    
    df = pd.read_csv(log_file)
    df = df.sort_values(by=['Generation'])
    figure, ax= plt.subplots()
    ax.plot(df['Generation'].to_list(), df['Min'].to_list(), label='Min')
    ax.errorbar(df['Generation'].to_list(), df['Mean'].to_list(), yerr = df["Std"].to_list() , label='Mean')
    ax.plot(df['Generation'].to_list(), df['Max'].to_list(), label='Max')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness over generations')
    ax.legend()
    ax.grid(True)
    plt.savefig(os.path.join(os.path.dirname(log_file), "_fitness.png"))
    plt.show()