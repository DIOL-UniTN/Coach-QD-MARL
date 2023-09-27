import gym
import sys
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
N_TRIALS = 25000

def evaluate():
    values = []
    e = gym.make("LunarLander-v2")
    _ = e.reset()
    done = False
    while not done:
        obs, _, done, _ = e.step(e.action_space.sample())
        values.append(obs)
    return values

if __name__ == "__main__":
    results = np.array(Parallel(4)(delayed(evaluate)() for _ in tqdm(range(N_TRIALS))))
    values = []
    for r in results:
        values.extend(r)
    values = np.array(values)
    with open("ll.txt", "w") as f:
        for i in range(len(values[0])):

            hist, intervals = np.histogram(values[:, i])
            hist = hist / sum(hist)
            hist = [f"{h:.2f}" for h in hist]
            f.write(f"Feature {i}: \n {hist}; \n{intervals}".replace("\n", ""))
            f.write(str(i) + " min  " + str(min(values[:, i])) + " max " + str(max(values[:, i]))+"\n")
            f.write(str(i) + "  mean " + str(np.mean(values[:, i])) + " std " + str(np.std(values[:, i])) + "\n")
            f.write(str(i) + "   " + str(np.quantile(values[:, i], 0.25)) + "   " + str(np.quantile(values[:, i], 0.75))+"\n")
            f.write("------------------\n")
            plt.figure()
            plt.hist(values[:, i], bins=100)
            plt.title("Feature {}".format(i))
            plt.savefig("leoBoundMo"+str(i)+".png", dpi=400)