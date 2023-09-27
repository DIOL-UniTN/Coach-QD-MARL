import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
import gym

from decisiontrees.conditions import ObliqueCondition
from util_processing_elements import OneHotEncoder


class Node:
    def __init__(self, condition):
        self.condition = condition
        self.left = None
        self.right = None

    def print(self, ind=0):
        if self.left is not None:
            string = f"{' ' * ind}if {self.condition}:\n" + self.left.print(
                ind + 4) + f"\n{' ' * ind}else:\n" + self.right.print(ind + 4)
            return string
        else:
            return " " * ind + f"out={str(self.condition)}"


def convert(string):
    root = None
    nodes = {}
    for l in string.split("\n"):
        if len(l) == 0:
            continue
        if "-->" in l:
            from_, branch, to = l.split(" ")
            if "True" in branch:
                nodes[from_].left = nodes[to]
            else:
                nodes[from_].right = nodes[to]
        else:
            id_, value = l.split(" [")
            value = value[:-1]

            if "visits" in value:
                pass
            nodes[id_] = Node(value)
            if len(nodes) == 1:
                root = nodes[id_]
    return root


mermaid = """
47627551707344[gt(input_[0], input_[5])]
47627784518096[2 (0.0 visits)]
47627551707344 -->|True| 47627784518096
47627551707824[lt(0.34142283463949896, mul(add(safediv(mul(input_[4], mul(input_[2], input_[3])), input_[7]), mul(-0.8580900404765415, 0.9759670513013132)), add(add(sub(input_[3], safediv(input_[6], safediv(-0.9950218748223119, input_[4]))), input_[0]), -0.7449174898974631)))]
47627551707344 -->|False| 47627551707824
47627784518192[0 (0.0 visits)]
47627551707824 -->|True| 47627784518192
47627784518000[3 (0.0 visits)]
47627551707824 -->|False| 47627784518000
"""


def makeVideo(envName):
    import pickle
    fls = os.listdir("logs/MountainCar-ME_pyRibs_7/29-05-2022_18-27-08_jlnjexmc/")
    for fl in fls:
        if "pkl" and "best_gen_" in fl:
            os.mkdir("video/" + fl.split(".")[0])
            fr = open("logs/MountainCar-ME_pyRibs_7/29-05-2022_18-27-08_jlnjexmc/" + fl, "rb")
            tree = pickle.load(fr)
            # print(tree.get_output())

            import gym
            env = gym.make(envName)
            path_of_video_with_name = "./video/" + fl.split(".")[0] + "/" + fl.split(".")[0] + ".mp4"

            # env =gym.make("MountainCar-v1")
            o = env.reset()
            video_recorder = None
            video_recorder = VideoRecorder(env, path_of_video_with_name, enabled=True)
            d = False
            while not d:
                env.render()
                video_recorder.capture_frame()
                s = tree.get_output(o)
                o, r, d, i = env.step(np.argmax(s))

            video_recorder.close()
            video_recorder.enabled = False
            env.close()


def normalizeObservation1(obs, envName):
    bounds = {"LunarLander-v2":
        [
            [-1., -0.25, -2., -2., -2., -0.5, 0., 0.],
            [1., 1.5, 2., 0.5, 2., 0.5, 1., 1.]
        ],
        "MountainCar-v0": [
            [-1.2, -0.07],
            [0.7, 0.07]
        ]
    }
    # print(envName)
    if not envName in bounds.keys():
        return obs

    norm_obs = []
    for i in range(len(obs)):
        norm_obs.append(((obs[i] - bounds[envName][0][i]) / (bounds[envName][1][i] - bounds[envName][0][i])))

    return np.array(norm_obs)


def normalizeObservation(obs, envName):
    bounds = {"LunarLander-v2":
        [
            [-0.153, 0.601, -0.501, -1.054, -0.185, -0.138, 0., 0.],
            [0.150, 1.364, 0.477, -0.302, 0.197, 0.146, 1., 1.]
        ],
        "MountainCar-v0": [
            [-1.2, -0.07],
            [0.7, 0.07]
        ]
    }

    if not envName in bounds.keys():
        return obs
    norm_obs = []
    for i in range(len(obs)):
        norm_obs.append(((obs[i] - bounds[envName][0][i]) / (bounds[envName][1][i] - bounds[envName][0][i])))

    return np.array(norm_obs)


def calcFitness(tree, envName, seeds, video=False):
    tmp = list()
    acs = np.zeros(4)
    for seed in seeds:
        env = gym.make(envName)
        # env =gym.make("MountainCar-v1")
        env.seed(seed)
        o = env.reset()
        o = normalizeObservation1(o, envName)
        d = False
        re = 0

        while not d:
            if video:
                env.render()

            actio = tree.get_output(o)
            # ifo = iftree(o)
            # acs[ifo] += 1
            # print(str(actio) + "    " + str(ifo) + "   " + str(actio == ifo))
            # print("### "+str(actio))
            o, r, d, i = env.step(actio)
            # o = normalizeObservation1(o, envName)
            re += r

        tmp.append(re)
        env.close()
    # print(acs)
    return tmp


def check_fitness(config):
    from tasks import fitness_function
    import pickle
    fr = open("logs/LunarLander-CMA_0/15-06-2022_15-48-42_rdmvdkxx/best_gen_75.pkl", "rb")
    tree = pickle.load(fr)
    fit = fitness_function(tree, config)
    return fit


def observe_space(envName):
    env = gym.make(envName)
    observations = [list() for i in range(8)]
    for seed in range(10000):
        env = gym.make(envName)
        env.seed(seed)
        o = env.reset()
        d = False
        re = 0
        while not d:
            # env.render()
            actio = env.action_space.sample()
            o, r, d, i = env.step(actio)
            for i in range(8):
                observations[i].append(o[i])
    return observations


def get_logdir_name():
    """
    Returns a name for the dir
    :returns: a name in the format 'dd-mm-yyyy_:mm:ss_<random string>'
    """
    time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    rand_str = "".join(np.random.choice([*string.ascii_lowercase], 8))
    return f"{time}_{rand_str}"


def iftree(i):
    '''

    print(input)
    if input[0] * 0.717 - 0.697 * input[1] < -0.229:
        print("first")
        return 2
    elif input[0] * 0.138 - 0.883 * input[1] < -0.389:
        print("second")
        return 2
    else:
        print("third")
        return 0
    '''
    if 0.401 * i[0] - 0.104 * i[1] + 0.495 * i[2] - 0.055 * i[3] - 0.69 * i[4] - 0.84 * i[5] - 0.2 * i[6] - 0.597 * i[
        7] < 0:
        if 0.448 * i[0] - 0.366 * i[1] + 0.431 * i[2] - 0.462 * i[3] - 0.693 * i[4] - 0.821 * i[5] + 0.461 * i[
            6] - 0.132 * i[7] < 0:
            return 1
        elif -0.101 * i[0] + 0.133 * i[1] - 0.791 * i[2] + 0.653 * i[3] - 0.207 * i[4] + 0.731 * i[5] + 0.068 * i[
            6] + 0.525 * i[7] < 0:
            return 2
        elif 0.12 * i[0] - 0.044 * i[1] - 0.772 * i[2] - 0.136 * i[3] - 0.169 * i[4] + 0.821 * i[5] - 0.573 * i[
            6] - 0.251 * i[7] < 0:
            return 0
        else:
            return 2
    else:
        return 3


def leoBound():
    pass


def tt(tree):
    root = tree._root
    """BFS search"""
    node = root
    fringe = [(0, node)]
    max_ = 0
    lf = ConstantLeafFactory(4)
    # ("interesting")
    while len(fringe) > 0:
        d, n = fringe.pop(0)
        # print(type(node))
        if isinstance(node, Leaf) or \
                n is None:
            continue

        if d > max_:
            max_ = d

        if not isinstance(n, Leaf):
            if isinstance(n.get_left(), Leaf):
                ind = int(str(n.get_left()).split(" ")[0])
                #print(ind)
                n.set_left(lf.create(ind))
            if isinstance(n.get_right(), Leaf):
                ind = int(str(n.get_right()).split(" ")[0])
                #print(ind)
                n.set_right(lf.create(ind))
            fringe.append((d + 1, n.get_left()))
            fringe.append((d + 1, n.get_right()))

    return root


def writeData(fn, data, size=10):
    with open(fn, "w") as f:
        hd = "x y"
        for k in data.keys():
            hd += " " + str(k)
        f.write(hd + "\n")
        for i in range(size):
            for j in range(size):
                l = str(i) + " " + str(j)
                for k in data.keys():
                    l += " {:3.2f}".format(data[k][j, i])
                f.write(l + "\n")


def avgm(ms):
    am = np.full((10, 10), np.nan)
    cm = np.full((10, 10), 0)
    for m in ms:
        for i in range(10):
            for j in range(10):
                if not np.isnan(m[i, j]):
                    # print(m[i,j])
                    if np.isnan(am[i, j]):
                        if i == 0 and j == 0:
                            # print("-- new --")
                            # print(m[i,j])
                            # print("-- end --")
                            pass
                        am[i, j] = m[i, j]
                        cm[i, j] += 1
                    else:

                        if i == 0 and j == 0:
                            pass
                            # print("-- pre --")
                            # print(m[i,j])
                        am[i, j] += m[i, j]
                        if i == 0 and j == 0:
                            pass
                            # print(m[i,j])
                            # print("-- post --")
                        cm[i, j] += 1

    for i in range(10):
        for j in range(10):

            if not np.isnan(am[i, j]):
                am[i, j] /= cm[i, j]
    return am, cm


def maxm(ms):
    am = np.full((10, 10), np.nan)
    for m in ms:
        for i in range(10):
            for j in range(10):
                if am[i, j] < m[i, j] or np.isnan(am[i, j]):
                    am[i, j] = m[i, j]
    return am


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # print(np.__version__)
    # obsers = observe_space("LunarLander-v2")
    # plt.boxplot(obsers)
    # plt.savefig("dist_lunar.png", dpi=800)
    # leoBound()
    ##for i in range(8):
    #   print(str(i) + "   " + str(np.quantile(obsers[i], 0.25)) + "   " + str(np.quantile(obsers[i], 0.75)))
    # print(mx)
    # print(mn)
    # bounds = {"LunarLander-v2": [mn, mx],
    #       "MountainCar-v0": [[-1.2, -0.07], [0.6, 0.07]]}

    import gym
    import cv2
    import string
    import numpy as np
    from tqdm import tqdm
    import os
    from datetime import datetime
    from sklearn.cluster import DBSCAN
    from joblib import Parallel, delayed
    from algorithms import genetic_programming
    from decisiontrees import *
    from time import time
    from tasks import *

    tm = time()
    config = {"Fitness": {
        "episodes": 100,
        "case": "train",
        "seeding": False,
        "seed": 0,
        "name": "GymTask",
        "n_actions": 3,
        "time_window": -1,
        "kwargs": {
            "env_name": "MountainCar-v0"
        }
    }
    }
    from experiment_launchers.pipeline import *

    # print(get_logdir_name())
    # −0.274 x − 0.543 v + −0.904 θ − 0.559 ω < −0.169
    '''
    lf = ConstantLeafFactory(2)
    left = lf.create(2)
    left1 = lf.create(2)
    right1 = lf.create(0)
    right = ObliqueCondition([0.138, -0.883, 0.389], left1, right1)
    root = ObliqueCondition([0.717, -0.697, 0.229], left, right)
    tree = DecisionTree(root)
    '''
    import pickle


    mats = [np.full((10,10), np.nan) for _ in range(5)]
    for i in range(5):
        print("run "+str(i))
        fold = "logs/er cartella/MountainCar-GE-ob_"+str(i)+"/fin/"
        fits = dict()
        with open("logs/er cartella/MountainCar-GE-ob_"+str(i)+"/fin/last.txt") as fitf:
            fitf.readline()
            for l in fitf:
                fits[l.split(" ")[-1].rstrip()] = float(l.split(" ")[-2])


        for fn in os.listdir(fold):
            if fn.endswith("pkl") and not fn.startswith("best"):
                ts = fold+fn

                #print(ts)
                with open(ts, "rb") as fts:
                    tree = pickle.load(fts)
                    if tree is None:
                        print("ehi")
                    #tree = RLDecisionTree(tree, 0)
                    #print(0)
                    pip = Pipeline(tree)
                    #print(1)
                    a = fit_training(pip, config)[0]
                    #print(2)
                    desc = a[1]  # self._get_descriptor(ind, entropy)
                    map_bound = [[0, 10], [0, 1]]
                    map_size = [10, 10]
                    thr = [abs((max(map_bound[i]) - min(map_bound[i])) / map_size[i]) for i in
                           range(len(map_size))]

                    desc = tuple([int((np.clip(desc[i], min(map_bound[i]), max(map_bound[i])) - min(map_bound[i])) / thr[i]) for i in range(len(map_size))])

                    print(str(a[1])+"   "+str(desc))
                    id = fn.split(".")[0]
                    if id in fits.keys():
                        mats[i][desc] = fits[id] if  np.isnan(mats[i][desc]) or mats[i][desc] < fits[id] else mats[i][desc]
                    else:
                        print("ehy")
                        mats[i][desc] = a[0] if np.isnan(mats[i][desc]) or mats[i][desc] < a[0]  else mats[i][desc]
    av, cv = avgm(mats)
    mv = maxm(mats)
    writeData("mc-ge-maps.txt", {"a": av, "m": mv, "c": cv})
    '''
    fold = "logs/er cartella/CartPole-GE-ob_0/08b/best_gen_7.pkl"
    with open(fold, "rb") as f:
        pipeline = pickle.load(f)
        eles = pipeline.get_elements()
        tree = DecisionTree(tt(eles[0]))
        encoder = eles[1]
        pipeline.set_elements((tree, encoder))
        # print(pip.get_elements())
        a = calcFitness(tree, "CartPole-v1", list(range(100)), False)  # fit_training(tree, config)
        print(a)
    '''
    # a = fit_training(Pipeline(tree), config)
    # a1 =
    # print(np.mean(a1))
    # for i in range(1):
    #    fit, tree = check_fitness(config)
    #    print(fit)
    # makeVideo("LunarLander-v2")
    # print(root.get_output())
