from importlib.resources import path
from experiment_launchers.pipeline import *
import pickle
import re
import os
import sys
import gym
import random
import numpy as np
from tqdm import tqdm
import string


def random_string():
    return "{}{}{}{}{}{}{}{}".format(*np.random.choice(list(string.ascii_lowercase), 10))


def count_spaces(current):
    spaces = 0
    for c in current:
        if c == " ":
            spaces += 1
        else:
            break
    return spaces


def get_tree(code, parent=None, branch=None):
    if len(code) == 0:
        return ""
    current = code[0]

    # Base case
    if "out" in current:
        return "{} -->|{}| {}[{}]\n".format(parent, branch, random_string(), current.split("=")[1])

    if not "if" in current:
        return get_tree(code[1:], parent, branch)

    # Recursion
    node_id = random_string()
    condition = current.replace("if ", "").replace(":", "").replace("  ", "")
    # print(condition)

    indentation_level = count_spaces(current)
    else_position = None

    for i, n in enumerate(code[1:]):
        if count_spaces(n) == indentation_level:
            else_position = i + 1
            break


    left_branch = get_tree(code[1:else_position], node_id, "True")
    right_branch = get_tree(code[else_position + 1:], node_id, "False")

    subtree = ""
    if parent is None:
        left_branch = "```mermaid\ngraph TD\n" + left_branch.replace(" -->", "[{}] -->".format(condition), 1)
    else:
        subtree = "{} -->|{}| {}[{}]\n".format(parent, branch, node_id, condition)
    subtree += left_branch + right_branch
    if parent is None:
        subtree += "```"
    return subtree


random.seed(0)
np.random.seed(0)
class Node:
    def __init__(self, id_, value=None, left_branch=None, right_branch=None, parent=None):
        self._id = id_
        self._value = value
        self._left_branch = left_branch
        self._right_branch = right_branch
        self._parent = parent
        self._visits = 0

    def __repr__(self):
        out = "{}[{}]\n".format(self._id, self._value, self._visits)
        if self._left_branch is not None:
            out += "{} -->|True| {}\n".format(self._id, self._left_branch._id)
            out += repr(self._left_branch)
        if self._right_branch is not None:
            out += "{} -->|False| {}\n".format(self._id, self._right_branch._id)
            out += repr(self._right_branch)
        return out

    def __str__(self):
        return repr(self)

    def get_output(self, input_):
        self._visits += 1
        if "_in_" in self._value or "<" in self._value or ">" in self._value:
            #print(self._value)
            val = re.sub(r"_in_([0-9]*)", "input_[\\1]", self._value)
            #print(val)
            branch = eval(val)
            if branch:
                return self._left_branch.get_output(input_)
            else:
                return self._right_branch.get_output(input_)
        else:
            return int(self._value)


def tree2Node(string):
    nodes = {}

    lines = string.split("\n")

    #print(string)
    for line in lines[2:-1]:
        if " -->" in line:
            left, right = line.split(" -->")
            true = "True" in right
            right = right.replace("|True| ", "").replace("|False| ", "")

            if "[" in left:
                left_id_, left_condition = left.split("[")
                left_condition = left_condition.replace("]", "")
            else:
                left_id_ = left
                left_condition = None

            if "[" in right:
                right_id_, right_condition = right.split("[")
                right_condition = right_condition.replace("]", "")
            else:
                right_id_ = right
                right_condition = None

            for id_, value in zip([left_id_, right_id_], [left_condition, right_condition]):
                if id_ not in nodes:
                    nodes[id_] = Node(id_)
                if value is not None:
                    nodes[id_]._value = value

            if true:
                nodes[left_id_]._left_branch = nodes[right_id_]
            else:
                nodes[left_id_]._right_branch = nodes[right_id_]
            nodes[right_id_]._parent = nodes[left_id_]
        else:
            # print(line)
            pass

    for n in nodes.values():
        if n._parent is None:
            return n


def calc_complexity_from_string(code, env, tree2Node=tree2Node):
    code = code.replace("/(1.0 - 0.0)", "").replace("- 0.0 ", "").replace("1.0 *", "").replace("+ 0.0 ", "").replace("- -","+ ")
    #print(code)
    tree = get_tree(code.split("\n"))
    #print(tree)
    root = tree2Node(tree)
    #print(code)
    mean_reward = []

    for episode in (range(5)):
        mean_reward.append(0)
        e = gym.make(env)
        e.seed(episode)
        obs = e.reset()
        done = False
        i = 0
        current_ep_reward = 0
        while not done:
            # e.render()
            action = root.get_output(obs) if root is not None else 0
            # action = root.get_output([i, *obs])
            i += 1

            obs, reward, done, _ = e.step(action)
            current_ep_reward += reward
        mean_reward[-1] += current_ep_reward

        e.close()
    # if np.mean(mean_reward) != 500:
    #     return 0
    #print(f"Mean reward: {np.mean(mean_reward)}")


    change = True
    if root is None:
        return 0,None,-200
    while change:
        change = False
        fringe = [root]

        while len(fringe) > 0:
            node = fringe.pop(0)
            if node._left_branch is not None and node._right_branch is not None:
                if node._left_branch._visits == 0:
                    # print(node, "has 0 visits")
                    if node._parent is not None:
                        if node == node._parent._left_branch:
                            node._parent._left_branch = node._right_branch
                        else:
                            node._parent._right_branch = node._right_branch
                    else:
                        root = node._right_branch
                    node._right_branch._parent = node._parent
                    fringe.append(node._right_branch)
                    change = True
                elif node._right_branch._visits == 0:
                    if node._parent is not None:
                        if node == node._parent._left_branch:
                            node._parent._left_branch = node._left_branch
                        else:
                            node._parent._right_branch = node._left_branch
                    else:
                        root = node._left_branch
                    node._left_branch._parent = node._parent
                    fringe.append(node._left_branch)
                    change = True
                else:
                    # print(node._left_branch._visits)
                    fringe.append(node._left_branch)
                    fringe.append(node._right_branch)
            else:
                # print("Not entered in {}".format(node))
                pass

    change = True
    while change:
        change = False
        fringe = [root]

        while len(fringe) > 0:
            node = fringe.pop(0)
            if node._left_branch is not None and node._right_branch is not None:
                # print(node._left_branch._value, len(node._left_branch._value))
                if node._left_branch._value == node._right_branch._value and len(node._left_branch._value) == 1:
                    node._value = node._left_branch._value
                    node._left_branch = None
                    node._right_branch = None
                    change = True
                else:
                    fringe.append(node._left_branch)
                    fringe.append(node._right_branch)

    l = 0
    no = 0
    nnao = 0
    ncnao = 0

    fringe = [root]
    nodes_seen = 0
    while len(fringe) > 0:
        node = fringe.pop(0)
        nodes_seen += 1

        if "_in_" in node._value:
            parts = node._value.replace("(", "").replace(")", "").replace("+-", "- ").replace("  ", " ").split(" ")
            # TODO:  <26-11-20, leonardo> # This works only for the type of trees used in the paper. Check it for different trees.
            l += 1
            for p in parts:
                l += 1
                if len(p) == 1 and p in ["+", "-", "*", "/", "<", ">"]:
                    no += 1
                    if p == "<" or p == ">":
                        nnao += 1
                        ncnao += 1
            """
            l += 4
            no += 2
            nnao += 2
            ncnao += 2
            """
            no += 1
            nnao += 1
            ncnao += 1
        else:
            l += 1

        if node._left_branch is not None:
            fringe.append(node._left_branch)
        if node._right_branch is not None:
            fringe.append(node._right_branch)
    # print(root)
    return (-0.2 + 0.2 * l + 0.5 * no + 3.4 * nnao + 4.5 * ncnao), root,np.mean(mean_reward)

class Node_m_to_p:
    def __init__(self, condition):
        self.condition = condition
        self.left = None
        self.right = None

    def print(self, ind=0):
        if self.left is not None:
            string = f"{' '*ind}if {self.condition}:\n" + self.left.print(ind + 4) + f"\n{' '*ind}else:\n" + self.right.print(ind + 4)
            return string
        else:
            return " "*ind + f"out={str(self.condition)}"


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
            id_, value = l.split("[")
            value = value[:-1]
            nodes[id_] = Node_m_to_p(value)
            if len(nodes) == 1:
                root = nodes[id_]
    return root

if __name__ == "__main__":
    #path_dir = "/home/matteo/marl_dts/src/logs/CartPole-ME_pyRibs_3/28-05-2022_18-38-45_xvkuotpf/"
    path_dir = "/home/matteo/marl_dts/src/logs/MountainCar-ME_pyRibs_7/29-05-2022_18-27-08_jlnjexmc"
    dir_list = os.listdir(path_dir)
    #dir_list = [f for f in dir_list if ".pkl" in f and "best" not in f]
    dir_list = [f for f in dir_list if ".pkl" in f]
    inter_list = []
    outmax = (0,None,-250)
    for f in dir_list:
        fw = open(os.path.join(path_dir, f), "rb")
        tree = pickle.load(fw)
        tree = str(tree)
        print(tree)
        tree = tree.replace(" [","[")
        tree = tree.replace("[0]","0")
        tree = tree.replace("[1]","1")
        tree = tree.replace("[2]","2")
        tree = tree.replace("[3]","3")
        tree = tree.replace("[4]","4")
        tree = tree.replace("[5]","5")
        tree = tree.replace("input","_in")
        tree = tree.replace("Node_m_to_p object at ","")
        tree = re.sub("\(.*?\)","",tree)
        #tree = re.sub("\<.*?\>","",tree)
        tree = tree.replace("\nOneHotEncoder\n\n","")
        tree = tree.replace("RLDecisionTree\n","")
        tree = tree[:tree.rfind('\n')]
        tree = tree[:tree.rfind('\n')]
        tree = tree[:tree.rfind('\n')]
        print(tree)
        root = convert(tree)
        #print(root)
        #print(f)
        out = calc_complexity_from_string(root.print(),"MountainCar-v0")
        if out[2] > outmax[2] or out[2] >= 475:
            print(fw)
            outmax=out
            print(out)
            inter_list.append(out[0])
            print("\n")
        fw.close()
    print(inter_list)
    lines = []
    with open(path_dir+"/all.txt") as f:
        lines = f.readlines()
    #lines.pop(0)
    lines = [l for l in lines if l != "\n" and "Gen Index Fitness" not in l]
    for i,l in enumerate(lines):
        lines[i] = lines[i].replace('(',"")
        lines[i] = lines[i].replace(')',"")
        lines[i] = lines[i].replace(',',"")
        lines[i] = lines[i].replace('\n',"")
        lines[i] = lines[i].split(" ")
        lines[i] = [int(lines[i][0]),[int(lines[i][1]),int(lines[i][2])],float(lines[i][3])]
    final_gen = max([item[0] for item in lines])
    lines = [item for item in lines if item[0]==final_gen]
    counter = 0
    for l in lines:
        if l[2]>= -110:
            counter += 1
    print("NUMBER OF TREES THAT RESOLVE THE TASK: ", counter)
    if len(inter_list)>0:
        print("MAX = ",max(inter_list))
        print("MIN = ",min(inter_list))
        print("MEAN = ",np.mean(inter_list))
        print("VARIANCE = ",np.var(inter_list))
    # fl = open("/home/matteo/marl_dts/src/logs/MountainCar-GP_4/14-06-2022_18-35-47_jxoemoti/2123402292240.pkl","rb")
    # tree = pickle.load(fl)
    # tree = str(tree)
    # #print(tree)
    # print(tree)
    # tree = tree.replace(" [","[")
    # tree = tree.replace("[0]","0")
    # tree = tree.replace("[1]","1")
    # tree = tree.replace("[2]","2")
    # tree = tree.replace("[3]","3")
    # tree = tree.replace("[4]","4")
    # tree = tree.replace("[5]","5")
    # tree = tree.replace("input","_in")
    # tree = re.sub("\(.*?\)","",tree)
    # tree = re.sub("\<.*?\>","",tree)
    # tree = tree.replace("\nOneHotEncoder\n\n","")
    # tree = tree.replace("RLDecisionTree\n","")
    # print(tree)
    # root = convert(tree)
    # #print(root.print())
    # # # print(calc_complexity_from_string(root.print(),"CartPole-v1"))


