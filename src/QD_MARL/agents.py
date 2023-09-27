class Agent:
    def __init__(self, name, squad, set_, tree, manual_policy, to_optimize):
        self._name = name
        self._squad = squad
        self._set = set_
        self._tree = tree.deep_copy() if tree is not None else None
        self._manual_policy = manual_policy
        self._to_optimize = to_optimize
        self._score = []

    def get_name(self):
        return self._name

    def get_squad(self):
        return self._squad

    def get_set(self):
        return self._set

    def to_optimize(self):
        return self._to_optimize

    def get_tree(self):
        return self._tree.deep_copy()

    def get_output(self, observation):
        if self._to_optimize:
            return self._tree.get_output(observation)
        else:
            return self._manual_policy.get_output(observation)

    def set_reward(self, reward):
        self._tree.set_reward(reward)
        self._score[-1] = reward

    def get_score_statistics(self, params):
        score_values = [score_dict[key] for score_dict in self._score for key in score_dict]
        return getattr(np, f"{params['type']}")(a=score_values, **params['params'])#Can't compare dicts with >

    def new_episode(self):
        self._score.append(0)

    def has_policy(self):
        return not self._manual_policy is None

    def __str__(self):
        return f"Name: {self._name}; Squad: {self._squad}; Set: {self._set}; Optimize: {str(self._to_optimize)}"
    
class CoachAgent(Agent):
    def __init__(self, name, squad, set_, tree, manual_policy, to_optimize):
        super().__init__(name, squad, set_, tree, manual_policy, to_optimize)

    def get_output(self, observation):
        return super().get_output(observation)
    
    def set_reward(self, reward):
        return super().set_reward(reward)
    
    def get_score_statistics(self, params):
        return super().get_score_statistics(params)
    
    def new_episode(self):
        return super().new_episode()
    
    def has_policy(self):
        return super().has_policy()
    
    def __str__(self):
        return super().__str__()
    
    def select_team(self):
        pass