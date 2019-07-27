import numpy as np
from colour import Color
import collections
import bisect


class States:
    
    def __init__(self, blocked_states=None, size=10):
        self.size=size
        self.states = np.mgrid[0:size, -0:size].reshape(2,-1).T
        
        
class Actions:
    
    def __init__(self, actions=None):
        if actions is None:
            left_right = np.array([[-1,0], [1,0]])
            up_down = left_right[:,::-1]
            self.actions = np.vstack((left_right, up_down))
        else:
            self.actions = actions
    

class Transitions:
    
    def __init__(self, states, actions, deterministic=1000):
        
        self.outcomes = np.clip(
                states.states[:,None] + actions.actions[None],
                a_min=0,
                a_max=states.size-1
            )
        self.deterministic_transitions = np.all(
                states.states[:,None,None,:] == self.outcomes[None],
                axis=3
            ).astype(int).transpose(0,2,1)
        
        self.action_probs = self.generate_action_probs(actions, deterministic)
        self.transitions = (self.action_probs[:,None] * self.deterministic_transitions[:,:,:,None]
                            ).mean(axis=1).transpose(0,2,1)
        
    def generate_action_probs(self, actions, deterministic):
        
        action_probs = []
        for i in range(len(actions.actions)):
            identity = np.zeros(len(actions.actions))
            identity[i] = deterministic
            ones = np.ones(len(actions.actions))
            action_probs.append(np.random.dirichlet(identity + ones))
        return np.array(action_probs)
    
    
class Rewards:
    
    def __init__(self, states, actions, state_rewards_dict):
        
        self.state_rewards = np.array(
            [state_rewards_dict[tuple(state)] if tuple(state) in state_rewards_dict
            else 0 for state in states.states]
            )
        
        self.rewards =(
                self.state_rewards[None,:] * np.ones(len(self.state_rewards))[:,None]
        )[:,None,:] * np.ones(len(actions.actions))[None,:,None]
        self.max_reward = self.rewards.max()
        self.min_reward = self.rewards.min()
            
    

class MDP:
    
    def __init__(self, states, actions, transitions, rewards):
        
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.values =(transitions.transitions * rewards.rewards).sum(axis=2).max(axis=1)
        
    def value_iteration_step(self, inplace=True):
        
        values =(
                self.values[None,:] * np.ones(len(self.values))[:,None]
        )[:,None,:] * np.ones(len(self.actions.actions))[None,:,None]
        new_values = (
                self.transitions.transitions * (self.rewards.rewards + values)
            ).sum(axis=2).max(axis=1)
        if inplace:
            self.values = new_values
        else:
            return new_values
        
    def value_iteration(self, max_iters=100, eps=.001):
        
        i = 0
        diff_size = np.inf
        while i < max_iters and diff_size > eps:
                        
            new_values = self.value_iteration_step(inplace=False)
            diff = new_values - self.values
            diff_size = np.sqrt(diff.dot(diff))
            self.values = new_values            
            i+=1
            
            
    def set_colors(self, n=100):
        
        red = Color("red")
        colors = list(red.range_to(Color("green"),n))
        reward_range = np.linspace(self.rewards.min_reward,
                                   self.rewards.max_reward, num=n)
        
        self.color_map = collections.OrderedDict(
                {reward_range[i]:colors[i].hex for i in range(n)}
            )
        
        
    def get_color(self, value):
        
        if value in self.color_map:
            return self.color_map[value]
        value = max(min(value, self.rewards.max_reward), self.rewards.min_reward)
        keys = list(self.color_map.keys())
        idx = bisect.bisect_left(keys, value)
        if abs(keys[idx]-value) < abs(keys[idx-1]-value):
            key = keys[idx]
        else:
            key = keys[idx-1]
        return self.color_map[key]
    
    
    def get_color_values(self):
        
        return [self.get_color(value) for value in self.values]
        
        
    def make_grid_world(self, show_policy=False):
        
        if show_policy:
            policy = self.get_policy()
        
        self.set_colors()
        colors = self.get_color_values()
        rounded_values = np.round(self.values,2)
                
        grid = "<table>"
        grid_num = 0
        for i in range(self.states.size):
            grid += "<tr>"; 
            for j in range(self.states.size):
                if show_policy:
                    val = ['&larr;', '&rarr;', '&uarr;', '&darr;'][policy[grid_num]]
                else:
                    val = rounded_values[grid_num]
                grid += f"<td align='center' style='background-color:{colors[grid_num]}'>{val}</td>"
                grid_num += 1
            grid += "</tr>"
        grid += '</table>'
        return grid
    

    
    def get_policy(self):
        
        values =(
                self.values[None,:] * np.ones(len(self.values))[:,None]
        )[:,None,:] * np.ones(len(self.actions.actions))[None,:,None]
        return (
                self.transitions.transitions * (self.rewards.rewards + values)
            ).sum(axis=2).argmax(axis=1)
        
"""
states = States()
actions = Actions()
transitions = Transitions(states, actions)

states_rewards_dict = {(0,0):1}
rewards = Rewards(states, actions, states_rewards_dict)
mdp = MDP(states, actions, transitions, rewards)
        
mdp.value_iteration()
grid = mdp.make_grid_world()
"""