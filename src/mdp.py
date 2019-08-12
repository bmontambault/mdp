import numpy as np
import seaborn as sns


class MDP:
    
    """
    MDP object that takes a set of states, a reward function, and discount
    factor. Optimal policies can then be found via either value iteration
    or policy iteration.
    Parameters:
        dict state_rewards_dict: (key,value) pairs of the form ((x,y), reward)
        list blocked_state_list: each element is an x,y pair indicating a state
                                 that cannot be traversed
        float discount: discount factor between 0 and 1; higher values weigh
                        future rewards closer to immediate rewards
        int size: size of the grid of actions
        list values: list of values for each state
        list policy: list of policies for each state
    """
    def __init__(self, state_rewards_dict={},
                 blocked_states_list=[],
                 discount=1, size=10,
                 values=None, policy=None):
        
        self.state_rewards_dict = state_rewards_dict
        self.blocked_states_list = blocked_states_list
        self.discount = discount
        self.size = size
        
        self.states = self.get_states_(size)
        self.actions = self.get_actions_()
        self.blocked_states = self.get_blocked_states_(
                self.states,
                self.blocked_states_list
            )
        
        self.transitions = self.get_transitions_(
                self.states,
                self.actions,
                self.blocked_states
            )
        self.rewards = self.get_rewards_(
                self.states,
                self.actions,
                state_rewards_dict
            )
        
        if values is None:
            self.values = np.zeros(shape=self.states.shape[0])
        else:
            self.values = values
        if policy is None:
            self.policy = np.random.randint(
                    0,
                    self.actions.shape[0]-1,
                    size=self.states.shape[0]
                )
        else:
            self.policy = policy
            
            
    """
    Take grid size and return (size, 2) array of state x,y coordinates
    Parameters:
        int size: size of grid
    Returns:
        array of size (size, 2), where each element is the x,y coordinates
        of a state
    """
    def get_states_(self, size):
        
        states = np.mgrid[0:size, -0:size].reshape(2,-1).T
        return states
            
    
    """
    Return a list of actions where each element is a pair of changes to x and
    y coordinates
    Returns:
        (5, 2) array of actions; left, right, up, down, stay
    """
    def get_actions_(self):
        
        left_right = np.array([[-1,0], [1,0]])
        up_down = left_right[:,::-1]
        stay = np.array([[0, 0]])
        actions = np.vstack((left_right, up_down, stay))
        return actions
    
    
    """
    Take an array of states and a list of blocked states and return an
    array of states with 0/1 indicating if the state can be traversed
    Parameters:
        array states: array of size (size, 2), where each element is the x,y
                      coordinates of a state
        list blocked_states_list: 
    Returns:
        array of where each element is a 0 or 1, indicating whether each state
        can be traversed
    """
    def get_blocked_states_(self, states, blocked_states_list):
        
        blocked_states = np.array(
            [0 if tuple(state) in blocked_states_list
            else 1 for state in states]
            )
        return blocked_states
    
    
    """
    Take arrays of states, actions and blocked states and return a (size, size)
    array where each element indicates (0/1) the ability to transition between
    two states
    Parameters:
        array states: array of size (size, 2), where each element is the x,y
                      coordinates of a state
        array actions: (5, 2) array of actions; left, right, up, down, stay
        array blocked_states: array of where each element is a 0 or 1,
                              indicating whether each state can be traversed
    Returns:
        (size, size) array where each element indicates whether each element
        (s, s') indicates whether s' can be reached from s
    """
    def get_transitions_(self, states, actions, blocked_states):
        
        states_new = np.clip(
                states[:,None] + actions[None,:],
                a_min=0,
                a_max=states.size-1
            )
        transitions = (
                states[None,:,None,:] == states_new[:,None]
                ).all(axis=3).transpose(0,2,1).astype(int)
        
        transitions = transitions * blocked_states[None,None,:]
        return transitions
    
    
    """
    Take arrays of states, actions a dictionary of rewards and returns an array
    with shape (#states, #actions, #states) where each element indicates the
    reward gained from moving from state s to state s' from action a
    Parameters:
        array states: array of size (size, 2), where each element is the x,y
                      coordinates of a state
        array actions: (5, 2) array of actions; left, right, up, down, stay
        dict state_rewards_dict: (key,value) pairs of the form ((x,y), reward)
    Returns:
        array with shape (#states, #actions, #states)
    """
    def get_rewards_(self, states, actions, state_rewards_dict):
        
        state_rewards = np.array(
            [state_rewards_dict[tuple(state)] if tuple(state) in state_rewards_dict
            else 0 for state in states]
            )
        
        rewards =(
                state_rewards[None,:] * np.ones(len(state_rewards))[:,None]
        )[:,None,:] * np.ones(len(actions))[None,:,None]
        return rewards
    
    
    """
    Expand list of values or policy of shape (#actions) to one of shape
    (#states, #actions, #states)
    Parameters:
        array actions: (5, 2) array of actions; left, right, up, down, stay
        array array: array of shape (#states) of either values or policies
    Returns:
        array with shape (#states, #actions, #states)
    """
    def expand_array_(self, actions, array):
        
        expanded_values =(
                array[None,:] * np.ones(array.shape)[:,None]
        )[:,None,:] * np.ones(actions.shape[0])[None,:,None]
        return expanded_values
    
    
    """
    Update values for each state given current state values
    Returns:
        array of shape (#states) where each element is the value of that state
    """
    def evaluate_values(self):
        
        expanded_values = self.expand_array_(self.actions, self.values)
        return (self.transitions*(self.rewards + self.discount*expanded_values)
                ).sum(axis=2).max(axis=1)
        
    
    """
    Find optimal policy given current state values
    Returns:
        array of shape (#states) where each element is the optimal policy at
        that state
    """
    def optimize_policy(self):
        
        expanded_values = self.expand_array_(self.actions, self.values)
        return (self.transitions*(self.rewards + self.discount*expanded_values)
                ).sum(axis=2).argmax(axis=1)
        
        
    def optimize_policy2(self):
        
        expanded_values = self.expand_array_(self.actions, self.values)
        transitions = np.nan_to_num(self.transitions / self.transitions.sum(axis=1)[:,None,:])
        return (transitions*(self.rewards + self.discount*expanded_values)
                ).sum(axis=2).argmax(axis=1)
        
        
    """
    Find optimal set values and policies for each state via value iteration
    Parameters:
        int max_iters: maximum number of iterations allowed before convergence
        float eps: maximum distance allowed for convergence
    Returns: None
    """
    def value_iteration(self, max_iters=100, eps=.001):
        
        i = 0
        diff_size = np.inf
        while i < max_iters and diff_size > eps:
            prev_values = self.values
            self.values = self.evaluate_values()
            diff = prev_values - self.values
            diff_size = np.sqrt(diff.dot(diff))
            i+=1
        self.policy = self.optimize_policy()
    
    
    """
    Update values for each state given current state values and the current
    policy
    Returns:
        array of shape (#states) where each element is the optimal policy at
        that state
    """
    def evaluate_policy_values(self):
        
        expanded_values = self.expand_array_(self.actions, self.values)
        values = (
                self.transitions*(self.rewards + self.discount*expanded_values)
            ).sum(axis=2)
        policy_values = np.array(
                [values[i][self.policy[i]] for i in range(values.shape[0])]
            )
        return policy_values
    
    
    """
    Update values until convergence given the current policy
    Parameters:
        int max_iters: maximum number of iterations allowed before convergence
        float eps: maximum distance allowed for convergence
    Returns: None
    """
    def policy_evaluation(self, max_iters=100, eps=.001):
        
        i = 0
        diff_size = np.inf
        while i < max_iters and diff_size > eps:
            prev_values = self.values
            self.values = self.evaluate_policy_values()
            diff = prev_values - self.values
            diff_size = np.sqrt(diff.dot(diff))
            i+=1
    
    
    """
    Find optimal policy given current values
    Returns:
        boolean indicating whether the policy has converged
    """
    def policy_improvement(self):
        
        prev_policy = self.policy
        self.policy = self.optimize_policy()
        stable = np.all(prev_policy == self.policy)
        return stable
    
    
    """
    Find optimal set values and policies for each state via policy iteration
    Parameters:
        int max_iters: maximum number of iterations allowed before convergence
        float eps: maximum distance allowed for convergence
    Returns: None
    """
    def policy_iteration(self, max_iters=100):
        
        stable = False
        i = 0
        while i < max_iters and not stable:
            self.policy_evaluation()
            stable = self.policy_improvement()
            i+=1
            
            
    def get_total_rewards(self):
        
        expanded_values = self.expand_array_(self.actions, self.values)
        total_rewards = self.rewards + expanded_values
        return total_rewards[0,0]
        
            
    def get_grid_total_rewards(self):
        
        total_rewards = self.get_total_rewards()
        grid_values = np.zeros(shape=(self.size, self.size))
        for i in range(len(total_rewards)):
            x,y = self.states[i]
            grid_values[x][y] = total_rewards[i]
        return grid_values
        
        
    def get_grid_policy(self):
        
        grid_values = np.zeros(shape=(self.size, self.size))
        for i in range(len(self.values)):
            x,y = self.states[i]
            grid_values[x][y] = self.policy[i]
        return grid_values

    
    def plot_grid_values(self):
        
        grid = self.get_grid_total_rewards()
        sns.heatmap(np.round(grid,2), annot=True, cbar=False)

"""
size = 5
state_rewards_dict = {(3,3):1, (0,0):2}
blocked_states_list = [(2,3), (2,4), (2,2)]
discount=.9
mdp = MDP(state_rewards_dict, blocked_states_list,
                 discount, size=size)
mdp.value_iteration()
"""