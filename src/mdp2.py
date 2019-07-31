import numpy as np
import seaborn as sns


class MDP:
    
    def __init__(self, state_rewards_dict={}, blocked_states_list=[], discount=1, size=10):
        
        self.state_rewards_dict = state_rewards_dict
        self.blocked_states_list = blocked_states_list
        self.discount = discount
        self.size = size
        
        self.states = self.get_states(size)
        self.actions = self.get_actions()
        self.blocked_states = self.get_blocked_states(self.states, self.blocked_states_list)
        
        self.transitions = self.get_transitions(self.states, self.actions, self.blocked_states)
        self.rewards = self.get_rewards(self.states, self.actions, state_rewards_dict)
        
        self.values = np.zeros(shape=self.states.shape[0])
        self.values_i = [self.values.copy()]
        
    
    def get_states(self, size):
        
        states = np.mgrid[0:size, -0:size].reshape(2,-1).T
        return states
            
            
    def get_actions(self):
        
        left_right = np.array([[-1,0], [1,0]])
        up_down = left_right[:,::-1]
        stay = np.array([[0, 0]])
        actions = np.vstack((left_right, up_down, stay))
        return actions
    
    
    def get_blocked_states(self, states, blocked_states_list):
        
        blocked_states = np.array(
            [0 if tuple(state) in blocked_states_list
            else 1 for state in states]
            )
        return blocked_states
    
    
    def get_transitions(self, states, actions, blocked_states):
        
        states_new = np.clip(
                states[:,None] + actions[None,:],
                a_min=0,
                a_max=states.size-1
            )
        transitions = (
                states[None,:,None,:] == states_new[:,None]
                ).all(axis=3).transpose(0,2,1).astype(int)
        
        return transitions * blocked_states[None,None,:]
    
    
    def get_rewards(self, states, actions, state_rewards_dict):
        
        state_rewards = np.array(
            [state_rewards_dict[tuple(state)] if tuple(state) in state_rewards_dict
            else 0 for state in states]
            )
        
        rewards =(
                state_rewards[None,:] * np.ones(len(state_rewards))[:,None]
        )[:,None,:] * np.ones(len(actions))[None,:,None]
        return rewards
    
    
    def get_values(self, prev_values, actions, rewards, transitions, blocked_states):
        
        expanded_values =(
                prev_values[None,:] * np.ones(prev_values.shape)[:,None]
        )[:,None,:] * np.ones(actions.shape[0])[None,:,None]
        new_values = (transitions * (rewards + self.discount*expanded_values)).sum(axis=2).max(axis=1)
        return new_values * blocked_states
    
    
    def value_iteration_step(self):
        
        self.values = self.get_values(self.values_i[-1], self.actions, self.rewards, self.transitions, self.blocked_states)
        self.values_i.append(self.values.copy())
        
    
    def value_iteration(self, max_iters=100, eps=.001):
        
        i = 0
        diff_size = np.inf
        while i < max_iters and diff_size > eps:
            
            self.value_iteration_step()
            prev_values = self.values_i[-2]            
            diff = prev_values - self.values
            diff_size = np.sqrt(diff.dot(diff))
            i+=1
            
    
    def get_policy_(self, prev_values, actions, rewards, transitions, blocked_states):
        
        expanded_values =(
                prev_values[None,:] * np.ones(prev_values.shape)[:,None]
        )[:,None,:] * np.ones(actions.shape[0])[None,:,None] * blocked_states[None,None,:]
        new_values = (transitions * (rewards + self.discount*expanded_values)).sum(axis=2).argmax(axis=1)
        return new_values
    
    
    def get_policy(self):
        return self.get_policy_(self.values, self.actions, self.rewards, self.transitions, self.blocked_states)
            
            
    def get_grid_values(self):
        
        grid_values = np.zeros(shape=(self.size, self.size))
        for i in range(len(self.values)):
            x,y = self.states[i]
            grid_values[x][y] = self.values[i]
        return grid_values
    
    
    def plot_grid_values(self):
        
        grid = self.get_grid_values()
        sns.heatmap(np.round(grid,2), annot=True, cbar=False)
        
        
    def get_grid_policy(self):
        
        policy = self.get_policy()
        grid_values = np.zeros(shape=(self.size, self.size))
        for i in range(len(self.values)):
            x,y = self.states[i]
            grid_values[x][y] = policy[i]
        return grid_values
    
    
    
    
    def policy_evaluation_step(self, prev_values, actions, rewards, transitions, blocked_states, policy):
        
        expanded_values =(
                prev_values[None,:] * np.ones(prev_values.shape)[:,None]
        )[:,None,:] * np.ones(actions.shape[0])[None,:,None] * blocked_states[None,None,:]
        new_values = (transitions * (rewards + self.discount*expanded_values)).sum(axis=2)
        new_values_policy = np.array([new_values[i][policy[i]] for i in range(new_values.shape[0])])
        return new_values_policy
    
    
    def policy_evaluation(self, policy, max_iters=100, eps=.001):
        
        i = 0
        diff_size = np.inf
        while i < max_iters and diff_size > eps:
            
            prev_values = self.values_i[-2]
            new_values = self.policy_evaluation_step(
                self.values_i[-1], self.actions, self.rewards, self.transitions, self.blocked_states, policy
                )
            self.values_i.append(new_values)
            self.values = new_values            
            diff = prev_values - self.values
            diff_size = np.sqrt(diff.dot(diff))
            i+=1
            
    
    def policy_improvement_step(self):
        
        new_policy = self.get_policy()
    
    
    
    

  
mdp = MDP({(6,6):1, (0,0):1}, [(2,2), (2,3), (1,3), (0,3), (2,1), (2,0)], discount=.8)
values = mdp.policy_evaluation()

#mdp.value_iteration()
#mdp.plot_grid_values()

#mdp.value_iteration_step()
#mdp.value_iteration_step()
#mdp.value_iteration_step()