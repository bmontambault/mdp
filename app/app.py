from flask import Flask,render_template,redirect,url_for,request
import os
import uuid
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime

import sys
sys.path.append('../src')
from mdp import MDP
from grid_world import make_grid_world

path=os.path.dirname(os.path.realpath(__file__))

app=Flask(__name__)
with open(path + '/secret_key.txt') as f:
    app.secret_key= f.read()
    


def get_mdp(request):
    
    data = json.loads(request.data)
    size = data['size']
    state_rewards_list = data['state_rewards_list']
    state_rewards_dict = {tuple(k):v for k,v in state_rewards_list}
    blocked_states_list = [tuple(s) for s in data['blocked_states_list']]
    discount = data['discount']
    
    values = np.array(data['values'])
    policy = np.array(data['policy'])
    
    mdp = MDP(state_rewards_dict, blocked_states_list,
                     discount, size, values, policy)
    return mdp
    
    
@app.route('/', methods=['GET', 'POST'])
def index():
    
    
    size = 10
    state_rewards_dict = {(6,6):1, (0,0):1}
    blocked_states_list = [(2, 3), (1, 3), (0, 3), (4, 8), (5, 8), (6, 8),
                           (5, 2), (6, 2), (7, 2), (8, 2), (8, 3), (8, 4)]
    discount=.9
    mdp = MDP(state_rewards_dict, blocked_states_list,
                     discount, size=size)
        
    min_reward = mdp.values.min()
    max_reward = mdp.values.max()
    
    table = make_grid_world(mdp.values, mdp.policy, mdp.blocked_states_list,
                            min_reward, max_reward)
    state_rewards_list = [[list(k),v] for k,v in state_rewards_dict.items()]
    return render_template("index.html",
                table=table,
                size=size,
                state_rewards_list=state_rewards_list,
                blocked_states_list=[list(s) for s in blocked_states_list],
                discount=discount,
                values=mdp.values.tolist(),
                policy=mdp.policy.tolist())
    
    

@app.route('/update', methods=['GET', 'POST'])
def update():
    
    data = json.loads(request.data)
    size = data['size']
    state_rewards_list = data['state_rewards_list']
    state_rewards_dict = {tuple(k):v for k,v in state_rewards_list}
    blocked_states_list = [tuple(s) for s in data['blocked_states_list']]
    discount = data['discount']
    started = data['started']
        
    values = np.array(data['values'])
    policy = np.array(data['policy'])
    
    if started:
        mdp = MDP(state_rewards_dict, blocked_states_list,
                     discount, size, values, policy)
    else:
        mdp = MDP(state_rewards_dict, blocked_states_list,
                     discount, size)
    
    min_reward = mdp.values.min()
    max_reward = mdp.values.max()
    table = make_grid_world(mdp.values, mdp.policy, mdp.blocked_states_list,
                            min_reward, max_reward)
    
    return json.dumps({'table': table, 'values': mdp.values.tolist(),
            'policy': mdp.policy.tolist()})
    

    
@app.route('/value_iteration', methods=['GET','POST'])
def value_iteration():
        
    size = 10
    state_rewards_dict = {(6,6):1, (0,0):1}
    blocked_states_list = [(2, 3), (1, 3), (0, 3), (4, 8), (5, 8), (6, 8),
                           (5, 2), (6, 2), (7, 2), (8, 2), (8, 3), (8, 4)]
    discount=.9
    mdp = MDP(state_rewards_dict, blocked_states_list,
                     discount, size=size)
        
    min_reward = mdp.values.min()
    max_reward = mdp.values.max()
    
    table = make_grid_world(mdp.values, mdp.policy, mdp.blocked_states_list,
                            min_reward, max_reward)
    state_rewards_list = [[list(k),v] for k,v in state_rewards_dict.items()]
    return render_template("value_iteration.html",
                table=table,
                size=size,
                state_rewards_list=state_rewards_list,
                blocked_states_list=[list(s) for s in blocked_states_list],
                discount=discount,
                values=mdp.values.tolist(),
                policy=mdp.policy.tolist())
    
    
@app.route('/value_iteration_step', methods=['GET', 'POST'])
def value_iteration_step():
    
    data = json.loads(request.data)
    size = data['size']
    state_rewards_list = data['state_rewards_list']
    state_rewards_dict = {tuple(k):v for k,v in state_rewards_list}
    blocked_states_list = [tuple(s) for s in data['blocked_states_list']]
    discount = data['discount']
    
    values = np.array(data['values'])
    policy = np.array(data['policy'])
    
    mdp = MDP(state_rewards_dict, blocked_states_list,
                     discount, size, values, policy)
    
    mdp.values = mdp.evaluate_values()
    mdp.policy = mdp.optimize_policy()
    
    min_reward = mdp.values.min()
    max_reward = mdp.values.max()
    table = make_grid_world(mdp.values, mdp.policy, mdp.blocked_states_list,
                            min_reward, max_reward)
    
    return json.dumps({'table': table, 'values': mdp.values.tolist(),
            'policy': mdp.policy.tolist()})
    
    

@app.route('/value_iteration_show_policy', methods=['GET', 'POST'])
def value_iteration_show_policy():
    
    data = json.loads(request.data)
    size = data['size']
    state_rewards_list = data['state_rewards_list']
    state_rewards_dict = {tuple(k):v for k,v in state_rewards_list}
    blocked_states_list = [tuple(s) for s in data['blocked_states_list']]
    discount = data['discount']
    
    values = np.array(data['values'])
    policy = np.array(data['policy'])
    
    mdp = MDP(state_rewards_dict, blocked_states_list,
                     discount, size, values, policy)
    
    min_reward = mdp.values.min()
    max_reward = mdp.values.max()
    table = make_grid_world(mdp.values, mdp.policy, mdp.blocked_states_list,
                            min_reward, max_reward, show_policy=True)
    return json.dumps({'table': table, 'values': mdp.values.tolist(),
            'policy': mdp.policy.tolist()})

    
    
@app.route('/policy_iteration', methods=['GET','POST'])
def policy_iteration():
        
    size = 10
    state_rewards_dict = {(6,6):1, (0,0):1}
    blocked_states_list = [(2, 3), (1, 3), (0, 3), (4, 8), (5, 8), (6, 8),
                           (5, 2), (6, 2), (7, 2), (8, 2), (8, 3), (8, 4)]
    discount=.9
    mdp = MDP(state_rewards_dict, blocked_states_list,
                     discount, size=size)
        
    min_reward = mdp.values.min()
    max_reward = mdp.values.max()
    
    table = make_grid_world(mdp.values, mdp.policy, mdp.blocked_states_list,
                            min_reward, max_reward, show_policy=True)
    state_rewards_list = [[list(k),v] for k,v in state_rewards_dict.items()]
    return render_template("policy_iteration.html",
                table=table,
                size=size,
                state_rewards_list=state_rewards_list,
                blocked_states_list=[list(s) for s in blocked_states_list],
                discount=discount,
                values=mdp.values.tolist(),
                policy=mdp.policy.tolist())

    
    
@app.route('/policy_evaluation_step', methods=['GET','POST'])
def policy_iteration_step():
        
    data = json.loads(request.data)
    size = data['size']
    state_rewards_list = data['state_rewards_list']
    state_rewards_dict = {tuple(k):v for k,v in state_rewards_list}
    blocked_states_list = [tuple(s) for s in data['blocked_states_list']]
    discount = data['discount']
    
    values = np.array(data['values'])
    policy = np.array(data['policy'])
    
    mdp = MDP(state_rewards_dict, blocked_states_list,
                     discount, size, values, policy)
    mdp.values = mdp.evaluate_policy_values()
    
    min_reward = mdp.values.min()
    max_reward = mdp.values.max()
    table = make_grid_world(mdp.values, mdp.policy, mdp.blocked_states_list,
                            min_reward, max_reward)
    
    return json.dumps({'table': table, 'values': mdp.values.tolist(),
            'policy': mdp.policy.tolist()})
    
    
@app.route('/policy_improvement', methods=['GET','POST'])
def policy_improvement():
        
    data = json.loads(request.data)
    size = data['size']
    state_rewards_list = data['state_rewards_list']
    state_rewards_dict = {tuple(k):v for k,v in state_rewards_list}
    blocked_states_list = [tuple(s) for s in data['blocked_states_list']]
    discount = data['discount']
    
    values = np.array(data['values'])
    policy = np.array(data['policy'])
    
    mdp = MDP(state_rewards_dict, blocked_states_list,
                     discount, size, values, policy)
    stable = mdp.policy_improvement()
    
    min_reward = mdp.values.min()
    max_reward = mdp.values.max()
    table = make_grid_world(mdp.values, mdp.policy, mdp.blocked_states_list,
                            min_reward, max_reward, show_policy=True)
    
    return json.dumps({'table': table, 'values': mdp.values.tolist(),
            'policy': mdp.policy.tolist()})
    
    


if __name__=="__main__":
    app.run()
