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
from mdp2 import MDP
from grid_world import make_grid_world

d = {}
path=os.path.dirname(os.path.realpath(__file__))

app=Flask(__name__)
with open(path + '/secret_key.txt') as f:
    app.secret_key= f.read()
    
    
@app.route('/value_iteration', methods=['GET','POST'])
def value_iteration():
        
    size = 10
    state_rewards_dict = {(6,6):1, (0,0):1}
    blocked_states_list = [(2,2), (2,3), (1,3), (0,3), (2,1), (2,0)]
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
    

    
    
@app.route('/policy_iteration', methods=['GET','POST'])
def policy_iteration():
    
        table = mdp.make_grid_world()
        return render_template("policy_iteration.html", table=table, size=states.size)
    
    
@app.route('/iterate', methods=['GET','POST'])
def iterate():
    
        mdp.value_iteration_step()
        table = mdp.make_grid_world()
        return json.dumps({'table':table})
    

@app.route('/policy', methods=['GET','POST'])
def policy():
    
        table = mdp.make_grid_world(show_policy=True)
        return json.dumps({'table':table})

    
if __name__=="__main__":
    app.run()
