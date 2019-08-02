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


path=os.path.dirname(os.path.realpath(__file__))

app=Flask(__name__)
with open(path + '/secret_key.txt') as f:
    app.secret_key= f.read()
    
    
@app.route('/value_iteration', methods=['GET','POST'])
def value_iteration():
        
    size = 10
    vi_state_rewards_dict = {(6,6):1, (0,0):1}
    vi_blocked_states_list = [(2,2), (2,3), (1,3), (0,3), (2,1), (2,0)]
    vi_discount=.8
    vi_mdp = MDP(vi_state_rewards_dict, vi_blocked_states_list,
                     vi_discount, size=size)
    
    min_reward = vi_mdp.rewards.max() * size**2
    max_reward = vi_mdp.rewards.min() * size**2
    vi_table = make_grid_world(vi_mdp.values, vi_mdp.policy, size, min_reward,
                               max_reward)
    state_rewards_list = [(k,v) for k,v in vi_state_rewards_dict.items()]
    return render_template("value_iteration.html", table=vi_table, size=size,
                               state_rewards_list=state_rewards_list,
                               blocked_states_list=vi_blocked_states_list,
                               discount=vi_discount,
                               values=vi_mdp.values.tolist(),
                               policy=vi_mdp.policy.tolist())
    
    
@app.route('/value_iteration_step', methods=['GET', 'POST'])
def value_iteration_step():
    
    print (request)
    data = request.json
    print (data)
    return ''
    

    
    
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
