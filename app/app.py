from flask import Flask,render_template,redirect,url_for,request
import os
import uuid
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime

import sys
path=os.path.dirname(os.path.realpath(__file__))
sys.path.append(path + '/../src')
from mdp import MDP
from grid_world import make_grid_world


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

    
@app.route('/policy_iteration_example', methods=['GET', 'POST'])
def policy_iteration_example():
    
    
    size = 5
    state_rewards_dict = {(3,3):1, (0,0):1}
    blocked_states_list = [(2,3), (2,4), (2,2)]
    discount=.9
    
    np.random.seed(443209)
    policy = np.random.randint(0,4, size=size**2)
    mdp = MDP(state_rewards_dict, blocked_states_list,
                     discount, size=size, policy=policy)        

    value_table1 = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list)
    policy_table1 = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list, show_policy=True)
    
    mdp.values = mdp.evaluate_policy_values()
    intermediate_table = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list)
    
    mdp.policy_evaluation()
    value_table2 = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list)
    
    mdp.policy_improvement()
    policy_table2 = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list, show_policy=True)
    
    mdp.policy_evaluation()
    value_table3 = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list)
    
    mdp.policy_improvement()
    policy_table3 = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list, show_policy=True)
    
    mdp.policy_iteration()
    value_table4 = make_grid_world(mdp.states, mdp.get_total_rewards(), mdp.policy, mdp.blocked_states_list)
    policy_table4 = make_grid_world(mdp.states, mdp.get_total_rewards(), mdp.policy, mdp.blocked_states_list, show_policy=True)
    
    state_rewards_list = [[list(k),v] for k,v in state_rewards_dict.items()]
    return render_template("policy_iteration_example.html",
                value_table1=value_table1,
                policy_table1=policy_table1,
                intermediate_table=intermediate_table,
                value_table2=value_table2,
                policy_table2=policy_table2,
                value_table3=value_table3,
                policy_table3=policy_table3,
                value_table4=value_table4,
                policy_table4=policy_table4,
                size=size,
                state_rewards_list=state_rewards_list,
                blocked_states_list=[list(s) for s in blocked_states_list],
                discount=discount,
                values=mdp.values.tolist(),
                policy=mdp.policy.tolist())


@app.route('/', methods=['GET', 'POST'])
def index():
    
    
    size = 10
    state_rewards_dict = {(6,6):1, (0,0):1}
    blocked_states_list = [(2, 3), (1, 3), (0, 3), (4, 8), (5, 8), (6, 8),
                           (5, 2), (6, 2), (7, 2), (8, 2), (8, 3), (8, 4)]
    discount=.9
    mdp = MDP(state_rewards_dict, blocked_states_list,
                     discount, size=size)
        
    table = make_grid_world(mdp.states, mdp.get_total_rewards(), mdp.policy, mdp.blocked_states_list)
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
    
    print(blocked_states_list)
    if started:
        mdp = MDP(state_rewards_dict, blocked_states_list,
                     discount, size, values, policy)
    else:
        mdp = MDP(state_rewards_dict, blocked_states_list,
                     discount, size)
    table = make_grid_world(mdp.states, mdp.get_total_rewards(), mdp.policy,
                            mdp.blocked_states_list)
    
    return json.dumps({'table': table, 'values': mdp.values.tolist(),
            'policy': mdp.policy.tolist()})
    

    
@app.route('/value_iteration', methods=['GET','POST'])
def value_iteration():
    
    size = 5
    state_rewards_dict = {(3,3):1, (0,0):2}
    blocked_states_list = [(2,3), (2,4), (2,2)]
    discount=.9
    mdp = MDP(state_rewards_dict, blocked_states_list,
                     discount, size=size)
        
    table1 = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list)
    
    mdp.values = mdp.evaluate_values()
    table2 = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list)
    
    mdp.values = mdp.evaluate_values()
    table3 = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list)
    
    
    mdp.values = mdp.evaluate_values()
    table4 = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list)
    
    
    mdp.value_iteration()
    table5 = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list)
    value_table6 = make_grid_world(mdp.states, mdp.get_total_rewards(), mdp.policy, mdp.blocked_states_list)
    policy_table6 = make_grid_world(mdp.states, mdp.get_total_rewards(), mdp.policy, mdp.blocked_states_list,
                                    show_policy=True)
            
    size = 10
    state_rewards_dict = {(6,6):1, (0,0):1}
    blocked_states_list = [(2, 3), (1, 3), (0, 3), (4, 8), (5, 8), (6, 8),
                           (5, 2), (6, 2), (7, 2), (8, 2), (8, 3), (8, 4)]
    discount=.9
    mdp = MDP(state_rewards_dict, blocked_states_list,
                     discount, size=size)
    
    table = make_grid_world(mdp.states, mdp.get_total_rewards(), mdp.policy, mdp.blocked_states_list)
    state_rewards_list = [[list(k),v] for k,v in state_rewards_dict.items()]
    return render_template("value_iteration.html",
                table1=table1,
                table2=table2,
                table3=table3,
                table4=table4,
                table5=table5,
                value_table6=value_table6,
                policy_table6=policy_table6,
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
    
    table = make_grid_world(mdp.states, mdp.get_total_rewards(), mdp.policy, mdp.blocked_states_list)
    
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
    
    table = make_grid_world(mdp.states, mdp.get_total_rewards(), mdp.policy, mdp.blocked_states_list, show_policy=True)
    return json.dumps({'table': table, 'values': mdp.values.tolist(),
            'policy': mdp.policy.tolist()})

    
    
@app.route('/policy_iteration', methods=['GET','POST'])
def policy_iteration():
    
    size = 5
    state_rewards_dict = {(3,3):1, (0,0):1}
    blocked_states_list = [(2,3), (2,4), (2,2)]
    discount=.9
    
    np.random.seed(443209)
    policy = np.random.randint(0,4, size=size**2)
    mdp = MDP(state_rewards_dict, blocked_states_list,
                     discount, size=size, policy=policy)        

    value_table1 = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list)
    policy_table1 = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list, show_policy=True)
    
    mdp.values = mdp.evaluate_policy_values()
    intermediate_table = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list)
    
    mdp.policy_evaluation()
    value_table2 = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list)
    
    mdp.policy_improvement()
    policy_table2 = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list, show_policy=True)
    
    mdp.policy_evaluation()
    value_table3 = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list)
    
    mdp.policy_improvement()
    policy_table3 = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list, show_policy=True)
    
    mdp.policy_iteration()
    value_table4 = make_grid_world(mdp.states, mdp.get_total_rewards(), mdp.policy, mdp.blocked_states_list)
    policy_table4 = make_grid_world(mdp.states, mdp.get_total_rewards(), mdp.policy, mdp.blocked_states_list, show_policy=True)
        
    size = 10
    state_rewards_dict = {(6,6):1, (0,0):1}
    blocked_states_list = [(2, 3), (1, 3), (0, 3), (4, 8), (5, 8), (6, 8),
                           (5, 2), (6, 2), (7, 2), (8, 2), (8, 3), (8, 4)]
    discount=.9
    mdp = MDP(state_rewards_dict, blocked_states_list,
                     discount, size=size)
    
    table = make_grid_world(mdp.states, mdp.get_total_rewards(), mdp.policy, mdp.blocked_states_list, show_policy=True)
    state_rewards_list = [[list(k),v] for k,v in state_rewards_dict.items()]
    return render_template("policy_iteration.html",
                value_table1=value_table1,
                policy_table1=policy_table1,
                intermediate_table=intermediate_table,
                value_table2=value_table2,
                policy_table2=policy_table2,
                value_table3=value_table3,
                policy_table3=policy_table3,
                value_table4=value_table4,
                policy_table4=policy_table4,
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
    
    table = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list)
    
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
    
    table = make_grid_world(mdp.states, mdp.values, mdp.policy, mdp.blocked_states_list,
                            show_policy=True)
    
    return json.dumps({'table': table, 'values': mdp.values.tolist(),
            'policy': mdp.policy.tolist()})
    
    


if __name__=="__main__":
    app.run()
