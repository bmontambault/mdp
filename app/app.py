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
from mdp import States, Actions, Transitions, Rewards, MDP


path=os.path.dirname(os.path.realpath(__file__))

app=Flask(__name__)
with open(path + '/secret_key.txt') as f:
    app.secret_key= f.read()
    
    
states = States()
actions = Actions()
transitions = Transitions(states, actions)
        
states_rewards_dict = {(5,5):1}
rewards = Rewards(states, actions, states_rewards_dict)
mdp = MDP(states, actions, transitions, rewards)
    
    
@app.route('/', methods=['GET','POST'])
def index():
    
        table = mdp.make_grid_world()
        return render_template("index.html", table=table, size=states.size)
    
    
@app.route('/value_iteration', methods=['GET','POST'])
def value_iteration():
    
        table = mdp.make_grid_world()
        return render_template("value_iteration.html", table=table, size=states.size)
    
    
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

    


