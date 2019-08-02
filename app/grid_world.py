from colour import Color
import collections
import bisect
import numpy as np


def get_color_map(n, min_reward, max_reward):
    
    red = Color("red")
    colors = list(red.range_to(Color("green"),n))
    reward_range = np.linspace(min_reward, max_reward, num=n)
    
    color_map = collections.OrderedDict(
            {reward_range[i]:colors[i].hex for i in range(n)}
        )
    return color_map
    
    
def get_color(value, color_map, min_reward, max_reward):
    
    if value in color_map:
        return color_map[value]
    value = max(min(value, max_reward), min_reward)
    keys = list(color_map.keys())
    idx = bisect.bisect_left(keys, value)
    if abs(keys[idx]-value) < abs(keys[idx-1]-value):
        key = keys[idx]
    else:
        key = keys[idx-1]
    return color_map[key]


def get_color_values(values, color_map, min_reward, max_reward):
    
    return [get_color(value, color_map, min_reward, max_reward) for value in values]
    
    
def make_grid_world(values, policy, n, min_reward, max_reward,
                    show_policy=False):
    
    color_map = get_color_map(n, min_reward, max_reward)
    colors = get_color_values(values, color_map, min_reward, max_reward)
    rounded_values = np.round(values,2)
            
    grid = "<table>"
    grid_num = 0
    for i in range(n):
        grid += "<tr>"; 
        for j in range(n):
            if show_policy:
                val = ['&larr;', '&rarr;', '&uarr;', '&darr;'][policy[grid_num]]
            else:
                val = rounded_values[grid_num]
            grid += f"<td align='center' style='background-color:{colors[grid_num]}'>{val}</td>"
            grid_num += 1
        grid += "</tr>"
    grid += '</table>'
    return grid