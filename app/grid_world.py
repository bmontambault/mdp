from colour import Color
import collections
import bisect
import numpy as np


class ColorMap:
    
    def __init__(self, min_reward, max_reward, n=100):
        
        red = Color("red")
        colors = list(red.range_to(Color("green"),n))
        reward_range = np.linspace(min_reward, max_reward, num=n)
        
        self.color_map = collections.OrderedDict(
                {reward_range[i]:colors[i].hex for i in range(n)}
        )
        
    
    def get_color(self, value):
        
        if value in self.color_map:
            return self.color_map[value]
        keys = list(self.color_map.keys())
        value = max(min(max(keys),value), min(keys))
        idx = bisect.bisect_left(keys, value)

        if abs(keys[idx]-value) < abs(keys[idx-1]-value):
            key = keys[idx]
        else:
            key = keys[idx-1]
        return self.color_map[key]

    
    
def make_grid_world(total_rewards, policy, blocked_states_list,
                    show_policy=False, n=100):
    
    
    min_reward = total_rewards.min()
    max_reward = total_rewards.max()
    color_map = ColorMap(min_reward, max_reward, n)
    colors = [color_map.get_color(r) for r in total_rewards]    
    rounded_total_rewards = np.round(total_rewards,2)
            
    size = int(np.sqrt(len(total_rewards)))
    grid = "<table>"
    grid_num = 0
    for i in range(size):
        grid += "<tr>"; 
        for j in range(size):
            if (i,j) in blocked_states_list:
                val = ''
                color = 'gray'
            elif show_policy:
                val = ['&larr;', '&rarr;', '&uarr;', '&darr;', 'X'][policy[grid_num]]
                color = colors[grid_num]
            else:
                val = rounded_total_rewards[grid_num]
                color = colors[grid_num]
            grid += f"<td align='center' id='{i}{j}' style='background-color:{color}'>{val}</td>"
            grid_num += 1
        grid += "</tr>"
    grid += '</table>'
    return grid