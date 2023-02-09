import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from env import State
from env import Environment

class Planner():

    def __init__(self, env):
        self.env = env
        self.log = []

    def initialize(self):
        self.env.reset()
        self.log = []

    def plan(self, gamma=0.9, threshold=0.0001):
        raise Exception("Planner have to implements plan method.")

    def transitions_at(self, state, action, map, init, DIR):
        transition_probs = self.env.transit_func_MP(state, action)
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            reward, _ = self.env.reward_func(next_state, map, init, DIR) # , step_change)
            yield prob, next_state, reward

    def dict_to_grid(self, state_reward_dict):
        grid = []
        for i in range(self.env.row_length):
            row = [0] * self.env.column_length
            grid.append(row)
        for s in state_reward_dict:
            grid[s.row][s.column] = state_reward_dict[s]

        return grid


class ValueIterationPlanner(Planner):

    def __init__(self, env): # ,     env_from_lbm):
        # super().__init__(env)

        self.env = env

        # self.env_from_lbm = env_from_lbm

        
    def plan(self, state_known, map, init, DIR, gamma=0.9, threshold=0.0001):
        self.initialize()

        actions = self.env.actions
        print(actions)
        # actions = self.env_from_lbm.actions
        # print(actions)
        V = {}
        A = {}

        self.state_known = state_known

        for s in self.env.states:
            # Initialize each state's expected reward.
            V[s] = 0

        while True:
            
            delta = 0
            self.log.append(self.dict_to_grid(V))
            
            # for s in V:
            for s in self.state_known:
                r_max = -10
                a_max = None
                max_action = None
                
                if not self.env.can_action_at(s):
                    continue
                expected_rewards = []

                for a in actions:
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a, map, init, DIR):
                        r += prob * (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                    if r>r_max: 
                        r_max = r
                        max_action = a
                max_reward = max(expected_rewards)
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward

                A[s] = max_action
                
            V_grid = self.dict_to_grid(V)
            # self.show_values(V_grid)
            # self.show(A, V)
            if delta < threshold:
            # if policy_pre == self.policy_2:
            
                break

        V_grid = self.dict_to_grid(V)
        
        return V_grid, V, A
    
    def show_values(self, V):
        
        # fig = plt.figure(figsize=(self.env.row_length, self.env.column_length))
        fig = plt.figure(figsize=(5, 5))
        
        sns.heatmap(V, square=True, cbar=False, annot=True, fmt='3.2f', cmap='autumn_r') # .invert_yaxis()
        plt.axis("off")
        plt.show()
    
    def to_arrows(self, policy, V):

        # fig = plt.figure(figsize=(self.env.row_length, self.env.column_length))

        chars = {(self.env.actions[3]): '>r', (self.env.actions[0]): '^b', (self.env.actions[2]): '<k', (self.env.actions[1]): 'vm', None: '.k'}

        # print({s: chars[a] for (s, a) in policy.items()})


        return self.draw_arrow({s: chars[a] for (s, a) in policy.items()}, V)

    def draw_arrow(self, grid, V):

        for s in V:
            # print("grid", self.env.grid[s.row][s.column])
            "State は 下向きに数が増えるので-row"
            if self.env.grid[s.row][s.column] == 1:
                # plt.plot(s.column, -s.row, '.r')
                pass
            # elif self.env.grid[s.row][s.column] == 9:
            #     plt.plot(s.column, -s.row, '.g')
            else:
                plt.plot(s.column, -s.row, grid.get(s, '.k'))

            # plt.plot(0, -4, ".y", markersize=10)

        # plt.show()

    # def show(self, hop_init, A, V, x, y, ww):
    def show(self, A, V, state, map,     DIR):
        size = -self.env.row_length
        # fig = plt.figure(figsize=(self.env.row_length, self.env.column_length))
        fig = plt.figure(figsize=(5, 5))
        
        plt.plot([-0.5, -0.5], [0.5, size+0.5], color='k')
        plt.plot([-0.5, -size-0.5], [size+0.5, size+0.5], color='k')
        plt.plot([-size-0.5, -0.5], [0.5, 0.5], color='k')
        plt.plot([-size-0.5, -size-0.5], [size+0.5, 0.5], color='k')
        
        
        import matplotlib.cm as cm
        import matplotlib.colors as colors

        pi = np.pi
        cmap = cm.binary
        cmap_data = cmap(np.arange(cmap.N))
        cmap_data[0, 3] = 0 # 0 のときのα値を0(透明)にする
        customized_gray = colors.ListedColormap(cmap_data)

        dem = [[0.0 for i in range(-size)] for i in range(-size)] #known or unknown
        x, y = np.mgrid[-0.5:-size+0.5:1, -0.5:-size:1]
        # x, y = np.mgrid[-0.5:-size-0.5:1, 0.5:-size+0.5:1]

        demGrid = plt.pcolor(x, -y, dem, vmax=1, cmap=plt.cm.BrBG, alpha=0.2)
        
        soil = [[0.0 for i in range(-size)] for i in range(-size)] #2Dgridmap(xw, yw)
        for ix in range(-size):
            for iy in range(-size):   
                if self.env.grid[ix][iy] == 9:
                    soil[ix][iy] = 1 #sandy terrain
                else:
                    soil[ix][iy] = 0 #hard ground

        soil = np.flip(soil, 1)
        soil = np.rot90(soil, k=1)
        terrain = plt.pcolor(x, -y, soil, vmax=1, cmap=plt.cm.Greys, alpha = 0.2)

        map  = np.flip(map, 1)
        map = np.rot90(map, k=1)
        known = plt.pcolor(x, -y, map, vmax=1, cmap=customized_gray)


        "----------"
        Node = ["A", "B", "C", "D", "O", "E", "F", "G"]
        # if self.env.grid[state.row][state.column] in Node:
        self.to_arrows(A, V)
        "----------"
        
        plt.plot(state.column, -state.row, ".y", markersize=10,     label = "Agent")
        # plt.plot(state.column, -state.row, ".y", markersize=8,     label = "Agent") # Master Paper

        # # tx = (8, 10, 12, 12, 14, 14, 16, 18)
        # # ty = (-14+0.3, -14+0.3, -14+0.3, -9+0.3, -9+0.3, -4+0.3, -4+0.3, -4+0.3)
        # tx = (9, 9, 12)
        # ty = (-6, -3, -3)
        # # plt.plot(tx, ty, "*m", markersize=5)
        # plt.plot(tx, ty, "*g", markersize=4+2,     label = "Node")
        # goal_x = (18)
        # # goal_x = (4) # Master Paper
        # goal_x = (12)
        # goal_y = (0.3)
        # plt.plot(goal_x, goal_y, "*r", markersize=4+2,     label = "Estimated Goal")

        # plt.plot([state.column, state.column+1], [-state.row, -state.row+1], linestyle = "--", color='y', alpha = 0.5)
        # plt.plot(state.column+1, -state.row+1, "*y", markersize=15,     label = "Estimated Dir")
        
        
        
        
        
        # plt.scatter(state.column+1, -state.row+1, marker = "*", color = "orange", s=300, label = "Estimated Dir")
        # plt.plot([state.column, state.column+1], [-state.row, -state.row+1], linestyle = "--", color='orange', alpha = 0.5)
        try:
            max_dir = max(DIR)
            if state.row+(-DIR[0]+DIR[1])*2/max_dir < 0 or state.column+(-DIR[2]+DIR[3])*2/max_dir < 0 or state.row+(-DIR[0]+DIR[1])*2/max_dir >= -size or state.column+(-DIR[2]+DIR[3])*2/max_dir >= -size:
                pass
            else:
                if self.env.NODELIST[state.row][state.column] in Node or self.env.NODELIST[state.row][state.column] == "x":
                    # plt.plot(state.column+(-DIR[2]+DIR[3])*2/max_dir, -state.row+(DIR[0]-DIR[1])*2/max_dir, "*y", markersize=10,     label = "Estimated") # , alpha = 0.5)
                    # plt.plot([state.column, state.column+(-DIR[2]+DIR[3])*2/max_dir], [-state.row, -state.row+(DIR[0]-DIR[1])*2/max_dir], linestyle = "--", color='y', alpha = 0.5)
                    plt.plot(state.column+(-DIR[2]+DIR[3])*2/max_dir, -state.row+(DIR[0]-DIR[1])*2/max_dir, marker = "*", color = "orange", markersize=10,     label = "Estimated") # , alpha = 0.5)
                    plt.plot([state.column, state.column+(-DIR[2]+DIR[3])*2/max_dir], [-state.row, -state.row+(DIR[0]-DIR[1])*2/max_dir], linestyle = "--", color='orange', alpha = 0.5)
        except:
            pass
        plt.legend(loc='upper left')


        # png_path = os.path.join(result_dir, "{0}.png".format(ww))
        # plt.savefig(png_path)
        
        plt.show()

    def obserb(self, init, size, map):
        
        init_x, init_y = init.row, init.column

        
        # for i in range(-1,2):
        # # for i in range(-4,7): # Add 0206
        #     if init_x+i < 0 or init_x+i >=size:
        #         continue
        #     for j in range(-1,2):
        #     # for j in range(0,1): # Add 0206
            
        #         if init_y+j < 0 or init_y+j >=size:
        #             continue
                
        #         map[init_x+i][init_y+j] = 0

        # # map[init_x][init_y] = 0 # 現在のマスのみ観測
                
        # return map

        if [init_x, init_y] == [2, 2]: # [6, 6]:
            # for i in range(-1,2):
            for i in range(-4,7): # Add 0206
                if init_x+i < 0 or init_x+i >=size:
                # if init_x+i >= 0 or init_x+i <size:
                    continue
                for j in range(-1,2):
                # for j in range(0,1): # Add 0206
                
                    if init_y+j < 0 or init_y+j >=size:
                    # if init_y+j >= 0 or init_y+j <size:
                        continue
                    
                    map[init_x+i][init_y+j] = 0
        else:
            map[init_x][init_y] = 0 # 現在のマスのみ観測
                
        return map


# if __name__ == "__main__":
#     from agent import Agent
#     from env_MP import State
#     from env_MP import Environment
#     import pprint

    

#     grid = [
#         [0, 0, 0, 0, 1],
#         [0, 9, 0, 9, 0],
#         [0, 0, 0, 0, 0],
#         [0, 9, 0, 9, 0],
#         [0, 0, 0, 0, 0]
#     ]
#     # grid = [

#     #     # Master Paper

#     #     [0, 0, 0, 0, 1],
#     #     [0, 9, 0, 9, 0],
#     #     [0, 0, 0, 0, 0],
#     #     [0, 9, 0, 9, 0],
#     #     [0, 0, 0, 0, 0]
#     # ]

#     grid = [

#         # [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 1],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "E", 0, "F",  0, "G"],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 9, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "D", 0, "O", 0, 0,  0, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         # [0, 0, 0, 0, 0, 0, 0, 0, "A", 0, "B", 0, "C", 0, 0, 0, 0,  0, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],

#         # Node = ["A", "B", "C", "D", "O", "E", "F", "G"]
#         # Node_direction = ["A*", "B*", "C*", "D*", "O*", "E*", "F*", "G*"]
        
#         # Environment (a)
#         [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9,    0,  9,   0,  9,   1],
#         [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9,    0,  9,   0,  9,   0],
#         [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9,    0,  9,   0,  9,   0],
#         [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9,    0,  9,   0,  9,  "G*"],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,   0,  0,   0,   0,   "E","E*","F","F*","G"],
#         [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9,    0, 9, 0,  9, 0],
#         [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9,    0, 9, 0,  9, 0],
#         [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9,    0, 9, 0,  9, 0],
#         [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9,   "O*", 9, 0,  9, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,   0,  0,  "D", "D*", "O", 0, 0,  0, 0],
#         [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9, 0, 9, 0,  9, 0],
#         [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9, 0, 9, 0,  9, 0],
#         [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9, 0, 9, 0,  9, 0],
#         [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,  "C*", 9, 0, 9, 0,  9, 0],
#         # [0, 0, 0, 0, 0, 0, 0, "A*","A",0,"B","B*","C",  0, 0, 0, 0,  0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0,"A","A*","B","B*","C",  0, 0, 0, 0,  0, 0],
#         [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],

#         # Environment (b)
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9,    0,  9,   0,  9,   1],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9,    0,  9,   0,  9,   0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9,    0,  9,   0,  9,   0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9,    0,  9,   0,  9,  "G*"],
#         # [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,   0,  0,   "O",   "O*",   "E","E*","F","F*","G"],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 9,  9,   0,  9,   0,   9,    0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9,    0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9,    0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   "D*",   9,   0, 9, 0,  9, 0],
#         # [0, 0, 0, 0, 0, 0, 0, 0, 0,  0,   "C",  "C*",  "D", 0, 0, 0, 0,  0, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9, 0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9, 0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   0,  9,   0,   9, 0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0,  9,   "B*",  9,  0, 9, 0, 9, 0,  9, 0],
#         # [0, 0, 0, 0, 0, 0, 0, 0,"A","A*","B",0,0,  0, 0, 0, 0,  0, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
#         # [0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0,  9, 0],
                
#         ]

#     "Master Paper"
#     grid = [

#         # Master Paper

#         # "A*は過去N個の情報[以前のNodeの発見方向から確率を割り振る]から報酬決定? A*のマスに1をおいても同じ"
#         # 未探索方向をreward=10にすれば以前Nodeを発見した確率が高い方向よりもみ探索方向を優先する

#         # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#         # [0, 9, 9, 0, 9, 9, 0, 9, 9, 9, 9, 9, 0],
#         # [0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0],
#         # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         # [0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0],
#         # [0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0],
#         # [0, 0, 0, 0, 0, 0, 0, "A*", 0, 0, 0, 9, 0],
#         # [0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0],
#         # [0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0],
#         # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         # [0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0],
#         # [0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0],
#         # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#         [0, 0, 0, 0, 0, 0, "", 0, 0, 0, 0, 0, 1],
#         [0, 9, 9, 0, 9, 9, "", 9, 9, 9, 9, 9, 0],
#         [0, 9, 9, 0, 9, 9, "", 9, 9, 0, 9, 9, 0],
#         [0, 0, 0, 0, 0, "", "", "", 0, 0, 0, 0, 0],
#         [0, 9, 9, 0, 9, 9, "", 9, 9, 0, 9, 9, 0],
#         [0, 9, 9, 0, 9, 9, "UP", 9, 9, 0, 9, 9, 0],
#         [0, 0, 0, 0, 0, "LEFT", 0, "RIGHT", 0, 0, 0, 9, 0],
#         [0, 9, 9, 0, 9, 9, "DOWN", 9, 9, 0, 9, 9, 0],
#         [0, 9, 9, 0, 9, 9, "", 9, 9, 0, 9, 9, 0],
#         [0, 0, 0, 0, 0, "", "", "", 0, 0, 0, 0, 0],
#         [0, 9, 9, 0, 9, 9, "", 9, 9, 0, 9, 9, 0],
#         [0, 9, 9, 0, 9, 9, "", 9, 9, 0, 9, 9, 0],
#         [0, 0, 0, 0, 0, "", "", "", 0, 0, 0, 0, 0]

        
        
#         # default
#         # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#         # [0, 9, 9, 0, 9, 9, 0, 9, 9, 9, 9, 9, 0],
#         # [0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0],
#         # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         # [0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0],
#         # [0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0],
#         # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0],
#         # [0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0],
#         # [0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0],
#         # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         # [0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0],
#         # [0, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, 0],
#         # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     ]

#     grid = [

#         # Master Paper

#         [0,      9,      "",       9, 1],
#         [9,      9,   "UP",       9, 9],
#         [0, "LEFT",      0, "RIGHT", 0],
#         [9,      9, "DOWN",       9, 9],
#         [0,      9,      "",       9, 0]
#     ]

#     # s = State
#     env = Environment(grid)
    
#     agent = Agent(env)
    
#     test = ValueIterationPlanner(env)

#     # print(test)

#     state = env.reset()
#     size = env.row_length


#     import copy
    
#     for nnn in range(1): #iteration number of smulation
        
#         # Initialize position of agent.
#         state = env.reset()
        
#         step=0
#         step_change = 0
#         threshold = 15 # 10
#         # epi = []
#         init_list = []
        
#         size = env.row_length

#         for s in env.states:
#             if env.grid[s.row][s.column] == 1:
#                 goal = s
            
#             "次にやること"
#             # if env.grid[s.row][s.column] == "A":
#             #     Node_A = s

        

#         dem = [[0.0 for i in range(size)] for i in range(size)] #2Dgridmap(xw, yw)
#         soil = [[0.0 for i in range(size)] for i in range(size)] #2Dgridmap(xw, yw)
        
#         map = [[0.0 for i in range(size)] for i in range(size)] #known or unknown
#         for ix in range(size):
#             for iy in range(size):
#                 map[ix][iy] = 1
#                 # soil[ix][iy] = 1 #sandy terrain
#                 if dem[ix][iy] <= 0.2:
#                     soil[ix][iy] = 1 #sandy terrain
#                 else:
#                     soil[ix][iy] = 0 #hard ground


#                 # map[ix][iy] = 0 # 全マス観測している場合

                
        
#         x, y = np.mgrid[-0.5:size+0.5:1,-0.5:size+0.5:1]
#         # x, y = np.mgrid[-0.5:-size+0.5:1, -0.5:-size:1]

#         # states_known = None
#         states_known = set() #empty set
#         for s in env.states:
#             if map[s.row][s.column] == 0:
#                 states_known.add(s)

#         a, v, aaa = test.plan(states_known, map)
        
#         "----- 初期位置 -----"
#         # test.show(aaa, v, state, map) # 一つ前の状態を表示させる場合コメントアウト




#         for ww in range(200): #number of steps
#             weight = 0.5

#             "----- 前後1マスのみ観測 -----"
#             # map = [[1.0 for i in range(size)] for i in range(size)] #known or unknown
#             "---------------------------"

#             map = test.obserb(state, size, map)
#             # print("map")
#             # pprint.pprint(map)
            
#             try:

#                 states_known = set() #empty set
#                 for s in env.states:
#                         if map[s.row][s.column] == 0:
#                             states_known.add(s)
                
#                 a, v, aaa = test.plan(states_known, map)

                
#                 new_pi = (aaa[state])

#                 state_pre = copy.copy(state) # 一つ前の状態を表示する時に使う
                
#                 # action = agent.policy(state, new_pi)
#                 action = new_pi # 上と同じ


#                 next_state, reward, done = env.step(state, action, map, step_change)
#                 # total_reward += reward
#                 state = next_state

#             except AttributeError:
#             # except:
#                 pi = None
#                 print("Error!")
#                 break
#             # except Exception as e:
#             #     print('=== エラー内容 ===')
#             #     print('type:' + str(type(e)))
#             #     print('args:' + str(e.args))
#             #     print('message:' + e.message)
#             #     print('e自身:' + str(e))
#             #     break

#             # if show_animation:
#             if not aaa:
#                 pass
#             else:

#                 test.show(aaa, v, state_pre, map) # 一つ前の状態を表示させる場合
#                 # test.show(aaa, v, state, map)

#                 "add"
#                 # test.show_values(a)

                
            
#             print("\n====================================================================\n")

#             "----- Add 0206 -----"
#             END_REQUIRE = ["UP", "DOWN", "LEFT", "RIGHT"]
#             "----- Add 0206 -----"
            
#             if state.row < 0 or state.column < 0 or state.row >= size or state.column >= size:
#                 test.show_values(a)
#                 print(state.row, state.column)
#                 print("Error!")
#                 break
#             elif state == goal:
#                 map = test.obserb(state, size, map)
#                 test.show(aaa, v, state, map) # 一つ前の状態を表示させる場合
#                 test.show_values(a)
#                 print("-----\nGoal!\n-----")
#                 # print(env.states)
#                 break



#             # Add 0206
#             elif grid[state.row][state.column] in END_REQUIRE:
#                 map = test.obserb(state, size, map)
#                 test.show(aaa, v, state, map) # 一つ前の状態を表示させる場合
#                 test.show_values(a)
#                 print("終了条件")
#                 break
#             # Add 0206