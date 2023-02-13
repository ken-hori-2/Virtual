import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class DEMO():

    def __init__(self, env):
        self.env = env

        self.BP = {}

    
    def show(self, state, map, Backed, DIR, trigar):
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
        
        "Add test-LBM"
        Node = ["A", "B", "C", "D", "O", "E", "F", "G",     "g"] # , "s"]
        test = [[0.0 for i in range(-size)] for i in range(-size)] #2Dgridmap(xw, yw) # Node
        # test = [[1.0 for i in range(-size)] for i in range(-size)] #2Dgridmap(xw, yw) # Node # here

        if Backed:
            # print("===============\n==================\nNULL")
            self.BP = Backed
            print("BP:", self.BP)

        for ix in range(-size):
            for iy in range(-size):   
                if self.env.grid[ix][iy] == 9:
                    soil[ix][iy] = 1 #sandy terrain
                else:
                    soil[ix][iy] = 0 #hard ground
                    "Add test-LBM"
                    
                    if self.env.NODELIST[ix][iy] in Node: # Node
                        test[ix][iy] = 0.5 # 1 #sandy terrain
                    # elif self.env.NODELIST[ix][iy] == "x":
                    #     test[ix][iy] = 0.1 #sandy terrain
                        # print("===== Node =====:", self.BP)

                    ##### Backed #####
                    
                    if self.env.NODELIST[ix][iy] in self.BP:
                        test[ix][iy] = 1
                        # print("Draw Backed")
                    ##### Backed #####
        # print("====== end ======")

        "Add"
        test = np.flip(test, 1)
        test = np.rot90(test, k=1)
        # lm = plt.pcolor(x, -y, test, vmax=1, cmap=plt.cm.Greens, alpha = 1.0) # マス目が消える
        lm = plt.pcolor(x, -y, test, vmax=1, cmap=plt.cm.BrBG, alpha = 1.0)
        # lm = plt.pcolor(x, -y, test, vmax=1, cmap=plt.cm.RdGy, alpha = 1.0) # here
        
        soil = np.flip(soil, 1)
        soil = np.rot90(soil, k=1)
        # terrain = plt.pcolor(x, -y, soil, vmax=1, cmap=plt.cm.Greys, alpha = 0.2)
        terrain = plt.pcolor(x, -y, soil, vmax=1, cmap=plt.cm.Greys, alpha = 0.5)
        # terrain = plt.pcolor(x, -y, soil, vmax=1, cmap=plt.cm.BuPu, alpha = 0.5) # here

        map  = np.flip(map, 1)
        map = np.rot90(map, k=1)
        known = plt.pcolor(x, -y, map, vmax=1, cmap=customized_gray)


        "----------"
        # Node = ["A", "B", "C", "D", "O", "E", "F", "G",     "g", "s"]
        Node = ["A", "B", "C", "D", "E", "F", "O", "g", "s"]
        # if self.env.grid[state.row][state.column] in Node:
        # self.to_arrows(A, V)
        "----------"
        
        # plt.plot(state.column, -state.row, ".y", markersize=10)
        if self.env.NODELIST[state.row][state.column] in Node:
            # plt.plot(state.column, -state.row, ".r", markersize=10)
            # plt.plot(state.column, -state.row, ".g", markersize=80, alpha = 0.2)
            
            plt.plot(state.column, -state.row, ".r", markersize=10,     label = "Agent")
            
        elif self.env.NODELIST[state.row][state.column] == "x":
            plt.plot(state.column, -state.row, "xb", markersize=5,     label = "Agent") # 10) # , alpha = 0.5)
        else:
            # plt.plot(state.column, -state.row, ".g", markersize=30, alpha = 0.2)

            plt.plot(state.column, -state.row, ".y", markersize=10,     label = "Agent")

        # Add
        # plt.plot(state.column, -state.row, ".y", markersize=80, alpha = 0.2)

        try:
            max_dir = max(DIR)
            if state.row+(-DIR[0]+DIR[1])*2/max_dir < 0 or state.column+(-DIR[2]+DIR[3])*2/max_dir < 0 or state.row+(-DIR[0]+DIR[1])*2/max_dir >= -size or state.column+(-DIR[2]+DIR[3])*2/max_dir >= -size:
                pass
            else:
                if self.env.NODELIST[state.row][state.column] in Node or self.env.NODELIST[state.row][state.column] == "x":

                    if not trigar and not self.env.NODELIST[state.row][state.column] == "g":
                        # plt.plot(state.column+(-DIR[2]+DIR[3])*2/max_dir, -state.row+(DIR[0]-DIR[1])*2/max_dir, "*y", markersize=10,     label = "Estimated") # , alpha = 0.5)
                        # plt.plot([state.column, state.column+(-DIR[2]+DIR[3])*2/max_dir], [-state.row, -state.row+(DIR[0]-DIR[1])*2/max_dir], linestyle = "--", color='y', alpha = 0.5)
                        plt.plot(state.column+(-DIR[2]+DIR[3])*2/max_dir, -state.row+(DIR[0]-DIR[1])*2/max_dir, marker = "*", color = "orange", markersize=10,     label = "Estimated") # , alpha = 0.5)
                        plt.plot([state.column, state.column+(-DIR[2]+DIR[3])*2/max_dir], [-state.row, -state.row+(DIR[0]-DIR[1])*2/max_dir], linestyle = "--", color='orange', alpha = 0.5)
        except:
            pass

        
        "Node, Goalの想定位置"
        tx = (8, 8, 8, 8, 8)
        ty = (-15+0.3, -12+0.3, -9+0.3, -6+0.3, -3+0.3)
        "----- 2d -----"
        tx = (8, 10, 10, 12, 12) # , 12, 14, 14, 16, 18)
        ty = (-14+0.3, -14+0.3, -9+0.3, -9+0.3, -4+0.3) # , -4+0.3, -4+0.3)
        plt.plot(tx, ty, "*g", markersize=4+2,     label = "Node")

        goal_x = (8)
        goal_y = (0.3)
        "----- 2d -----"
        goal_x = (14)
        # goal_x = (4)
        goal_y = (-4+0.3)
        plt.plot(goal_x, goal_y, "*r", markersize=4+2,     label = "Goal")

        
        # plt.legend(loc='upper right')
        plt.legend(loc='upper left')


        # png_path = os.path.join(result_dir, "{0}.png".format(ww))
        # plt.savefig(png_path)
        
        plt.show()

    def obserb(self, init, size, map):
        
        init_x, init_y = init.row, init.column

        # Node = ["A", "B", "C", "D", "O", "E", "F", "G",     "g",     "x"]
        # Node = ["A", "B", "C", "D", "O", "E", "F", "G",     "g",     "x",          "s"] # "s"を追加
        Node = ["A", "B", "C", "D", "E", "F", "O",   "g",     "x",          "s"] # "s"を追加

        if self.env.NODELIST[init_x][init_y] in Node: #交差点のみ前後一マス観測
            for i in range(-1,2):
                if init_x+i < 0 or init_x+i >=size:
                # if init_x+i >= 0 or init_x+i <size:
                    continue
                for j in range(-1,2):
                
                    if init_y+j < 0 or init_y+j >=size:
                    # if init_y+j >= 0 or init_y+j <size:
                        continue
                    
                    map[init_x+i][init_y+j] = 0
        map[init_x][init_y] = 0 # 現在のマスのみ観測
                
        return map




    def viz(self, viz):
        # fig = plt.figure(figsize=(3, 3))

        # viz.plot()
        # viz.plot.line(subplots=True, layout=(3, 1), grid=False, figsize=(5+2, 5), style=['-', '--', '-.', ':']) # , sharey=True) # 2, 2))
        viz.plot.line(subplots=True, layout=(3, 1), grid=False, figsize=(5, 5), style=['-', '--', '-.', ':'])

        plt.show()

    def bp_viz(self, Attribute):
        
        # Attribute[:5].plot.bar()
        # Attribute.plot.bar(subplots=True, layout=(1, 3), grid=False, figsize=(5+2, 5)) # , sharey=True) # 2, 2))
        Attribute.plot.bar(subplots=True, layout=(1, 3), grid=False, figsize=(5, 5))
        # self.Attribute.plot.bar()
        plt.show()