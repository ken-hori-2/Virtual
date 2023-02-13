from pprint import pprint
import numpy as np
import pprint
from refer import Property
import copy


class Algorithm_exp():

    def __init__(self, *arg):
        
        self.state = arg[0] # state
        self.env = arg[1] # env
        self.agent = arg[2] # agent
        self.NODELIST = arg[3] # NODELIST
        self.Observation = arg[4]
        self.refer = Property()
        self.total_stress = 0
        self.stress = 0
        self.Stressfull = 2.0
        self.COUNT = 0
        self.done = False
        self.TRIGAR = False
        self.STATE_HISTORY = []
        self.bp_end = False
        self.test_s = 0
        self.n_m = arg[5]

        self.RATE = arg[6]

        self.test = arg[7]

    def hierarchical_model_X(self): # 良い状態ではない時に「戻るタイミングは半信半疑」とした時のストレス値の蓄積の仕方

        self.End_of_O = True # ○の連続が途切れたのでTrue

        self.M += 1
        print("===== 🌟🌟🌟🌟🌟 =====")
        print("total : ", round(self.total_stress, 3))
        self.Σ = 1
        print("Save's Σ : ", self.Σ)
        print("[M, n] : ", self.M, self.n)
        print("[befor] total : ", round(self.total_stress, 3))
        print("m/m+n=", self.M/(self.M+self.n))
        self.total_stress += self.Σ *1.0* (self.M/(self.M+self.n)) # n=5,0.2 # ここ main
        # self.total_stress += self.Σ # row
        print("[after] total : ", round(self.total_stress, 3))
        self.STATE_HISTORY.append(self.state)
        self.TOTAL_STRESS_LIST.append(self.total_stress)

        "基準距離, 割合の可視化"
        self.standard_list.append(self.test_s)
        # self.rate_list.append(self.n/(self.M+self.n))    # ○
        self.rate_list.append(self.M/(self.M+self.n))      # ×

        "基準距離を可視化に反映させないver.はコメントアウト"
        # self.total_stress -= self.test_s # ×分は蓄積したので、基準距離分は一旦リセット
        "基準距離を可視化に反映させないver.はコメントアウト"

        print("[-基準距離] total : ", round(self.total_stress, 3))
        self.test_s = 0
        print("===== 🌟🌟🌟🌟🌟 =====")

        return True
    
    def match(self):
        # self.test_s = 0 # 0203 コメントアウト 1行動早くリセットされてしまう
        print("\n============================\n🤖 🔛　アルゴリズム切り替え\n============================")

    def nomatch(self, test, DIR):
        if self.grid[self.state.row][self.state.column] == 5:
            print("===== 交差点! 🚥　🚙　✖️ =====")
            if self.state not in self.CrossRoad:
                print("===== 未探索の交差点! 🚥　🚙　✖️ =====")
                self.CrossRoad.append(self.state)

        print("CrossRoad : {}".format(self.CrossRoad))

        print("事前情報にないNode!!!!!!!!!!!!")
        if self.NODELIST[self.state.row][self.state.column] == "x":
            true_or_false = self.hierarchical_model_X()

            # if self.M/(self.M+self.n) >= 0.5 + self.RATE: # 0.5: # here
            if self.M/(self.M+self.n) >= 0.5 + self.RATE:

                "----- Add -----"
                # test.show(self.state, self.map)

                self.TRIGAR = True
                self.test.show(self.state, self.map, {}, DIR,     self.TRIGAR)





                self.threshold()

    def threshold(self):
        # self.TRIGAR = True # 上に移動
        print("=================")
        print("FULL ! MAX! 🔙⛔️", self.retry_num, self.rrr)
        print("=================")

        "----- Add mark -----"
        self.env.mark(self.state, self.TRIGAR)
        
        
        "----- Add 2D Back x-----"
        if self.retry_num >=1: # here
            if self.rrr < 5:
                print("NODE POSITION x :", self.NODE_POSITION_x)
                self.state = self.NODE_POSITION_x
            else:
                self.state = self.NODE_POSITION
            self.rrr += 1
        else:
            self.state = self.NODE_POSITION # here
        "----- Add 2D Back x-----"
        # self.state = self.NODE_POSITION # here

        print(f"🤖 State:{self.state}")
        # self.total_stress = 0
        self.STATE_HISTORY.append(self.state)
        self.TOTAL_STRESS_LIST.append(self.total_stress)
        print(f"Total Stress:{self.total_stress}")

        "基準距離, 割合の可視化"
        self.standard_list.append(self.test_s)
        # self.rate_list.append(self.n/(self.M+self.n))    # ○
        self.rate_list.append(self.M/(self.M+self.n))      # ×

        self.test_s = 0

        "----- min cost cal -----"
        self.move_step = 0
        "----- min cost cal -----"
        
        "----- Add 1216 -----"
        self.Backed_just_before = True
        "ここで結局 True になっている"

    def lost_state(self):
        self.TRIGAR = True
        print("=================")
        print("LOST! 🔙⛔️", self.retry_num, self.rrr)
        print("=================")

        "----- Add mark -----"
        self.env.mark(self.state, self.TRIGAR)
        
        "----- Add 2D Back x-----"
        if self.retry_num >=1:
            if self.rrr < 5:
                print("NODE POSITION x :", self.NODE_POSITION_x)
                self.state = self.NODE_POSITION_x
            else:
                self.state = self.NODE_POSITION
            self.rrr += 1
        else:
            self.state = self.NODE_POSITION # here
        "----- Add 2D Back x-----"
        # self.state = self.NODE_POSITION # here

        self.STATE_HISTORY.append(self.state)
        self.TOTAL_STRESS_LIST.append(self.total_stress)
        print(f"🤖 State:{self.state}")

        "基準距離, 割合の可視化"
        self.standard_list.append(self.test_s)
        # self.rate_list.append(self.n/(self.M+self.n))    # ○
        self.rate_list.append(self.M/(self.M+self.n))      # ×

        # self.total_stress = 0
        self.test_s = 0

        "----- min cost cal -----"
        self.move_step = 0
        "----- min cost cal -----"

    def all_explore(self, Returned_state):
        self.env.mark_all(Returned_state)
        print("終了します")
        self.All_explore = False
        # self.total_stress = 0

        "----- min cost cal -----"
        self.move_step = 0
        # self.old_to_advance = self.NODELIST[self.state.row][self.state.column]
        "----- min cost cal -----"
        
    def Explore(self, STATE_HISTORY, state, TRIGAR, total_stress, grid, CrossRoad, x, TOTAL_STRESS_LIST, Backed_just_before, standard_list, rate_list, map_viz_test, DIR): # , PERMISSION):

        self.STATE_HISTORY = STATE_HISTORY
        self.state = state
        self.TRIGAR = False # TRIGAR
        pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()
        pprint.pprint(PERMISSION)
        self.NODE_POSITION = state
        # self.map_unexp_area = map_unexp_area
        self.lost = False
        self.grid = grid
        self.CrossRoad = CrossRoad
        GOAL = False
        self.total_stress = total_stress
        self.stress = 0
        index = Node.index("s")
        self.TOTAL_STRESS_LIST = TOTAL_STRESS_LIST
        self.standard_list = standard_list
        self.rate_list = rate_list
        self.move_step = 0
        self.old_to_advance = self.NODELIST[self.state.row][self.state.column]
        self.Backed_just_before = Backed_just_before

        
        "-----"
        self.retry_num = x # here
        self.rrr = 0 # here
        self.NODE_POSITION_x = state # here

        self.index = Node.index(self.NODELIST[self.state.row][self.state.column])
        "----- Add -----"
        self.map = map_viz_test
        # import sys
        # sys.path.append('/Users/ken/Desktop/src/YouTube/mdp')
        
        # from map_viz import DEMO
        # test = DEMO(self.env)
        self.pre_action = None
        "->main.pyに移動"

        size = self.env.row_length
        dem = [[0.0 for i in range(size)] for i in range(size)] #2Dgridmap(xw, yw)
        soil = [[0.0 for i in range(size)] for i in range(size)] #2Dgridmap(xw, yw)
        # map = [[0.0 for i in range(size)] for i in range(size)] #known or unknown
        # for ix in range(size):
        #     for iy in range(size):
        #         self.map[ix][iy] = 1
        #         # soil[ix][iy] = 1 #sandy terrain
        #         if dem[ix][iy] <= 0.2:
        #             soil[ix][iy] = 1 #sandy terrain
        #         else:
        #             soil[ix][iy] = 0 #hard ground


        #         # map[ix][iy] = 0 # 全マス観測している場合

                
        
        x, y = np.mgrid[-0.5:size+0.5:1,-0.5:size+0.5:1]
        states_known = set() #empty set
        for s in self.env.states:
            if self.map[s.row][s.column] == 0:
                states_known.add(s)
        "----- Add -----"

        while not self.done:

            "----- Add -----"
            self.map = self.test.obserb(self.state, size, self.map)
            # print("map")
            # pprint.pprint(map)
            
            try:
                states_known = set() #empty set
                for s in self.env.states:
                        if self.map[s.row][s.column] == 0:
                            states_known.add(s)
            except AttributeError:
            # except:
                pi = None
                print("Error!")
                break

            # test.show(state, map)
            "----- Add -----"


            if self.Backed_just_before: # 直前で戻っていた場合 これはbp.pyにてself.Backed_just_before = Trueを追加する
                __a = self.n_m[self.state.row][self.state.column] # -> ここは戻る場所決定で決めた場所を代入というか戻った後はこの関数に入るので現在地を代入
                print(f"🤖 State:{self.state}")
                pprint.pprint(self.n_m)
                print("__a = ", __a)
                try:
                    self.n = __a[0] # nを代入
                    self.M = __a[1] # mを代入
                except AttributeError: # here
                    print("Error!")
                    # break
                print(f"[n, m] = {self.n, self.M}")
                self.phi = [self.n, self.M]
                print("👍 (exp) phi = ", self.phi)
                print("phi(%) = ", self.n/(self.n+self.M), 1-self.n/(self.n+self.M))
                self.Backed_just_before = False # Add 1216



            print("\n========== 🌟 {}steps ==========".format(self.COUNT+1))
            print(f"🤖 State:{self.state}")
            print("stress : {}".format(self.stress))

            "----- Add 0203 -----"
            # self.move_step += 1 # here
            # print("move step = ", self.move_step)
            "----- Add 0203 -----"

            # if not self.crossroad:
            self.map_unexp_area = self.env.map_unexp_area(self.state)
            if self.map_unexp_area:
                print("un explore area ! 🤖 ❓❓")
                
                # if self.total_stress + self.stress >= 0:
                if self.test_s + self.stress >= 0:
                    
                    # 蓄積量(傾き)
                    ex = (self.n/(self.n+self.M))
                    ex = -2*ex+2

                    "----- Add ----"
                    # ex = 1.0 # 蓄積量の階層化は一旦ナシ

                    try:
                        # self.test_s += round(self.stress/float(Arc[index-1]), 3) # 2)
                        self.test_s += round(self.stress/float(Arc[self.index-1]), 3) *ex # here 元々はこっち

                        if not self.NODELIST[self.state.row][self.state.column] in pre:
                            self.move_step += 1
                    # except:
                    except Exception as e:
                        print('=== エラー内容 ===')
                        print('type:' + str(type(e)))
                        print('args:' + str(e.args))
                        print('message:' + e.message)
                        print('e自身:' + str(e))
                        self.test_s += 0
                        # self.move_step += 0

                    print("Arc to the next node : {}".format(Arc[index-1]))

                if self.NODELIST[self.state.row][self.state.column] in pre:

                    print("🪧 NODE : ⭕️")
                    
                    if self.NODELIST[self.state.row][self.state.column] == "g":
                        print("🤖 GOALに到達しました。")
                        GOAL = True
                        self.STATE_HISTORY.append(self.state)
                        self.TOTAL_STRESS_LIST.append(self.total_stress)

                        "基準距離, 割合の可視化"
                        self.standard_list.append(self.test_s)
                        # self.rate_list.append(self.n/(self.M+self.n))    # ○
                        self.rate_list.append(self.M/(self.M+self.n))      # ×

                        "----- Add -----"
                        # test.show(self.state, self.map)
                        self.test.show(self.state, self.map, {}, DIR,     self.TRIGAR)
                        
                        break

                    self.match()



                    # self.pre_action = self.action

                    "本当はここで self.Backed_just_before = True にするが、これは引き継いでいないのでそのままでもOK"
                    break # Advanceに移行する？
                else:
                    
                    print("🪧 NODE : ❌")
                    print("no match!")

                    self.nomatch(self.test, DIR)

                    
            if self.NODELIST[self.state.row][self.state.column] == "x": # here
                self.NODE_POSITION_x = self.state

            "ここはあってもあまり変わらない? 一応知っているところにたどり着いたらストレスを減少させておくが、たぶんmark=1になっている場所には進まないから意味ない"
            # if self.NODELIST[self.state.row][self.state.column] in pre:
            #     index = Node.index(self.NODELIST[self.state.row][self.state.column])
            #     print("<{}> match !".format(self.NODELIST[self.state.row][self.state.column]))
            #     print("事前のArc : {}".format(Arc[index]))
            #     # self.total_stress = 0
            #     "expで増加した分だけ減少"
            #     # if self.total_stress - self.test_s >= 0:
            #     #     self.total_stress -= self.test_s
            #     # else:
            #     #     self.total_stress = 0
            #     self.test_s = 0
            #     "expで増加した分だけ減少"

            print("PERMISSION : {}".format(PERMISSION[index][0]))

            print("\n===== test_s[基準距離]:", self.test_s)
            # if self.total_stress >= self.Stressfull: # or self.M/(self.M+self.n) >= 0.5:
            if self.test_s >= 2.0:
                
                "----- Add -----"
                # test.show(self.state, self.map)
                self.test.show(self.state, self.map, {}, DIR,     self.TRIGAR)
                
                
                
                self.threshold()


                "----- Add -----"
                # test.show(self.state, self.map)
                self.test.show(self.state, self.map, {}, DIR,     self.TRIGAR)

                continue






            print(f"Total Stress:{self.total_stress}")
            print("trigar : {}".format(self.TRIGAR))
            self.STATE_HISTORY.append(self.state)
            self.TOTAL_STRESS_LIST.append(self.total_stress)

            "基準距離, 割合の可視化"
            self.standard_list.append(self.test_s)
            # self.rate_list.append(self.n/(self.M+self.n))    # ○
            self.rate_list.append(self.M/(self.M+self.n))      # ×

            "----- Add -----"
            # test.show(self.state, self.map)
            self.test.show(self.state, self.map, {}, DIR,     self.TRIGAR)
            if self.Backed_just_before:
                __a = self.n_m[self.state.row][self.state.column] # -> ここは戻る場所決定で決めた場所を代入というか戻った後はこの関数に入るので現在地を代入
                print(f"🤖 State:{self.state}")
                pprint.pprint(self.n_m)
                print("__a = ", __a)
                try:
                    self.n = __a[0] # nを代入
                    self.M = __a[1] # mを代入
                # except AttributeError:
                #     print("Error!")
                #     # break
                except: # here
                    pass
                print(f"[n, m] = {self.n, self.M}")
                self.phi = [self.n, self.M]
                print("👍 (exp) phi = ", self.phi)
                print("phi(%) = ", self.n/(self.n+self.M), 1-self.n/(self.n+self.M))
                self.Backed_just_before = False
            "----- Add -----"



            # self.action, self.bp_end, self.All_explore, self.TRIGAR, self.Reverse, self.lost = self.agent.policy_exp(self.state, self.TRIGAR)
            self.action, self.bp_end, self.All_explore, self.TRIGAR, self.Reverse, self.lost = self.agent.mdp_exp(self.state, self.TRIGAR,     states_known, self.map, self.grid, DIR)
            

            "----- Add -----"
            self.pre_action = self.action


            if self.lost:
                self.lost_state()
                "----- Add -----"
                # test.show(self.state, self.map) # here
                self.test.show(self.state, self.map, {}, DIR,     self.TRIGAR)
                
            print("All explore : {}".format(self.All_explore))
            if self.All_explore:
                
                self.all_explore(state)

                "----- Add -----"
                # test.show(self.state, self.map)


                break
            
            if not self.lost:
                # self.next_state, self.stress, self.done = self.env._move(self.state, self.action, self.TRIGAR)
                self.next_state, self.stress, self.done = self.env.step(self.state, self.action, self.TRIGAR)
                self.prev_state = self.state # 1つ前のステップを保存 -> 後でストレスの減少に使う
                self.state = self.next_state

                "Add"
                # self.Backed_just_before = False
            else:
                self.lost = False

            "----- Add -----"
            # test.show(self.state, self.map)

            
            if self.COUNT > 150:
                break
            self.COUNT += 1
            
        if self.done:
            print("GOAL")

        return self.total_stress, self.STATE_HISTORY, self.state, self.TRIGAR, self.CrossRoad, GOAL, self.TOTAL_STRESS_LIST, self.move_step, self.old_to_advance, self.phi, self.standard_list, self.rate_list, self.test_s, self.map, self.pre_action