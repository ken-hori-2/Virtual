
import math
from refer import Property
import copy
import pprint
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson

class Algorithm_bp():

    def __init__(self, *arg):
        
        self.state = arg[0] # state
        self.env = arg[1] # env
        self.agent = arg[2] # agent
        self.NODELIST = arg[3] # NODELIST
        self.Observation = arg[4]
        self.refer = Property()
        self.total_stress = 0
        self.stress = 0
        self.Stressfull = 8 # 10 # 4
        self.COUNT = 0
        self.done = False
        self.TRIGAR = False
        self.TRIGAR_REVERSE = False
        self.BACK = False
        self.BACK_REVERSE = False
        self.on_the_way = False
        self.bf = True
        self.STATE_HISTORY = []
        self.BPLIST = []
        self.PROB = []
        self.Arc = []
        self.OBS = []
        self.Storage_Arc = []
        self.SAVE = []

        "============================================== Visualization ver. との違い =============================================="
        self.Node_l = ["s", "A", "B", "C", "D", "E", "F", "O", "g", "x"]
        # self.COST = move_cost_cal()
        self.backed = []
        self.Unbacked = self.Node_l
        "============================================== Visualization ver. との違い =============================================="
        self.n_m = arg[5]

        self.test = arg[6]

    def move_cost_cal(self):
        "----- move cost -----"
        print("bp-----=========================================================================================\n")
        print("mv_copy : ", self.move_cost_result_copy)
        self.move_cost_result_copy[self.move_cost_result_copy == np.inf] = np.nan
        print("mv_copy inf -> nan : ", self.move_cost_result_copy)
        print(type(self.move_cost_result_copy))

        self.demo = copy.copy(self.move_cost_result_copy)
        
        print("-----")
        self.move_cost_result_copy = pd.Series(self.move_cost_result_copy, index=self.Node_l) # index=self.Unbacked)
        print("mv_copy -> pandas + add index: ", self.move_cost_result_copy)
        print(type(self.move_cost_result_copy))
        print("-----")
        self.move_cost_result_copy.dropna(inplace=True)
        print("mv_copy drop nan: ", self.move_cost_result_copy)
        print(type(self.move_cost_result_copy))
        print("-----")
        
        try:
            self.move_cost_result_copy.drop(index=["x"], inplace=True)
        except:
            # Add 0203
            # self.move_cost_result_copy.drop(index=[self.NODELIST[self.state.row][self.state.column]], inplace=True) # というかこっちのみで良い気がする
            pass


        print("mv_copy drop x: ", self.move_cost_result_copy)
        # # print(self.move_cost_result_copy["s"])
        # # self.move_cost_result_copy.drop(index=self.backed, columns=self.backed, inplace=True)
        self.move_cost_result_copy.drop(index=self.backed, inplace=True) # mv_copyはXの行成分抽出 = npの配列 -> 再度pandasでindex追加しているのでindexのみ削除で大丈夫
        print("mv_copy drop backed: ", self.move_cost_result_copy)


        print(f"test_bp_st : \n{self.test_bp_st}")
        self.test_bp_st.dropna(inplace=True)
        print("test bp st drop nan : ", self.test_bp_st)
        print("-----")
        # self.test_bp_st.drop(index=["x"], inplace=True)
        # print(self.test_bp_st)

        self.test_bp_st.drop(index=self.backed, inplace=True)
        print("Storage : ", self.test_bp_st)

        print("bp-----=========================================================================================\n")
        "----- move cost -----"

    def next_position_decision(self):
        # self.BPLIST = self.test_bp_st
                        
        # self.next_position, self.next_attribute = self.agent.back_position(self.test_bp_st, self.Attribute)
        self.next_attribute = self.agent.back_position(self.test_bp_st, self.Attribute)
        
        "============================================== Visualization ver. との違い =============================================="
        print("===== TEST 1114 =====")
        # print(self.BPLIST)
        # print(self.test_bp_st)
        # print(self.next_attribute["STATE"][self.next_attribute.index[0]]) # self.next_position)
        print(self.next_attribute["STATE"][0])
        print(self.next_attribute.index[0])          # これと
        print(self.next_attribute["STATE"].index[0]) # これは同じ
        print("===== TEST 1114 =====")

        # print(type(self.test_bp_st)) # self.BPLIST))
        # # print(type(self.next_attribute["STATE"][self.next_attribute.index[0]])) # self.next_position))
        # print(type(self.next_attribute["STATE"][0])) # self.next_position))
        # # next_lm = (self.BPLIST[self.BPLIST == self.next_attribute["STATE"][self.next_attribute.index[0]]]) # self.next_position])
        # # next_lm = (self.test_bp_st[self.test_bp_st == self.next_attribute["STATE"][self.next_attribute.index[0]]]) # self.next_position])
        # # next_lm = (self.test_bp_st[self.test_bp_st == self.next_attribute["STATE"][0]]) # self.next_position])
        # next_lm = self.next_attribute["STATE"][0]
        # print("next_lm : ", next_lm)
        # # # print("next_lm : ", next_lm.index[0])
        # # print("test:", self.next_attribute["STATE"].index[0])
        # # print("test:", self.next_attribute["STATE"][0])
        # # print("test:", self.next_attribute.index[0])

        # "test index"
        # # self.test_index = self.BPLIST.index.get_loc(next_lm.index[0])
        # # print("self.BPLIST.index : ", self.BPLIST.index.get_loc(next_lm.index[0]))
        # print("self.test_bp_st.index : ", self.test_bp_st.index.get_loc(self.next_attribute.index[0])) # next_lm.index[0]))

        self.Backed_Node = self.next_attribute.index[0] # next_lm.index[0]
        print("Backed Node : ", self.Backed_Node)

    def Finished_returning(self, Node):
        __a = self.n_m[self.state.row][self.state.column] # -> ここは戻る場所決定で決めた場所を代入というか戻った後はこの関数に入るので現在地を代入
        pprint.pprint(self.n_m)
        print("__a = ", __a)
        self.n = __a[0] # nを代入
        self.M = __a[1] # mを代入
        print(f"[n, m] = {self.n, self.M}")
        self.phi = [self.n, self.M]
        print("👍 (bp) phi = ", self.phi)
        print("phi(%) = ", self.n/(self.n+self.M), 1-self.n/(self.n+self.M))

        
        self.backed.append(self.Backed_Node)
        self.Unbacked = [i for i in Node if i not in self.backed]
        print("----- Backed : ", self.backed)
        print("----- UnBacked : ", self.Unbacked)
        print("----- test_bp_st -----")
        print("削除前")
        print("Storage : ", self.test_bp_st)
        print("削除後")
        # self.test_bp_st.drop(index=next_lm.index, inplace=True)
        "一旦printして可視化するためだけのもの"
        self.test_bp_st_copy = copy.copy(self.test_bp_st_pre)
        self.test_bp_st_copy.dropna(inplace=True)
        self.test_bp_st_copy.drop(index=self.backed, inplace=True)
        print("Storage : ", self.test_bp_st_copy)
        print("----- test_bp_st -----")
        print("----- move cost -----")
        print("削除前")
        print("mv_copy : ", self.move_cost_result_copy)
        print("削除後")
        
        "Add 1116 一旦printして可視化するためだけのもの 今は下の方でやっているが、こっちで可視化用のcopy2でやってもいい"
        self.move_cost_result_copy2 = copy.copy(self.move_cost_result)
        self.move_cost_result_copy2 = pd.Series(self.move_cost_result_copy2, index=self.Node_l) # index=self.Unbacked)
        self.move_cost_result_copy2.drop(index=self.backed, columns=self.backed, inplace=True)
        self.move_cost_result_copy2[self.move_cost_result_copy2 == np.inf] = np.nan
        self.move_cost_result_copy2.dropna(inplace=True)
        self.move_cost_result_copy2.drop(index=["x"], inplace=True)
        print("mv_copy2 : ", self.move_cost_result_copy2)
        # print("mv_copy : ", self.move_cost_result_copy)
        " ↑ or ↓ "
        # "----- これだとinplace=Trueでも、この後アルゴリズムを抜けるのでこの結果はリセットされてしまい反映されない -----"
        # self.move_cost_result_copy.drop(index=self.backed, inplace=True) # mv_copyはXの行成分抽出 = npの配列 -> 再度pandasでindex追加しているのでindexのみ削除で大丈夫
        # "-> エラー 上でindexのdropを既に削除しているのでエラーになる"
        # print("mv_copy drop backed: ", self.move_cost_result_copy)
        # print("mv_copy : ", self.move_cost_result_copy)
        "-----------------------------------------------------------------------------------------------"

        print("----- move cost -----")

        # self.BPLIST, self.w, self.OBS, self.Attribute = self.agent.back_end(self.BPLIST, self.next_position, self.w, self.OBS, self.test_index, self.move_cost_result, self.Attribute, self.next_attribute)
        self.Attribute = self.agent.back_end(self.Attribute, self.next_attribute)
        self.BACK =True
        print("🔚 ARRIVE AT BACK POSITION (戻り終わりました。)")
        print(f"🤖 State:{self.state}")
        print("OBS : {}".format(self.OBS))

        # self.total_stress = 0
        "⚠️ 要検討 ⚠️ 戻った時にどのくらい減少させるか test_s = 進んだ分だけ減少させるか = その場所までのストレスまで減少させるか"
        print("⚠️ total : {}".format(self.total_stress))
        delta_s = self.Observation[self.state.row][self.state.column]
        delta_s = round(abs(1.0-delta_s), 3)
        if delta_s > 2:
            delta_s = 1.0
        "1217 コメントアウト"
        # if self.total_stress - delta_s >= 0:
        #     self.total_stress -= delta_s
        # else:
        #     self.total_stress = 0

        print("⚠️ delta_s : {}".format(delta_s))
        print("⚠️ total : {}".format(self.total_stress))
        "-- 追加部分 --"
        "============================================== Visualization ver. との違い =============================================="

        self.STATE_HISTORY.append(self.state)
        self.TOTAL_STRESS_LIST.append(self.total_stress)

        "基準距離, 割合の可視化"
        self.test_s = 0
        self.standard_list.append(self.test_s)
        # self.rate_list.append(self.n/(self.M+self.n))    # ○
        self.rate_list.append(self.M/(self.M+self.n))      # ×

        self.TRIGAR = False
        self.TRIGAR_REVERSE = False

    def BP(self, STATE_HISTORY, state, TRIGAR, OBS, BPLIST, action, Add_Advance, total_stress, SAVE_ARC, TOTAL_STRESS_LIST, move_cost_result, test_bp_st_pre, move_cost_result_X, standard_list, rate_list, map_viz_test, Attribute):
        self.STATE_HISTORY = STATE_HISTORY
        self.state = state
        self.TRIGAR = TRIGAR
        self.OBS = OBS
        self.BPLIST = BPLIST
        self.Advance_action = action
        self.bf = True
        self.state_history_first = True
        self.Add_Advance = Add_Advance
        self.Backed_just_before = False
        self.total_stress = total_stress
        self.SAVE_ARC = SAVE_ARC
        self.first_pop = True
        self.BackPosition_finish = False
        pre, Node, Arc, Arc_sum, PERMISSION = self.refer.reference()
        self.TOTAL_STRESS_LIST = TOTAL_STRESS_LIST
        self.standard_list = standard_list
        self.rate_list = rate_list
        "============================================== Visualization ver. との違い =============================================="
        self.move_cost_result_pre = move_cost_result # self.l
        self.test_bp_st_pre = test_bp_st_pre
        # self.BPLIST = pd.Series(self.BPLIST, index=self.Node_l)
        self.test_bp_st = copy.copy(self.test_bp_st_pre)
        print("===== Storage : ", self.test_bp_st)
        X = self.Node_l.index("x") # self.new)
        self.move_cost_result = move_cost_result_X # shortest_path(self.move_cost_result_pre, indices=X, directed=False) # bpで使う
        self.move_cost_result_copy = copy.deepcopy(self.move_cost_result)




        print("-----\nbp.py first Attribute", Attribute)
        self.Attribute = Attribute
        print("move cost は含まれていない->weightのみadv.pyで追加\n-----")
        "============================================== Visualization ver. との違い =============================================="
        
        "----- Add -----"
        self.map = map_viz_test
        # import sys
        # sys.path.append('/Users/ken/Desktop/src/YouTube/mdp')
        
        # from map_viz import DEMO
        # test = DEMO(self.env)
        "->main.pyに移動"
        
        size = self.env.row_length
        # dem = [[0.0 for i in range(size)] for i in range(size)] #2Dgridmap(xw, yw)
        # soil = [[0.0 for i in range(size)] for i in range(size)] #2Dgridmap(xw, yw)
        # map = [[0.0 for i in range(size)] for i in range(size)] #known or unknown
        # for ix in range(size):
        #     for iy in range(size):
        #         map[ix][iy] = 1
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

            
            print("===== Attribute① =====")
            print(self.Attribute)

            # "----- Add 0203 -----"
            # try:
            #     print(self.Attribute)
            #     import matplotlib.pyplot as plt
            #     # print(self.Attribute["stress"])
            #     self.Attribute[:5].plot.bar()
            #     # self.Attribute.plot.bar()
            #     plt.show()
            # except:
            #     print("weight cross エラー")
            # "----- Add 0203 -----"

            

            "============================================== Visualization ver. との違い =============================================="
            "戻る行動の可視化ver.の場合はここにReverseが入る"
            "============================================== Visualization ver. との違い =============================================="

            if self.BACK or self.bf:
                    try:
                        
                        if self.bf: # ストレスが溜まってから初回
                            self.w = self.OBS
                            print(f"🥌 WEIGHT = {self.w}")
                            print("SAVE ARC : {}".format(self.SAVE_ARC))

                            if self.Add_Advance:

                                # ユークリッド距離
                                # self.Arc = [math.sqrt((self.BPLIST[-1].row - self.BPLIST[x].row) ** 2 + (self.BPLIST[-1].column - self.BPLIST[x].column) ** 2) for x in range(len(self.BPLIST))]

                                self.move_cost_cal()

                            print(f"🥌 WEIGHT = {self.w}")
                            print("👟 Arc[移動コスト]:{}".format(self.move_cost_result_copy))
                            print("📂 Storage {}".format(self.test_bp_st))






                            "----- 0130 -----"
                            # # Landmark = "A"
                            # # print(self.Attribute)
                            # # self.Attribute.loc[f"{Landmark}", "move cost"] = 100
                            # # print("===== Attribute =====")
                            # # print(self.Attribute)
                            # # print("===== Attribute =====")
                            # # print(self.demo)
                            # # print("===== Attribute =====")
                            # # print(self.move_cost_result_copy)
                            # # print(self.move_cost_result_copy["A"])
                            print("===== Attribute =====")
                            self.Attribute["move cost"] = self.move_cost_result_copy
                            print("===== Attribute Change =====")
                            print(self.Attribute)
                            "----- 0130 -----"
                            "============================================== Visualization ver. との違い =============================================="  
                        else:
                            print(f"🥌 WEIGHT = {self.w}")
                            print("👟 Arc[移動コスト]:{}".format(self.move_cost_result_copy))
                            print("📂 Storage {}".format(self.test_bp_st))
                        self.bf = False
                        self.BACK = False
                        
                        "----- 0130 -----"
                        # print("===== Attribute =====")
                        # self.Attribute["move cost"] = self.move_cost_result_copy
                        # print("===== Attribute Change =====")
                        # print(self.Attribute)
                        "----- 0130 -----"


                        "----- Add Move Cost -----"

                        print("next position decision")

                        self.next_position_decision()




                        "----- Add 0203 -----"
                        try:
                            print(self.Attribute)
                            # import matplotlib.pyplot as plt
                            # self.Attribute[:5].plot.bar()
                            # # self.Attribute.plot.bar()
                            # plt.show()
                            self.test.bp_viz(self.Attribute)
                        except:
                            print("weight cross エラー")
                        "----- Add 0203 -----"

                        
                        # print(f"========Decision Next State=======\n⚠️  NEXT POSITION:{self.next_position}\n==================================")
                        NP = self.next_attribute["STATE"][0]
                        print(f"========Decision Next State=======\n⚠️  NEXT POSITION:\n{NP}\n==================================")
                        NP = self.next_attribute["STATE"][self.next_attribute.index][0]
                        print(f"========Decision Next State=======\n⚠️  NEXT POSITION:\n{NP}\n==================================")
                        NP = self.next_attribute["STATE"][self.next_attribute.index] # [0]]
                        print(f"========Decision Next State=======\n⚠️  NEXT POSITION:\n{NP}\n==================================")
                        NP = self.next_attribute["STATE"]
                        print(f"========Decision Next State=======\n⚠️  NEXT POSITION:\n{NP}\n==================================")
                        # NP = self.next_attribute["Attribute"]
                        # print(f"========Decision Next State=======\n⚠️  NEXT POSITION:\n{NP}\n==================================")
                        self.on_the_way = True 
                    except:
                    # except Exception as e:
                    #     print('=== エラー内容 ===')
                    #     print('type:' + str(type(e)))
                    #     print('args:' + str(e.args))
                    #     print('message:' + e.message)
                    #     print('e自身:' + str(e))
                        print("ERROR!")
                        print("リトライ行動終了！")
                        print(" = 戻り切った状態 🤖🔚")
                        self.BackPosition_finish = True
                        break
            try:

                # if self.state == self.next_position:
                # if self.state == self.next_attribute["STATE"][self.next_attribute.index[0]]:
                if self.state == self.next_attribute["STATE"][0]:
                    print("===== back end =====")
                    print(self.state, type(self.state))
                    # print(self.next_attribute["STATE"][self.next_attribute.index])
                    print(self.next_attribute["STATE"])
                    # print(self.next_attribute["STATE"][self.next_attribute.index[0]], type(self.next_attribute["STATE"][self.next_attribute.index[0]]))
                    print(self.next_attribute["STATE"][0], type(self.next_attribute["STATE"][0]))
                    print("===== back end =====")

                    self.Backed_just_before = True

                    self.Finished_returning(Node)

                    print("===== bp.py Attribute =====")
                    print(self.Attribute)
                    print("===== bp.py Attribute =====")

                    "----- Add -----"
                    # test.show(self.state, self.map)
                    self.test.show(self.state, self.map, self.backed, {})

                    "----- Add 0203 -----"
                    # self.total_stress = 0
                    "----- Add 0203 -----"

                    

                    print("\n============================\n🤖 🔛　アルゴリズム切り替え\n============================")
                    break

                else:
                    if self.on_the_way:
                        self.on_the_way = False
                    else:
                        print("🔛 On the way BACK")
            except:
            # except Exception as e:
            #         print('=== エラー内容 ===')
            #         print('type:' + str(type(e)))
            #         print('args:' + str(e.args))
            #         print('message:' + e.message)
            #         print('e自身:' + str(e))
                    print("state:{}".format(self.state))
                    print("これ以上戻れません。 終了します。")
                    break # expansion 無しの場合は何回も繰り返さない
                
            print(f"🤖 State:{self.state}")
            if not self.state_history_first:
                self.STATE_HISTORY.append(self.state)
                self.TOTAL_STRESS_LIST.append(self.total_stress)

                "基準距離, 割合の可視化"
                self.test_s = 0
                self.standard_list.append(self.test_s)
                # self.rate_list.append(self.n/(self.M+self.n))    # ○
                self.rate_list.append(self.M/(self.M+self.n))      # ×

                "----- Add -----"
                # test.show(self.state, self.map)
                self.test.show(self.state, self.map, self.backed, {})

            print(f"Total Stress:{self.total_stress}")
            print("TRIGAR : {}".format(self.TRIGAR))
            self.state_history_first = False
            # self.state = self.next_position
            # self.state = self.next_attribute["STATE"][self.next_attribute.index[0]]
            self.state = self.next_attribute["STATE"][0]
            
            "----- Add -----"
            # test.show(self.state, self.map)
            
            print("COUNT : {}".format(self.COUNT))
            if self.COUNT > 100:
                print("\n######## BREAK ########\n")
                # breakではなくて、戻る場所に戻れないから別の戻る場所にするとか
                print("\n📂 Storage {}\n\n\n".format(self.BPLIST))
                break
            self.COUNT += 1
        self.COUNT = 0

        return self.total_stress, self.STATE_HISTORY, self.state, self.OBS, self.BackPosition_finish, self.TOTAL_STRESS_LIST, self.move_cost_result_pre, self.test_bp_st_pre, self.Backed_just_before, self.standard_list, self.rate_list, self.map