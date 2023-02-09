
from lib2to3.pytree import Node
from sre_parse import State
import numpy as np
from itertools import count
import pprint




class Property():
    def __init__(self, *arg):

        
        self.pre = np.array([

                            # [5, "g"],
                            # [5, "D"],
                            # [5, "C"],
                            # [5, "B"],
                            # [5, "A"],
                            # [5, "O"],
                            # [0, "s"]
                            # 1205
                            # [2, "g"],
                            # [2, "D"],
                            # [2, "C"],
                            # [2, "B"],
                            # [2, "A"],
                            # [2, "O"],
                            # [0, "s"]

                            
                            # [3, "g"],
                            # [3, "D"],
                            # [3, "C"],
                            # [3, "B"],
                            # [3, "A"],
                            # [3, "O"],
                            # [0, "s"]
                            # 0203
                            [3, "g"],
                            [3, "O"],
                            [3, "F"],
                            [3, "E"],
                            [3, "D"],
                            [3, "C"],
                            [3, "B"],
                            [3, "A"],
                            [0, "s"]
                            
                            ])

    
    def reference(self):
        
        
        Node = self.pre[:, 1]
        Arc = self.pre[:, 0]
        # print(Node)
        # print(Arc)
        Node = Node.tolist()
        Arc = Arc.tolist()
        num = [float(i) for i in Arc]
        # print(type(num))
        Arc_sum = sum(num)
        # print(Arc_sum)

        PERMISSION = [
                
                
                # [Arc_sum-float(Arc[5])-float(Arc[4])-float(Arc[3])-float(Arc[2])-float(Arc[1])-float(Arc[0])],
                # [Arc_sum-float(Arc[5])-float(Arc[4])-float(Arc[3])-float(Arc[2])-float(Arc[1])],
                # [Arc_sum-float(Arc[5])-float(Arc[4])-float(Arc[3])-float(Arc[2])],
                # [Arc_sum-float(Arc[5])-float(Arc[4])-float(Arc[3])],
                # [Arc_sum-float(Arc[5])-float(Arc[4])],
                # [Arc_sum-float(Arc[5])],
                # [Arc_sum]

                
                [0],
                [3],
                [6],
                [9],
                [12],
                [15],
                [18],
                [21],
                [24]
        ]

        return self.pre, Node, Arc, Arc_sum, PERMISSION

if __name__ == "__main__":
   test = Property()
   test.reference