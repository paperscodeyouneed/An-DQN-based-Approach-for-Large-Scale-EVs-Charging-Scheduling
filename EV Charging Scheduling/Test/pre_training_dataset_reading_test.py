
import pickle
import numpy as np
from _Environment.environment import *

if __name__ == '__main__':

    start = time.time()
    res_ = []
    with open("../Data/experience/EV_Experience/ev_experience", "rb") as f:
        try:
            i = 0
            while True:
                res = pickle.load(f)
                res_.append(res[0]["next_sel_ev_number"])
                # print(i, res[0]["next_sel_ev_number"])
                i += 1
                if i % 100000 == 0:
                    print(i)
        except EOFError:
            pass
    print(time.time() - start)
    print(max(res_))


# 读取EV的 100,000 经验需要 20s
# 读取CS的 100,000 经验需要 2.6s
