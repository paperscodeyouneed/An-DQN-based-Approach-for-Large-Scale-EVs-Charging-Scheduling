
import pickle

file_before = open("formal_ave_queue_b", "rb+")
res_before = pickle.load(file_before)

file_after = open("formal_ave_queue", "rb+")
res_after = pickle.load(file_after)

import matplotlib.pyplot as plt
import numpy as np

print(res_before)

