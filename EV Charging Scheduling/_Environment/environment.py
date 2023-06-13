# The .py file implements the experiment environment for the large-scale EV (electric vehicle) charging scheduling algo
# All the methods are efficient-tested on PC with 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz   2.30 GHz  RAM 32 GB.

# import all related packages

import _pickle as cpickle
import abc
import argparse
import array
import bs4
import calendar
import copy
import collections
import csv
import datetime
import decimal
import decorator
import enum
import itertools
import jinja2
import logging
import lxml
import math
import matplotlib.pyplot as plt
import numba as nb
import numbers
import numexpr
import numpy as np
import os
import operator
import pandas as pd
import pickle
import ptan
import ptan.ignite
import random
import scipy
import six
import skimage
import sklearn
import symbol
import symtable
import time
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchtext
import urllib
import urllib3
import warnings
import wheel

from IPython.display import clear_output
from collections import deque, defaultdict, namedtuple
from numba import jit
from scipy.special import log_softmax, softmax
from torch.nn.utils import clip_grad, clip_grad_norm, clip_grad_norm_, clip_grad_value_
from typing import Callable, Counter, Dict, List, NamedTuple, Tuple

# mask out warnings
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")


# around 79 methods are defined in _Environment
class Environment(object):

    """
    Notes:
        ev -> electric vehicle
        cp -> charging price
        cs -> charging station
    """

    # 70 ms ± 269 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    def __init__(
            self,
            filename_of_cp_information: str = "../Data/original_dataset/ev_cp_34.csv",
            filename_of_cs_information: str = "../Data/original_dataset/charging_stations.csv",
            filename_of_ev_information: str = "../Data/original_dataset/ev_data_1000.csv",
            charging_device_amount: int = 5,
            t=False,
            s=True) -> None:
        assert filename_of_cs_information is not None
        assert filename_of_cp_information is not None
        assert filename_of_ev_information is not None
        # reading data
        self.__filename_of_cp_information = filename_of_cp_information
        self.__filename_of_cs_information = filename_of_cs_information
        self.__filename_of_ev_information = filename_of_ev_information
        self.__charging_device_amount = charging_device_amount
        self.__cp_data = pd.read_csv(
            self.__filename_of_cp_information, delimiter=",").iloc[:, 1:].values
        self.__cs_data = pd.read_csv(
            self.__filename_of_cs_information, delimiter=";", header=None).iloc[:, 1:].values
        self.__ev_data = pd.read_csv(
            self.__filename_of_ev_information, delimiter=",").iloc[:, [1, 2, 3, 4]].values
        # pre-cleaning data
        self.__ev_x = [j for i in self.__ev_data[:, 0:1] for j in i]
        self.__ev_y = [j for i in self.__ev_data[:, 1:2] for j in i]
        self.__ev_ld = [j for i in self.__ev_data[:, 2:3] for j in i]
        self.__ev_ct = [j for i in self.__ev_data[:, 3:4] for j in i]
        self.__cs_x = [j for i in self.__cs_data[:, 0:1] for j in i]
        self.__cs_y = [j for i in self.__cs_data[:, 1:2] for j in i]
        self.__cp = [j for i in self.__cp_data[:, 0:1] for j in i]
        # define some variables/data structures for constant information (I)
        self.__distance = np.zeros((len(self.__ev_data), len(self.__cs_data)))
        self.__reachability = np.zeros(
            (len(self.__ev_data), len(self.__cs_data)))
        self.__reachable_cs_for_ev = []
        self.__schedulable_ev_list = []
        self.__transfer_n_o = dict()
        self.__transfer_o_n = dict()
        # initialize variables/data structures define in part (I)
        for i in range(len(self.__ev_data)):
            reachable_cs_of_ev_ = []
            for j in range(len(self.__cs_data)):
                distance = np.sqrt(
                    (self.__ev_x[i] - self.__cs_x[j]) ** 2 + (self.__ev_y[i] - self.__cs_y[j]) ** 2)
                if distance < self.__ev_ld[i]:
                    self.__reachability[i][j] = 1.0
                    reachable_cs_of_ev_.append(j)
                else:
                    self.__reachability[i][j] = 0.0
                self.__distance[i][j] = distance
            self.__reachable_cs_for_ev.append(reachable_cs_of_ev_)
            if np.sum(self.__reachability[i]) != 0:
                self.__schedulable_ev_list.append(i)
        for i in range(len(self.__schedulable_ev_list)):
            self.__transfer_n_o[self.__schedulable_ev_list[i]] = i
            self.__transfer_o_n[i] = self.__schedulable_ev_list[i]
        # define some variables/data structures for volatile information (II)
        self.__si = np.zeros(
            (len(
                self.__cs_data), self.__charging_device_amount, len(
                self.__schedulable_ev_list)), dtype=int)
        self.__sip = np.zeros(
            (len(
                self.__cs_data),
                self.__charging_device_amount),
            dtype=int)
        self.__cd_st = np.zeros(
            (len(self.__cs_data), self.__charging_device_amount))
        self.__cd_wt = np.zeros(
            (len(self.__cs_data), self.__charging_device_amount))
        self.__ev_cs = np.zeros(len(self.__cs_data))
        self.__ev_n = np.zeros(
            (len(self.__cs_data), self.__charging_device_amount))
        self.__time = np.zeros(
            (len(self.__cs_data), self.__charging_device_amount, 3))
        self.__time_for_ev = np.zeros(
            (len(
                self.__cs_data), self.__charging_device_amount, len(
                self.__schedulable_ev_list), 3))
        self.__trace = np.zeros(
            (len(self.__schedulable_ev_list) + 1, 5), dtype=int)
        self.__whether_ev_was_scheduled = np.array(
            len(self.__schedulable_ev_list) * [0.0])
        self.__average_speed = 1.0
        self.__backup_for_schedulable_ev_number = copy.deepcopy(
            self.__schedulable_ev_list)
        self.__index = 0
        self.__not_scheduled_ev = copy.deepcopy(self.__schedulable_ev_list)
        self.__scheduled_ev = []
        self.__qt = 0.0
        self.__st = 0.0
        self.__tt = 0.0
        self.__hour_charging_ev = 0
        self.t = t
        self.s = s
        self.__basedir = r"E:\EV_Charging_Scheduling"
        self.filelists = []
        self.whitelist = ['php', 'py']
        # state construct assistance
        self.__ev_left_travel_distance = np.array(
            [self.__ev_ld[i] for i in self.__backup_for_schedulable_ev_number])
        self.__ev_expecting_charging_time = np.array(
            [self.__ev_ct[i] for i in self.__backup_for_schedulable_ev_number])
        self.__distance_between_ev_and_cs = np.array(
            self.__distance)[self.__backup_for_schedulable_ev_number]
        # self.__mask_matrix = np.ones((903, 36))

    # 6.09 ms ± 29.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    def calculate(self) -> None:
        self.__cd_st = np.zeros(
            (len(self.__cs_data), self.__charging_device_amount))
        self.__cd_wt = np.zeros(
            (len(self.__cs_data), self.__charging_device_amount))
        self.__ev_n = np.zeros(
            (len(self.__cs_data), self.__charging_device_amount))
        self.__qt = 0.0
        self.__st = 0.0
        self.__time = np.zeros(
            (len(self.__cs_data), self.__charging_device_amount, 3))
        self.__time_for_ev = np.zeros(
            (len(
                self.__cs_data), self.__charging_device_amount, len(
                self.__schedulable_ev_list), 3))
        for i in range(len(self.__cs_data)):
            for j in range(self.__charging_device_amount):
                for k in range(int(self.__sip[i][j])):
                    if self.t:
                        index = int(self.__transfer_o_n[i])
                    else:
                        index = int(self.__si[i][j][k])
                    if (self.__distance[int(self.__si[i][j][k])]
                            [i] / self.__average_speed) < self.__cd_wt[i][j]:
                        self.__qt += self.__cd_wt[i][j] - \
                            self.__distance[int(self.__si[i][j][k])][i]
                        self.__time_for_ev[i][j][k][0] = self.__cd_wt[i][j] - \
                            self.__distance[int(self.__si[i][j][k])][i]
                        self.__time_for_ev[i][j][k][1] = 0
                        self.__time_for_ev[i][j][k][2] = self.__ev_ct[index]
                        self.__time[i][j][0] += self.__cd_wt[i][j] - \
                            self.__distance[int(self.__si[i][j][k])][i]
                        self.__time[i][j][2] += self.__ev_ct[index]
                        self.__cd_wt[i][j] += self.__ev_ct[index]
                    else:
                        self.__qt += 0
                        self.__st += self.__distance[int(
                            self.__si[i][j][k])][i] - self.__cd_wt[i][j]
                        self.__time_for_ev[i][j][k][0] = 0
                        self.__time_for_ev[i][j][k][1] = self.__distance[int(
                            self.__si[i][j][k])][i] - self.__cd_wt[i][j]
                        self.__time_for_ev[i][j][k][2] = self.__ev_ct[index]
                        self.__time[i][j][1] += self.__distance[int(
                            self.__si[i][j][k])][i] - self.__cd_wt[i][j]
                        self.__time[i][j][2] += self.__ev_ct[index]
                        self.__cd_st[i][j] += self.__distance[int(
                            self.__si[i][j][k])][i] - self.__cd_wt[i][j]
                        self.__cd_wt[i][j] = self.__distance[int(
                            self.__si[i][j][k])][i] + self.__ev_ct[index]
        return None

    # 4.38 µs ± 30.4 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops
    # each)
    def calculate_update(self, cs: int, cd: int) -> None:
        i = cs
        j = cd
        k = int(self.__sip[i][j]) - 1
        if self.t is True:
            index = int(self.__transfer_o_n[i])
        else:
            index = int(self.__si[i][j][k])
        subtract = self.__cd_wt[i][j] - \
            self.__distance[int(self.__si[i][j][k])][i]
        if self.__distance[int(self.__si[i][j][k])][i] < self.__cd_wt[i][j]:
            self.__qt += subtract
            self.__time_for_ev[i][j][k][0] = subtract
            self.__time_for_ev[i][j][k][1] = 0
            self.__time_for_ev[i][j][k][2] = self.__ev_ct[index]
            self.__time[i][j][0] += subtract
            self.__time[i][j][2] += self.__ev_ct[index]
            self.__cd_wt[i][j] += self.__ev_ct[index]
        else:
            self.__st += -subtract
            self.__time_for_ev[i][j][k][0] = 0
            self.__time_for_ev[i][j][k][1] = -subtract
            self.__time_for_ev[i][j][k][2] = self.__ev_ct[index]
            self.__time[i][j][1] += -subtract
            self.__time[i][j][2] += self.__ev_ct[index]
            self.__cd_st[i][j] += -subtract
            self.__cd_wt[i][j] = self.__distance[int(
                self.__si[i][j][k])][i] + self.__ev_ct[index]
        return None

    # 10.8 µs ± 102 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops
    # each)
    def count(self) -> None:
        self.get_file()
        total_line = 0
        for filelist in self.filelists:
            total_line += self.count_line(filelist)
        print('total lines:', total_line)
        return None

    # 165 µs ± 1.69 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    @staticmethod
    def count_line(f_name) -> int:
        count = 0
        for file_line in open(f_name,
                              encoding="utf=8").readlines():
            if file_line != '':
                count += 1
        print(f_name + '----', count)
        return count

    # 31.6 µs ± 91.6 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops
    # each)
    def find_position(self, ev_number: int) -> Tuple:
        assert ev_number in self.__scheduled_ev
        for i in range(len(self.__trace)):
            if self.__trace[i][1] == ev_number:
                return self.__trace[i]

    # 3.39 µs ± 25.1 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops
    # each)
    def get_average_charging_time(self, cs_number: int) -> float:
        assert cs_number in range(len(self.__cs_data))
        c_time = 0
        num = 0
        for i in range(self.__charging_device_amount):
            c_time += self.__time[cs_number][i][2]
            num += self.__sip[cs_number][i]
        average_charging_time = c_time / num
        return average_charging_time

    # 7.61 µs ± 22 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    def get_average_distance_of_ev_to_cs(self, ev_number: int) -> float:
        assert ev_number in self.__backup_for_schedulable_ev_number
        distance = self.__distance[ev_number]
        sum_ = 0
        count = 0
        for i in range(len(distance)):
            if self.__reachability[ev_number][i] == 1.0:
                sum_ += distance[i]
                count += 1
        sum_ /= count
        return sum_

    # 3.97 µs ± 29.4 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops
    # each)
    def get_average_first_k_distance_of_ev_to_cs(
            self, ev_number: int, k: int) -> float:
        assert ev_number in self.__backup_for_schedulable_ev_number
        distance = self.__distance[ev_number]
        sum_ = 0
        count = 0
        for i in range(len(distance)):
            if self.__reachability[ev_number][i] == 1.0:
                sum_ += distance[i]
                count += 1
                if count >= k:
                    break
        sum_ /= count
        return sum_

    # 3.25 µs ± 48.5 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops
    # each)
    def get_average_idling_time(self, cs_number: int) -> float:
        assert cs_number in range(len(self.__cs_data))
        i_time = 0
        num = 0
        for i in range(self.__charging_device_amount):
            i_time += self.__time[cs_number][i][1]
            num += self.__sip[cs_number][i]
        average_idle_time = i_time / num
        return average_idle_time

    # 3.34 µs ± 16.2 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops
    # each)
    def get_average_queueing_time(self, cs_number: int) -> float:
        assert cs_number in range(len(self.__cs_data))
        q_time = 0
        num = 0
        for i in range(self.__charging_device_amount):
            q_time += self.__time[cs_number][i][0]
            num += self.__sip[cs_number][i]
        average_queueing_time = q_time / num
        return average_queueing_time

    # 91.2 µs ± 256 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    def get_average_time(self) -> Tuple:
        ave_queue = self.__qt / len(self.__scheduled_ev)
        ave_idle = self.__st / (len(self.__cs_data) *
                                self.__charging_device_amount)
        ave_charging = 0
        for i in range(len(self.__scheduled_ev)):
            ave_charging += self.__ev_ct[self.__scheduled_ev[i]]
        ave_charging /= len(self.__scheduled_ev)
        return ave_idle, ave_queue, ave_charging

    # 182 µs ± 843 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    def get_average_time_for_cs(self) -> Tuple:
        average_charging_time = np.zeros(len(self.__cs_data))
        average_idling_time = np.zeros(len(self.__cs_data))
        average_queueing_time = np.zeros(len(self.__cs_data))
        for i in range(len(self.__cs_data)):
            average_charging_time_temp = 0.0
            average_idling_time_temp = 0.0
            average_queueing_time_temp = 0.0
            for j in range(self.__charging_device_amount):
                average_queueing_time_temp += self.__time[i][j][0]
                average_idling_time_temp += self.__time[i][j][1]
                average_charging_time_temp += self.__time[i][j][2]
            average_charging_time_temp /= self.__charging_device_amount
            average_idling_time_temp /= self.__charging_device_amount
            average_queueing_time_temp /= self.__charging_device_amount
            average_charging_time[i] = average_charging_time_temp
            average_idling_time[i] = average_idling_time_temp
            average_queueing_time[i] = average_queueing_time_temp
        return average_queueing_time, average_idling_time, average_charging_time

    # 1.42 µs ± 4.28 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops
    # each)
    def get_best_charging_device(self, cs_number: int) -> int:
        assert cs_number in range(len(self.__cs_data))
        best_slot = int(sum(self.__sip[cs_number]) %
                        self.__charging_device_amount)
        return best_slot

    # 4.38 µs ± 19.1 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops
    # each)
    def get_best_charging_device_(self, cs_number: int) -> int:
        assert cs_number in range(len(self.__cs_data))
        o = [
            self.get_occupied_time_of_charging_device(
                cs_number, i) for i in range(
                self.__charging_device_amount)]
        return int(np.argmin(np.array(o)))

    # 93.4 ns ± 0.399 ns per loop (mean ± std. dev. of 7 runs, 10,000,000
    # loops each)
    def get_brief_time_matrix(self) -> np.ndarray:
        ret = self.__time
        return ret

    # 92.1 ns ± 0.2 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops
    # each)
    def get_charging_device_count_on_one_charging_station(self) -> int:
        return self.__charging_device_amount

    # 91.7 ns ± 0.179 ns per loop (mean ± std. dev. of 7 runs, 10,000,000
    # loops each)
    def get_charging_device_number(self):
        return self.__charging_device_amount

    # 1.62 µs ± 2.11 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops
    # each)
    def get_charging_ev_number_for_concrete_cs(self, cs_number: int) -> int:
        assert cs_number in range(len(self.__cs_data))
        sum_ = 0
        for i in range(self.__charging_device_amount):
            sum_ += self.__sip[cs_number][i]
        return sum_

    # 115 ns ± 0.241 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops
    # each)
    def get_charging_station_number(self) -> int:
        return len(self.__cs_data)

    # 154 ns ± 0.568 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops
    # each)
    def get_charging_time(self, ev_number: int) -> float:
        assert ev_number in self.__backup_for_schedulable_ev_number
        return self.__ev_ct[ev_number]

    # 345 ns ± 0.954 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops
    # each)
    def get_charging_price(self, cs_number: int):
        assert cs_number in range(len(self.__cs_data))
        price = self.__cp[cs_number]
        return price

    # 343 ns ± 4.97 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops
    # each)
    def get_cs_x_coordination(self, cs_number: int):
        assert cs_number in range(len(self.__cs_data))
        x = self.__cs_x[cs_number]
        return x

    # 340 ns ± 0.443 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops
    # each)
    def get_cs_y_coordination(self, cs_number: int):
        assert cs_number in range(len(self.__cs_data))
        y = self.__cs_y[cs_number]
        return y

    # 410 µs ± 2.59 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    def get_current_cs_state(self, sel_ev_number: int) -> np.ndarray:
        ev_left_travel_distance = self.__ev_left_travel_distance
        ev_expecting_charging_time = self.__ev_expecting_charging_time
        distance_between_ev_and_cs = self.__distance_between_ev_and_cs
        average_queueing_time = np.concatenate(
            (self.get_average_time_for_cs()[0], np.array([0.0, 0.0])), axis=-1)
        average_idling_time = np.concatenate(
            (self.get_average_time_for_cs()[1], np.array([0.0, 0.0])), axis=-1)
        charging_number_on_every_cs = np.concatenate(
            (np.sum(self.__sip, axis=-1), np.array([0.0, 0.0])), axis=-1)
        occupied_time_of_every_cs = np.concatenate((np.sum(
            self.__cd_wt, axis=1) / self.__charging_device_amount, np.array([0.0, 0.0])))
        ev_part = np.concatenate((distance_between_ev_and_cs,
                                  ev_expecting_charging_time.reshape(-1,
                                                                     1),
                                  ev_left_travel_distance.reshape(-1,
                                                                  1),
                                  ),
                                 axis=-1)[self.__transfer_n_o[sel_ev_number]].reshape(1,
                                                                                      -1)
        cs_part = np.concatenate((average_queueing_time.reshape(1, -1),
                                  average_idling_time.reshape(1, -1),
                                  charging_number_on_every_cs.reshape(1, -1),
                                  occupied_time_of_every_cs.reshape(1, -1)),
                                 axis=0)
        cs_state = np.concatenate((ev_part, cs_part), axis=0)[np.newaxis, :]
        return cs_state

    # 429 µs ± 1.93 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    def get_current_ev_state(self) -> np.ndarray:
        ev_left_travel_distance = self.__ev_left_travel_distance
        ev_expecting_charging_time = self.__ev_expecting_charging_time
        distance_between_ev_and_cs = self.__distance_between_ev_and_cs
        average_queueing_time = np.concatenate(
            (self.get_average_time_for_cs()[0], np.array([0.0, 0.0])), axis=-1)
        average_idling_time = np.concatenate(
            (self.get_average_time_for_cs()[1], np.array([0.0, 0.0])), axis=-1)
        charging_number_on_every_cs = np.concatenate(
            (np.sum(self.__sip, axis=-1), np.array([0.0, 0.0])), axis=-1)
        occupied_time_of_every_cs = np.concatenate(
            (np.sum(self.__cd_wt, axis=1) / self.__charging_device_amount, np.array([0.0, 0.0])))
        ev_part = np.concatenate((distance_between_ev_and_cs,
                                  ev_expecting_charging_time.reshape(-1, 1),
                                  ev_left_travel_distance.reshape(-1, 1),),
                                 axis=-1)
        cs_part = np.concatenate((average_queueing_time.reshape(1, -1),
                                  average_idling_time.reshape(1, -1),
                                  charging_number_on_every_cs.reshape(1, -1),
                                  occupied_time_of_every_cs.reshape(1, -1)), axis=0)
        ev_state = np.concatenate((ev_part, cs_part), axis=0)[np.newaxis, :]
        # ev_state = ev_state * self.__mask_matrix
        return ev_state

    # 23.4 µs ± 61.7 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops
    # each)
    def get_current_schedulable_ev(self) -> np.ndarray:
        result = np.array(self.__schedulable_ev_list, dtype=np.int)
        return result

    # 94.9 ns ± 0.411 ns per loop (mean ± std. dev. of 7 runs, 10,000,000
    # loops each)
    def get_detailed_time_matrix(self) -> np.ndarray:
        ret = self.__time_for_ev
        return ret

    # 615 ns ± 1.8 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops
    # each)
    def get_distance(self, ev_number: int, cs_number: int) -> float:
        assert ev_number in self.__backup_for_schedulable_ev_number
        assert cs_number in range(len(self.__cs_data))
        distance = self.__distance[ev_number][cs_number]
        return distance

    # 156 ns ± 0.889 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops
    # each)
    def get_ev_charging_time(self, ev_number: int) -> float:
        assert ev_number in self.__backup_for_schedulable_ev_number
        return self.__ev_ct[ev_number]

    # 154 ns ± 0.276 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops
    # each)
    def get_ev_left_travel_distance(self, ev_number: int) -> float:
        assert ev_number in self.__backup_for_schedulable_ev_number
        return self.__ev_ld[ev_number]

    # 216 ns ± 1.06 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops
    # each)
    def get_ev_position(self, ev_number: int) -> Tuple:
        assert ev_number in self.__backup_for_schedulable_ev_number
        x = self.__ev_x[ev_number]
        y = self.__ev_y[ev_number]
        return x, y

    # 153 ns ± 0.412 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops
    # each)
    def get_ev_x_coordination(self, ev_number: int) -> float:
        assert ev_number in self.__backup_for_schedulable_ev_number
        x = self.__ev_x[ev_number]
        return x

    # 157 ns ± 0.33 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops
    # each)
    def get_ev_y_coordination(self, ev_number: int) -> float:
        assert ev_number in self.__backup_for_schedulable_ev_number
        y = self.__ev_y[ev_number]
        return y

    # 6.2 µs ± 5.14 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops
    # each)
    def get_file(self) -> None:
        for parent, dir_names, filenames in os.walk(self.__basedir):
            for filename in filenames:
                ext = filename.split('.')[-1]
                if ext in self.whitelist:
                    self.filelists.append(os.path.join(parent, filename))
        return None

    # 90.6 ns ± 0.211 ns per loop (mean ± std. dev. of 7 runs, 10,000,000
    # loops each)
    def get_hour_charging_ev(self):
        return self.__hour_charging_ev

    # 134 ns ± 6.7 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops
    # each)
    def get_how_many_ev_have_been_scheduled(self) -> int:
        return len(self.__scheduled_ev)

    # 114 ns ± 0.378 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops
    # each)
    def get_how_many_ev_need_to_be_scheduled(self) -> int:
        return len(self.__not_scheduled_ev)

    # 90.9 ns ± 0.81 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops
    # each)
    def get_idling_time(self) -> float:
        return self.__st

    # 91.6 ns ± 0.687 ns per loop (mean ± std. dev. of 7 runs, 10,000,000
    # loops each)
    # def get_mask_matrix(self) -> np.array:
    #     return self.__mask_matrix

    # 90.7 ns ± 0.207 ns per loop (mean ± std. dev. of 7 runs, 10,000,000
    # loops each)
    def get_not_scheduled_ev(self) -> list:
        return self.__not_scheduled_ev

    # 245 ns ± 0.581 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops
    # each)
    def get_occupied_time_of_charging_device(
            self, cs_number: int, charging_device_number: int) -> float:
        return self.__cd_wt[cs_number][charging_device_number]

    # 92.3 ns ± 0.526 ns per loop (mean ± std. dev. of 7 runs, 10,000,000
    # loops each)
    def get_queueing_time(self):
        return self.__qt

    # 42.8 µs ± 113 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    def get_reachable_cs_list_for_ev(self, ev_number: int) -> np.ndarray:
        assert ev_number in self.__backup_for_schedulable_ev_number
        reachable_cs_list_for_ev = np.array(
            self.__reachable_cs_for_ev)[ev_number]
        return reachable_cs_list_for_ev

    # 568 ns ± 1.96 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops
    # each)
    def get_reachability_of_an_ev(self, ev_number: int) -> np.ndarray:
        reachability = np.array(self.__reachability[ev_number])
        return reachability

    # 92.3 ns ± 0.256 ns per loop (mean ± std. dev. of 7 runs, 10,000,000
    # loops each)
    def get_reachability_matrix(self) -> np.ndarray:
        return self.__reachability

    # 691 ns ± 13.9 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops
    # each)
    def get_reward(self, q: float = 1.0, i: float = 0.0) -> float:
        reward = self.__qt / len(self.__scheduled_ev) * q + self.__st / \
            (len(self.__cs_data) * self.__charging_device_amount) * i
        return reward

    # 3.72 µs ± 27.6 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops
    # each)
    def get_reward_for_cs(self, cs_number: int, ev_number: int) -> float:
        assert ev_number in self.__backup_for_schedulable_ev_number
        assert cs_number in range(len(self.__cs_data))
        distance = self.__distance[ev_number][cs_number]
        occupied = np.sum(self.__cd_wt[cs_number]) / \
            self.__charging_device_amount
        reward = occupied + distance  # 找那些到那里+充电时间少的排队时间自然就少
        return reward

    # 4.11 µs ± 20.2 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops
    # each)
    def get_reward_for_ev(
            self,
            ev_number: int,
            cs_number: int,
            cd_number: int,
            q: float = 1.0,
            i: float = 0.0) -> float:
        assert ev_number in self.__backup_for_schedulable_ev_number
        assert cs_number in range(len(self.__cs_data))
        assert cd_number in range(self.__charging_device_amount)
        idle = 0
        queue = 0
        for pointer in range(int(self.__sip[cs_number][cd_number])):
            if self.__si[cs_number][cd_number][pointer] == ev_number:
                queue = self.__time_for_ev[cs_number][cd_number][pointer][0]
                idle = self.__time_for_ev[cs_number][cd_number][pointer][1]
                break
        weighted_reward = idle * i + queue * q
        return weighted_reward

    # 23.8 µs ± 565 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    def get_schedulable_ev(self) -> np.ndarray:
        result = np.array(
            self.__backup_for_schedulable_ev_number,
            dtype=np.int)
        return result

    # 93.9 ns ± 0.44 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops
    # each)
    def get_scheduled_ev(self) -> list:
        return self.__scheduled_ev

    # 91.9 ns ± 0.307 ns per loop (mean ± std. dev. of 7 runs, 10,000,000
    # loops each)
    def get_sip(self) -> np.ndarray:
        return self.__sip

    # 610 ns ± 2.79 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops
    # each)
    def get_travel_time_from_ev_to_cs(
            self, ev_number: int, cs_number: int) -> float:
        assert ev_number in self.__backup_for_schedulable_ev_number
        assert cs_number in range(len(self.__cs_data))
        return self.__distance[ev_number][cs_number]

    # 806 µs ± 10.2 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    @staticmethod
    def global_settings():
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        seed = 777
        torch.manual_seed(seed=seed)
        np.random.seed(seed)
        random.seed(seed)

    # 137 ns ± 0.562 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops
    # each)
    def is_done(self) -> bool:
        boolean_value = len(self.__not_scheduled_ev) == 0
        return boolean_value

    # 146 ms ± 512 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    def optimize(self) -> List:
        new_result = []
        for i in range(len(self.__cs_x)):
            sub_ = []
            for j in range(self.__charging_device_amount):
                if self.__sip[i][j] == 0:
                    continue
                if self.__sip[i][j] != 0:
                    r_in_slot_ = self.optimize_(i, j)
                    sub_.append(r_in_slot_)
            new_result.append(sub_)
        self.reset()
        for i in range(len(new_result)):
            for j in range(len(new_result[i])):
                for k in range(len(new_result[i][j])):
                    self.step(new_result[i][j][k], i, j)
        return new_result

    # 227 µs ± 1.47 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    def optimize_(self, cs_number: int, cd_number: int) -> List:
        assert cs_number in range(len(self.__cs_data))
        assert cd_number in range(self.__charging_device_amount)
        assert self.__sip[cs_number][cd_number] != 0
        si = self.__si[cs_number][cd_number]
        sip = self.__sip[cs_number][cd_number]
        distance = []
        for i in range(sip):
            distance.append(self.get_distance(si[i], cs_number))
        ev_no_list = []
        occupy_list = []
        final = []
        cs_occupy = 0
        for i in range(sip):
            charging_time = self.get_ev_charging_time(si[i])
            travel_time = self.get_distance(si[i], cs_number)
            ev_no_list.append(si[i])
            occupy_list.append(charging_time + 2 * travel_time)
        first_ev = True
        for i in range(sip):
            if cs_occupy == 0:
                ev_index = distance.index(min(distance))
                ev_no = ev_no_list[ev_index]
            else:
                ev_index = occupy_list.index(min(occupy_list))
                ev_no = ev_no_list[ev_index]
            ev_no_list.remove(ev_no)
            occupy_list.remove(occupy_list[ev_index])
            final.append(ev_no)
            if first_ev:
                cs_occupy += self.__distance[ev_no][cs_number] + \
                    self.get_ev_charging_time(ev_no)
                first_ev = False
            else:
                if cs_occupy >= self.__distance[ev_no][cs_number]:
                    cs_occupy += self.get_ev_charging_time(ev_no)
                else:
                    cs_occupy = self.__distance[ev_no][cs_number] + \
                        self.get_ev_charging_time(ev_no)
            for i_ in range(len(occupy_list)):
                ev = ev_no_list[i_]
                cs = cs_number
                if cs_occupy >= self.__distance[ev][cs]:
                    occupy_list[ev_no_list.index(
                        ev)] = self.get_ev_charging_time(ev)
                else:
                    occupy_list[ev_no_list.index(ev)] = (self.get_ev_charging_time(
                        ev) + 2 * (self.__distance[ev][cs] - cs_occupy))
        return final

    # more than 193 ns ± 0.532 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def print_scheduling_consequence_info(self) -> None:
        itn = self.__charging_device_amount * len(self.__cs_data)
        qtn = len(self.__backup_for_schedulable_ev_number)
        print(
            "|||",
            "average idling time = {:.3f}".format(
                self.__st / itn),
            "|||",
            "average queueing time = {:.3f}".format(
                self.__qt / qtn),
            "|||",
            end="\t")
        return None

    # more than 193 ns ± 0.532 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    def print_scheduling_info(self) -> None:
        for i in range(len(self.__cs_data)):
            self.print_scheduling_info_for_one_cs(i)
        return None

    # 85.4 µs ± 902 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    def print_scheduling_info_for_one_cs(self, cs_number: int) -> None:
        assert cs_number in range(len(self.__cs_data))
        for cd in range(self.__charging_device_amount):
            print("CS--{:5s},SLOT--{:5s}--({:})---->".format(str(cs_number),
                  str(cd), int(self.__sip[cs_number][cd])), end=" ")
            for k in range(int(self.__sip[cs_number][cd])):
                print(int(self.__si[cs_number][cd][k]), end="\t")
            print("")
        return None

    # 72 µs ± 807 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    def print_scheduling_consequence_list(self) -> list:
        assert self.is_done() is True
        result = []
        all_ev = 0
        for i in range(self.get_charging_station_number()):
            for j in range(self.__charging_device_amount):
                for k in range(self.__sip[i][j]):
                    temp = [self.__si[i][j][k], i, j]
                    result.append(temp)
                    all_ev += 1
                    if (all_ev + 1) % 20 == 0:
                        print()
        return result

    # more than 5.24 µs ± 69.9 ns per loop (mean ± std. dev. of 7 runs,
    # 100,000 loops each)
    def print_state_of_cs(self, sel_ev_number: int) -> None:
        assert sel_ev_number in self.__backup_for_schedulable_ev_number
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        print(self.get_current_cs_state(sel_ev_number))
        return None

    # more than 5.09 µs ± 21.1 ns per loop (mean ± std. dev. of 7 runs,
    # 100,000 loops each)
    def print_state_of_ev(self) -> None:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        print(self.get_current_ev_state())
        return None

    # 68.7 ms ± 335 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    def reset(self):
        self.__init__(
            self.__filename_of_cp_information,
            self.__filename_of_cs_information,
            self.__filename_of_ev_information)

    # 53 µs ± 458 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    def ret_scheduling_consequence_list(self) -> list:
        assert self.is_done() is True
        result = []
        for i in range(self.get_charging_station_number()):
            for j in range(self.__charging_device_amount):
                for k in range(self.__sip[i][j]):
                    temp = [self.__si[i][j][k], i, j]
                    result.append(temp)
        return result

    # 151 ns ± 0.324 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops
    # each)
    def transfer_ev_no_to_order(self, ev_number: int) -> int:
        assert ev_number in self.__schedulable_ev_list
        return self.__transfer_n_o[ev_number]

    # 209 ns ± 0.985 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops
    # each)
    def transfer_ev_order_to_no(self, ev_order: int) -> int:
        assert ev_order <= len(self.__schedulable_ev_list) + 1
        return self.__transfer_o_n[ev_order]

    # 19.3 µs ± 182.5 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops
    # each)
    def step(self, ev_number: int, cs_number: int,
             charging_device_number: int) -> None:
        assert self.__reachability[ev_number][cs_number] == 1.0
        assert ev_number in self.__not_scheduled_ev
        pointer = int(self.__sip[cs_number][charging_device_number])
        self.__si[cs_number][charging_device_number][pointer] = ev_number
        self.__sip[cs_number][charging_device_number] += 1
        self.__scheduled_ev.append(ev_number)
        self.__not_scheduled_ev.remove(ev_number)
        self.__trace[self.__index][0] = self.__index  # count from 0
        self.__trace[self.__index][1] = ev_number
        self.__trace[self.__index][2] = cs_number
        self.__trace[self.__index][3] = charging_device_number
        self.__trace[self.__index][4] = pointer
        self.__index += 1
        self.__whether_ev_was_scheduled[self.__transfer_n_o[ev_number]] = 1.0
        self.__ev_n[cs_number][charging_device_number] += 1
        self.__ev_cs[cs_number] += 1
        if self.__cd_wt[cs_number][charging_device_number] <= 60:
            self.__hour_charging_ev += 1
        if not self.s:
            self.calculate()
        else:
            self.calculate_update(cs_number, charging_device_number)
        # self.__mask_matrix[self.transfer_ev_no_to_order(ev_number)] *= 0
        return None

    # 19.3 µs ± 182.5 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    def un_step(self, ev_number: int) -> None:
        assert ev_number in self.__scheduled_ev
        index, ev_number, cs_number, cd_number, scheduling_order = self.find_position(
            ev_number)
        for i in range(scheduling_order, self.__sip[cs_number][cd_number]):
            self.__si[cs_number][cd_number][i] = self.__si[cs_number][cd_number][i + 1]
        self.__si[cs_number][cd_number][self.__sip[cs_number][cd_number]] = 0
        self.__sip[cs_number][cd_number] -= 1
        for i in range(index, len(self.__scheduled_ev)):
            self.__trace[i] = self.__trace[i + 1]
        self.__scheduled_ev.remove(ev_number)
        self.__not_scheduled_ev.append(ev_number)
        self.__index -= 1
        self.__whether_ev_was_scheduled[self.__transfer_n_o[ev_number]] = 0.0
        self.__ev_n[cs_number][cd_number] -= 1
        self.__ev_cs[cs_number] -= 1
        if self.s:
            self.calculate_update(cs_number, cd_number)
        else:
            self.calculate()
        # self.__mask_matrix[self.transfer_ev_no_to_order(ev_number)] += 1
        return None

    # executing time is need not test
    def v_generate_idling_time_according_to_exp_v1(self, experience):
        self.reset()  # exp==[real ev number, cs number, cd number]
        for exp in experience:
            ev = exp[0]
            cs = exp[1]
            cd = exp[2]
            self.step(ev, cs, cd)
        res = self.get_idling_time() / (self.get_charging_station_number() * self.__charging_device_amount)
        self.reset()
        return res

    # executing time is need not test
    def v_generate_idling_time_according_to_exp_v2(self, experience):
        self.reset()  # exp==[cs number, distance between ev and cs, relative ev number]
        for exp in experience:
            ev = self.transfer_ev_order_to_no(exp[2])
            cs = exp[0]
            cd = self.get_best_charging_device(cs)
            self.step(ev, cs, cd)
        res = self.get_idling_time() / (self.get_charging_station_number() * self.__charging_device_amount)
        self.reset()
        return res

    # executing time is need not test
    def v_generate_queueing_time_according_to_exp_v1(self, experience):
        self.reset()  # exp==[real ev number, cs number, cd number]
        for exp in experience:
            ev = exp[0]
            cs = exp[1]
            cd = exp[2]
            self.step(ev, cs, cd)
        res = self.get_queueing_time() / len(self.__backup_for_schedulable_ev_number)
        self.reset()
        return res

    # executing time is need not test
    def v_generate_queueing_time_according_to_exp_v2(self, experience):
        self.reset()  # exp==[cs number, distance between ev and cs, relative ev number]
        for exp in experience:
            ev = self.transfer_ev_order_to_no(exp[2])
            cs = exp[0]
            cd = self.get_best_charging_device(cs)
            self.step(ev,
                      cs,
                      cd)
        res = self.get_queueing_time() / len(self.__backup_for_schedulable_ev_number)
        self.reset()
        return res

    # executing time is need not test
    def v_generate_scheduled_ev_in_an_hour_according_to_exp_v1(self, experience):
        self.reset()  # exp==[real ev number, cs number, cd number]
        for exp in experience:
            ev = exp[0]
            cs = exp[1]
            cd = exp[2]
            self.step(ev, cs, cd)
        res = self.get_hour_charging_ev()
        self.reset()
        return res

    # executing time is need not test
    def v_generate_scheduled_ev_in_an_hour_according_to_exp_v2(self, experience):
        self.reset()  # exp==[cs number, distance between ev and cs, relative ev number]
        for exp in experience:
            ev = self.transfer_ev_order_to_no(exp[2])
            cs = exp[0]
            cd = self.get_best_charging_device(cs)
            self.step(ev, cs, cd)
        res = self.get_hour_charging_ev()
        self.reset()
        return res

    # executing time is need not test
    def v_get_ait_aqt_eih_v1(self, experience):
        self.reset()  # exp==[real ev number, cs number, cd number]
        for exp in experience:
            ev = exp[0]
            cs = exp[1]
            cd = exp[2]
            self.step(ev, cs, cd)
        ait = self.get_idling_time() / (self.get_charging_station_number() * self.__charging_device_amount)
        aqt = self.get_queueing_time() / len(self.__backup_for_schedulable_ev_number)
        eih = self.get_hour_charging_ev()
        self.reset()
        return ait, aqt, eih

    # executing time is need not test
    def v_get_ait_aqt_eih_v2(self, experience):
        self.reset()  # exp==[cs number, distance between ev and cs, relative ev number]
        for exp in experience:
            ev = self.transfer_ev_order_to_no(exp[2])
            cs = exp[0]
            cd = self.get_best_charging_device(cs)
            self.step(ev, cs, cd)
        ait = self.get_idling_time() / (self.get_charging_station_number() * self.__charging_device_amount)
        aqt = self.get_queueing_time() / len(self.__backup_for_schedulable_ev_number)
        eih = self.get_hour_charging_ev()
        self.reset()
        return ait, aqt, eih

    # executing time is need not test
    def verify_scheduling_calculate_correctness_v1(self, experience, result):
        assert type(experience) == list  # exp==[real ev number, cs number, cd number]
        self.reset()
        for exp in experience:
            ev = exp[0]
            cs = exp[1]
            cd = exp[2]
            self.step(ev, cs, cd)
        self.reset()
        return result - self.get_average_time()[1] < 1e-2

    # executing time is need not test
    def verify_scheduling_calculate_correctness_v2(self, experience, result):
        assert type(experience) == list  # exp==[cs number, distance between ev and cs, relative ev number]
        self.reset()
        for exp in experience:
            ev = self.transfer_ev_order_to_no(exp[2])
            cs = exp[0]
            cd = self.get_best_charging_device(cs)
            self.step(ev, cs, cd)
        self.reset()
        return result - self.get_average_time()[1] < 1e-2
