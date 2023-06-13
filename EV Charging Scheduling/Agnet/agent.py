import pickle

import numpy as np
import torch

from _Environment.environment import *
from Models.CS_mdoel import *
from Models.EV_model import *
from Utils.data_handler import *
from Utils.dataset_constructor import *
from Utils.ou_noise import *
from Utils.prioritized_replay_buffer import *
from Utils.replay_buffer import *
from Utils.segment_tree import *


class Agent(object):

    """
    define a class used as an agent for reinforcement learning
    """
    def __init__(self,
                 batch_size: int,
                 target_update_: int,
                 gamma: float = 0.6,
                 alpha: float = 0.2,
                 beta: float = 0.6,
                 prior_eps: float = 1e-6,
                 device = "cuda"):
        """ common class utils """
        self.env = Environment()
        self.ds_constructor = DatasetConstructor()
        """ replay buffer """
        # EV-part
        self.er_ev = ReplayBuffer(
            is_ev=True,
            batch_size=batch_size,
            size=89900
        )
        self.dh_ev = DataHandler(is_ev=True)
        # CS-part
        self.er_cs = ReplayBuffer(is_ev=False,
                                  batch_size=batch_size,
                                  size=89900)
        self.dh_cs = DataHandler(is_ev=False)
        """ hyperparameters """
        self.batch_size = batch_size
        self.target_hard_update_ = target_update_
        self.alpha = alpha
        self.beta = beta
        self.prior_eps = prior_eps
        self.pre_training_round = 80000
        self.cs_gamma = gamma
        self.ev_gamma = gamma
        """ common variables """
        self.device = "cuda"
        self.is_test = False
        self.trajectory_len = len(self.env.get_schedulable_ev())
        self.reachability_matrix = torch.Tensor(self.env.get_reachability_matrix()).to(self.device)
        self.total_steps = 0
        self.update_cnt = 0
        """ models """
        # EV-part
        self.ev_model = EV_SELECTING_MODEL().to(self.device)
        self.ev_model_target = EV_SELECTING_MODEL().to(self.device)
        self.ev_model_target.load_state_dict(self.ev_model.state_dict())
        # CS-part
        self.cs_model = CS_SELECTING_MODEL().to(self.device)
        """ optimizers for model-optimization """
        self.cs_optim = optim.Adam(self.cs_model.parameters(), lr=0.001)
        self.ev_optim = optim.Adagrad(self.ev_model.parameters(), lr=0.0001)
        self.init_storage_pool()

    def init_storage_pool(self) -> None:
        # storage pool for ev experience
        self.ev_current_state_ = list()
        self.ev_action_ = list()
        self.ev_reward_ = list()
        self.ev_next_state_ = list()
        self.ev_done_flag_ = list()
        self.ev_current_ev_number_ = list()
        self.ev_current_cs_number_ = list()
        self.ev_current_cd_number_ = list()
        self.ev_current_sel_ev_number_ = list()
        self.ev_next_sel_ev_number_ = list()
        # storage pool for cs experience
        self.cs_current_state_ = list()
        self.cs_action_ = list()
        self.cs_reward_ = list()
        self.cs_next_state_ = list()
        self.cs_done_flag_ = list()
        self.cs_current_ev_number_ = list()
        self.cs_current_cs_number_ = list()
        self.cs_current_cd_number_ = list()
        self.cs_current_sel_ev_number_ = list()
        self.cs_next_sel_ev_number_ = list()

    def compute_loss(self,
                     ev_sample,
                     cs_sample):
        """ get info of EVs """
        ev_current_state = torch.FloatTensor(ev_sample["current_state"]).to(self.device)
        ev_action = torch.LongTensor(ev_sample["action"]).reshape(-1, 1).to(self.device)  # 相对次序 动作选择
        ev_reward = torch.FloatTensor(ev_sample["reward"]).reshape(-1, 1).to(self.device)
        ev_next_state = torch.FloatTensor(ev_sample["next_state"]).to(self.device)
        ev_done = torch.LongTensor(ev_sample["done"]).reshape(-1, 1).to(self.device)
        ev_next_sel_ev_number = torch.LongTensor(ev_sample["next_sel_ev_number"]).reshape(-1, 1).to(self.device)  # 相对次序 动作选择

        """ compute loss for EV net """
        ev_current_q = self.ev_model(ev_current_state).gather(1, ev_action)  # 预测排队时间
        ev_next_q = self.ev_model_target(ev_next_state).gather(1, ev_next_sel_ev_number)
        ev_mask = 1 - ev_done
        ev_target_q_value = (ev_reward + ev_mask * self.ev_gamma * ev_next_q)
        # ev_target_q_value = ev_reward
        ev_loss = f.smooth_l1_loss(ev_current_q, ev_target_q_value)

        """ get info of CSs """
        cs_current_state = torch.FloatTensor(cs_sample["current_state"]).to(self.device)
        cs_action = torch.LongTensor(cs_sample["action"]).reshape(-1, 1).to(self.device)
        cs_reward = torch.FloatTensor(cs_sample["reward"]).reshape(-1, 1).to(self.device)

        """ compute loss for CS net """
        cs_current_q_value = self.cs_model(cs_current_state).gather(1, cs_action)
        cs_target_q_value = cs_reward
        cs_loss = f.smooth_l1_loss(cs_current_q_value, cs_target_q_value)

        """ return the losses """
        return ev_loss, cs_loss

    def select_action(self):
        ev_current_state = self.env.get_current_ev_state()
        ev_val_list = self.ev_model_target(torch.FloatTensor(ev_current_state).unsqueeze(dim=0).to(self.device)).cpu().squeeze(dim=0).data.numpy()
        ev_num = []
        ev_val = []
        for i in range(len(ev_val_list)):
            if self.env.transfer_ev_order_to_no(i) in self.env.get_not_scheduled_ev():
                ev_num.append(self.env.transfer_ev_order_to_no(i))
                ev_val.append(ev_val_list[i])
        ev_sel = int(ev_num[np.array(ev_val).argmin()])  # real ev number

        cs_current_state = self.env.get_current_cs_state(ev_sel)
        cs_val_list = self.cs_model(torch.FloatTensor(cs_current_state).unsqueeze(dim=0).to(self.device)).cpu().squeeze(dim=0).data.numpy()
        cs_num = []
        cs_val = []
        for i in range(self.env.get_charging_station_number()):
            if self.env.get_reachability_of_an_ev(ev_sel)[i] == 1.0:
                cs_val.append(cs_val_list[i] + self.env.get_charging_ev_number_for_concrete_cs(i) * 2.5)
                cs_num.append(i)
        cs_sel = int(cs_num[np.array(cs_val).argmin()])
        return ev_sel, cs_sel, ev_current_state, cs_current_state  # real ev_number

    def step(self,
             ev_sel,  # real ev number
             cs_sel):
        cd_sel = self.env.get_best_charging_device_(cs_sel)
        self.env.step(ev_sel, cs_sel, cd_sel)
        ev_reward = self.env.get_reward_for_ev(ev_sel, cs_sel, cd_sel)
        cs_reward = self.env.get_reward_for_cs(cs_sel, ev_sel)
        done = self.env.is_done()
        self.total_steps += 1
        return ev_reward, cs_reward, done

    def update_model(self):
        ev_samples = self.er_ev.sample_batch()
        cs_samples = self.er_cs.sample_batch()
        """ for both """
        ev_loss, cs_loss = self.compute_loss(ev_samples, cs_samples)
        """ for EV model """
        self.ev_optim.zero_grad()
        ev_loss.backward()
        self.ev_optim.step()
        """ for CS model """
        self.cs_optim.zero_grad()
        cs_loss.backward()
        self.cs_optim.step()
        return cs_loss.data, ev_loss.data

    def target_hard_update(self):
        self.ev_model_target.load_state_dict(self.ev_model.state_dict())

    def train(self):
        self.env.reset()
        print("reading data...")
        self.ds_constructor.construct_ev_dataset()
        self.ds_constructor.construct_cs_dataset()
        print("data reading finished...")
        pre_train_ev_loss = []
        pre_train_cs_loss = []
        pre_train_ev_queue = []
        pre_train_ev_idle = []
        pre_train_ev_c_n = []

        # 预训练阶段
        print("moving data...")
        self.dh_ev.move_file_to_buffer(self.er_ev, 100000)
        self.dh_cs.move_file_to_buffer(self.er_cs,  100000)
        print("")
        if len(self.er_ev) >= self.batch_size and len(self.er_cs) >= self.batch_size:
            print("Pre-training stage for {} times.".format(self.pre_training_round))
            for i in range(self.pre_training_round):
                # 进行预训练操作
                print("pre-training round {0}".format(i))
                cs_loss, ev_loss = self.update_model()
                pre_train_ev_loss.append(ev_loss)
                pre_train_cs_loss.append(cs_loss)
                if i % 40 == 0:
                    for j in range(self.trajectory_len):
                        ev_sel, cs_sel, current_ev_state, cs_current_state = self.select_action()  # real ev number
                        ev_reward, cs_reward, done = self.step(ev_sel, cs_sel)  # real ev number
                        if done:
                            self.env.print_scheduling_consequence_info()
                            print("\n")
                            ev_c_n = self.env.get_hour_charging_ev()
                            idle = self.env.get_average_time()[0]
                            queue = self.env.get_average_time()[1]
                            self.env.reset()
                            pre_train_ev_idle.append(idle)
                            pre_train_ev_queue.append(queue)
                            pre_train_ev_c_n.append(ev_c_n)
                self.update_cnt += 1
                if self.update_cnt % self.target_hard_update_ == 0:
                    self.target_hard_update()
                print("cs_loss = {0}, ev_loss = {1}".format(cs_loss, ev_loss))

        train_ev_loss = []
        train_cs_loss = []
        # 优化前结果
        train_ev_queue_bo = []
        train_ev_idle_bo = []
        train_ev_c_n_bo = []
        # 优化后结果
        train_ev_queue_ao = []
        train_ev_idle_ao = []
        train_ev_c_n_ao = []

        # 正式训练阶段
        print("Stage formal-training")
        for i in range(1, 2001):
            if i % 100 == 0:
                print("moving data...")
                self.dh_ev.move_file_to_buffer(self.er_ev, 100000)
                self.dh_cs.move_file_to_buffer(self.er_cs, 100000)
            print(" formal training process, round {0}".format(i))
            first_round = True
            self.env.reset()
            self.init_storage_pool()
            start = time.time()
            for j in range(self.trajectory_len):  # 根据当前神经网络执行一次全部电动汽车调度
                print(j)
                ev_sel, cs_sel, ev_current_state, cs_current_state = self.select_action()  ## ev_sel 当前是real number
                cd_sel = self.env.get_best_charging_device_(cs_sel)
                ev_reward, cs_reward, done = self.step(ev_sel, cs_sel)

                """ 保存EV中间状态信息 到数据库 """
                ev_current_state = ev_current_state
                ev_action = self.env.transfer_ev_no_to_order(ev_sel)  # 数据库中存储的是相对编号  也即实际动作
                ev_reward = ev_reward
                if not first_round:
                    ev_next_state = ev_current_state
                    self.ev_next_state_.append(ev_next_state)
                if self.env.is_done():
                    ev_next_state = ev_current_state
                    self.ev_next_state_.append(ev_next_state)
                ev_done = self.env.is_done()
                ev_current_ev_number = ev_sel  # real ev number
                ev_current_cs_number = cs_sel
                ev_current_cd_number = cd_sel
                ev_current_sel_ev_number = ev_sel
                if not first_round:
                    ev_next_sel_ev_number = ev_action
                    self.ev_next_sel_ev_number_.append(ev_next_sel_ev_number)
                if self.env.is_done():
                    ev_next_sel_ev_number = ev_action
                    self.ev_next_sel_ev_number_.append(ev_next_sel_ev_number)
                """ append all ev-side information into _ array """
                self.ev_current_state_.append(ev_current_state)
                self.ev_action_.append(ev_action)
                self.ev_reward_.append(ev_reward)
                self.ev_done_flag_.append(ev_done)
                self.ev_current_ev_number_.append(ev_current_ev_number)
                self.ev_current_cs_number_.append(ev_current_cs_number)
                self.ev_current_cd_number_.append(ev_current_cd_number)
                self.ev_current_sel_ev_number_.append(ev_current_sel_ev_number)

                """ 保存CS中间状态信息 到数据库 """
                cs_current_state = cs_current_state
                cs_action = cs_sel
                cs_reward = cs_reward
                if not first_round:
                    cs_next_state = cs_current_state
                    self.cs_next_state_.append(cs_next_state)
                if self.env.is_done():
                    cs_next_state = cs_current_state
                    self.cs_next_state_.append(cs_next_state)
                cs_done = self.env.is_done()
                cs_current_ev_number = ev_sel
                cs_current_cs_number = cs_sel
                cs_current_cd_number = cd_sel
                cs_current_sel_ev_number = ev_sel
                if not first_round:
                    cs_next_sel_ev_number = ev_sel
                    self.cs_next_sel_ev_number_.append(cs_next_sel_ev_number)
                if self.env.is_done():
                    cs_next_sel_ev_number = ev_sel
                    self.cs_next_sel_ev_number_.append(cs_next_sel_ev_number)
                """ append all cs-side information into _ array """
                self.cs_current_state_.append(cs_current_state)
                self.cs_action_.append(cs_action)
                self.cs_reward_.append(cs_reward)
                self.cs_done_flag_.append(cs_done)
                self.cs_current_ev_number_.append(cs_current_ev_number)
                self.cs_current_cs_number_.append(cs_current_cs_number)
                self.cs_current_cd_number_.append(cs_current_cd_number)
                self.cs_current_sel_ev_number_.append(cs_current_sel_ev_number)

                # 第一轮结束
                first_round = False

            # 优化调度结果
            if self.env.is_done():
                # train_ev_idle_bo.append(self.env.get_idling_time() / 170)
                # train_ev_queue_bo.append(self.env.get_queueing_time() / 899)
                # train_ev_c_n_bo.append(self.env.get_hour_charging_ev())
                # print(self.env.print_scheduling_consequence_info(), end="  ")
                self.env.optimize()
                # print(self.env.print_scheduling_consequence_info())
                # train_ev_idle_ao.append(self.env.get_idling_time() / 170)
                # train_ev_queue_ao.append(self.env.get_queueing_time() / 899)
                # train_ev_c_n_ao.append(self.env.get_hour_charging_ev())
            print(time.time() - start)
            exit(0)
            if self.env.is_done() and self.env.get_average_time()[1] < 22:
                """ for ev part """
                for e in range(899):
                    self.dh_ev.store(
                        current_state=self.ev_current_state_[e],
                        action=self.ev_action_[e],
                        reward=self.ev_reward_[e],
                        next_state=self.ev_next_state_[e],
                        done=self.ev_done_flag_[e],
                        current_ev_number=self.ev_current_ev_number_[e],
                        current_cs_number=self.ev_current_cs_number_[e],
                        current_cd_number=self.ev_current_cd_number_[e],
                        current_sel_ev_number=self.ev_current_sel_ev_number_[e],
                        next_sel_ev_number=self.ev_next_sel_ev_number_[e]
                    )
                self.dh_ev.calculate_discounted_q_value()
                self.dh_ev.move_temp_to_buffer(self.er_ev)
                self.dh_ev.reset()

            if self.env.is_done() and self.env.get_average_time()[1] < 22:
                """ for cs part """
                for c in range(899):
                    self.dh_cs.store(
                        current_state=self.cs_current_state_[c],
                        action=self.cs_action_[c],
                        reward=self.cs_reward_[c],
                        next_state=self.cs_next_state_[c],
                        done=self.ev_done_flag_[c],
                        current_ev_number=self.cs_current_ev_number_[c],
                        current_cs_number=self.cs_current_cs_number_[c],
                        current_cd_number=self.cs_current_cd_number_[c],
                        current_sel_ev_number=self.cs_current_sel_ev_number_[c],
                        next_sel_ev_number=self.cs_next_sel_ev_number_[c]
                    )
                self.dh_cs.calculate_discounted_q_value()
                self.dh_cs.move_temp_to_buffer(self.er_cs)
                self.dh_cs.reset()

            if len(self.er_cs) >= self.batch_size and len(self.er_ev) >= self.batch_size:
                print("stage ---> training for 500 times...")
                for it in range(500):
                    cs_loss, ev_loss = self.update_model()
                    train_cs_loss.append(cs_loss)
                    train_ev_loss.append(ev_loss)
                    self.update_cnt += 1
                    if self.update_cnt % self.target_hard_update_:
                        self.target_hard_update()

        # 将实验结果存入预训练数据集合
        with open("../Data/results/pre_train_ev_loss", "ab+") as f:
            pickle.dump(pre_train_ev_loss, f)
            f.close()

        with open("../Data/results/pre_train_cs_loss", "ab+") as f:
            pickle.dump(pre_train_cs_loss, f)
            f.close()

        with open("../Data/results/pre_train_queue", "ab+") as f:
            pickle.dump(pre_train_ev_queue, f)
            f.close()

        with open("../Data/results/pre_train_idle", "ab+") as f:
            pickle.dump(pre_train_ev_idle, f)
            f.close()

        with open("../Data/results/pre_train_ev_c_n", "ab+") as f:
            pickle.dump(pre_train_ev_c_n, f)
            f.close()

        with open("../Data/results/formal_ave_idle_b", "ab+") as f:
            pickle.dump(train_ev_idle_bo, f)
            f.close()

        with open("../Data/results/formal_ave_idle", "ab+") as f:
            pickle.dump(train_ev_idle_ao, f)
            f.close()

        with open("../Data/results/formal_ave_queue", "ab+") as f:
            pickle.dump(train_ev_queue_ao, f)
            f.close()

        with open("../Data/results/formal_ave_queue_b", "ab+") as f:
            pickle.dump(train_ev_queue_bo, f)
            f.close()

        with open("../Data/results/formal_ev_c_n", "ab+") as f:
            pickle.dump(train_ev_c_n_ao, f)
            f.close()

        with open("../Data/results/formal_ev_c_n_b", "ab+") as f:
            pickle.dump(train_ev_c_n_bo, f)
            f.close()

if __name__ == '__main__':
    agent = Agent(256, 100)
    agent.train()
