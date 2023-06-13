
from _Environment.environment import *
from Utils.data_handler import *


class DatasetConstructor(object):

    def __init__(self):
        """ construct experience dataset for ev and cs simultaneously """
        self.env = Environment()
        self.dh_ev = DataHandler(is_ev=True)
        self.dh_cs = DataHandler(is_ev=False)
        """ reading raw experience """
        with open("../Data/trimmed_pre_trained_data_information △/exp0_original_trajectory", "r") as file:
            self.experience = file.read()
        """ define target storage file """
        self.ev_exp_file = "../Data/experience/EV_Experience/ev_experience"
        self.cs_exp_file = "../Data/experience/CS_Experience/cs_experience"
        """ acquiring pre-trained data raw array as original trajectory """
        self.pre_train_ev_data = self.dh_ev.squeeze_pretrain_data()  # 数组化后为一个 (826, 899, 3) 大小的数组
        self.pre_train_cs_data = self.dh_cs.squeeze_pretrain_data()  # 数组化后为一个 (826, 899, 3) 大小的数组

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

    def construct_ev_dataset(self):
        if os.path.getsize(self.ev_exp_file) == 0: # 如果没有进行经验的存储
            for i in range(len(self.pre_train_ev_data)): # 对当前读取到的所有经验 每一组经验都是不同的
                print(i)
                # 中的每一个trajectory
                first_round = True
                self.env.reset()
                for j in range(len(self.pre_train_ev_data[i])): # 这个循环执行899次
                    """ 数据格式为 [ [cs, dis, relative ev] ... [] ] """
                    ev = self.pre_train_ev_data[i][j][2]  # 选择动作 相对编号
                    cs = self.pre_train_ev_data[i][j][0]
                    cd = self.env.get_best_charging_device(cs)
                    current_state = self.env.get_current_ev_state()
                    self.env.step(self.env.transfer_ev_order_to_no(ev), cs, cd)
                    action = ev
                    reward = self.env.get_reward_for_ev(self.env.transfer_ev_order_to_no(ev), cs, cd)
                    if not first_round:
                        next_state = current_state
                        self.ev_next_state_.append(next_state)
                    if self.env.is_done():
                        next_state = current_state
                        self.ev_next_state_.append(next_state)
                    done = self.env.is_done()
                    current_ev_number = ev
                    current_cs_number = cs
                    current_cd_number = cd
                    current_sel_ev_number = ev
                    if not first_round:
                        next_sel_ev_number = ev
                        self.ev_next_sel_ev_number_.append(next_sel_ev_number)
                    if self.env.is_done():
                        next_sel_ev_number = ev
                        self.ev_next_sel_ev_number_.append(next_sel_ev_number)
                    first_round = False
                    """ append one-step data into storage pool """
                    self.ev_current_state_.append(current_state)
                    self.ev_action_.append(action)
                    self.ev_reward_.append(reward)
                    self.ev_done_flag_.append(done)
                    self.ev_current_ev_number_.append(current_ev_number)
                    self.ev_current_cs_number_.append(current_cs_number)
                    self.ev_current_cd_number_.append(current_cd_number)
                    self.ev_current_sel_ev_number_.append(current_sel_ev_number)
                if self.env.is_done():
                    for e in range(len(self.ev_done_flag_)):
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
                    self.dh_ev.move_temp_to_file()
                    self.dh_ev.reset()
                self.init_storage_pool()

    def construct_cs_dataset(self):
        if os.path.getsize(self.cs_exp_file) == 0:  # 如果没有进行经验的存储
            for i in range(len(self.pre_train_cs_data)):  # 对当前读取到的所有的数据
                print(i)
                # 中的每一个trajectory
                first_round = True
                self.env.reset()
                for j in range(len(self.pre_train_cs_data[i])):  # 这个循环执行899次
                    """ 数据格式为 [ [cs, dis, relative ev] ... [] ] """
                    ev = self.pre_train_cs_data[i][j][2]
                    cs = self.pre_train_cs_data[i][j][0]
                    cd = self.env.get_best_charging_device(cs)
                    current_state = self.env.get_current_cs_state(self.env.transfer_ev_order_to_no(ev))
                    self.env.step(self.env.transfer_ev_order_to_no(ev), cs, cd)
                    action = cs
                    reward = self.env.get_reward_for_ev(self.env.transfer_ev_order_to_no(ev), cs, cd)
                    if not first_round:
                        next_state = current_state
                        self.cs_next_state_.append(next_state)
                    if self.env.is_done():
                        next_state = current_state
                        self.cs_next_state_.append(next_state)
                    done = self.env.is_done()
                    current_ev_number = ev ## 相对序号
                    current_cs_number = cs
                    current_cd_number = cd
                    current_sel_ev_number = ev
                    if not first_round:
                        next_sel_ev_number = ev
                        self.cs_next_sel_ev_number_.append(next_sel_ev_number)
                    if self.env.is_done():
                        next_sel_ev_number = ev
                        self.cs_next_sel_ev_number_.append(next_sel_ev_number)
                    first_round = False
                    """ append one-step data into storage pool """
                    self.cs_current_state_.append(current_state)
                    self.cs_action_.append(action)
                    self.cs_reward_.append(reward)
                    self.cs_done_flag_.append(done)
                    self.cs_current_ev_number_.append(current_ev_number)  # 相对序号
                    self.cs_current_cs_number_.append(current_cs_number)
                    self.cs_current_cd_number_.append(current_cd_number)
                    self.cs_current_sel_ev_number_.append(current_sel_ev_number)
                if self.env.is_done():
                    for c in range(len(self.cs_done_flag_)):
                        self.dh_cs.store(
                            current_state=self.cs_current_state_[c],
                            action=self.cs_action_[c],
                            reward=self.cs_reward_[c],
                            next_state=self.cs_next_state_[c],
                            done=self.cs_done_flag_[c],
                            current_ev_number=self.cs_current_ev_number_[c],
                            current_cs_number=self.cs_current_cs_number_[c],
                            current_cd_number=self.cs_current_cd_number_[c],
                            current_sel_ev_number=self.cs_current_sel_ev_number_[c],
                            next_sel_ev_number=self.cs_next_sel_ev_number_[c]
                        )
                    self.dh_cs.calculate_discounted_q_value()
                    self.dh_cs.move_temp_to_file()
                    self.dh_cs.reset()
                self.init_storage_pool()

if __name__ == '__main__':
    dc = DatasetConstructor()
    dc.construct_ev_dataset()
    dc.construct_cs_dataset()
