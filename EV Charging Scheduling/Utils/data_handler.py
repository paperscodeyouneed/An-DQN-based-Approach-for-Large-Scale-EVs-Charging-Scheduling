import numpy as np

from _Environment.environment import *


class DataHandler:

    def __init__(self,
                 is_ev: bool = True,
                 gamma: float = 0.6):
        self.is_ev = is_ev
        self.gamma = gamma
        self.env = Environment()
        self.all_part = True

        self.ev_amount = self.env.get_schedulable_ev().size                                     # 899
        self.cs_amount = self.env.get_charging_station_number()                                 # 34
        self.cd_amount = self.env.get_charging_device_number()                                  # 5

        self.ev_state_shape = self.env.get_current_ev_state().shape                             # (1, 5, 36)
        self.cs_state_shape = self.env.get_current_cs_state(sel_ev_number=0).shape              # (1, 903, 36)

        if self.is_ev:
            self.state_shape = (self.ev_amount, *self.ev_state_shape)
        else:
            self.state_shape = (self.ev_amount, *self.cs_state_shape)
        # 定义训练数据临时存储空间
        self.current_state = np.zeros(self.state_shape, dtype=np.float16)
        self.action = np.zeros(self.ev_amount, dtype=int)
        self.reward = np.zeros((self.cs_amount, self.cd_amount, self.ev_amount), dtype=np.float16)
        self.reward_ = np.zeros(shape=(899,), dtype=np.float64)
        self.next_state = np.zeros(self.state_shape, dtype=np.float16)
        self.done = np.zeros(self.ev_amount, dtype=bool)
        self.current_ev_number = np.zeros(self.ev_amount, dtype=int)
        self.current_cs_number = np.zeros(self.ev_amount, dtype=int)
        self.current_cd_number = np.zeros(self.ev_amount, dtype=int)
        self.current_sel_ev_number = np.zeros(self.ev_amount, dtype=int)
        self.next_sel_ev_number = np.zeros(self.ev_amount, dtype=int)
        self.reward_pointer = np.zeros((self.cs_amount, self.cd_amount), dtype=int)
        self.scheduling_order = 0
        self.scheduling_index = []

    # 下一步增加全局折扣计算
    def calculate_discounted_q_value(self):
        if self.is_ev and not self.all_part:  # (1-step discounted)
            for i in range(self.cs_amount):
                for j in range(self.cd_amount):
                    if self.reward_pointer[i][j] >= 2:
                        for k in range(self.reward_pointer[i][j]-1):
                            # 从前往后计算
                            self.reward[i][j][k] += (
                                self.gamma * self.reward[i][j][k+1]
                            )
                    else:
                        if self.reward_pointer[i][j] == 1:
                            self.reward[i][j][0] += (self.gamma * self.reward[i][j][1])
        if self.is_ev and self.all_part:  # (1-step discounted)
            for i in range(len(self.reward_) - 1):
                self.reward_[i] += self.gamma * self.reward_[i+1]
        pass

    # FILE --> REPLAY BUFFER (ANY TIME)
    def move_file_to_buffer(self,
                            replay_buffer,
                            experience_quantity=50000):
        # move experience from file to replay buffer
        if self.is_ev:
            filename = "../Data/experience/EV_Experience/ev_experience"
        else:
            filename = "../Data/experience/CS_Experience/cs_experience"
        i = 0
        with open(filename, "rb+") as file:
            while i < experience_quantity:
                i += 1
                if i % 10000 == 0:
                    print(i)
                try:
                    data = cpickle.load(file)
                    current_state = data[0]["current_state"]
                    action = data[0]["action"]
                    reward = data[0]["reward"]
                    next_state = data[0]["next_state"]
                    done = data[0]["done"]
                    current_ev_number = data[0]["current_ev_number"]
                    current_cs_number = data[0]["current_cs_number"]
                    current_cd_number = data[0]["current_cd_number"]
                    current_sel_ev_number = data[0]["current_sel_ev_number"]
                    next_sel_ev_number = data[0]["next_sel_ev_number"]
                    replay_buffer.store(
                        current_state=current_state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=done,
                        current_ev_number=current_ev_number,
                        current_cs_number=current_cs_number,
                        current_cd_number=current_cd_number,
                        current_sel_ev_number=current_sel_ev_number,
                        next_sel_ev_number=next_sel_ev_number
                    )
                except EOFError:
                    pass
        return None

    # TEMP --> FILE (PRE-TRAINING)
    def move_temp_to_file(self):
        if self.is_ev:
            filename = "../Data/experience/EV_Experience/ev_experience"
        else:
            filename = "../Data/experience/CS_Experience/cs_experience"
        for i in range(len(self.scheduling_index)):
            cs_number = self.scheduling_index[i][1]  # 选择的充电站编号
            cd_number = self.scheduling_index[i][2]  # 选择的充电桩编号
            ord_in_cd = self.scheduling_index[i][3]  # 在充电桩内的排队次序
            sched_ord = self.scheduling_index[i][4]  # 当前电车的调度次序

            current_state = self.current_state[sched_ord]
            action = self.action[sched_ord]

            if not self.all_part:
                reward = self.reward[cs_number, cd_number, ord_in_cd]
            else:
                reward = self.reward_[sched_ord]

            next_state = self.next_state[sched_ord]
            done = self.done[sched_ord]
            current_ev_number = self.current_ev_number[sched_ord]
            current_cs_number = self.current_cs_number[sched_ord]
            current_cd_number = self.current_cd_number[sched_ord]
            current_sel_ev_number = self.current_sel_ev_number[sched_ord]
            next_sel_ev_number = self.next_sel_ev_number[sched_ord]

            data_dict = [{
                'current_state': current_state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'current_ev_number': current_ev_number,
                'current_cs_number': current_cs_number,
                'current_cd_number': current_cd_number,
                'current_sel_ev_number': current_sel_ev_number,
                'next_sel_ev_number': next_sel_ev_number
            }]

            with open(filename, "ab+") as file:
                cpickle.dump(data_dict, file)

    # TEMP --> REPLAY BUFFER (TRAIN)
    def move_temp_to_buffer(self,
                            replay_buffer):
        for i in range(len(self.scheduling_index)):

            cs_number = self.scheduling_index[i][1]
            cd_number = self.scheduling_index[i][2]
            or_in_slt = self.scheduling_index[i][3]
            sched_ord = self.scheduling_index[i][4]

            current_state = self.current_state[sched_ord]
            action = self.action[sched_ord]

            if not self.all_part:
                reward = self.reward[cs_number, cd_number, or_in_slt]
            else:
                reward = self.reward_[sched_ord]

            next_state = self.next_state[sched_ord]
            done = self.done[sched_ord]
            current_ev_number = self.current_ev_number[sched_ord]
            current_cs_number = self.current_cs_number[sched_ord]
            current_cd_number = self.current_cd_number[sched_ord]
            current_sel_ev_number = self.current_sel_ev_number[sched_ord]
            next_sel_ev_number = self.next_sel_ev_number[sched_ord]

            transition = [
                current_state, action, reward, next_state, done, current_ev_number,
                current_cs_number, current_cd_number, current_sel_ev_number, next_sel_ev_number
            ]

            replay_buffer.store(*transition)

    @staticmethod
    def squeeze_pretrain_data():
        with open("../Data/trimmed_pre_trained_data_information △/exp0_original_trajectory") as file:
            pre_train_data = eval(file.read())
        return pre_train_data

    def reset(self):
        self.__init__(is_ev=self.is_ev,
                      gamma=self.gamma)

    # a single transition --> TEMP (any time)
    def store(self,
              current_state: np.ndarray,
              action: np.int,
              reward: np.float,
              next_state: np.ndarray,
              done: np.bool,
              current_ev_number: np.int,
              current_cs_number: np.int,
              current_cd_number: np.int,
              current_sel_ev_number: np.int,
              next_sel_ev_number: np.int):
        # pointer 指示当前的经验应该存储在哪里 pointer ∈ [0, 898] 每存储一个经验后pointer加一
        pointer = self.reward_pointer[current_cs_number][current_cd_number]
        self.current_state[self.scheduling_order] = current_state  # scheduling order 指代当前经验在分类部分中各自存储的位置，它等价于调度顺序
        self.action[self.scheduling_order] = action

        if not self.all_part:
            self.reward[current_cs_number, current_cd_number, pointer] = reward
        else:
            self.reward_[self.scheduling_order] = reward

        self.next_state[self.scheduling_order] = next_state
        self.done[self.scheduling_order] = done
        self.current_ev_number[self.scheduling_order] = current_ev_number
        self.current_cs_number[self.scheduling_order] = current_cs_number
        self.current_cd_number[self.scheduling_order] = current_cd_number
        self.current_sel_ev_number[self.scheduling_order] = current_sel_ev_number
        self.next_sel_ev_number[self.scheduling_order] = next_sel_ev_number

        self.scheduling_index.append([
            current_ev_number,
            current_cs_number,
            current_cd_number,
            pointer,
            self.scheduling_order
        ])

        self.scheduling_order += 1
        self.reward_pointer[current_cs_number][current_cd_number] += 1
        return None
