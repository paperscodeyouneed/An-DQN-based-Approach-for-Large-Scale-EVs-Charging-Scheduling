
from _Environment.environment import *


class ReplayBuffer(object):

    """
    a simple replay buffer implemented by np.array
    """

    def __init__(self,
                 is_ev: bool = True,
                 size: int = 100000,
                 batch_size: int = 2560) -> None:
        self.env = Environment()
        if is_ev:
            shape = self.env.get_current_ev_state().shape
        else:
            shape = self.env.get_current_cs_state(0).shape
        self.current_state = np.zeros([size, *shape], dtype=np.float16)
        self.action = np.zeros([size], dtype=np.int)
        self.reward = np.zeros([size], dtype=np.float16)
        self.next_state = np.zeros([size, *shape], dtype=np.float16)
        self.done = np.zeros([size], dtype=np.bool)
        self.current_ev_number = np.zeros([size], dtype=np.int)
        self.current_cs_number = np.zeros([size], dtype=np.int)
        self.current_cd_number = np.zeros([size], dtype=np.int)
        self.current_sel_ev_number = np.zeros([size], dtype=np.int)
        self.next_sel_ev_number = np.zeros([size], dtype=np.int)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0

    def store(self,
              current_state: np.ndarray,
              action: int,
              reward: float,
              next_state: np.ndarray,
              done: bool,
              current_ev_number: int,
              current_cs_number: int,
              current_cd_number: int,
              current_sel_ev_number: int,
              next_sel_ev_number: int):
        self.current_state[self.ptr] = current_state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.current_ev_number[self.ptr] = current_ev_number
        self.current_cs_number[self.ptr] = current_cs_number
        self.current_cd_number[self.ptr] = current_cd_number
        self.current_sel_ev_number[self.ptr] = current_sel_ev_number
        self.next_sel_ev_number[self.ptr] = next_sel_ev_number

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        idx = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            current_state=self.current_state[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            next_state=self.next_state[idx],
            done=self.done[idx],
            current_ev_number=self.current_ev_number[idx],
            current_cs_number=self.current_cs_number[idx],
            current_cd_number=self.current_cd_number[idx],
            current_sel_ev_number=self.current_sel_ev_number[idx],
            next_sel_ev_number=self.next_sel_ev_number[idx]
        )

    def __len__(self):
        return self.size
