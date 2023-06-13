
from _Environment.environment import *
from Utils.replay_buffer import *
from Utils.segment_tree import SegmentTree, SumSegmentTree, MinSegmentTree


class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self,
                 is_ev: bool = True,
                 size: int = 10000,
                 batch_size: int = 256,
                 alpha: float = 0.6) -> None:
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(is_ev=is_ev,
                                                      size=size,
                                                      batch_size=batch_size)
        self.max_priority = 1.0
        self.tree_ptr = 0
        self.alpha = alpha

        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

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
        """Store experience and priority."""
        super().store(current_state,
                      action,
                      reward,
                      next_state,
                      done,
                      current_ev_number,
                      current_cs_number,
                      current_cd_number,
                      current_sel_ev_number,
                      next_sel_ev_number)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self,
                     beta: float = 0.4):
        assert len(self) >= self.batch_size
        assert beta > 0
        """Sample a batch of experiences."""
        indices = self._sample_proportional()

        current_state = self.current_state[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        next_state = self.next_state[indices]
        done = self.done[indices]
        current_ev_number = self.current_ev_number[indices]
        current_cs_number = self.current_cs_number[indices]
        current_cd_number = self.current_cd_number[indices]
        current_sel_ev_number = self.current_sel_ev_number[indices]
        next_sel_ev_number = self.next_sel_ev_number[indices]

        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        return dict(
            current_state=current_state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            current_ev_number=current_ev_number,
            current_cs_number=current_cs_number,
            current_cd_number=current_cd_number,
            current_sel_ev_number=current_sel_ev_number,
            next_sel_ev_number=next_sel_ev_number,
            weights=weights,
            indices=indices
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight
