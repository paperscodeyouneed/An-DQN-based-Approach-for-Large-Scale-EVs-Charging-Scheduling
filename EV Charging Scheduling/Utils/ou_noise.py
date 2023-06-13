
from _Environment.environment import *


# define a class used to add noise data
class OUNoise(object):

    def __init__(self,
                 size: int,
                 mu: float = 0.0,
                 theta: float = 0.15,
                 sigma: float = 0.2) -> None:
        """initialize the parameters and noise process"""
        assert size > 0
        self.state = np.float64(0)
        self.mu = mu * np.ones(shape=size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self) -> None:
        """ initialize parameters and noise process"""
        assert size > 0
        self.state = copy.copy(x=self.mu)
        return None

    def sample(self) -> np.ndarray:
        """ update internal state and return it as a noise sample """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state
