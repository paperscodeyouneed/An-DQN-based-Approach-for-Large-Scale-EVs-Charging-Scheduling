
from _Environment.environment import *

env = Environment()
OUTPUT_SHAPE = env.get_schedulable_ev().size  # 899


class NoisyLayer(nn.Module):

    """
    Noisy linear layer module for NoisyNet
    """
    def __init__(self,
                 in_features,
                 out_features,
                 std_init: float = 0.5):
        super(NoisyLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """
        reset the trainable network parameters (factorized gaussian noise)
        """
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.uniform_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """
        set scale to make noise (factorized gaussian noise)
        """
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        """
        make new noise
        """
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
         y = (wμ + wσ ⊙ wε) * x + bμ + bσ ⊙ bε
        """
        return f.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon
        )


class Residual(nn.Module):

    def __init__(self,
                 input_channels,
                 num_channels,
                 use_1x1conv=False,
                 strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
            stride=strides
        )
        self.conv2 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1
        )
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=num_channels,
                kernel_size=1,
                stride=strides
            )
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_features=num_channels)
        self.bn2 = nn.BatchNorm2d(num_features=num_channels)

    def forward(self, x):
        y = f.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return f.relu(y)


class EV_SELECTING_MODEL(nn.Module):

    # input -> (batch_size, 1, 903, 36)
    # output -> (batch_size, 899)

    def __init__(self):

        super(EV_SELECTING_MODEL, self).__init__()
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(8, 8), stride=4),
            nn.ReLU(),
            Residual(input_channels=8, num_channels=8),
            Residual(input_channels=8, num_channels=8),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        )
        self.middle_layer = nn.Sequential(
            nn.Dropout2d(0.5),
            NoisyLayer(6976, 4096),
            NoisyLayer(4096, 2048)
        )
        self.advantage_layer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU6(),
            nn.Linear(1024, 1024),
            nn.ReLU6(),
            nn.Linear(1024, len(env.get_schedulable_ev()))
        )
        self.value_layer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU6(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self,
                x_in: torch.Tensor) -> torch.Tensor:
        o = self.convolutional_layer(x_in)
        o = o.reshape(-1, o.size()[1] * o.size()[2] * o.size()[3])
        o = self.middle_layer(o)
        a = self.advantage_layer(o)
        v = self.value_layer(o)
        q = v + a - (a.mean(dim=-1, keepdim=True))
        return q
