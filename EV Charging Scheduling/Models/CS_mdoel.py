
from _Environment.environment import *


class CS_SELECTING_MODEL(nn.Module):

    # input -> (batch_size, 1, 5, 36)
    # output -> (batch_size, 34)

    def __init__(self):
        super(CS_SELECTING_MODEL, self).__init__()
        self.env = Environment()
        output_shape = self.env.get_charging_station_number()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=16,
                               kernel_size=(3, 6))
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=(2, 4))
        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=(2, 4))
        self.fc_1 = nn.Linear(1600, 1024)
        self.fc_2 = nn.Linear(1024, 512)
        self.fc_3 = nn.Linear(512, 128)
        self.fc_4 = nn.Linear(128, out_features=output_shape)

    def forward(self,
                x_in: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x_in)
        x = f.relu(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = self.conv3(x)
        x = f.relu(x)
        conv_out_size = x.size()[1] * x.size()[2] * x.size()[3]
        x = x.view(-1, conv_out_size)
        x = f.relu(self.fc_1(x))
        x = f.relu(self.fc_2(x))
        x = f.relu(self.fc_3(x))
        x = self.fc_4(x)
        return x
