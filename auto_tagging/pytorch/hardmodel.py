from torch import nn
from torchsummary import summary
from torch import Tensor
from ResidualBlock import ResidualBlock
import config


class ResCnn(nn.Module):
    def __init__(
        self,
        classes_num: int,
        image_size: tuple,
        in_channels: int = 1,
        pull_stride: int = 2,
        pull_kernel_size: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        dropout_p: float = 0.2,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.layer1 = nn.Sequential(
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            nn.MaxPool2d(kernel_size=pull_kernel_size, stride=pull_stride),
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(16, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.MaxPool2d(kernel_size=pull_kernel_size, stride=pull_stride),
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.MaxPool2d(kernel_size=pull_kernel_size, stride=pull_stride),
        )

        self.layer4 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.MaxPool2d(kernel_size=pull_kernel_size, stride=pull_stride),
        )

        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(
            in_features=int(image_size[0] / 2**4 * image_size[1] / 2**4 * 128),
            out_features=classes_num,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn(self.conv(x)))

        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x


if __name__ == "__main__":
    custom_res_cnn = ResCnn(
        in_channels=config.IN_CHANNELS,
        classes_num=config.CLASSES_NUM,
        image_size=config.IMAGE_SIZE,
        pull_kernel_size=config.PULL_KERNEL_SIZE,
        pull_stride=config.PULL_STRIDE,
        kernel_size=config.KERNEL_SIZE,
        stride=config.STRIDE,
        dropout_p=config.DROPOUT_P,
    ).to(config.DEVICE)
    print(summary(custom_res_cnn, (1, *config.IMAGE_SIZE)))
