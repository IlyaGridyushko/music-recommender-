import torch
from torch import nn
from torchsummary import summary
from torch import Tensor
from ResidualBlock import ResidualBlock
import config


class ResCnn(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        classes_num: int = 10,
        kernel_size: int = 3,
        stride: int = 1,
        pull_kernel_size: int = 2,
        pull_stride: int = 2,
        image_size: tuple = (128, 128),
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(
            kernel_size=pull_kernel_size,
            stride=pull_stride,
        )

        self.layer1 = ResidualBlock(in_channels=16, out_channels=32, stride=stride)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(
            kernel_size=pull_kernel_size,
            stride=pull_stride,
        )

        self.layer2 = ResidualBlock(in_channels=32, out_channels=64, stride=stride)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(
            kernel_size=pull_kernel_size,
            stride=pull_stride,
        )

        self.layer3 = ResidualBlock(in_channels=64, out_channels=128, stride=stride)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(
            kernel_size=pull_kernel_size,
            stride=pull_stride,
        )

        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(
            in_features=int(image_size[0] / 2**4 * image_size[1] / 2**4 * 128),
            out_features=classes_num,
        )

    def forward(self, x) -> Tensor:
        out = self.max_pool(self.relu(self.bn1(self.conv1(x))))

        out = self.max_pool(self.relu(self.layer1(out)))

        out = self.max_pool(self.relu(self.layer2(out)))

        out = self.max_pool(self.relu(self.layer3(out)))

        out = self.flatten(out)
        out = self.dropout(out)
        out = self.linear(out)
        return out


if __name__ == "__main__":
    custom_res_cnn = ResCnn(
        in_channels=config.IN_CHANNELS,
        classes_num=config.CLASSES_NUM,
        kernel_size=config.KERNEL_SIZE,
        stride=config.STRIDE,
        pull_kernel_size=config.PULL_KERNEL_SIZE,
        pull_stride=config.PULL_STRIDE,
        image_size=config.IMAGE_SIZE,
        dropout_p=config.DROPOUT_P,
    ).to(config.DEVICE)
    print(summary(custom_res_cnn, (1, *config.IMAGE_SIZE)))
