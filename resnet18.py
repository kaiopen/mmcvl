import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.short_cut = nn.Sequential()
        if stride != 1 or in_channel != in_channel:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.short_cut(x)
        out = F.relu(out)

        return out


class ResNet18_grd(nn.Module):
    def __init__(self, ResBlock=ResBlock, pic_channel=3, out_feature=128):
        super(ResNet18_grd, self).__init__()
        self.pic_channel = pic_channel
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.pic_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.fc_1 = nn.Linear(512, out_feature)
        self.fc_2 = nn.Linear(512, out_feature//2)
        self.fc_3 = nn.Linear(512, out_feature//4)
        self.fc_4 = nn.Linear(512, out_feature//8)
        self.fc_5 = nn.Linear(512, out_feature//16)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.last_conv = nn.Sequential(
            nn.Conv2d(512,768,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(),
        )

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        multi_resolution = []
        out = self.conv1(x)
        out = self.max_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out_1 = self.fc_1(out)
        multi_resolution.append(out_1)
        out_2 = self.fc_2(out)
        multi_resolution.append(out_2)
        out_3 = self.fc_3(out)
        multi_resolution.append(out_3)
        out_4 = self.fc_4(out)
        multi_resolution.append(out_4)
        out_5 = self.fc_5(out)
        multi_resolution.append(out_5)
        return multi_resolution


class ResNet18_sat(nn.Module):
    def __init__(self, ResBlock=ResBlock, pic_channel=3):
        super(ResNet18_sat, self).__init__()
        self.pic_channel = pic_channel
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.pic_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.last_conv = nn.Sequential(
            nn.Conv2d(512,768,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(),
        )

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        multi_scale = []
        out = self.conv1(x)
        multi_scale.append(out)
        out = self.max_pool(out)
        out = self.layer1(out)
        multi_scale.append(out)
        out = self.layer2(out)
        multi_scale.append(out)
        out = self.layer3(out)
        multi_scale.append(out)
        out = self.layer4(out)
        out = self.last_conv(out)
        return multi_scale[0], multi_scale[1], multi_scale[2], multi_scale[3], out


# if __name__=="__main__":
#     a = torch.randn((2,5,512,512))
#     model = ResNet18_grd(ResBlock, 5, 128)
#     multi_resolution = model(a)
#     print(multi_resolution[0].shape)
#     print(multi_resolution[1].shape)
#     print(multi_resolution[2].shape)
#     print(multi_resolution[3].shape)
#     print(multi_resolution[4].shape)
#
