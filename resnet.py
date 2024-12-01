import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models.resnet import ResNet18_Weights

class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        # 加载预训练的ResNet18模型
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # self.resnet = models.resnet18()
        keep_pro = 0.
        # 移除avgpool和fc层
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()
        self.drop1 = nn.Dropout(keep_pro)
        self.drop2 = nn.Dropout(keep_pro)
        self.drop3 = nn.Dropout(keep_pro)
        self.drop4 = nn.Dropout(keep_pro)
        self.drop5 = nn.Dropout(keep_pro)

    def forward(self, x):
        # 第一个卷积层和池化层的输出
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.drop1(x)
        conv1_out = x

        # 第一个卷积块的输出
        x = self.resnet.layer1(x)
        x = self.drop2(x)
        layer1_out = x

        # 第二个卷积块的输出
        x = self.resnet.layer2(x)
        x = self.drop3(x)
        layer2_out = x

        # 第三个卷积块的输出
        x = self.resnet.layer3(x)
        x = self.drop4(x)
        layer3_out = x

        # 第四个卷积块的输出
        x = self.resnet.layer4(x)
        x = self.drop5(x)
        layer4_out = x

        return conv1_out, layer1_out, layer2_out, layer3_out, layer4_out

