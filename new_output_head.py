import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, in_channels=4097, out_channels=1024):
        super(Decoder, self).__init__()
        
        # deconv1 and conv1, height, width: 16*16 -> 32*32
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1_1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        # deconv2 and conv2, height, width: 32*32 -> 64*64
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        # deconv3 and conv3, height, width: 64*64 -> 128*128
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3_1 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # deconv4 and conv4, height, width: 128*128 -> 256*256
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4_1 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # deconv5 and conv5, height, width: 256*256 -> 512*512
        self.deconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5_1 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        # final layer to get to 512*512
        self.final_deconv = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_conv = nn.Conv2d(in_channels=80, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, sat512, sat256, sat128, sat64, sat32):
        # Upsample and convolve
        x = self.deconv1(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        
        x = self.deconv2(x)
        sat32_upsampled = F.interpolate(sat32, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, sat32_upsampled], dim=1)
        x = self.conv2_1(x)
        x = self.conv2_2(x)

        x = self.deconv3(x)
        sat64_upsampled = F.interpolate(sat64, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, sat64_upsampled], dim=1)
        
        x = self.conv3_1(x)
        x = self.conv3_2(x)

        x = self.deconv4(x)
        sat128_upsampled = F.interpolate(sat128, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, sat128_upsampled], dim=1)

        x = self.conv4_1(x)
        x = self.conv4_2(x)

        x = self.deconv5(x)
        sat256_upsampled = F.interpolate(sat256, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, sat256_upsampled], dim=1)
        
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        
        x = self.final_deconv(x)
        sat512_upsampled = F.interpolate(sat512, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, sat512_upsampled], dim=1)
        # print(x.shape)
        x = self.final_conv(x)

        return x
