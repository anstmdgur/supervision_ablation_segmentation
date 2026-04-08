import torch
from torch import nn
import torchvision.models as models
from torch.nn import functional as F

def select_model(config):
    model_name = config.get('model')
    backbone = FrozenResNet34()
    model_dict = {
        'unet': Unet,
        'attention_unet': attentionUnet,
        'unet_plus': Unetplus,
    }
    model = model_dict[model_name]
    return model(backbone)

class FrozenResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # for param in resnet.parameters():
        #     param.requires_grad = False
            
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 64
        self.enc1 = nn.Sequential(resnet.maxpool, resnet.layer1)         # 64
        self.enc2 = resnet.layer2                                        # 128
        self.enc3 = resnet.layer3                                        # 256
        self.enc4 = resnet.layer4                                        # 512

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return e0, e1, e2, e3, e4

    #외부에서 model.train()을 호출해도 백본은 eval()을 유지하도록 강제
    # def train(self, mode=True):
    #     super().train(False)
    #     return self


class Unet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
        # ResNet-34 채널 수(512, 256, 128, 64, 64)
        self.dec4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4_conv1 = conv_block(256 + 256, 256)
        self.dec4_conv2 = conv_block(256, 256)

        self.dec3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3_conv1 = conv_block(128 + 128, 128)
        self.dec3_conv2 = conv_block(128, 128)

        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2_conv1 = conv_block(64 + 64, 64)
        self.dec2_conv2 = conv_block(64, 64)

        self.dec1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1_conv1 = conv_block(64 + 64, 64) 
        self.dec1_conv2 = conv_block(64, 64)
        
        self.final_upconv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv1 = conv_block(32, 32)
        self.final_conv2 = conv_block(32, 32)
        self.final = nn.Conv2d(32, 1, kernel_size=1, stride=1)

    def forward(self, x):

        e0, e1, e2, e3, e4 = self.backbone(x)
        

        dec4 = self.dec4(e4) #upconv
        dec4 = torch.cat((dec4, e3), dim=1) #skip connection
        dec4 = self.dec4_conv1(dec4) #conv
        dec4 = self.dec4_conv2(dec4)

        dec3 = self.dec3(dec4)
        dec3 = torch.cat((dec3, e2), dim=1)
        dec3 = self.dec3_conv1(dec3)
        dec3 = self.dec3_conv2(dec3)

        dec2 = self.dec2(dec3)
        dec2 = torch.cat((dec2, e1), dim=1)
        dec2 = self.dec2_conv1(dec2)
        dec2 = self.dec2_conv2(dec2)

        dec1 = self.dec1(dec2)
        dec1 = torch.cat((dec1, e0), dim=1)
        dec1 = self.dec1_conv1(dec1)
        dec1 = self.dec1_conv2(dec1)

        out = self.final_upconv(dec1)
        out = self.final_conv1(out)
        out = self.final_conv2(out)

        out = self.final(out)
        return out
    
class attentionUnet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.conv_1_g4 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv_1_x4 = nn.Conv2d(256, 128, kernel_size=1, stride=2) #downsampling 해서 g4와 크기 및 채널을 맞춰줌
        self.psi_4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        self.conv_1_g3 = nn.Conv2d(256, 64, kernel_size=1, stride=1)
        self.conv_1_x3 = nn.Conv2d(128, 64, kernel_size=1, stride=2)
        self.psi_3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        self.conv_1_g2 = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        self.conv_1_x2 = nn.Conv2d(64, 32, kernel_size=1, stride=2)
        self.psi_2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        
        self.out_dec3 = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        self.out_dec2 = nn.Conv2d(128, 1, kernel_size=1, stride=1)
        self.out_dec1 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
            
        # ResNet-34 채널 수(512, 256, 128, 64, 64)
        self.dec4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4_conv1 = conv_block(256 + 256, 256)
        self.dec4_conv2 = conv_block(256, 256)

        self.dec3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3_conv1 = conv_block(128 + 128, 128)
        self.dec3_conv2 = conv_block(128, 128)

        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2_conv1 = conv_block(64 + 64, 64)
        self.dec2_conv2 = conv_block(64, 64)

        self.dec1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1_conv1 = conv_block(64 + 64, 64) 
        self.dec1_conv2 = conv_block(64, 64)
        
        self.final_upconv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv1 = conv_block(32, 32)
        self.final_conv2 = conv_block(32, 32)
        self.final = nn.Conv2d(32, 1, kernel_size=1, stride=1)

    def forward(self, x):

        e0, e1, e2, e3, e4 = self.backbone(x)
        

        dec4 = self.dec4(e4)
        g4=self.conv_1_g4(e4) #g4는 conv feature map. channel을 512->128
        x4=self.conv_1_x4(e3) #x4는 skip connection으로 넘어온 feature map. 해상도 1/2, channel 256->128
        psi_4=self.psi_4(g4+x4)
        psi_4_upsampled = F.interpolate(psi_4, size=e3.size()[2:], mode='bilinear', align_corners=False)
        attention_e3 = e3 * psi_4_upsampled

        dec4 = torch.cat((dec4, attention_e3), dim=1)
        dec4 = self.dec4_conv1(dec4)
        dec4 = self.dec4_conv2(dec4)
        out3 = self.out_dec3(dec4) #deep supervision


        dec3 = self.dec3(dec4)
        g3=self.conv_1_g3(dec4)
        x3=self.conv_1_x3(e2)
        psi_3=self.psi_3(g3+x3)
        psi_3_upsampled = F.interpolate(psi_3, size=e2.size()[2:], mode='bilinear', align_corners=False)
        attention_e2 = e2 * psi_3_upsampled

        dec3 = torch.cat((dec3, attention_e2), dim=1)
        dec3 = self.dec3_conv1(dec3)
        dec3 = self.dec3_conv2(dec3)
        out2 = self.out_dec2(dec3)


        dec2 = self.dec2(dec3)
        g2=self.conv_1_g2(dec3)
        x2=self.conv_1_x2(e1)
        psi_2=self.psi_2(g2+x2)
        psi_2_upsampled = F.interpolate(psi_2, size=e1.size()[2:], mode='bilinear', align_corners=False)
        attention_e1 = e1 * psi_2_upsampled

        dec2 = torch.cat((dec2, attention_e1), dim=1)
        dec2 = self.dec2_conv1(dec2)
        dec2 = self.dec2_conv2(dec2)
        out1 = self.out_dec1(dec2)


        dec1 = self.dec1(dec2)
        dec1 = torch.cat((dec1, e0), dim=1)
        dec1 = self.dec1_conv1(dec1)
        dec1 = self.dec1_conv2(dec1)

        out = self.final_upconv(dec1)
        out = self.final_conv1(out)
        out = self.final_conv2(out)

        out = self.final(out)
        return out,out1,out2,out3 #0 1 2 3 순서로 해상도가 낮아짐


class Unetplus(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.conv2_1 = nn.Conv2d(256,128, kernel_size=1, stride=1)
        self.dense2_1 = conv_block(128 + 128, 128)
        self.dense2_2 = conv_block(128, 128)

        self.conv1_1 = nn.Conv2d(128,64, kernel_size=1, stride=1)
        self.dense1_1 = conv_block(64 + 64, 64)
        self.dense1_1_2 = conv_block(64, 64)

        self.conv1_2 = nn.Conv2d(128,64, kernel_size=1, stride=1)
        self.dense1_2 = conv_block(64 + 64 + 64, 64)
        self.dense1_2_2 = conv_block(64, 64)

        self.dense0_1 = conv_block(64 + 64, 64)
        self.dense0_1_2 = conv_block(64, 64)

        self.dense0_2 = conv_block(64 + 64 + 64, 64)
        self.dense0_2_2 = conv_block(64, 64)

        self.dense0_3 = conv_block(64 + 64 + 64 + 64, 64)
        self.dense0_3_2 = conv_block(64, 64)


        self.out_x0_1 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        self.out_x0_2 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        self.out_x0_3 = nn.Conv2d(64, 1, kernel_size=1, stride=1)

            
        # ResNet-34 채널 수(512, 256, 128, 64, 64)
        self.dec4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4_conv1 = conv_block(256 + 256, 256)
        self.dec4_conv2 = conv_block(256, 256)

        self.dec3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3_conv1 = conv_block(128 + 128 + 128, 128)
        self.dec3_conv2 = conv_block(128, 128)

        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2_conv1 = conv_block(64 + 64 + 64 + 64, 64)
        self.dec2_conv2 = conv_block(64, 64)

        self.dec1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1_conv1 = conv_block(64 + 64 + 64 + 64 + 64, 64) 
        self.dec1_conv2 = conv_block(64, 64)
        
        self.final_upconv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv1 = conv_block(32, 32)
        self.final_conv2 = conv_block(32, 32)
        self.final = nn.Conv2d(32, 1, kernel_size=1, stride=1)

    def forward(self, x):

        e0, e1, e2, e3, e4 = self.backbone(x)
        

        dec4 = self.dec4(e4) #upconv
        dec4 = torch.cat((dec4, e3), dim=1) #skip connection
        dec4 = self.dec4_conv1(dec4) #conv
        dec4 = self.dec4_conv2(dec4)

        dec3 = self.dec3(dec4)

        x2_1 = self.conv2_1(e3)
        x2_1 = F.interpolate(x2_1, size=e2.size()[2:], mode='bilinear', align_corners=False)   
        x2_1 = torch.cat((x2_1, e2), dim=1)
        x2_1 = self.dense2_1(x2_1)
        x2_1 = self.dense2_2(x2_1)

        dec3 = torch.cat((dec3, x2_1, e2), dim=1)
        dec3 = self.dec3_conv1(dec3)
        dec3 = self.dec3_conv2(dec3)

        dec2 = self.dec2(dec3)

        x1_1 = self.conv1_1(e2)
        x1_1 = F.interpolate(x1_1, size=e1.size()[2:], mode='bilinear', align_corners=False)
        x1_1 = torch.cat((x1_1, e1), dim=1)
        x1_1 = self.dense1_1(x1_1)
        x1_1 = self.dense1_1_2(x1_1)

        x1_2 = self.conv1_2(x2_1)
        x1_2 = F.interpolate(x1_2, size=e1.size()[2:], mode='bilinear', align_corners=False)
        x1_2 = torch.cat((x1_2, e1, x1_1), dim=1)
        x1_2 = self.dense1_2(x1_2)
        x1_2 = self.dense1_2_2(x1_2)

        dec2 = torch.cat((dec2, e1, x1_1, x1_2), dim=1)
        dec2 = self.dec2_conv1(dec2)
        dec2 = self.dec2_conv2(dec2)

        dec1 = self.dec1(dec2)

        x0_1 = F.interpolate(e1, size=e0.size()[2:], mode='bilinear', align_corners=False)
        x0_1 = torch.cat((x0_1, e0), dim=1)
        x0_1 = self.dense0_1(x0_1)
        x0_1 = self.dense0_1_2(x0_1)

        out_x0_1 = self.out_x0_1(x0_1)

        x0_2 = F.interpolate(x1_1, size=e0.size()[2:], mode='bilinear', align_corners=False)
        x0_2 = torch.cat((x0_2, e0, x0_1), dim=1)
        x0_2 = self.dense0_2(x0_2)
        x0_2 = self.dense0_2_2(x0_2)

        out_x0_2 = self.out_x0_2(x0_2)

        x0_3 = F.interpolate(x1_2, size=e0.size()[2:], mode='bilinear', align_corners=False)
        x0_3 = torch.cat((x0_3, e0, x0_1, x0_2), dim=1)
        x0_3 = self.dense0_3(x0_3)
        x0_3 = self.dense0_3_2(x0_3)

        out_x0_3 = self.out_x0_3(x0_3)

        dec1 = torch.cat((dec1, e0, x0_1, x0_2, x0_3), dim=1)
        dec1 = self.dec1_conv1(dec1)
        dec1 = self.dec1_conv2(dec1)

        out = self.final_upconv(dec1)
        out = self.final_conv1(out)
        out = self.final_conv2(out)

        out = self.final(out)
        return out, out_x0_3, out_x0_2, out_x0_1 #4 3 2 1 순서로 더 deep한 supervision