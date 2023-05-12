import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.anchors import Anchors

from nets.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

class conv_block(nn.Module):
    def __init__(self,in_c,out_c):
        super(conv_block,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=(3,3),stride=1,padding=1,padding_mode='reflect'),
            nn.BatchNorm2d(out_c),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=(3, 3), stride=1, padding=1, padding_mode='reflect',bias = False),
            nn.BatchNorm2d(out_c),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Upsample(nn.Module):
    def __init__(self,channel):
        super(Upsample, self).__init__()
        self.conv1 = nn.Conv2d(channel,channel//2,kernel_size=(1,1),stride=1)

    def forward(self,x):
        x = F.interpolate(x,scale_factor=2,mode='nearest')
        x = self.conv1(x)
        return x
class Upsample_1(nn.Module):
    def __init__(self,channel):
        super(Upsample_1, self).__init__()
        self.conv1 = nn.Conv2d(channel,channel//2,kernel_size=(1,1),stride=1)

    def forward(self,x):
        x = F.interpolate(x,size=[75,75],mode='nearest')
        x = self.conv1(x)
        return x

class UNET(nn.Module):
    def __init__(self,in_channel=128,out_channel=128):
        super(UNET, self).__init__()
        self.layer9 = conv_block(2048,2048)
        self.layer10 = Upsample(2048)
        self.layer11 = conv_block(1024,1024)
        self.layer12 = Upsample_1(1024)
        self.layer13 = conv_block(512,512)
        self.layer14 = Upsample(512)
        self.layer15 = conv_block(256,256)
        self.layer16 = Upsample(256)
        self.layer17 = conv_block(128,128)
        self.layer18 = Upsample(128)
        self.layer19 = conv_block(64,64)
        self.layer20 = nn.Conv2d(64,3,kernel_size=(1,1),stride=1)
        self.act = nn.Sigmoid()

    def forward(self,p3, p4, p5):
        x = self.layer9(p5)
        x_p5 = p5 + x
        x = self.layer10(x)
        x = self.layer11(x)
        x_p4 = p4 + x
        x = self.layer12(x)
        x = self.layer13(x)
        x_p3 = p3 + x
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)
        out = self.act(x)
        return x_p3,x_p4,x_p5,out
class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        _, _, h4, w4 = C4.size()
        _, _, h3, w3 = C3.size()

        # 75,75,512 -> 75,75,256
        P3_x = self.P3_1(C3)
        # 38,38,1024 -> 38,38,256
        P4_x = self.P4_1(C4)
        # 19,19,2048 -> 19,19,256
        P5_x = self.P5_1(C5)

        # 19,19,256 -> 38,38,256
        P5_upsampled_x = F.interpolate(P5_x, size=(h4, w4))
        # 38,38,256 + 38,38,256 -> 38,38,256
        P4_x = P5_upsampled_x + P4_x
        # 38,38,256 -> 75,75,256
        P4_upsampled_x = F.interpolate(P4_x, size=(h3, w3))
        # 75,75,256 + 75,75,256 -> 75,75,256
        P3_x = P3_x + P4_upsampled_x

        # 75,75,256 -> 75,75,256
        P3_x = self.P3_2(P3_x)
        # 38,38,256 -> 38,38,256
        P4_x = self.P4_2(P4_x)
        # 19,19,256 -> 19,19,256
        P5_x = self.P5_2(P5_x)

        # 19,19,2048 -> 10,10,256
        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        # 10,10,256 -> 5,5,256
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1  = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1   = nn.ReLU()

        self.conv2  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2   = nn.ReLU()

        self.conv3  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3   = nn.ReLU()

        self.conv4  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4   = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=24, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1  = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1   = nn.ReLU()

        self.conv2  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2   = nn.ReLU()

        self.conv3  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3   = nn.ReLU()

        self.conv4  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4   = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, height, width, channels = out1.shape

        out2 = out1.view(batch_size, height, width, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class Resnet(nn.Module):
    def __init__(self, phi, pretrained=False):
        super(Resnet, self).__init__()
        self.edition = [resnet18, resnet34, resnet50, resnet101, resnet152]
        model = self.edition[phi](pretrained)
        del model.avgpool
        del model.fc
        self.model = model

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        feat1 = self.model.layer2(x)
        feat2 = self.model.layer3(feat1)
        feat3 = self.model.layer4(feat2)

        return [feat1,feat2,feat3]

class retinanet(nn.Module):
    def __init__(self, num_classes, phi, pretrained=False, fp16=False):
        super(retinanet, self).__init__()
        self.pretrained = pretrained
        #-----------------------------------------#
        #   取出三个有效特征层，分别是C3、C4、C5
        #   假设输入图像为600,600,3
        #   当我们使用resnet50的时候
        #   C3     75,75,512
        #   C4     38,38,1024
        #   C5     19,19,2048
        #-----------------------------------------#
        self.backbone_net = Resnet(phi, pretrained)
        fpn_sizes = {
            0: [128, 256, 512],
            1: [128, 256, 512],
            2: [512, 1024, 2048],
            3: [512, 1024, 2048],
            4: [512, 1024, 2048],
        }[phi]
        self.u_net = UNET()
        #-----------------------------------------#
        #   经过FPN可以获得5个有效特征层分别是
        #   P3     75,75,256
        #   P4     38,38,256
        #   P5     19,19,256
        #   P6     10,10,256
        #   P7     5,5,256
        #-----------------------------------------#
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        #----------------------------------------------------------#
        #   将获取到的P3, P4, P5, P6, P7传入到
        #   Retinahead里面进行预测，获得回归预测结果和分类预测结果
        #   将所有特征层的预测结果进行堆叠
        #----------------------------------------------------------#
        self.regressionModel        = RegressionModel(256)
        self.classificationModel    = ClassificationModel(256, num_classes=num_classes)
        self.anchors = Anchors()
        self._init_weights(fp16)

    def _init_weights(self, fp16):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        
        prior = 0.01
        self.classificationModel.output.weight.data.fill_(0)
        if fp16:
            self.classificationModel.output.bias.data.fill_(-2.9)
        else:
            self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

    def forward(self, inputs):
        #-----------------------------------------#
        #   取出三个有效特征层，分别是C3、C4、C5
        #   C3     75,75,512
        #   C4     38,38,1024
        #   C5     19,19,2048
        #-----------------------------------------#
        p3, p4, p5 = self.backbone_net(inputs)
        p3, p4, p5,unet_out = self.u_net(p3, p4, p5)

        #-----------------------------------------#
        #   经过FPN可以获得5个有效特征层分别是
        #   P3     75,75,256
        #   P4     38,38,256
        #   P5     19,19,256
        #   P6     10,10,256
        #   P7     5,5,256
        #-----------------------------------------#
        features = self.fpn([p3, p4, p5])

        #----------------------------------------------------------#
        #   将获取到的P3, P4, P5, P6, P7传入到
        #   Retinahead里面进行预测，获得回归预测结果和分类预测结果
        #   将所有特征层的预测结果进行堆叠
        #----------------------------------------------------------#
        regression      = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        classification  = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(features)

        return features, regression, classification, anchors,unet_out
