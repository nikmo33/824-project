import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from aspp import build_aspp
from Loss import Decoder
from backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=8, num_classes=10,
                 sync_bn=False, freeze_bn=False):
        super(DeepLab, self).__init__()

        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        normalize,
        ])
        self.transform = transform

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        return x


if __name__ == "__main__":
    model = DeepLab(backbone='resnet', output_stride=8)
    model.eval()    
    image = model.transform(Image.open('demo_img.jpg').convert('RGB'))
    input = image.unsqueeze(0)
    output = model.forward(input)
    dec = Decoder(10)
    seg_out, depth_out = dec.forward(output)